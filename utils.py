import secrets
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import scipy.signal
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

import scipy.stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC

VALID_BASES = {"scroll-up", "scroll-down", "swipe-left", "swipe-right", "texting"}
VALID_USERS = {"a", "b", "s"}
VALID_SENSORS_PER_FILE = {"acc", "gyr"}
VARIANT_TO_FINGER_STYLE = {
    "thumb":  ("thumb", "na"),
    "index":  ("index", "na"),
    "two":    ("na",    "two_handed"),
    "single": ("na",    "single_handed"),
}
SIX_AXES = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")


# -------------------------------------------------------------------------- #
# Next phases candidate helpers (EDA, Scaling, Feature Engineering, Evaluation)
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
# Raw CSV → merged 6-axis documents, 
# -------------------------------------------------------------------------- #

def parse_filename(name: str) -> dict:
    """Parses a raw session CSV filename into metadata fields.

    Expected format (7 underscore-separated tokens):
        {base}_{variant}_{hand}_{sr}_{sensor}_{primary}_{user}.csv

    Args:
        name: CSV filename or path.

    Returns:
        Dict with gesture_id, finger, typing_style, hand, sr, sensor (acc|gyr),
        primary, user. The runtime fields `session_id`, `spontaneous` and the
        merged `sensor="AccGyr"` are filled by `build_document`.

    Raises:
        ValueError: If the filename does not match the expected format.
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 7:
        raise ValueError(f"Bad filename {name!r}: expected 7 tokens, got {len(parts)}")
    base, variant, hand, sr, sensor, primary, user = parts
    if base not in VALID_BASES:
        raise ValueError(f"Bad base in {name!r}: {base!r}")
    if variant not in VARIANT_TO_FINGER_STYLE:
        raise ValueError(f"Bad variant in {name!r}: {variant!r}")
    if base == "texting" and variant not in {"two", "single"}:
        raise ValueError(f"texting requires two/single, got {variant!r}")
    if base != "texting" and variant not in {"thumb", "index"}:
        raise ValueError(f"{base} requires thumb/index, got {variant!r}")
    if sensor not in VALID_SENSORS_PER_FILE:
        raise ValueError(f"Bad sensor in {name!r}: {sensor!r}")
    if user not in VALID_USERS:
        raise ValueError(f"Bad user in {name!r}: {user!r}")
    finger, typing_style = VARIANT_TO_FINGER_STYLE[variant]
    return {
        "gesture_id":   base,
        "finger":       finger,
        "typing_style": typing_style,
        "hand":         int(hand),
        "sr":           int(sr),
        "sensor":       sensor,
        "primary":      int(primary),
        "user":         user,
    }


def session_key(meta: dict) -> tuple:
    """Returns the pairing key (everything except `sensor`)."""
    return (meta["gesture_id"], meta["finger"], meta["typing_style"],
            meta["hand"], meta["primary"], meta["user"])


def discover_csvs(data_root) -> list:
    """Returns every CSV under data_root, recursively, sorted."""
    return sorted(Path(data_root).rglob("*.csv"))


def pair_acc_gyr(csv_paths: Iterable) -> Iterator:
    """Pairs acc/gyr files that share the same session key.

    Args:
        csv_paths: Iterable of CSV paths.

    Yields:
        (shared_meta, acc_path, gyr_path) for each complete pair. Files with
        an unparseable name or no counterpart are skipped with a warning.
    """
    by_key: dict = defaultdict(dict)
    for path in csv_paths:
        try:
            meta = parse_filename(Path(path).name)
        except ValueError as exc:
            print(f"SKIP {Path(path).name}: {exc}")
            continue
        by_key[session_key(meta)][meta["sensor"]] = (Path(path), meta)

    for key, sensors in by_key.items():
        missing = {"acc", "gyr"} - sensors.keys()
        if missing:
            print(f"SKIP {key}: missing {missing}")
            continue
        acc_path, acc_meta = sensors["acc"]
        gyr_path, _        = sensors["gyr"]
        shared = {k: v for k, v in acc_meta.items() if k != "sensor"}
        yield shared, acc_path, gyr_path


def load_acc_csv(path) -> pd.DataFrame:
    """Loads a raw accelerometer CSV (Epoch,X,Y,Z) and renames axes to acc_*.

    The wearable occasionally emits multiple samples sharing the same Epoch
    millisecond (~25% of rows). Rather than dropping them (data loss) or
    keeping duplicate keys (which break `merge_asof`), each duplicate is
    nudged forward to `last + 1 ms`. The cumulative drift stays well within
    the 10 ms tolerance used by `merge_acc_gyr`, so acc↔gyr alignment is
    preserved.
    """
    df = pd.read_csv(path, encoding='utf-8-sig', engine='python')
    df = df.rename(columns={"X": "acc_x", "Y": "acc_y", "Z": "acc_z"})
    df = df.sort_values("Epoch")

    new_epochs = []
    last_val = -1
    for val in df["Epoch"]:
        last_val = max(val, last_val + 1)
        new_epochs.append(last_val)

    df["Epoch"] = new_epochs
    return df.reset_index(drop=True)


def load_gyr_csv(path) -> pd.DataFrame:
    """Loads a raw gyroscope CSV (Epoch,X,Y,Z) and renames axes to gyr_*.

    Same monotonic-epoch fix as `load_acc_csv` — duplicate timestamps are
    nudged by 1 ms to keep `merge_asof` happy without dropping samples.
    """
    df = pd.read_csv(path, encoding='utf-8-sig', engine='python')
    df = df.rename(columns={"X": "gyr_x", "Y": "gyr_y", "Z": "gyr_z"})
    df = df.sort_values("Epoch")

    new_epochs = []
    last_val = -1
    for val in df["Epoch"]:
        last_val = max(val, last_val + 1)
        new_epochs.append(last_val)

    df["Epoch"] = new_epochs
    return df.reset_index(drop=True)


def merge_acc_gyr(acc_df: pd.DataFrame, gyr_df: pd.DataFrame,
                  tolerance_ms: int = 10) -> pd.DataFrame:
    """Aligns acc/gyr by Epoch with merge_asof; drops rows missing any axis."""
    merged = pd.merge_asof(acc_df, gyr_df, on="Epoch",
                            direction="nearest", tolerance=tolerance_ms)
    return (merged
              .dropna(subset=list(SIX_AXES))
              .reset_index(drop=True))


# -------------------------------------------------------------------------- #
# Filtering, mongo connection , encoding 
# -------------------------------------------------------------------------- #

def apply_continuous_filter(df: pd.DataFrame, order: int = 4,
                              wn: float = 0.2) -> pd.DataFrame:
    """Zero-phase Butterworth lowpass on the 6 axes using SOS format.
    
    Args:
        df: DataFrame containing the SIX_AXES columns.
        order: Butterworth order. Defaults to 4.
        wn: Normalized critical frequency 2*fc/fs. fc=10Hz @ fs=100Hz → 0.2.

    Returns:
        A copy of df with the 6 axes filtered. Other columns are preserved.
    """
    sos = butter(order, wn, btype="lowpass", output="sos")
    out = df.copy()

    for col in SIX_AXES:
        data = pd.to_numeric(df[col], errors='coerce').values
        
        # Replace NaNs with 0 to avoid issues with sosfiltfilt
        data = pd.Series(data).interpolate(limit_direction='both').fillna(0).values
        out[col] = sosfiltfilt(sos, data, padlen=0)        
    return out


def build_document(df: pd.DataFrame, meta: dict) -> dict:
    """Builds the MongoDB document from a 6-axis DataFrame and shared metadata."""
    return {
        "session_id":   secrets.token_hex(12),
        "data":         {col: df[col].tolist() for col in SIX_AXES},
        "gesture_id":   meta["gesture_id"],
        "finger":       meta["finger"],
        "typing_style": meta["typing_style"],
        "hand":         meta["hand"],
        "sr":           meta["sr"],
        "sensor":       "AccGyr",
        "primary":      meta["primary"],
        "spontaneous":  0,
        "user":         meta["user"],
        "datetime":     datetime.now(timezone.utc),
    }


def mongo_connect(uri: str = "mongodb://localhost:27017/",
                   db: str = "aiot_gestures",
                   collection: str = "sessions",
                   reset: bool = False) -> Collection:
    """Returns a MongoDB collection handle, optionally clearing existing docs."""
    db = db.strip().replace(" ", "_")
    collection = collection.strip().replace(" ", "_")
    client = MongoClient(uri)
    coll = MongoClient(uri)[db][collection]
    if reset:
        coll.delete_many({})
    coll.create_index([("session_id", ASCENDING)], unique=True)
    return coll


def insert_documents(coll: Collection, docs: Iterable) -> int:
    """Inserts each document; returns the number successfully inserted."""
    inserted = 0
    for doc in docs:
        try:
            coll.insert_one(doc)
            inserted += 1
        except DuplicateKeyError as exc:
            tag = f"{doc.get('user')}/{doc.get('gesture_id')}/{doc.get('finger')}"
            print(f"SKIP insert ({tag}): duplicate session_id ({exc})")
    return inserted


# -------------------------------------------------------------------------- #
# Phase 3 & 4 Helpers (EDA, Scaling, Feature Engineering, Evaluation)
# -------------------------------------------------------------------------- #

OUT_FIG = Path("outputs/figures")
OUT_RES = Path("outputs/results")
OUT_CACHE = Path("outputs/cache")

for p in (OUT_FIG, OUT_RES, OUT_CACHE):
    p.mkdir(parents=True, exist_ok=True)


def sliding_window_pd(
        df,
        ws=200,
        overlap=100,
        w_type="hann",
        w_center=True,
        print_stats=True
) -> list:
    """Applies the sliding window algorithm to the DataFrame rows.

    Args:
        df: The DataFrame with all the values that will be inserted to the
            sliding window algorithm.
        ws: The window size in number of samples.
        overlap: The hop length in number of samples.
        w_type: The windowing function.
        w_center: If False, set the window labels as the right edge of the
            window index. If True, set the window labels as the center of the
            window index.
        print_stats: Print statistical inferences from the process. Defaults
            to False.

    Returns:
        A list of DataFrames each one corresponding to a produced window.
    """
    windows_counter = 0
    windows_list = list()
    # min_periods: minimum number of observations in window required to have a value 
    # For a window that is specified by an integer, min_periods will default to the size of the window.
    for window in df.rolling(window=ws, step=overlap, min_periods=ws,
                             win_type=w_type, center=w_center):
        if window[window.columns[0]].count() >= ws:
            if print_stats:
                print("Print Window:", windows_counter)
                print("Number of samples:", window[window.columns[0]].count())
            windows_list.append(window)
        windows_counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))

    return windows_list


def save_figure(fig, name: str, dpi: int = 150) -> Path:
    """Saves a matplotlib figure to outputs/figures/as PNG.

    Args:
        fig: The matplotlib Figure object to save.
        name: The filename stem (without extension).
        dpi: Resolution in dots per inch. Defaults to 150.

    Returns:
        The absolute Path of the saved file.
    """
    path = OUT_FIG / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def apply_filter(
        arr,
        order=5,
        wn=0.1,
        filter_type="lowpass"
) -> np.ndarray:
    """Applies a zero-phase Butterworth filter to a multi-axis signal.

    Args:
        arr: The initial NumPy signal array values.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {'lowpass', 'highpass', 'bandpass',
            'bandstop'}

    Returns:
        NumPy array with the filtered signal.
    """
    sos = scipy.signal.butter(N=order, Wn=wn, btype=filter_type, output="sos")
    return scipy.signal.sosfiltfilt(sos=sos, x=arr, padlen=0)


def filter_instances(instances_list, order, wn, filter_type) -> list:
    """Applies `apply_filter` to a list of windowed DataFrames.

    Args:
        instances_list: List of DataFrames.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter.

    Returns:
        List of filtered DataFrames.
    """
    filtered_instances_list = []
    for item in instances_list:
        filtered_instance = item.apply(apply_filter,
                                       args=(order, wn, filter_type))
        filtered_instances_list.append(filtered_instance)
    print("Number of filtered instances in the list:",
          len(filtered_instances_list))
    return filtered_instances_list


def flatten_instances_df(instances_list: list) -> pd.DataFrame:
    """Flattens each instance and returns a DataFrame of flattened rows.

    Args:
        instances_list: The list of DataFrames to flatten.

    Returns:
        DataFrame whose rows are the flattened instances.
    """
    flattened = [item.to_numpy().flatten() for item in instances_list]
    return pd.DataFrame(flattened)


def are_lists_equal(list1: list, list2: list) -> bool:
    return set(list1) == set(list2)


def df_rebase(df: pd.DataFrame, target_list: list, ref_list: list) -> pd.DataFrame:
    """Reorders and renames DataFrame columns to project-standard names.

    Args:
        df: The pandas DataFrame.
        target_list: Source column names in the desired order.
        ref_list: Replacement column names (same length as `target_list`).

    Returns:
        DataFrame with the new column order and names.
    """
    print("Initial columns:", list(df.columns))
    if are_lists_equal(list(df.columns), ref_list):
        pass
    else:
        if len(target_list) == len(ref_list):
            df = df[target_list]
            df = df.rename(columns=dict(zip(target_list, ref_list)))
        else:
            print("The length of the target list and the reference list is not equal.")
    print("Processed columns:", list(df.columns))
    return df


def rename_df_column_values(
    np_array: np.ndarray,
    y: list,
    columns_names: tuple = ("acc_x", "acc_y", "acc_z")
):
    """Builds a DataFrame with a label column whose values are replaced by
    the index of each unique label.

    Args:
        np_array: 2D NumPy array.
        y: List of labels.
        columns_names: Names for the value columns.

    Returns:
        DataFrame with the values and the integer-encoded `y` column.
    """
    arr_y = np.array(y)
    unique_values_list = np.unique(arr_y)
    df = pd.DataFrame(np_array, columns=columns_names)
    df["y"] = y
    for idx, x in enumerate(unique_values_list):
        df["y"] = np.where(df["y"] == x, idx, df["y"])
    return df


def save_results(df, name: str) -> Path:
    """Saves a DataFrame as CSV in outputs/results/.

    Args:
        df: The pandas DataFrame to save.
        name: The filename stem (without extension).

    Returns:
        The absolute Path of the saved CSV.
    """
    path = OUT_RES / f"{name}.csv"
    df.to_csv(path, index=False)
    return path



#Is it better for our task?
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler


def MixedScaling(train_mask, test_mask, windows):
    feature_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

    Unfold3D = np.array([win[feature_cols].values for win in windows])
    n_windows, n_timesteps, n_features = Unfold3D.shape

    X_train_3d = Unfold3D[train_mask]
    X_test_3d = Unfold3D[test_mask]

    X_train_2d = X_train_3d.reshape(-1, n_features)
    X_test_2d = X_test_3d.reshape(-1, n_features)

    scaler = ColumnTransformer(transformers=[
        ('accel_scale', StandardScaler(), [0, 1, 2]),  # Προστέθηκε το [0, 1, 2]
        ('gyro_scale', RobustScaler(), [3, 4, 5])  # Προστέθηκε το [3, 4, 5]
    ])

    X_train_2d_scaled = scaler.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler.transform(X_test_2d)

    train_set = X_train_2d_scaled.reshape(X_train_3d.shape[0], n_timesteps * n_features)
    test_set = X_test_2d_scaled.reshape(X_test_3d.shape[0], n_timesteps * n_features)

    print(f"Έτοιμα Train δεδομένα με σχήμα: {train_set.shape}")
    print(f"Έτοιμα Test δεδομένα με σχήμα: {test_set.shape}")

    return train_set, test_set


#------------------- test ML models ----------------#

def GradientBoosting(train_set,train_labels,test_set,test_labels):
    gb_model = HistGradientBoostingClassifier(random_state=42)
    gb_model.fit(train_set, train_labels)
    print("✔️ Το Gradient Boosting εκπαιδεύτηκε επιτυχώς.")

    y_pred = gb_model.predict(test_set)


    return gb_model,y_pred

def SVM(train_set,train_labels,test_set,test_labels):
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)

    print("Έναρξη εκπαίδευσης SVM... Παρακαλώ περιμένετε.")
    svm_model.fit(train_set, train_labels)
    print("✔️ Το SVM εκπαιδεύτηκε επιτυχώς.")


    y_pred = svm_model.predict(test_set)

    return svm_model,y_pred

def LogicRegression(train_set,train_labels,test_set,test_labels):
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

    print("Έναρξη εκπαίδευσης Logistic Regression... Παρακαλώ περιμένετε.")
    lr_model.fit(train_set, train_labels)
    print("✔️ Η Λογιστική Παλινδρόμηση εκπαιδεύτηκε επιτυχώς.")

    y_pred = lr_model.predict(test_set)

    return lr_model,y_pred

# Fine Tunning Grid Search ---> do that to go faster
def Fine_Tunning(model_name,train_set,train_labels,test_set,test_labels):
    X_all = np.vstack((train_set, test_set))
    y_all = np.concatenate((train_labels, test_labels))

    test_fold = np.zeros(X_all.shape[0])
    test_fold[:train_set.shape[0]] = -1
    ps = PredefinedSplit(test_fold=test_fold)

    if model_name == 'gradient_boosting':
        estimator = HistGradientBoostingClassifier(random_state=42)
        param_grid = {
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, None],
            'l2_regularization': [0.0, 1.0, 10.0]
        }

    elif model_name == 'svm':
        estimator = SVC(random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01],
            'kernel': ['rbf', 'linear']
        }

    elif model_name == 'logistic_regression':
        estimator = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],  # Αντίστροφη ποινή regularization
            'penalty': ['l2'],  # L2 Regularization (Ridge)
            'solver': ['lbfgs', 'sag']
        }
    else:
        raise ValueError("Λάθος model_name. Επιλέξτε 'gradient_boosting', 'svm' ή 'logistic_regression'")

    print(f"\n==================================================")
    print(f" ΕΝΑΡΞΗ GRID SEARCH: {model_name.upper()}")
    print(f"==================================================")

    # 4. Αρχικοποίηση και εκτέλεση του GridSearchCV
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=ps,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_all, y_all)

    print(f"✔️ Το Grid Search για το {model_name} ολοκληρώθηκε!")
    print(f"-> Καλύτερες Παράμετροι: {grid_search.best_params_}")
    print(f"-> Καλύτερο Validation F1-Score: {grid_search.best_score_:.4f}\n")

    return grid_search.best_estimator_
