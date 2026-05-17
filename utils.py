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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import f_classif

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


def extract_features(window: pd.DataFrame, fs: int = 100) -> dict:
    """Extracts time-domain, spectral, and cross-channel features for a window.

    Per axis (6 axes): mean, std, rms, min, max, median, IQR, skewness,
    kurtosis, ZCR, MAD, dominant freq, spectral energy, mean freq, spectral
    entropy. Per sensor (acc/gyr): SMA, vector-magnitude mean/std, pairwise
    axis correlations. Total approximately 102 features.

    Args:
        window: DataFrame with the 6 sensor columns (``acc_x``..``gyr_z``)
            and ``ws`` rows.
        fs: Sampling frequency in Hz. Defaults to 100.

    Returns:
        Dict mapping ``{axis}_{feat}`` (and ``{sensor}_{feat}`` for
        cross-channel) to scalar values.
    """
    feats = {}
    for col in SIX_AXES:
        x = window[col].values
        feats[f"{col}_mean"]   = x.mean()
        feats[f"{col}_std"]    = x.std()
        feats[f"{col}_rms"]    = np.sqrt((x ** 2).mean())
        feats[f"{col}_min"]    = x.min()
        feats[f"{col}_max"]    = x.max()
        feats[f"{col}_median"] = np.median(x)
        feats[f"{col}_iqr"]    = np.percentile(x, 75) - np.percentile(x, 25)
        feats[f"{col}_skew"]   = scipy.stats.skew(x)
        feats[f"{col}_kurt"]   = scipy.stats.kurtosis(x)
        feats[f"{col}_zcr"]    = ((x[:-1] * x[1:]) < 0).mean()
        feats[f"{col}_mad"]    = np.mean(np.abs(x - x.mean()))
        X = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), 1 / fs)
        p = X ** 2 / max((X ** 2).sum(), 1e-12)
        feats[f"{col}_dom_freq"]     = freqs[np.argmax(X)]
        feats[f"{col}_spec_energy"]  = (X ** 2).sum()
        feats[f"{col}_mean_freq"]    = (freqs * p).sum()
        feats[f"{col}_spec_entropy"] = -(p[p > 0] * np.log2(p[p > 0])).sum()
    for sensor in ("acc", "gyr"):
        x = window[f"{sensor}_x"]
        y = window[f"{sensor}_y"]
        z = window[f"{sensor}_z"]
        feats[f"{sensor}_sma"]     = (x.abs() + y.abs() + z.abs()).mean()
        vm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        feats[f"{sensor}_vm_mean"] = vm.mean()
        feats[f"{sensor}_vm_std"]  = vm.std()
        feats[f"{sensor}_corr_xy"] = np.corrcoef(x, y)[0, 1]
        feats[f"{sensor}_corr_xz"] = np.corrcoef(x, z)[0, 1]
        feats[f"{sensor}_corr_yz"] = np.corrcoef(y, z)[0, 1]
    return feats


def anova_rank_features(X: pd.DataFrame, y) -> pd.DataFrame:
    """Ranks features by ANOVA F-statistic.

    F = MSB / MSW. High F means class means are far apart relative to
    within-class scatter, so the feature separates classes well. Low F
    (around 1) means the feature carries no discriminative information.

    HAR caveat: windows from the same subject are not independent. Use
    F only for ranking; do not take p-values at face value.

    Args:
        X: Feature DataFrame (N rows by F columns). Train set only;
            the held-out subject must not be present.
        y: 1D label array of length N.

    Returns:
        DataFrame with columns ``feature``, ``F``, ``p_value`` sorted by
        ``F`` descending. Constant (zero-variance) columns yield ``F=NaN``;
        ``find_highly_correlated`` drops those downstream.
    """
    import warnings
    with warnings.catch_warnings():
        # sklearn warns when a column has zero variance (UserWarning) and
        # numpy warns about the resulting 0/0 division (RuntimeWarning).
        # Both are expected here: the NaN F-score is the signal that
        # find_highly_correlated uses to drop the column.
        warnings.filterwarnings("ignore", message="Features .* are constant.")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        F, p = f_classif(X.values, y)
    return (pd.DataFrame({"feature": X.columns, "F": F, "p_value": p})
              .sort_values("F", ascending=False)
              .reset_index(drop=True))


def correlation_diagnostic(X: pd.DataFrame, threshold: float = 0.9) -> dict:
    """Computes Pearson and Spearman correlation matrices plus a high-pairs
    diagnostic table.

    Classifies each highly correlated pair as:
      - ``linear``               if both ``|r_pearson|`` and ``|r_spearman|``
        exceed the threshold,
      - ``monotonic_nonlinear``  if only the Spearman magnitude exceeds it
        (rank order is preserved but the relationship is not linear),
      - ``outlier_driven``       if only the Pearson magnitude exceeds it
        (a few extreme points inflate the linear correlation).

    Pairs with both magnitudes below the threshold are treated as
    independent and omitted from the diagnostic table.

    Args:
        X: Feature DataFrame (numeric columns only).
        threshold: Absolute correlation cutoff for "highly correlated"
            pairs. Defaults to 0.9.

    Returns:
        Dict with keys:
          - ``pearson``    (DataFrame): Pearson correlation matrix.
          - ``spearman``   (DataFrame): Spearman correlation matrix.
          - ``high_pairs`` (DataFrame): columns ``a``, ``b``,
            ``r_pearson``, ``r_spearman``, ``kind``; sorted by the larger
            of the two absolute correlations, descending. Empty if no
            pair clears the threshold.
    """
    p = X.corr(method="pearson")
    s = X.corr(method="spearman")
    cols = list(p.columns)
    rows = []
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            rp = p.iat[i, j]
            rs = s.iat[i, j]
            ap, as_ = abs(rp), abs(rs)
            if ap <= threshold and as_ <= threshold:
                continue
            if ap > threshold and as_ > threshold:
                kind = "linear"
            elif as_ > threshold:
                kind = "monotonic_nonlinear"
            else:
                kind = "outlier_driven"
            rows.append({"a": a, "b": b, "r_pearson": rp,
                         "r_spearman": rs, "kind": kind})
    high_pairs = pd.DataFrame(rows)
    if not high_pairs.empty:
        high_pairs["abs_max"] = (high_pairs[["r_pearson", "r_spearman"]]
                                   .abs().max(axis=1))
        high_pairs = (high_pairs.sort_values("abs_max", ascending=False)
                                .drop(columns="abs_max")
                                .reset_index(drop=True))
    return {"pearson": p, "spearman": s, "high_pairs": high_pairs}


def find_highly_correlated(X: pd.DataFrame, anova_ranking: pd.DataFrame,
                            threshold: float = 0.95) -> list:
    """Returns the features to drop so that no two surviving features have
    absolute Pearson correlation above ``threshold``.

    Zero-variance ("constant") features are always dropped: their ANOVA F is
    undefined (``NaN``), their correlations with every other column are
    ``NaN``, and they break downstream scalers. For the remaining columns
    the member of each correlated pair with the lower ANOVA F-score is
    marked for removal, keeping the better single-feature discriminator.
    When scores tie, the alphabetically larger name is dropped so the
    output is deterministic.

    Args:
        X: Feature DataFrame whose columns will be considered for
            pruning.
        anova_ranking: DataFrame produced by ``anova_rank_features``
            containing at least the columns ``feature`` and ``F``.
            Features absent from this table are treated as having
            ``F = 0``.
        threshold: Absolute Pearson correlation cutoff. Pairs at or
            below this magnitude are left alone. Defaults to 0.95.

    Returns:
        Sorted list of feature names to drop from ``X``. Applying
        ``X.drop(columns=result)`` leaves only the highest-F member of
        each correlated cluster and removes any constant columns.
    """
    f_score = dict(zip(anova_ranking["feature"], anova_ranking["F"]))
    corr = X.corr().abs()
    cols = list(corr.columns)
    drop = {c for c in cols if not np.isfinite(f_score.get(c, 0.0))}
    for i, a in enumerate(cols):
        if a in drop:
            continue
        for j in range(i + 1, len(cols)):
            b = cols[j]
            if b in drop:
                continue
            r = corr.iat[i, j]
            if not np.isfinite(r) or r <= threshold:
                continue
            fa = f_score.get(a, 0.0)
            fb = f_score.get(b, 0.0)
            if fa > fb:
                drop.add(b)
            elif fb > fa:
                drop.add(a)
            else:
                drop.add(max(a, b))
    return sorted(drop)


def evaluate_classifier(y_true, y_pred) -> dict:
    """Computes a standard classifier metrics dictionary.

    Reports both weighted and macro F1: the weighted variant tracks
    overall hit rate when classes are imbalanced, while the macro
    variant gives every class the same weight and surfaces minority-
    class failures.

    Args:
        y_true: Ground-truth labels (1D array-like).
        y_pred: Predicted labels (1D array-like).

    Returns:
        Dict with keys ``accuracy``, ``precision``, ``recall``,
        ``f1_weighted``, and ``f1_macro``. Precision and recall use
        weighted averaging. All metrics use ``zero_division=0`` so
        absent classes contribute zero rather than raising.
    """
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":      recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
    }
