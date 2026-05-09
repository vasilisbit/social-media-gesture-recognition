from os import listdir
from os.path import isfile, join
import secrets
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import scipy
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

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


def sliding_window_pd(
        df,
        ws=500,
        overlap=250,
        w_type="hann",
        w_center=True,
        print_stats=False
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
    counter = 0
    windows_list = list()
    # min_periods: Minimum number of observations in window required to have
    # a value;
    # For a window that is specified by an integer, min_periods will default
    # to the size of the window.
    for window in df.rolling(window=ws, step=overlap, min_periods=ws,
                             win_type=w_type, center=w_center):
        if window[window.columns[0]].count() >= ws:
            if print_stats:
                print("Print Window:", counter)
                print("Number of samples:", window[window.columns[0]].count())
            windows_list.append(window)
        counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))

    return windows_list


def apply_filter(
        arr,
        order=5,
        wn=0.1,
        filter_type="lowpass"
) -> np.ndarray:
    """Applies filter to the multi-axis signal.

    Args:
        arr: The initial NumPy signal array values.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’,
            ‘bandstop’}

    Returns:
        NumPy Array with the filtered signal.
    """
    fbd_filter = scipy.signal.butter(N=order, Wn=wn, btype=filter_type,
                                     output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos=fbd_filter, x=arr, padlen=0)

    return filtered_signal


def filter_instances(
        instances_list,
        order,
        wn,
        filter_type
) -> list:
    """Applies filter to a list of windows (each window is a DataFrame).

    Args:
        instances_list: List of DataFrames.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’,
            ‘bandstop’}

    Returns:

    """
    filtered_instances_list = list()
    for item in instances_list:
        filtered_instance = item.apply(apply_filter,
                                       args=(order, wn, filter_type)
                                       )
        filtered_instances_list.append(filtered_instance)
    print("Number of filtered instances in the list:",
          len(filtered_instances_list)
          )

    return filtered_instances_list


def flatten_instances_df(instances_list: list) -> pd.DataFrame:
    """Flattens each instance and create a DataFrame with the whole flattened
        instances.

    Args:
        instances_list: The list of DataFrames to be flattened

    Returns:
        A DataFrame that includes the whole flattened DataFrames
    """
    flattened_instances_list = list()
    for item in instances_list:
        instance = item.to_numpy().flatten()
        flattened_instances_list.append(instance)
    df = pd.DataFrame(flattened_instances_list)

    return df


def df_rebase(
        df: pd.DataFrame,
        target_list: list,
        ref_list: list
) -> pd.DataFrame:
    """Changes the order and name of DataFrame columns to the project's needs
        for readability.

    Args:
        df: The pandas DataFrame.
        order_list: List object that contains the proper order of the default
             column names.
        ref_list: List object that contains the renaming list based
            on the project needs.

    Returns:
        A DataFrame with the new columns order and names.
    """
    print("Initial columns:", list(df.columns))

    if are_lists_equal(list(df.columns), ref_list):
        pass

    else:
        if len(target_list) == len(ref_list): 
            # keep and re-order only the necessary columns of the initial DataFrame
            df = df[target_list]
            rename_dict = dict(zip(target_list, ref_list))
            df = df.rename(columns=rename_dict)  # rename the columns
        else:
            print("The length of the target list and the reference list is not equal.")

    print("Processed columns:", list(df.columns))

    return df


def rename_df_column_values(
    np_array: np.ndarray, 
    y: list, 
    columns_names: tuple = ("acc_x", "acc_y", "acc_z")
):
    """Creates a DataFrame with a "y" label column and replaces the values of the y with the index
    of the unique values of y.

    Args:
        np_array: 2D NumPy array.
        y: List with the y labels
        columns_names: List with the DF columns names.

    Returns:
        DataFrame with the multi-axes values and the target labels column.
    """
    arr_y = np.array(y)  # list to numpy array
    unique_values_list = np.unique(arr_y)  # unique list of values

    df = pd.DataFrame(np_array, columns=columns_names)
    df["y"] = y

    # replace the row item value in the y column of the df, with its index in the unique list
    for idx, x in enumerate(unique_values_list):
        df["y"] = np.where(df["y"] == x, idx, df["y"])

    return df


def are_lists_equal(
    list1: list, 
    list2: list
) -> bool:
    return set(list1) == set(list2)


def encode_labels(instances_list) -> np.ndarray:
    """Encodes target labels.

    Args:
        instances_list: List of instances to be encoded.

    Returns:
        The encoded array.
    """
    le = preprocessing.LabelEncoder()
    le.fit(instances_list)
    instances_arr = le.transform(instances_list)

    return instances_arr


def list_files_in_folder(folder_path) -> list:
    """Returns a list of all CSV files within the specified folder.

    Args:
        folder_path (str): The directory path to search for files.

    Returns:
        list: A list containing the filenames (strings) of all files
              in the directory that end with the '.csv' extension.
    """
    files_list = list()
    for f in listdir(folder_path):
        if isfile(join(folder_path, f)):
            if f.endswith(".csv"):
                files_list.append(f)

    return files_list


# -------------------------------------------------------------------------- #
# Raw CSV → merged 6-axis MongoDB documents (functional)
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

# σε καποιο αρχειο υπαχει περιεργος χαρακτηρας
def load_acc_csv(path) -> pd.DataFrame:
    """Loads a raw accelerometer CSV (Epoch,X,Y,Z) and renames axes to acc_*."""
    return (pd.read_csv(path, encoding='utf-8-sig', engine='python')
              .rename(columns={"X": "acc_x", "Y": "acc_y", "Z": "acc_z"})
              .sort_values("Epoch")
              .reset_index(drop=True))


def load_gyr_csv(path) -> pd.DataFrame:
    """Loads a raw gyroscope CSV (Epoch,X,Y,Z) and renames axes to gyr_*."""
    return (pd.read_csv(path)
              .rename(columns={"X": "gyr_x", "Y": "gyr_y", "Z": "gyr_z"})
              .sort_values("Epoch")
              .reset_index(drop=True))


def merge_acc_gyr(acc_df: pd.DataFrame, gyr_df: pd.DataFrame,
                  tolerance_ms: int = 10) -> pd.DataFrame:
    """Aligns acc/gyr by Epoch with merge_asof; drops rows missing any axis."""
    merged = pd.merge_asof(acc_df, gyr_df, on="Epoch",
                            direction="nearest", tolerance=tolerance_ms)
    return (merged
              .dropna(subset=list(SIX_AXES))
              .reset_index(drop=True))

# Διωρθωμενη ΣΟΣ καθαριζει τα αρχεια απο περιεργουσ χαρακτηρες
def apply_continuous_filter(df: pd.DataFrame, order: int = 4,
                              wn: float = 0.2) -> pd.DataFrame:
    """Zero-phase Butterworth lowpass on the 6 axes (D5).

    Args:
        df: DataFrame containing the SIX_AXES columns.
        order: Butterworth order. Defaults to 4.
        wn: Normalized critical frequency 2*fc/fs. fc=10Hz @ fs=100Hz → 0.2.

    Returns:
        A copy of df with the 6 axes filtered. Other columns are preserved.
    """
    b, a = butter(order, wn, btype="lowpass", output="ba")
    out = df.copy()
#καθαρισμα δεδομενων
    for col in SIX_AXES:
        clean_col = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        data = pd.to_numeric(clean_col, errors='coerce').values
        data = np.nan_to_num(data)
        out[col] = filtfilt(b, a, data, padlen=0)
    return out


def load_session(acc_path, gyr_path) -> pd.DataFrame:
    """Loads, merges and lowpass-filters one acc/gyr pair into a 6-axis frame."""
    return apply_continuous_filter(
        merge_acc_gyr(load_acc_csv(acc_path), load_gyr_csv(gyr_path))
    )


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

# Processing data
def ProcessingData(pairs):
    all_users_dicts = {}
    for meta, acc_p, gyr_p in pairs:
        df_acc = load_acc_csv(acc_p)
        df_gyr = load_gyr_csv(gyr_p)
        merged_df = merge_acc_gyr(df_acc, df_gyr)

        user_map = {'a': 'Alex', 'b': 'Vasilis', 's': 'Stamy'}
        user_name = user_map.get(meta['user'], meta['user'])

        gesture = meta['gesture_id']
        hand_label = meta['finger']

        if user_name not in all_users_dicts:
            all_users_dicts[user_name] = {}
        if gesture not in all_users_dicts[user_name]:
            all_users_dicts[user_name][gesture] = {}

        all_users_dicts[user_name][gesture][hand_label] = merged_df

    return all_users_dicts