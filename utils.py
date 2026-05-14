import secrets
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from scipy.signal import butter, sosfiltfilt
import pandas as pd
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
    """Loads a raw accelerometer CSV (Epoch,X,Y,Z) and renames axes to acc_*."""
    df = pd.read_csv(path, encoding='utf-8-sig', engine='python')
    df = df.rename(columns={"X": "acc_x", "Y": "acc_y", "Z": "acc_z"})
    df = df.sort_values("Epoch")
    
    # Ensure strict monotonic increase by at least 1 ms
    new_epochs = []
    last_val = -1
    for val in df["Epoch"]:
        last_val = max(val, last_val + 1)
        new_epochs.append(last_val)
        
    df["Epoch"] = new_epochs
    return df.reset_index(drop=True)


def load_gyr_csv(path) -> pd.DataFrame:
    """Loads a raw gyroscope CSV (Epoch,X,Y,Z) and renames axes to gyr_*."""
    df = pd.read_csv(path)
    df = df.rename(columns={"X": "gyr_x", "Y": "gyr_y", "Z": "gyr_z"})
    
    # Sort first to ensure chronological processing
    df = df.sort_values("Epoch")
    
    # Ensure strict monotonic increase by at least 1 ms
    new_epochs = []
    last_val = -1
    for val in df["Epoch"]:
        val = max(val, last_val + 1)
        new_epochs.append(val)
        last_val = val
        
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

 
