from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

AXIS_COLORS = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
ACC_COLS = ("acc_x", "acc_y", "acc_z")
GYR_COLS = ("gyr_x", "gyr_y", "gyr_z")


def plot_instance_time_domain(df: pd.DataFrame):
    """Visualizes the movement instance to a plot in time domain.

    Args:
        df: The DataFrame to be visualized in time domain.

    Returns:

    """
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20).legend(fontsize=18)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)


def plot_instance_3d(
        df: pd.DataFrame,
        axes_list: tuple = ("acc_x", "acc_y", "acc_z")
):
    """Plots a 3-axes DataFrame in 3D graph.

    Args:
        df: The DataFrame to be plotted in 3D.
        axes_list: Tuple with the 3-axis values. For gyroscope axes should
            be: ("gyr_x", "gyr_y", "gyr_z")

    Returns:

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # print the plot in 3D

    xs = df[axes_list[0]]
    ys = df[axes_list[1]]
    zs = df[axes_list[2]]

    ax.scatter(xs, ys, zs, color='green', s=50, alpha=0.6, edgecolors='w')

    ax.set_xlabel(axes_list[0])
    ax.set_ylabel(axes_list[1])
    ax.set_zlabel(axes_list[2])


def plot_np_instance(
        np_array: np.ndarray,
        columns_list: list
):
    """Plot NumPy instance using DataFrames (pandas). It first transforms the
        array into
    DataFrame with its corresponding columns naming, and then, it plots the
        DataFrame in time domain.

    Args:
        np_array: The NumPy array to be transformed.
        columns_list: The columns list that the DataFrame and the plot will
            have.

    Returns:

    """
    df = pd.DataFrame(np_array, columns=columns_list)
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)


def plot_heatmap(df: pd.DataFrame):
    """Visualizes the heatmap of the DataFrame's values.

    Args:
        df: A DataFrame.

    Returns:

    """
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, cmap='plasma')


def plot_scatter_pca(
        df: pd.DataFrame,
        c_name: str,
        cmap_set: str = "plasma"
):
    """Visualizes the values of the component columns of the DataFrame
    according to its column that includes the labels.

    Args:
        df: The DataFrame that contains the transformed data after the PCA
            procedure.
        c_name: The name of the column that includes the labels.
        cmap_set: The format of the plot.

    Returns:

    """
    if len(df.columns) == 3:
        plt.style.use('classic')
        plt.figure(figsize=(16, 8))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df[c_name], cmap=cmap_set)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
    elif len(df.columns) == 4:
        plt.style.use('classic')
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df[c_name], cmap=cmap_set)
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
    else:
        print("The DataFrame has more than 4 columns.")


# -------------------------------------------------------------------------- #
# Static PNG visualizations for full merged sessions
# -------------------------------------------------------------------------- #

def session_label(meta: dict) -> str:
    """Builds a short human-readable label, e.g. 'scroll-up | thumb | user=a'."""
    parts = [meta["gesture_id"], meta["finger"], meta["typing_style"], f"user={meta['user']}"]
    return " | ".join(p for p in parts if p != "na")


def session_slug(meta: dict) -> str:
    """Builds a filesystem-safe slug for a session."""
    return (f"{meta['gesture_id']}_{meta['finger']}_{meta['typing_style']}"
            f"_h{meta['hand']}_p{meta['primary']}_sr{meta['sr']}_{meta['user']}")


def _time_axis(df: pd.DataFrame, sr: int) -> pd.Series:
    """Returns time in seconds. Uses Epoch if present, else sample index / sr."""
    if "Epoch" in df.columns and len(df) > 0:
        return (df["Epoch"] - df["Epoch"].iloc[0]) / 1000.0
    return pd.Series(df.index / sr)


def plot_session_timeseries(df: pd.DataFrame, meta: dict, out_dir) -> Path:
    """Saves a 2x3 grid of time-domain plots for the 6 axes of a session."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    t = _time_axis(df, meta["sr"])
    rows = [(ACC_COLS, "Accelerometer (g)"),
            (GYR_COLS, "Gyroscope (deg/s)")]
    for row_i, (cols, ylabel) in enumerate(rows):
        for col_i, col in enumerate(cols):
            ax = axes[row_i, col_i]
            ax.plot(t, df[col], color=AXIS_COLORS[col[-1]], linewidth=0.6)
            ax.set_title(col)
            ax.grid(alpha=0.3)
            if col_i == 0:
                ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"6-axis time domain — {session_label(meta)}", fontsize=12)
    fig.tight_layout()
    out = out_dir / f"timeseries_{session_slug(meta)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_session_3d(df: pd.DataFrame, meta: dict, out_dir) -> Path:
    """Saves side-by-side static 3D scatters for acc and gyr of a session."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5))
    panels = [("Accelerometer", ACC_COLS),
              ("Gyroscope",     GYR_COLS)]
    for i, (sensor_name, cols) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]],
                    c=df.index, cmap="viridis", s=2, alpha=0.6)
        ax.set_title(sensor_name)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_zlabel(cols[2])
    fig.suptitle(f"3D trajectory — {session_label(meta)}", fontsize=12)
    fig.tight_layout()
    out = out_dir / f"scatter3d_{session_slug(meta)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_session(df: pd.DataFrame, meta: dict, out_dir):
    """Saves both time-domain and 3D-scatter plots for a single session."""
    return plot_session_timeseries(df, meta, out_dir), plot_session_3d(df, meta, out_dir)


def visualize_all(sessions: Iterable, out_dir) -> int:
    """Runs visualize_session for every (df, meta); returns the count visualized."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for df, meta in sessions:
        visualize_session(df, meta, out_dir)
        count += 1
    return count
