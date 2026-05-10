from pathlib import Path
from typing import Iterable
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from utils import VALID_BASES

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
    """Builds a filesystem-safe slug for a session.

    If meta carries `acc_csv`/`gyr_csv` (the original CSV stems), they are used
    verbatim so the saved figure name traces back to the source files. Falls
    back to the constructed key otherwise.
    """
    if "acc_csv" in meta and "gyr_csv" in meta:
        return f"{meta['acc_csv']}__{meta['gyr_csv']}"
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


def plot_session_3d(df: pd.DataFrame, meta: dict, out_dir, t_vmax: float = None) -> Path:
    """Saves side-by-side static 3D scatters for acc and gyr of a session.

    Color encodes elapsed time in seconds, normalized to [0, t_vmax] so the
    colormap is comparable across sessions. If t_vmax is None, falls back to
    this session's own duration.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = _time_axis(df, meta["sr"])
    vmax = t_vmax if t_vmax is not None else (float(t.iloc[-1]) if len(t) else 1.0)

    fig = plt.figure(figsize=(12, 5))
    panels = [("Accelerometer", ACC_COLS),
              ("Gyroscope",     GYR_COLS)]
    sc = None
    for i, (sensor_name, cols) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sc = ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]],
                    c=t, cmap="viridis", vmin=0, vmax=vmax, s=2, alpha=0.6)
        ax.set_title(sensor_name)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_zlabel(cols[2])
    fig.suptitle(f"3D trajectory — {session_label(meta)}", fontsize=12)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.7, pad=0.1)
        cbar.set_label("Time (s)")
    out = out_dir / f"scatter3d_{session_slug(meta)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_session(df: pd.DataFrame, meta: dict, out_dir, t_vmax: float = None,
                       window: int = None):
    """Saves per-session figures: acc-3, gyr-3, and 3D scatter.

    The combined 6-axis figure is now produced *across users* by
    `plot_grouped_sixaxis`, so it is no longer emitted here.
    If `window` is given, only the first `window` samples of `df` are plotted.
    """
    if window is not None:
        df = df.iloc[:window].reset_index(drop=True)
    return (plot_session_acc_3axis(df, meta, out_dir),
            plot_session_gyr_3axis(df, meta, out_dir),
            plot_session_3d(df, meta, out_dir, t_vmax=t_vmax))


def _group_key(meta: dict) -> str:
    """Filesystem-safe key shared by all users of one gesture variant."""
    return (f"{meta['gesture_id']}_{meta['finger']}"
            f"_{meta['hand']}_{meta['sr']}_{meta['primary']}")


def plot_grouped_sixaxis(group_sessions, out_dir, name_suffix: str = "") -> Path:
    """One figure per gesture-variant, with users laid out in columns.

    Top row: 3-axis accelerometer overlay per user.
    Bottom row: 3-axis gyroscope overlay per user.
    All inputs in `group_sessions` must share gesture/finger/typing_style/hand/sr/primary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_sessions = sorted(group_sessions, key=lambda x: x[1]["user"])
    n_users = len(group_sessions)
    fig, axes = plt.subplots(2, n_users, figsize=(5 * n_users, 7),
                              sharex=True, squeeze=False)
    meta0 = group_sessions[0][1]
    for col_i, (df, meta) in enumerate(group_sessions):
        t = _time_axis(df, meta["sr"])
        ax_a, ax_g = axes[0, col_i], axes[1, col_i]
        for c in ACC_COLS:
            ax_a.plot(t, df[c], color=AXIS_COLORS[c[-1]], linewidth=0.8, label=c)
        for c in GYR_COLS:
            ax_g.plot(t, df[c], color=AXIS_COLORS[c[-1]], linewidth=0.8, label=c)
        ax_a.set_title(f"user={meta['user']}")
        ax_g.set_xlabel("Time (s)")
        for ax in (ax_a, ax_g):
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[0, 0].set_ylabel("Accelerometer (g)")
    axes[1, 0].set_ylabel("Gyroscope (deg/s)")
    fig.suptitle(
        f"6-axis per user — {meta0['gesture_id']} / {meta0['finger']} / "
        f"typing_style={meta0['typing_style']}",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / f"sixaxis_{_group_key(meta0)}{name_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_all(sessions: Iterable, out_dir, window: int = None) -> int:
    """Runs per-session plots and grouped sixaxis plots.

    Returns the number of sessions visualized. Per-session: acc3, gyr3, scatter3d.
    Per group (sessions sharing gesture/finger/typing_style/hand/sr/primary):
    one combined sixaxis figure with users in columns.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = list(sessions)
    if window is not None:
        sessions = [(df.iloc[:window].reset_index(drop=True), meta) for df, meta in sessions]
    t_vmax = max(
        (float(_time_axis(df, meta["sr"]).iloc[-1]) for df, meta in sessions if len(df)),
        default=1.0,
    )
    count = 0
    for df, meta in sessions:
        visualize_session(df, meta, out_dir, t_vmax=t_vmax)
        count += 1
    groups = defaultdict(list)
    for df, meta in sessions:
        key = (meta["gesture_id"], meta["finger"], meta["typing_style"],
               meta["hand"], meta["sr"], meta["primary"])
        groups[key].append((df, meta))
    for grp in groups.values():
        plot_grouped_sixaxis(grp, out_dir)
    return count


# -------------------------------------------------------------------------- #
# EXPERIMENTAL — overlay-style time-domain plots (acc-3, gyr-3, combined 6).
# Safe to delete this whole section if the visual style is not useful.
# -------------------------------------------------------------------------- #

def plot_session_acc_3axis(df: pd.DataFrame, meta: dict, out_dir, name_suffix: str = "") -> Path:
    """Single plot with the 3 accelerometer axes overlaid on one axis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    t = _time_axis(df, meta["sr"])
    for col in ACC_COLS:
        ax.plot(t, df[col], color=AXIS_COLORS[col[-1]], linewidth=0.8, label=col)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accelerometer (g)")
    ax.set_title(f"Accelerometer 3-axis — {session_label(meta)}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    stem = meta.get("acc_csv", session_slug(meta))
    out = out_dir / f"acc3_{stem}{name_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_session_gyr_3axis(df: pd.DataFrame, meta: dict, out_dir,
                           smooth_window: int = 15, name_suffix: str = "") -> Path:
    """Three stacked subplots (one per gyroscope axis) sharing the time axis.

    Stacking avoids the three traces overlapping into an unreadable mass.
    A rolling-mean smoothing (window=`smooth_window` samples, ~150 ms at 100 Hz)
    is applied so the line reads as a continuous trajectory rather than the raw
    high-frequency oscillation. Set smooth_window=1 to disable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    t = _time_axis(df, meta["sr"])
    for ax, col in zip(axes, GYR_COLS):
        y = df[col].rolling(smooth_window, center=True, min_periods=1).mean()
        ax.plot(t, y, color=AXIS_COLORS[col[-1]], linewidth=1.2, label=col)
        ax.set_ylabel(f"{col} (deg/s)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Gyroscope 3-axis — {session_label(meta)}", fontsize=12)
    fig.tight_layout()
    stem = meta.get("gyr_csv", session_slug(meta))
    out = out_dir / f"gyr3_{stem}{name_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_session_6axis(df: pd.DataFrame, meta: dict, out_dir, name_suffix: str = "") -> Path:
    """Combined 6-axis view: acc and gyr stacked, sharing the time axis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t = _time_axis(df, meta["sr"])
    for col in ACC_COLS:
        ax_a.plot(t, df[col], color=AXIS_COLORS[col[-1]], linewidth=0.8, label=col)
    for col in GYR_COLS:
        ax_g.plot(t, df[col], color=AXIS_COLORS[col[-1]], linewidth=0.8, label=col)
    ax_a.set_ylabel("Accelerometer (g)")
    ax_g.set_ylabel("Gyroscope (deg/s)")
    ax_g.set_xlabel("Time (s)")
    for ax in (ax_a, ax_g):
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=3)
    fig.suptitle(f"6-axis (acc + gyr) — {session_label(meta)}", fontsize=12)
    fig.tight_layout()
    if "acc_csv" in meta:
        stem = meta["acc_csv"].replace("_acc_", "_").replace("_acc", "")
    else:
        stem = session_slug(meta)
    out = out_dir / f"sixaxis_{stem}{name_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
def visualize_session_overlays(df: pd.DataFrame, meta: dict, out_dir):
    """Convenience wrapper that saves all three overlay plots for a session."""
    return (plot_session_acc_3axis(df, meta, out_dir),
            plot_session_gyr_3axis(df, meta, out_dir),
            plot_session_6axis(df, meta, out_dir))



#------------------------------Our Implementation-----------------------------------------#

# Plot 3 axis Window to visualize The data for Each User
def plot(ax, df, sensor, user_name, hand):
    if not df.empty:
        df_plot = df.copy()

        cols = [f'{sensor}_x', f'{sensor}_y', f'{sensor}_z']
        for col in cols:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

        ax.plot(df_plot.index, df_plot[f'{sensor}_x'], label='X', color='red', lw=1)
        ax.plot(df_plot.index, df_plot[f'{sensor}_y'], label='Y', color='green', lw=1)
        ax.plot(df_plot.index, df_plot[f'{sensor}_z'], label='Z', color='blue', lw=1)


        ax.legend(loc='upper right', fontsize='small', framealpha=0.5)

    if hand == 0:
        ax.set_xlabel("Sample Thumb")
    else:
        ax.set_xlabel("Sample Index")

    ax.set_title(f"{user_name} | {sensor.upper()}")
    ax.grid(True, alpha=0.2)
def Plot_3axis_window2(all_users_dicts, gesture, limit=128):

    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    fig.suptitle(f"3-Axis Analysis (Thumb vs Index): {gesture}", fontsize=20, fontweight='bold')

    user_names = ['Vasilis', 'Stamy', 'Alex']

    for col, name in enumerate(user_names):
        u_dict = all_users_dicts.get(name, {})
        gesture_data = u_dict.get(gesture, {})
        if gesture == 'texting':
            df_t = gesture_data.get('na', pd.DataFrame())
            df_i = None
        else:
            df_t = gesture_data.get('thumb', pd.DataFrame())
            df_i = gesture_data.get('index', pd.DataFrame())

        df_t = df_t.iloc[:limit]
        for row, sensor in enumerate(['acc', 'gyr']):
            ax = axes[row, col]
            ax.cla()
            plot(ax, df_t, sensor, user_names[col],0)

        if df_i is not None and not df_i.empty:
            df_i = df_i.iloc[:limit]
            for row, sensor in enumerate(['acc', 'gyr']):

                ax = axes[row + 2, col]
                ax.cla()
                plot(ax, df_i, sensor, user_names[col], 1)
        else:
            axes[2, col].set_visible(False)
            axes[3, col].set_visible(False)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
def Plot_3axis(all_users_dicts):

    for gesture in VALID_BASES:
        Plot_3axis_window2(all_users_dicts, gesture)


#plot 6 axis gyr + accelerometer
def plot_6_axis(ax, df, user_name, hand):
    if not df.empty:
        df_plot = df.copy()
        configs = [
            ('acc', '-'),
            ('gyr', '--')
        ]
        colors = ['red', 'green', 'blue']
        axes_labels = ['x', 'y', 'z']
        for sensor, ls in configs:
            cols = [f'{sensor}_x', f'{sensor}_y', f'{sensor}_z']
            for i, col in enumerate(cols):
                if col in df_plot.columns:
                    data_col = pd.to_numeric(df_plot[col], errors='coerce')
                    label = f"{sensor.upper()}_{'XYZ'[i]}"
                    ax.plot(df_plot.index, data_col, label=label, color=colors[i], linestyle=ls, lw=1.2)

    ax.set_title(f"{user_name} | 6-Axis Combined")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Sample Thumb" if hand == 0 else "Sample Index")
    ax.legend(loc='upper right', fontsize='small', ncol=2)
def Plot_6axis_window2(all_users_dicts, gesture, limit=128):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"6-Axis Combined Visualization: {gesture}", fontsize=20, fontweight='bold')

    user_names = ['Vasilis', 'Stamy', 'Alex']

    for col, name in enumerate(user_names):
        u_dict = all_users_dicts.get(name, {})
        gesture_data = u_dict.get(gesture, {})
        df_t = gesture_data.get('thumb' if gesture != 'texting' else 'na', pd.DataFrame()).iloc[:limit]
        plot_6_axis(axes[0, col], df_t, name, 0)
        df_i = gesture_data.get('index', pd.DataFrame()).iloc[:limit] if gesture != 'texting' else None
        if df_i is not None and not df_i.empty:
            plot_6_axis(axes[1, col], df_i, name, 1)
        else:
            axes[1, col].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
def Plot_6axis(all_users_dicts):
    for gesture in VALID_BASES:
        Plot_6axis_window2(all_users_dicts, gesture)