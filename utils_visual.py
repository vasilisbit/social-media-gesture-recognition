from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from utils import VALID_BASES, load_acc_csv, load_gyr_csv, merge_acc_gyr, save_figure


def plot_instance_time_domain(df: pd.DataFrame):
    """Visualizes the movement instance to a plot in time domain.

    Args:
        df: The DataFrame to be visualized in time domain.

    Returns:

    """
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20).legend(fontsize=18)
    plt.xlabel("Sample", fontsize=20)
    plt.ylabel("Axes", fontsize=20)


def plot_instance_3d(df: pd.DataFrame, axes_list: tuple = ("acc_x", "acc_y", "acc_z")):
    """Plots a 3-axes DataFrame in 3D graph.

    Args:
        df: The DataFrame to be plotted in 3D.
        axes_list: Tuple with the 3-axis values. For gyroscope axes should
            be: ("gyr_x", "gyr_y", "gyr_z")

    Returns:

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # print the plot in 3D

    xs = df[axes_list[0]]
    ys = df[axes_list[1]]
    zs = df[axes_list[2]]

    ax.scatter(xs, ys, zs, color="green", s=50, alpha=0.6, edgecolors="w")

    ax.set_xlabel(axes_list[0])
    ax.set_ylabel(axes_list[1])
    ax.set_zlabel(axes_list[2])


def plot_np_instance(np_array: np.ndarray, columns_list: list):
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
    plt.xlabel("Sample", fontsize=20)
    plt.ylabel("Axes", fontsize=20)


def plot_heatmap(df: pd.DataFrame):
    """Visualizes the heatmap of the DataFrame's values.

    Args:
        df: A DataFrame.

    Returns:

    """
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, cmap="plasma")


def plot_scatter_pca(df: pd.DataFrame, c_name: str, cmap_set: str = "plasma"):
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
        plt.style.use("classic")
        plt.figure(figsize=(16, 8))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df[c_name], cmap=cmap_set)
        plt.xlabel("First principal component")
        plt.ylabel("Second Principal Component")
    elif len(df.columns) == 4:
        plt.style.use("classic")
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df[c_name], cmap=cmap_set
        )
        ax.set_xlabel("First principal component")
        ax.set_ylabel("Second Principal Component")
        ax.set_zlabel("Third Principal Component")
    else:
        print("The DataFrame has more than 4 columns.")


# -------------------------------------------------------------------------- #
# Original Data visualization (3-axis and 6-axis) for each user and gesture
# -------------------------------------------------------------------------- #

def load_data_for_visualization(pairs: list) -> dict:
    """Processes the raw CSV files and returns a nested dict of DataFrames."""
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


# Plot 3 axis Window to visualize The data for Each User
def plot(ax, df, sensor, user_name, hand):
    if not df.empty:
        df_plot = df.copy()

        cols = [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"]
        for col in cols:
            df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")

        ax.plot(df_plot.index, df_plot[f"{sensor}_x"], label="X", color="red", lw=1)
        ax.plot(df_plot.index, df_plot[f"{sensor}_y"], label="Y", color="green", lw=1)
        ax.plot(df_plot.index, df_plot[f"{sensor}_z"], label="Z", color="blue", lw=1)

        ax.legend(loc="upper right", fontsize="small", framealpha=0.5)

    if hand == 0:
        ax.set_xlabel("Sample Thumb")
    else:
        ax.set_xlabel("Sample Index")

    ax.set_title(f"{user_name} | {sensor.upper()}")
    ax.grid(True, alpha=0.2)


def Plot_3axis_window2(all_users_dicts, gesture, limit=128, out_dir=None):

    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    fig.suptitle(
        f"3-Axis Analysis (Thumb vs Index): {gesture}", fontsize=20, fontweight="bold"
    )

    user_names = ["Vasilis", "Stamy", "Alex"]

    for col, name in enumerate(user_names):
        u_dict = all_users_dicts.get(name, {})
        gesture_data = u_dict.get(gesture, {})
        if gesture == "texting":
            df_t = gesture_data.get("na", pd.DataFrame())
            df_i = None
        else:
            df_t = gesture_data.get("thumb", pd.DataFrame())
            df_i = gesture_data.get("index", pd.DataFrame())

        df_t = df_t.iloc[:limit]
        for row, sensor in enumerate(["acc", "gyr"]):
            ax = axes[row, col]
            ax.cla()
            plot(ax, df_t, sensor, user_names[col], 0)

        if df_i is not None and not df_i.empty:
            df_i = df_i.iloc[:limit]
            for row, sensor in enumerate(["acc", "gyr"]):

                ax = axes[row + 2, col]
                ax.cla()
                plot(ax, df_i, sensor, user_names[col], 1)
        else:
            axes[2, col].set_visible(False)
            axes[3, col].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"3axis_{gesture}.png", dpi=150, bbox_inches="tight")
    plt.show()


def Plot_3axis(all_users_dicts, out_dir=None):

    for gesture in VALID_BASES:
        Plot_3axis_window2(all_users_dicts, gesture, out_dir=out_dir)


# plot 6 axis gyr + accelerometer
def plot_6_axis(ax, df, user_name, hand):
    if not df.empty:
        df_plot = df.copy()
        configs = [("acc", "-"), ("gyr", "--")]
        colors = ["red", "green", "blue"]
        axes_labels = ["x", "y", "z"]
        for sensor, ls in configs:
            cols = [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"]
            for i, col in enumerate(cols):
                if col in df_plot.columns:
                    data_col = pd.to_numeric(df_plot[col], errors="coerce")
                    label = f"{sensor.upper()}_{'XYZ'[i]}"
                    ax.plot(
                        df_plot.index,
                        data_col,
                        label=label,
                        color=colors[i],
                        linestyle=ls,
                        lw=1.2,
                    )

    ax.set_title(f"{user_name} | 6-Axis Combined")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Sample Thumb" if hand == 0 else "Sample Index")
    ax.legend(loc="upper right", fontsize="small", ncol=2)


def Plot_6axis_window2(all_users_dicts, gesture, limit=128, out_dir=None):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"6-Axis Combined Visualization: {gesture}", fontsize=20, fontweight="bold"
    )

    user_names = ["Vasilis", "Stamy", "Alex"]

    for col, name in enumerate(user_names):
        u_dict = all_users_dicts.get(name, {})
        gesture_data = u_dict.get(gesture, {})
        df_t = gesture_data.get(
            "thumb" if gesture != "texting" else "na", pd.DataFrame()
        ).iloc[:limit]
        plot_6_axis(axes[0, col], df_t, name, 0)
        df_i = (
            gesture_data.get("index", pd.DataFrame()).iloc[:limit]
            if gesture != "texting"
            else None
        )
        if df_i is not None and not df_i.empty:
            plot_6_axis(axes[1, col], df_i, name, 1)
        else:
            axes[1, col].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"6axis_{gesture}.png", dpi=150, bbox_inches="tight")
    plt.show()


def Plot_6axis(all_users_dicts, out_dir=None):
    for gesture in VALID_BASES:
        Plot_6axis_window2(all_users_dicts, gesture, out_dir=out_dir)


# -------------------------------------------------------------------------- #
# EDA Visualizations 
# -------------------------------------------------------------------------- #

def plot_average_durations(df_all: pd.DataFrame):
    """Plots the average total recording time per gesture class.

    For each (gesture, finger) pair the mean session duration is computed
    and then stacked into a single bar per gesture. The swipe / scroll
    gestures were recorded twice (`thumb` and `index`, ~150 s each); the
    stack makes the full ~300 s per class visible. `texting` has a single
    `na` finger value and shows as a single segment.

    Args:
        df_all: The DataFrame containing the continuous raw time-series data.
            Must include `session_id`, `gesture_id`, `finger`, and `sr`.
    """
    durations = []
    for sid, sub in df_all.groupby("session_id"):
        sr = float(sub["sr"].iloc[0])
        durations.append({
            "gesture_id": sub["gesture_id"].iloc[0],
            "finger":     sub["finger"].iloc[0],
            "duration_s": len(sub) / sr,
        })

    dur_df = pd.DataFrame(durations)

    # Mean duration per (gesture, finger), then stack fingers per gesture so
    # the bar shows the full recording time (thumb + index ~= 300 s for the
    # swipe/scroll classes; texting has finger="na" and shows a single segment).
    pivot = (dur_df.groupby(["gesture_id", "finger"])["duration_s"]
                   .mean()
                   .unstack(fill_value=0.0))

    finger_colors = {"thumb": "#1f77b4", "index": "#ff7f0e", "na": "#2ca02c"}
    colors = [finger_colors.get(c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white")

    fig.suptitle("Average Duration per Session (by Class)", fontsize=15)
    ax.set_ylabel("Duration (s)")
    ax.set_xlabel("gesture_id")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="finger", loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "eda_class_durations")


# X (red), Y (green), Z (blue) — shared by KDE and histogram views
_AXIS_COLORS = ('#d62728', '#2ca02c', '#1f77b4')


def plot_signal_histograms(df_all: pd.DataFrame):
    """Plots, for each (gesture, finger) group, a combined figure with:

    - top row: KDE for accelerometer and gyroscope (X/Y/Z overlaid)
    - middle row: per-axis histograms `acc_x`, `acc_y`, `acc_z`
    - bottom row: per-axis histograms `gyr_x`, `gyr_y`, `gyr_z`

    Histograms reuse the same X=red / Y=green / Z=blue scheme as the KDE.

    Args:
        df_all: The DataFrame containing the continuous raw time-series data.
    """
    sensors = (("Accelerometer", ("acc_x", "acc_y", "acc_z")),
               ("Gyroscope",     ("gyr_x", "gyr_y", "gyr_z")))

    for (gesture, finger), group_df in df_all.groupby(['gesture_id', 'finger']):
        if group_df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Signal Distributions — {gesture.upper()} ({finger.title()})",
                     fontsize=15, fontweight='bold')

        for ax_, (title, cols) in zip(axes, sensors):
            for c, col in enumerate(cols):
                color = _AXIS_COLORS[c]
                label = col.split("_")[-1].upper()
                # Histogram (density-scaled to share the KDE y-axis) underneath
                sns.histplot(group_df[col].dropna(), ax=ax_, bins=60,
                             stat="density", color=color, alpha=0.25,
                             edgecolor=None, label=None)
                # KDE on top
                sns.kdeplot(data=group_df[col], ax=ax_, color=color,
                            fill=True, alpha=0.3, label=label)
            ax_.set_title(title)
            ax_.set_xlabel("Value")
            ax_.set_ylabel("Density")
            ax_.grid(True, alpha=0.3)
            ax_.legend()

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, f"eda_signal_dist_{gesture}_{finger}")


def plot_windows_per_class(all_labels: list): 
    """ Plots the count of windows per class after segmentation to visualize class balance.
    
    Args:
        all_labels: A list of class labels corresponding to each segmented window.
    """ 
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.countplot(x=all_labels, ax=ax)
    ax.set_title("Window Count per Class Post-Segmentation")
    save_figure(fig, "eda_segmented_class_balance")


def plot_classifier_evaluation(model_name, test_labels, y_pred, classes, cmap="Blues"):
    """Print a classifier evaluation report and plot its confusion matrix.

    Emits the standard scikit-learn ``classification_report``, then walks the
    confusion matrix row by row to print per-class true positive, false
    positive, false negative and true negative counts. Finally draws a
    confusion-matrix display with ``cmap`` colouring and shows the figure
    inline.

    Args:
        model_name: Display name of the model, used in headings and the
            figure title.
        test_labels: 1D array of ground-truth labels for the held-out set.
        y_pred: 1D array of predicted labels of the same length as
            ``test_labels``.
        classes: Ordered iterable of class identifiers; used both for
            ``confusion_matrix(labels=...)`` and as display labels on the
            plot, so the matrix rows / columns line up with the caller's
            expected ordering.
        cmap: Matplotlib colormap name passed to
            :class:`ConfusionMatrixDisplay`. Defaults to ``"Blues"``.

    Notes:
        Previously named ``ModelEvaluation``. The old name is kept as a
        module-level alias so existing notebooks continue to call
        ``utils_visual.ModelEvaluation(...)`` without edits.
    """
    print(f"\n==================================================")
    print(f" STATISTICAL EVALUATION: {model_name}")
    print(f"==================================================")

    print("\n[1] Classification Report:")
    print(classification_report(test_labels, y_pred))

    cm = confusion_matrix(test_labels, y_pred, labels=classes)

    print("[2] True/False Positive and Negative Breakdown per Gesture:")
    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        print(f"  - Gesture '{class_name}':")
        print(f"    - True Positives  (TP): {tp}  (correct prediction of this gesture)")
        print(f"    - False Positives (FP): {fp}  (other gestures misclassified as '{class_name}')")
        print(f"    - False Negatives (FN): {fn}  (actual '{class_name}' that was missed)")
        print(f"    - True Negatives  (TN): {tn}  (correct rejection of unrelated gestures)\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap, values_format="d", ax=ax, colorbar=False)

    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()


# Back-compat alias. Original name from the first time-series notebook draft;
# kept so existing call sites (e.g. utils_visual.ModelEvaluation) keep resolving.
ModelEvaluation = plot_classifier_evaluation
