from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from utils import VALID_BASES


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


# ------------------------------Our Implementation-----------------------------------------#


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
