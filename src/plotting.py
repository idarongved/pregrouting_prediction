"""Plotting functions for use in other scripts"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("./src/figures_styles.mplstyle")


def plot_histograms(
    df: pd.DataFrame, features: list[str], savepath: Path, binsize: int = 10
) -> None:
    """
    Plots histograms for the specified features in a DataFrame and saves the plot to a
    file.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - features (list[str]): The list of feature names to plot.
    - savepath (Path): The path to save the plot.
    - binsize (int): The number of bins for the histograms. Default is 10.

    Returns:
    - None
    """
    num_features = len(features)
    nrows = int(num_features / 3) + (num_features % 3 > 0)
    ncols = 3 if num_features > 2 else num_features

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        axs[i].hist(
            df[feature], bins=binsize, alpha=0.5, edgecolor="black", color="lightgrey"
        )
        axs[i].set_title(feature, fontsize=15)
        axs[i].tick_params(labelsize=15)

    for ax in axs[num_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath)


def plot_barplots(df: pd.DataFrame, features: list[str], savepath: Path) -> None:
    """
    Plots bar plots for the specified categorical features in a DataFrame and saves the plot to a file.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - features (list[str]): The list of feature names to plot.
    - savepath (Path): The path to save the plot.

    Returns:
    - None
    """
    num_features = len(features)
    nrows = int(num_features / 3) + (num_features % 3 > 0)
    ncols = 3 if num_features > 2 else num_features

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        values = df[feature].value_counts()
        axs[i].bar(values.index, values.values, color="lightgrey", edgecolor="black")
        axs[i].set_title(feature, fontsize=15)
        axs[i].tick_params(labelsize=15)
        axs[i].tick_params(axis="x", labelrotation=90)

    for ax in axs[num_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath)


# def plot_correlation_matrix(
#     savepath: Path, data: pd.DataFrame, features: list[str], figure_width: float = 6.3
# ) -> None:
#     corr = (data[features]).corr().round(2)
#     mask = np.zeros_like(corr, dtype=bool)
#     mask[np.triu_indices_from(mask)] = True
#     fig, ax = plt.subplots(figsize=(2 * figure_width, 2 * figure_width))
#     cmap = sns.color_palette("vlag", as_cmap=True)
#     sns.heatmap(
#         corr,
#         cmap=cmap,
#         mask=mask,
#         center=0,
#         square=True,
#         linewidths=0.5,  # type: ignore
#         cbar_kws={"shrink": 0.5},
#         annot=True,
#         annot_kws={"size": 8},
#         ax=ax,
#     )
#     plt.savefig(savepath)


def plot_correlation_matrix(
    savepath: Path,
    data: pd.DataFrame,
    features: list[str],
    figure_width: float = 6.3,
    highlight_features: list[str] = [],
) -> None:
    corr = (data[features]).corr().round(2)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(2 * figure_width, 2 * figure_width))
    cmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(
        corr,
        cmap=cmap,
        mask=mask,
        center=0,
        square=True,
        linewidths=0.5,  # type: ignore
        cbar_kws={"shrink": 0.5},
        annot=True,
        annot_kws={"size": 8},
        ax=ax,
    )
    if highlight_features:
        # Get the current tick labels
        tick_labels = ax.get_xticklabels()

        # Find the indices of the tick labels corresponding to the highlight features
        highlight_indices = [
            i
            for i, label in enumerate(tick_labels)
            if label.get_text() in highlight_features
        ]

        # Set the color of the highlight tick labels to red
        for index in highlight_indices:
            tick_labels[index].set_color("red")

    plt.savefig(savepath)
    plt.show()
