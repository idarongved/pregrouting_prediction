"""Plotting functions for use in other scripts"""


from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import PredictionErrorDisplay, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

plt.style.use("./src/config/figures_styles.mplstyle")


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
    # nrows = int(num_features / 3) + (num_features % 3 > 0)
    ncols = 5 if num_features > 5 else num_features
    nrows = int(np.ceil(num_features / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        axs[i].hist(
            df[feature], bins=binsize, alpha=0.5, edgecolor="black", color="lightgrey"
        )
        axs[i].set_title(feature, fontsize=15)
        axs[i].tick_params(labelsize=15)
        axs[i].grid(True, axis="y", alpha=0.5)

    for ax in axs[num_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath)


def plot_barplots(df: pd.DataFrame, features: list[str], savepath: Path) -> None:
    """
    Plots bar plots for the specified categorical features in a DataFrame and saves the
    plot to a file.

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
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)

    for ax in axs[num_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath)


def plot_features(
    df: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    savepath: Path,
    binsize: int = 10,
) -> None:
    """
    Plots histograms for numerical features and bar plots for categorical features.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - num_features (list[str]): The list of numerical feature names to plot.
    - cat_features (list[str]): The list of categorical feature names to plot.
    - savepath (Path): The path to save the plot.
    - binsize (int): The number of bins for the histograms. Default is 10.

    Returns:
    - None
    """
    total_features = len(num_features) + len(cat_features)
    ncols = 4 if total_features > 4 else total_features
    nrows = int(np.ceil(total_features / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
    axs = axs.flatten()

    # Plotting numerical features
    for i, feature in enumerate(num_features):
        axs[i].hist(
            df[feature], bins=binsize, alpha=0.5, edgecolor="black", color="lightgrey"
        )
        axs[i].set_title(feature, fontsize=15)
        axs[i].tick_params(labelsize=15)
        axs[i].grid(True, axis="y", alpha=0.5)

    # Plotting categorical features
    for i, feature in enumerate(cat_features, start=len(num_features)):
        values = df[feature].value_counts()
        axs[i].bar(values.index, values.values, color="lightgrey", edgecolor="black")
        axs[i].set_title(feature, fontsize=15)
        axs[i].tick_params(labelsize=15)
        axs[i].tick_params(axis="x", labelrotation=90)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].grid(False)

    for ax in axs[total_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath)


def plot_correlation_matrix(
    savepath: Path,
    data: pd.DataFrame,
    features: list[str],
    figure_width: float = 6.3,
    highlight_features: list[str] = [],
    threshold_corr_value: float = 0.1,
    threshold_feature: str = "Total grout take",
) -> None:
    corr: pd.DataFrame = (data[features]).corr().round(2)
    # Filter the correlation matrix based on the threshold
    mask_features = abs(corr[threshold_feature]) > threshold_corr_value
    corr = corr[mask_features]
    corr.to_csv("test.csv")
    corr = corr.loc[:, corr.index.tolist()]  # same num features vertical and horizontal

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(2 * figure_width, 2 * figure_width))
    cmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        mask=mask,
        center=0,
        square=True,
        linewidths=0.5,  # type: ignore
        cbar_kws={"shrink": 0.5},
        annot=True,
        annot_kws={"size": 10},
        ax=ax,
    )
    # Set bigger fontsize for tick labels for this plot exclusively
    ax.tick_params(axis="both", labelsize=12)

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

    plt.savefig(savepath, bbox_inches="tight")


def plot_scatters(
    data: pd.DataFrame,
    feature_pairs: list[tuple[str, str]],
    savepath: Path,
    figure_width: float = 6.3,
    marker_size: int = 40,
    alpha: float = 0.5,
) -> None:
    num_features = len(feature_pairs)
    nrows = int(num_features / 3) + (num_features % 3 > 0)
    ncols = 3 if num_features > 2 else num_features

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figure_width, figure_width / 2),
    )
    for i, pair in enumerate(feature_pairs):
        # axs[i].scatter(data[pair[0]], data[pair[1]], s=marker_size, alpha=alpha)
        sns.regplot(
            data=data,
            x=pair[0],
            y=pair[1],
            ax=axs[i],
            scatter_kws={"s": marker_size, "alpha": alpha, "color": "#1f77b4"},
            color="red",
            order=1,
            # label="Linear regression line, 95% ci"
        )
        axs[i].set_xlabel(pair[0])
        axs[i].set_ylabel(pair[1])
        # axs[i].legend()
    fig.tight_layout()
    plt.savefig(savepath)


def plot_scatter(
    data: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    savepath: Path,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[int, int] | None = None,
    figure_width: float = 3.15,
    marker_size: int = 40,
    alpha: float = 0.5,
) -> None:
    fig, ax = plt.subplots(figsize=(figure_width, figure_width))
    sns.regplot(
        data=data,
        x=x_feature,
        y=y_feature,
        ax=ax,
        scatter_kws={"s": marker_size, "alpha": alpha, "color": "Gray"},
        color="red",
        order=1,
    )
    if xlim is not None:
        ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
        ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    plt.savefig(savepath)


def plot_feature_importances(
    save_path: Path,
    model: RandomForestRegressor,
    feature_names: list[str],
    num_bars: int = 8,
    figure_width: float = 3.15,
) -> None:
    """Plotting feature importance for tree based methods.

    Choose the num_bars features with heighest importance.
    """
    num_bars = len(feature_names) if len(feature_names) < num_bars else num_bars
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)[
        0:num_bars
    ]
    fig, ax = plt.subplots(figsize=(figure_width, 1.5 * figure_width))
    forest_importances = pd.Series(importances, index=feature_names).sort_values(
        ascending=False
    )
    forest_importances = forest_importances[0:num_bars]  # constrain the bars
    forest_importances.plot(kind="bar", yerr=std, ax=ax, color="Gray")  # 8b9dc3")
    ax.set_xticklabels(
        forest_importances.index,
        rotation=90,
        # rotation_mode="anchor",
        # verticalalignment="top",
    )
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity (MDI)")
    ax.set_xlabel("Feature variables")
    plt.tight_layout()
    plt.grid(True, axis="y", alpha=0.5)
    plt.savefig(save_path)


def plot_pred_error(
    df,
    model_name,
    label,
    savepath,
    xlim=(0, 70000),
    ylim=(0, 70000),
    tick_density=10000,
    figure_width=3.15,
):
    fig, ax = plt.subplots(figsize=(figure_width, figure_width))

    sns.scatterplot(
        data=df,
        x="Ground truth",
        y="Prediction",
        hue="Cement type",
        ax=ax,
        palette={"industrycement": "gray", "microcement": "#aa3a3d"},
    )
    sns.lineplot(
        x=[xlim[0], xlim[1]],
        y=[ylim[0], ylim[1]],
        color="black",
        linestyle="--",
    )
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(np.arange(xlim[0], xlim[1], tick_density))
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_yticks(np.arange(ylim[0], ylim[1], tick_density))

    # ax.set_xticklabels(ax.get_xticks(), rotation=90)

    y_true = df["Ground truth"]
    y_pred = df["Prediction"]

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    ax.set_title(
        f"{model_name} predicting: {label}. \nScores. R2: {r2:.2f}. MSE: {mse:.2e} "
    )

    plt.grid(True, alpha=0.5)
    fig.savefig(savepath)


def make_predictions(
    clf_pipelines: List[Pipeline],
    features: pd.DataFrame,
    labels: pd.Series,
    cv_splits: int,
) -> Tuple[List[pd.DataFrame], List[str]]:
    dfs = []
    model_names = []

    for clf in clf_pipelines:
        y_predicted = cross_val_predict(clf, features, labels, cv=cv_splits)

        binary_coloring = features["Cement type"]

        df_error = pd.DataFrame(
            {
                "Prediction": y_predicted,
                "Ground truth": labels,
                "Cement type": binary_coloring,
            }
        )

        dfs.append(df_error)

        # get the name of the last step of the pipeline if it's named
        if isinstance(clf, Pipeline):
            model_name = type(clf["classifier"]).__name__
        else:
            model_name = clf.__class__.__name__

        model_names.append(model_name)

    return dfs, model_names


def plot_pred_error_models(
    dfs: list[pd.DataFrame],
    model_names: list[str],
    label: str,
    savepath: Path,
    xlim=(0, 70000),
    ylim=(0, 70000),
    tick_density=10000,
    figure_width=3.15,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(figure_width * 3, figure_width))

    for ax, df, model_name in zip(axes, dfs, model_names):
        sns.scatterplot(
            data=df,
            x="Ground truth",
            y="Prediction",
            hue="Cement type",
            ax=ax,
            palette={"industrycement": "gray", "microcement": "#aa3a3d"},
        )
        sns.lineplot(
            x=[xlim[0], xlim[1]],
            y=[ylim[0], ylim[1]],
            color="black",
            linestyle="--",
            ax=ax,
        )
        if xlim:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_xticks(np.arange(xlim[0], xlim[1], tick_density))
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_yticks(np.arange(ylim[0], ylim[1], tick_density))

        y_true = df["Ground truth"]
        y_pred = df["Prediction"]

        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        ax.set_title(
            f"{model_name} predicting: {label}. \nScores. R2: {r2:.2f}. RMSE: {rmse:.2f}"
        )

        ax.grid(True, alpha=0.5)

    fig.savefig(savepath)


# Use this palette in the previous function:
def plot_time_series(
    df,
    temp_cols,
    precip_cols,
    savepath,
    figure_width=3.15,
):
    # Get the 'vlag' color palette
    color_palette = sns.color_palette("vlag", 4)
    # You may want to convert this palette to a list of RGB tuples for matplotlib:
    # color_palette = color_palette.as_hex()

    fig, ax1 = plt.subplots(figsize=(figure_width * 2, figure_width))

    # Create a color iterator
    color_iter = iter(color_palette)

    # Plot temperature data
    for col in temp_cols:
        ax1.plot(df.index, df[col], color=next(color_iter), label=col, linewidth=0.7)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (deg. C)")
    ax1.legend(loc="upper left")

    # Create a second y-axis for precipitation data
    ax2 = ax1.twinx()

    # Plot precipitation data
    for col in precip_cols:
        ax2.plot(df.index, df[col], color=next(color_iter), label=col, linewidth=0.7)

    ax2.set_ylabel("Precipitation (mm)")
    ax2.legend(loc="upper right")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(savepath)


def plot_grouting_data(
    data: pd.DataFrame,
    x_name: str,
    y_names: list[str],
    savepath: Path,
    figure_width=3.15,
) -> None:
    data = data.loc[data[x_name] < 56000, :]

    fig, axes = plt.subplots(
        figsize=(2 * figure_width, figure_width), nrows=3, ncols=1, sharex=True
    )

    for ax, feature in zip(axes, y_names):
        ax.plot(data[x_name], data[feature], color="Gray")
        ax.scatter(data[x_name], data[feature], color="#aa3a3d", s=1.5)
        # ax.stem(data[x_name], data[feature])
        ax.set_ylabel(feature, rotation=45)
        ax.grid(visible=True)

    plt.xlabel("Profile number")
    plt.tight_layout()
    plt.savefig(savepath)


# Legacy functions
#################################################################################


def plot_prediction_error(
    y_pred, y_true, savepath, binary_feature, xlim=None, ylim=None, figure_width=3.15
):
    """
    NOTE: OLD SCIKIT LEARN BASED VERSION

    Generate a prediction error plot from predicted values and true values and save it
    to a file.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        The predicted target values for the input samples.
    y_true : array-like of shape (n_samples,)
        The true target values for the input samples.
    savepath : str
        The file path to save the plot.
    xlim : tuple, optional, default: None
        The x-limits of the plot.
    ylim : tuple, optional, default: None
        The y-limits of the plot.
    """
    fig, ax = plt.subplots(figsize=(figure_width, figure_width))

    PredictionErrorDisplay.from_predictions(
        y_true,
        y_pred,
        kind="actual_vs_predicted",
        ax=ax,
        scatter_kwargs={
            "alpha": 0.5,
            "s": 12,
            # "color": ["red" if b else "blue" for b in binary_feature],
            "color": np.where(binary_feature, "red", "blue"),
            "label": [
                "Industry cement" if b else "Micro cement" for b in binary_feature
            ],
        },
    )
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(np.arange(xlim[0], xlim[1], 10000))
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_yticks(np.arange(ylim[0], ylim[1], 10000))

    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    ax.set_title(f"Score of trained model \nR2: {r2:.2f}. MSE: {mse:.2e} ")

    plt.grid(True, alpha=0.5)
    # handles, labels = pred_disp.ax_.collections[0].legend_elements()
    # ax.legend(handles, labels, loc="best")
    fig.savefig(savepath)


def plot_scatter2(
    data: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    hue_feature: str,
    savepath: Path,
    figure_width: float = 3.15,
    marker_size: int = 40,
    alpha: float = 0.5,
) -> None:
    sns.set_style("whitegrid")
    fig = sns.lmplot(
        data=data,
        x=x_feature,
        y=y_feature,
        hue=hue_feature,
        scatter_kws={"s": marker_size, "alpha": alpha},
        # markers=["o", "s", "D"],
        height=figure_width,
        aspect=1.5,
        legend=False,
    )
    ax = fig.axes[0, 0]
    ax.set_xlim(xmin=0, xmax=70000)
    ax.set_ylim(ymin=0, ymax=250)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend(title=hue_feature, loc="upper left")
    fig.tight_layout()
    plt.savefig(savepath)
