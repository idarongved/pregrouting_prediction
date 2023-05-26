"""Train and eval a machine learning model"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from plotting_lib import (
    make_predictions,
    plot_feature_importances,
    plot_pred_error,
    plot_pred_error_models,
    plot_prediction_error,
)
from rich import print as pprint
from rich.console import Console
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    ShuffleSplit,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    QuantileTransformer,
    StandardScaler,
    quantile_transform,
)
from sklearn.tree import DecisionTreeRegressor

from src.utility import (
    all_relevant_features,
    auto_features_take,
    auto_features_time,
    encode_categorical_features,
    train_features_manual_domain,
    train_features_no_previous,
    train_features_small,
)

# CONSTANTS
##########################################

# MOST IMPORTANT CONSTANTS TO CHOOSE: LABEL, TRAINING_FEATURES, MODEL_TYPE_PARAMS
# TO SPEED THINGS UP: TURN OFF PLOTTING

# data
LABEL = "Grouting time"  # "Total grout take" "Grouting time"
# choose between: train_features_manual_domain, all_relevant_features, auto_features_take
# auto_features_time, train_features_no_previous, train_features_small
TRAINING_FEATURES = auto_features_time
MWD_DATA = "longholes"  # longholes, blastholes - choose which MWD-dataset to use
OUTLIER_REMOVAL = False

# model - general
MODEL_TYPE_PARAMS = "extra_trees"  # random_forest extra_trees
# NOTE: In addition you have to uncomment the right model when the model is defined
TEST_SIZE = 0.2
SPLITTING_SEED = 0
MODEL_SEED = 0
CV_SPLITS = 3
OPTIMIZED_METRIC = "rmse"  # hyperparameter optimized metric

RUN_CROSS_VALIDATION = True

# model - investigation
RUN_TRAIN_TEST_SPLIT = False  # run train test split instead of cross validation
LABEL_TRANSFORM = False  # used in label transform experimentation in train-test-split
ML_INVESTIGATION_PLOTS = True

# plotting and analysis
PLOTTING = True  # plot or not
FEATURE_IMPORTANCE = True  # feature importance plot
PREDICTION_INTERVAL = False  # examplify prediction intevals
PLOT_THREE_MODELS = True

# data paths
path_model_ready_data_blasting = Path(
    "./data/model_ready/model_ready_dataset_blasting.xlsx"
)
path_model_ready_data_longholes = Path(
    "./data/model_ready/model_ready_dataset_longholes.xlsx"
)
path_model_ready_data_longholes_outlier = Path(
    "./data/model_ready/model_ready_dataset_longholes_outlier.xlsx"
)  # dataset with outliers_removed

# plot paths
label_short = ("_").join(LABEL.split(" "))
path_feature_importance = Path(f"./plots/feature_importance_{label_short}.png")
path_result_scatter = Path(f"./plots/result_scatter_{label_short}.png")
path_result_scatter_residuals = Path(
    f"./plots/result_scatter_residuals_{label_short}.png"
)
path_3_models = Path(f"./plots/3_models_{label_short}.png")


console = Console()  # pretty printing

if RUN_TRAIN_TEST_SPLIT and PLOTTING:
    warnings.warn(
        "Both RUN_TRAIN_TEST_SPLIT and PLOTTING are set to True. Plotting is not set up\
            for running with train-test-split, only for crossvalidation."
    )

# READ IN DATA, CHOOSE FEATURES AND LABEL
######################################################

# choose MWD-data from previous blastholes or on face grouting holes
if MWD_DATA == "blastholes":
    df = pd.read_excel(path_model_ready_data_blasting, header=0, index_col=0)
elif MWD_DATA == "longholes":
    df = pd.read_excel(path_model_ready_data_longholes, header=0, index_col=0)
else:
    raise ValueError("not valid drilling type")

df = df.sample(
    frac=1, random_state=42
)  # to break sequences in data for crossvalidation, same as shuffling in train_test_sp


# choose features and label
# df = df[df["Cement type"]=="Industrisement"] #Mikrosement
features = df[TRAINING_FEATURES]
labels = df[LABEL]

# TODO: try to separate the label in two and classify binary

# One-hot-encoding of categorical variables
features_encoded = pd.get_dummies(features)
# features_encoded = encode_categorical_features(features)  # alternative


# DEFINE THE MODEL PIPELINE
########################################################

# get best params from hyperparameter optimization
path_params = Path(
    "./src/config",
    f"best_params_{MODEL_TYPE_PARAMS}_{OPTIMIZED_METRIC}_{label_short}.yaml",
)
# Load the YAML file into a dictionary
with open(path_params, "r") as file:
    optimized_params = yaml.safe_load(file)

# define pipeline
clf_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            ExtraTreesRegressor(
                verbose=False, n_jobs=-1, random_state=MODEL_SEED, **optimized_params
            ),
            # Ridge(random_state=MODEL_SEED),
            # LinearRegression(),
            # LGBMRegressor(verbose=0, n_jobs=-1, random_state=MODEL_SEED),
            # RandomForestRegressor(verbose=False, n_jobs=-1, random_state=MODEL_SEED, **optimized_params),
            # HistGradientBoostingRegressor(verbose=False, random_state=MODEL_SEED),
            # KNeighborsRegressor(n_neighbors=5, **optimized_params),
            # DummyRegressor(strategy="mean"),
        ),
    ],
    verbose=False,
)

# TRAIN A REGRESSION MODEL USING TRAIN TEST SPLIT
########################################################

if RUN_TRAIN_TEST_SPLIT:
    # # split data
    x_train, x_val, y_train, y_val = train_test_split(
        features_encoded,
        labels,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=SPLITTING_SEED,
    )

    fitted_clf = clf_pipeline.fit(x_train, y_train)
    y_predict = fitted_clf.predict(x_val)

    r2 = r2_score(y_val, y_predict)
    rmse = mean_squared_error(y_val, y_predict, squared=False)

    pprint(f"R2 from train-test-split. Testsize: {TEST_SIZE}. R2: {r2}")
    pprint(f"RMSE from train-test-split. Testsize: {TEST_SIZE}. RMSE: {rmse}")

    if ML_INVESTIGATION_PLOTS:
        from yellowbrick.model_selection import LearningCurve
        from yellowbrick.regressor import PredictionError, ResidualsPlot

        model = ExtraTreesRegressor(
            random_state=MODEL_SEED, **optimized_params
        )  # LinearRegression()

        # residualsplot to show trends in residuals. Should have a centered histogram
        visualizer = ResidualsPlot(model)

        # visualizer = ResidualsPlot(LinearRegression())
        visualizer.fit(x_train, y_train)
        visualizer.score(x_val, y_val)
        visualizer.show(outpath="./plots/residualplot.png")
        plt.close()

        # prediction error plot - show identity line vs fitted line. Show potential for transformation
        visualizer = PredictionError(model)
        visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(x_val, y_val)  # Evaluate the model on the test data
        visualizer.show(
            outpath="./plots/prediction_error_plot.png"
        )  # Finalize and render the figure
        plt.close()

        # learning curve
        visualizer = LearningCurve(model, scoring="r2")
        visualizer.fit(features_encoded, labels)  # Fit the data to the visualizer
        visualizer.show(
            outpath="./plots/learningcurve.png"
        )  # Finalize and render the figure
        plt.close()

    if LABEL_TRANSFORM:  # experiments with label transform
        # Transforming the target
        clf_pipeline_trans = TransformedTargetRegressor(
            # regressor=clf_pipeline, func=np.sqrt, inverse_func=np.square
            #     transformer=QuantileTransformer(
            #         output_distribution="normal", n_quantiles=235))
            regressor=clf_pipeline,
            func=np.square,
            inverse_func=np.sqrt,
        )

        fitted_clf = clf_pipeline_trans.fit(x_train, y_train)
        y_predict = fitted_clf.predict(x_val)

        r2 = r2_score(y_val, y_predict)
        rmse = mean_squared_error(y_val, y_predict, squared=False)

        pprint(f"R2 transformed from train-test-split. Testsize: {TEST_SIZE}. R2: {r2}")
        pprint(f"RMSE from train-test-split. Testsize: {TEST_SIZE}. RMSE: {rmse}")
        exit()

# TRAIN A REGRESSION MODEL USING CROSS VALIDATION
########################################################

if RUN_CROSS_VALIDATION:
    metrics = {
        "neg_mean_squared_error": "neg_mean_squared_error",
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    # train with cross validation and report R2, MSE, RMSE
    # control the splitting process with ShuffleSplit
    splitter = ShuffleSplit(
        n_splits=CV_SPLITS, test_size=TEST_SIZE, random_state=SPLITTING_SEED
    )
    scores = cross_validate(
        clf_pipeline,
        features_encoded,
        labels,
        cv=splitter,
        n_jobs=-1,
        scoring=metrics,
        return_train_score=True,
        return_estimator=True,
    )
    trained_clf = scores["estimator"][1]["classifier"]  # choosing regressor from fold 2
    r2 = scores["test_r2"]
    mse = scores["test_neg_mean_squared_error"]
    rmse = scores["test_neg_root_mean_squared_error"]

    metrics = dict(
        R2=r2,
        R2_mean=np.mean(r2),
        R2_std=np.std(r2),
        MSE=np.mean(mse),
        RMSE=np.mean(rmse),
    )
    console.print("Metrics: ", metrics)

# SUMMARY STATISTICS AND PLOTS
########################################################

# TODO: log experiment to mlflow

if PLOTTING:
    # plot a distribution plot of predicted and true values
    y_predicted = cross_val_predict(
        clf_pipeline, features_encoded, labels, cv=CV_SPLITS
    )
    # binary_coloring = features_encoded["Cement type_Industrisement"]
    binary_coloring = features["Cement type"]

    df_error = pd.DataFrame(
        {
            "Prediction": y_predicted,
            "Ground truth": labels,
            "Cement type": binary_coloring,
        }
    )

    if LABEL == "Total grout take":
        console.print("[green]Plotting pred error grout take")
        plot_pred_error(
            df_error,
            # type(clf_pipeline["classifier"]).__name__,  # type: ignore
            type(clf_pipeline["classifier"]).__name__,  # type: ignore
            LABEL,
            path_result_scatter,
            xlim=(0, 70000),
            ylim=(0, 70000),
            tick_density=10000,
            figure_width=6.3,
        )

        if PLOT_THREE_MODELS:
            console.print("[green]Plotting pred error grout take - 3 models")
            classifiers = [
                DecisionTreeRegressor(random_state=MODEL_SEED),
                RandomForestRegressor(
                    verbose=False,
                    n_jobs=-1,
                    random_state=MODEL_SEED,
                ),
                ExtraTreesRegressor(
                    verbose=False,
                    n_jobs=-1,
                    random_state=MODEL_SEED,
                    **optimized_params,
                ),
            ]
            pipelines = []

            for clf in classifiers:
                pipeline = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("classifier", clf),
                    ],
                    verbose=False,
                )
                pipelines.append(pipeline)
            dfs, model_names = make_predictions(
                pipelines, features, features_encoded, labels, CV_SPLITS
            )

            plot_pred_error_models(
                dfs, model_names, LABEL, path_3_models, tick_density=20000
            )

    if LABEL == "Grouting time":
        console.print("[green]Plotting pred error grout time")
        plot_pred_error(
            df_error,
            type(clf_pipeline["classifier"]).__name__,  # type: ignore
            LABEL,
            path_result_scatter,
            xlim=(0, 40),
            ylim=(0, 30),
            tick_density=5,
            figure_width=6.3,
        )

        if PLOT_THREE_MODELS:
            console.print("[green]Plotting pred error grout time - 3 models")
            classifiers = [
                DecisionTreeRegressor(random_state=MODEL_SEED),
                RandomForestRegressor(
                    verbose=False,
                    n_jobs=-1,
                    random_state=MODEL_SEED,
                ),
                ExtraTreesRegressor(
                    verbose=False,
                    n_jobs=-1,
                    random_state=MODEL_SEED,
                    **optimized_params,
                ),
            ]
            pipelines = []

            for clf in classifiers:
                pipeline = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("classifier", clf),
                    ],
                    verbose=False,
                )
                pipelines.append(pipeline)
            dfs, model_names = make_predictions(
                pipelines, features, features_encoded, labels, CV_SPLITS
            )

            plot_pred_error_models(
                dfs,
                model_names,
                LABEL,
                path_3_models,
                xlim=(0, 40),
                ylim=(0, 30),
                tick_density=5,
            )
    if FEATURE_IMPORTANCE:
        # plot feature importance
        console.print("[green]Plotting feature importance")
        feature_names = features_encoded.columns.tolist()
        plot_feature_importances(
            path_feature_importance, trained_clf, feature_names, num_bars=10
        )

# CALCULATING PREDICTION INTERVALS
#######################################################

if PREDICTION_INTERVAL:
    # Calculate the residuals
    residuals = labels - y_predicted

    # Estimate the standard deviation of the residuals
    std_dev = np.std(residuals)

    # Define the z-score for a 95% confidence level
    z_score = 1.96

    # Calculate the upper and lower bounds of the prediction intervals
    upper_bound = y_predicted + (z_score * std_dev)
    lower_bound = y_predicted - (z_score * std_dev)

    print("Uncertainty (z x std): ", z_score * std_dev)

    # Calculate the differences between upper and lower bounds
    differences = upper_bound - lower_bound

    # Calculate the mean of the differences
    mean_difference = np.mean(differences)
    median_difference = np.median(differences)

    print("Mean Difference:", mean_difference)
    print("Median Difference:", median_difference)

    # pprint(f"Prediction interval. Lower bound: {lower_bound}. Upper bound: {upper_bound}.")

    # Calculate the minimum, maximum, mean, median, and standard deviation of the upper bounds
    upper_bound_min = np.min(upper_bound)
    upper_bound_max = np.max(upper_bound)
    upper_bound_mean = np.mean(upper_bound)
    upper_bound_median = np.median(upper_bound)
    upper_bound_std = np.std(upper_bound)

    # Calculate the minimum, maximum, mean, median, and standard deviation of the lower bounds
    lower_bound_min = np.min(lower_bound)
    lower_bound_max = np.max(lower_bound)
    lower_bound_mean = np.mean(lower_bound)
    lower_bound_median = np.median(lower_bound)
    lower_bound_std = np.std(lower_bound)

    print("Upper Bound Summary:")
    print("Min:", upper_bound_min)
    print("Max:", upper_bound_max)
    print("Mean:", upper_bound_mean)
    print("Median:", upper_bound_median)
    print("Standard Deviation:", upper_bound_std)
    print()

    print("Lower Bound Summary:")
    print("Min:", lower_bound_min)
    print("Max:", lower_bound_max)
    print("Mean:", lower_bound_mean)
    print("Median:", lower_bound_median)
    print("Standard Deviation:", lower_bound_std)
