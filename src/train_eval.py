"""Train and eval a machine learning model"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from plotting_lib import (
    plot_feature_importances,
    plot_pred_error,
    plot_prediction_error,
)
from rich import print as pprint
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
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
    StandardScaler,
    quantile_transform,
)

from src.utility import (
    encode_categorical_features,
    train_features_chosen,
    train_features_max,
    train_features_no_previous,
    train_features_small,
)

# CONSTANTS
##########################################
LABEL = "Total grout take"  # "Total grout take" "Grouting time"
LABEL_TRANSFORM = "log"  # log, quantile, no
TRAINING_FEATURES = train_features_chosen
TEST_SIZE = 0.3
SPLITTING_SEED = 0
MODEL_SEED = 0
CV_SPLITS = 4
MWD_DATA = "longholes"  # longholes, blastholes - choose which MWD-dataset to use
TRAIN_TEST_SPLIT = False

# data paths
path_model_ready_data_blasting = Path(
    "./data/model_ready/model_ready_dataset_blasting.xlsx"
)
path_model_ready_data_longholes = Path(
    "./data/model_ready/model_ready_dataset_longholes.xlsx"
)

# plot paths
path_feature_importance = Path(
    f"./plots/feature_importance_{('_').join(LABEL.split(' '))}.png"
)
path_result_scatter = Path(f"./plots/result_scatter_{('_').join(LABEL.split(' '))}.png")
path_result_scatter_residuals = Path(
    f"./plots/result_scatter_residuals_{('_').join(LABEL)}.png"
)

# READ IN DATA, CHOOSE FEATURES AND LABEL
######################################################
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

# TODO: check out label transform. The labels have not a gaussian distribution: https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py
if LABEL_TRANSFORM == "log":
    labels = np.log1p(labels)
    plt.hist(labels, bins=20)
    plt.savefig("./data/temporary/label_transform_log.png")

elif LABEL_TRANSFORM == "quantile":
    labels = quantile_transform(
        labels.to_frame(), output_distribution="normal", n_quantiles=340
    ).squeeze()
    plt.hist(labels, bins=20)
    plt.savefig("./data/temporary/label_transform_quantiile.png")


# One-hot-encoding of categorical variables
features_encoded = pd.get_dummies(features)
# features_encoded = encode_categorical_features(features)  # alternative

# TODO: possible feature engineering with ugini? Eventually in a separat script


# TRAIN A REGRESSION MODEL USING TRAIN TEST SPLIT
########################################################


# define pipeline including a feature encoder for one-hot-encoding of categorical variables
clf_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            ExtraTreesRegressor(verbose=False, n_jobs=-1, random_state=MODEL_SEED),
            # LGBMRegressor(verbose=0, n_jobs=-1, random_state=MODEL_SEED),
            # RandomForestRegressor(verbose=False, n_jobs=-1, random_state=MODEL_SEED),
            # HistGradientBoostingRegressor(verbose=False, random_state=MODEL_SEED),
            # KNeighborsRegressor(n_neighbors=5),
            # DummyRegressor(strategy="mean"),
        ),
    ],
    verbose=False,
)

if TRAIN_TEST_SPLIT:
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

    pprint(f"R2 from train-test-split. Testsize: {TEST_SIZE}. R2: {r2}")
    exit()

# TRAIN A REGRESSION MODEL USING CROSS VALIDATION
########################################################


metrics = {
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "r2": "r2",
}


# train with cross validation and report R2, MSE, RMSE
# control the splitting process with ShuffleSplit
splitter = ShuffleSplit(n_splits=CV_SPLITS, test_size=0.25, random_state=SPLITTING_SEED)
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

# SUMMARY STATISTICS AND PLOTS
########################################################

pprint("Metrics: ", metrics)

# TODO: log experiment to mlflow

# plot feature importance
feature_names = features_encoded.columns.tolist()
plot_feature_importances(
    path_feature_importance, trained_clf, feature_names, num_bars=10
)

# plot a distribution plot of predicted and true values
y_predicted = cross_val_predict(clf_pipeline, features_encoded, labels, cv=CV_SPLITS)
binary_coloring = features_encoded["Cement type_Industrisement"]
binary_coloring = features["Cement type"]

df_error = pd.DataFrame(
    {"Prediction": y_predicted, "Ground truth": labels, "Cement type": binary_coloring}
)

if LABEL == "Total grout take":
    # plot_pred_error(
    #     df_error,
    #     type(clf_pipeline["classifier"]).__name__,  # type: ignore
    #     LABEL,
    #     path_result_scatter,
    #     xlim=(0, 70000),
    #     ylim=(0, 70000),
    #     tick_density = 10000,
    #     figure_width=6.3,
    # )

    # plot_pred_error(
    #     df_error,
    #     type(clf_pipeline["classifier"]).__name__,  # type: ignore
    #     LABEL,
    #     path_result_scatter,
    #     xlim=(-3, 3),
    #     ylim=(-3, 3),
    #     tick_density=2,
    #     figure_width=6.3,
    # )

    plot_pred_error(
        df_error,
        type(clf_pipeline["classifier"]).__name__,  # type: ignore
        LABEL,
        path_result_scatter,
        xlim=(7, 12),
        ylim=(7, 12),
        tick_density=5,
        figure_width=6.3,
    )

if LABEL == "Grouting time":
    # plot_pred_error(
    #     df_error,
    #     type(clf_pipeline["classifier"]).__name__,  # type: ignore
    #     LABEL,
    #     path_result_scatter,
    #     xlim=(0, 40),
    #     ylim=(0, 30),
    #     tick_density=5,
    #     figure_width=6.3,
    # )

    plot_pred_error(
        df_error,
        type(clf_pipeline["classifier"]).__name__,  # type: ignore
        LABEL,
        path_result_scatter,
        xlim=(1, 4),
        ylim=(1, 4),
        tick_density=1,
        figure_width=6.3,
    )
