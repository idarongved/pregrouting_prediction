"""Train and eval a machine learning model"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as pprint
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.plotting import (
    plot_feature_importances,
    plot_pred_error,
    plot_prediction_error,
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
TRAINING_FEATURES = train_features_chosen
TEST_SIZE = 0.3
SPLITTING_SEED = 42
MODEL_SEED = 42
CV_SPLITS = 3

path_model_ready_data = Path("./data/model_ready/model_ready_dataset.xlsx")
path_feature_importance = Path(
    f"./plots/feature_importance_{('_').join(LABEL.split(' '))}.png"
)
path_result_scatter = Path(f"./plots/result_scatter_{('_').join(LABEL.split(' '))}.png")
path_result_scatter_residuals = Path(
    f"./plots/result_scatter_residuals_{('_').join(LABEL)}.png"
)

# READ IN DATA, CHOOSE FEATURES AND LABEL
######################################################
df = pd.read_excel(path_model_ready_data, header=0, index_col=0)
df = df.sample(
    frac=1, random_state=42
)  # to break sequences in data for crossvalidation, same as shuffling in train_test_sp

# choose features and label
# df = df[df["Cement type"]=="Industrisement"] #Mikrosement
features = df[TRAINING_FEATURES]
labels = df[LABEL]

# One-hot-encoding of categorical variables
features_encoded = pd.get_dummies(features)
# features_encoded = encode_categorical_features(features)  # alternative

# possible feature engineering with ugini? Eventually in a separat script

# # split data
# x_train, x_val, y_train, y_val = train_test_split(
#     features,
#     labels,
#     test_size=TEST_SIZE,
#     shuffle=True,
#     stratify=labels,
#     random_state=SPLITTING_SEED,
# )

# TRAIN A REGRESSION MODEL USING CROSS VALIDATION
########################################################

# define pipeline including a feature encoder for one-hot-encoding of categorical variables
clf_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            ExtraTreesRegressor(verbose=False, n_jobs=-1, random_state=MODEL_SEED),
            # RandomForestRegressor(verbose=False, n_jobs=-1, random_state=MODEL_SEED),
            # HistGradientBoostingRegressor(verbose=False, random_state=MODEL_SEED),
            # KNeighborsRegressor(n_neighbors=5),
            # DummyRegressor(strategy="mean"),
        ),
    ],
    verbose=False,
)

metrics = {
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "r2": "r2",
}


# train with cross validation and report R2, MSE, RMSE
scores = cross_validate(
    clf_pipeline,
    features_encoded,
    labels,
    cv=CV_SPLITS,
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

# # log experiment to mlflow

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

plot_pred_error(
    df_error,
    type(clf_pipeline["classifier"]).__name__,  # type: ignore
    LABEL,
    path_result_scatter,
    xlim=(0, 70000),
    ylim=(0, 70000),
    figure_width=6.3,
)
