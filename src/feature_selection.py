"""
Finding optimum features and labels in an automated way
- Investigating feature engineering
- Investigating label transform
- Feature selection - investigating optimum features

Run: python src/feature_selection.py

Output: List of selected features
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from featurewiz import featurewiz
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import quantile_transform
from yellowbrick.model_selection import RFECV, DroppingCurve

# from verstack import FeatureSelector
from src.utility import all_relevant_features

# Constants
#####################################################
# Choose label and auto feature method
# In addition you need to choose model type for the RFECV method
# The FEATUREVIZ method use the model XGBOOST by default
LABEL = "Grouting time"  # "Total grout take" "Grouting time"
FEATUREVIZ = True
YELLOWBRICK_RFECV = False
MODEL_TYPE = "extra_trees"  # random_forest extra_trees

VISUALIZE_LABEL_TRANSFORM = False
MODEL_SEED = 0
SPLITTING_SEED = 0
CV_SPLITS = 3
TEST_SIZE = 0.2
OPTIMIZED_METRIC = "rmse"

path_model_ready_data_longholes = Path(
    "./data/model_ready/model_ready_dataset_longholes.xlsx"
)

# READ IN DATA
####################################################
df = pd.read_excel(path_model_ready_data_longholes, header=0, index_col=0)
# df = df.loc[:, train_features_max]

labels = df[LABEL]
features = df[all_relevant_features]
features_encoded = pd.get_dummies(features)
features_and_label = pd.concat([features, labels], axis=1)

# label_short = ("_").join(LABEL.split(" "))
# # get best params for model from hyperparameter optimization
# path_params = Path(
#     "./src/config", f"best_params_{MODEL_TYPE}_{OPTIMIZED_METRIC}_{label_short}.yaml"
# )
# # Load the YAML file into a dictionary
# with open(path_params, "r") as file:
#     optimized_params = yaml.safe_load(file)

model = ExtraTreesRegressor(random_state=MODEL_SEED, n_jobs=-1) #, **optimized_params)

# LABEL TRANSFORM - VISUALIZATION
#####################################################
# check out label transform. The labels have not a gaussian distribution: https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py
# visualize transform

if VISUALIZE_LABEL_TRANSFORM:
    # log
    plt.hist(np.log1p(labels), bins=20)
    plt.savefig("./data/temporary/label_transform_log.png")

    # quantile transform
    plt.hist(
        quantile_transform(
            labels.to_frame(), output_distribution="normal", n_quantiles=340
        ).squeeze(),
        bins=20,
    )
    plt.savefig("./data/temporary/label_transform_quantiile.png")

    # sqrt transform
    plt.hist(np.sqrt(labels), bins=20)
    plt.savefig("./data/temporary/label_transform_quantiile.png")


# FEATURE ENGINEERING
#######################################################
# TODO: possible feature engineering.
# One option is with ugini? Eventually in a separat script


# FEATURE SELECTION
##########################################################

# FEATURE VIZ
########################
# TODO: consider to add a category_encoder
if FEATUREVIZ:
    features, df_result = featurewiz(
        features_and_label,
        target=LABEL,
        corr_limit=0.40,
        # feature_engg="interactions",
        verbose=2,
    )

    print("Num. features selected: ", len(features))
    print("Num. features originally: ", len(df.columns))
    print("---------------------------------------------")
    print(features)
    pd.Series(features).to_csv(
        "./data/temporary/selected_features.csv", index_label=False
    )
    df_result.to_excel("./data/temporary/feature_selection_process.xlsx")


# YELLOWBRICK RFECV
############################################################
# check out: https://www.scikit-yb.org/en/latest/quickstart.html

if YELLOWBRICK_RFECV:
    print("Starting RFECV")

    # RFECV
    # Instantiate RFECV visualizer with Extratrees
    splitter = ShuffleSplit(
        n_splits=CV_SPLITS, test_size=TEST_SIZE, random_state=SPLITTING_SEED
    )
    visualizer = RFECV(model, cv=splitter, scoring="r2", kwargs={"verbose": 2})

    visualizer.fit(features_encoded, labels)  # Fit the data to the visualizer
    visualizer.show(outpath="./plots/rfecv_plot.png")  # Finalize and render the figure
    plt.close()

    mask_features = visualizer.support_
    print("Selected features: ", all_relevant_features[mask_features])

    # DROPPING CURVE
    # Initialize visualizer with estimator
    visualizer = DroppingCurve(model)

    # Fit the data to the visualizer
    visualizer.fit(features_encoded, labels)
    # Finalize and render the figure
    visualizer.show(outpath="./plots/feature_dropping_curve.png")
    plt.close()

# VERSTACK. TODO: run in a separate test environment where I only read in the data
##########################

# if VERSTACK:

#     FS = FeatureSelector(objective = 'regression', auto = True)
#     selected_feats = FS.fit_transform(X, y)
#     print(selected_feats)#     print(selected_feats)
