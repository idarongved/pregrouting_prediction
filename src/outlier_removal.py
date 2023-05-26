"""Functionality to detect, visualize and remove multivariate outliers"""

from pathlib import Path

import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from rich.console import Console

from src.utility import train_features_manual_domain

# TODO: check out other methods in pyod. Ecod etc. https://towardsdatascience.com/how-to-perform-multivariate-outlier-detection-in-python-pyod-for-machine-learning-b0a9c557a21c

# Constants
#####################################################
LABEL = "Total grout take"  # "Total grout take" "Grouting time"
TRAINING_FEATURES = train_features_manual_domain
# Set a confidence threshold before removing an outlier
THRESHOLD = 0.8
OUTLIER_REMOVAL_METHOD = "hard_values"  # lof, hard_values, isolation_forest

path_model_ready_data = Path("./data/model_ready/model_ready_dataset_longholes.xlsx")
path_model_ready_data_outlier = Path(
    "./data/model_ready/model_ready_dataset_longholes_outlier.xlsx"
)
console = Console()

# GET DATA
##########################################################

df = pd.read_excel(path_model_ready_data, header=0, index_col=0)
df = df.sample(
    frac=1, random_state=42
)  # to break sequences in data for crossvalidation

# choose features and label
features = df[TRAINING_FEATURES]
features = pd.get_dummies(features)  # one-hot-encode categorical variables
labels = df[LABEL]
features_labels = pd.concat([features, labels], axis=1)

# INVESTIGATING MULTI FEATURE OUTLIER REMOVAL WITH DIFFERENT TECHNIQUES
#########################################################

# WITH LOF (Local outlier factor)
########################################
# Initialize
lof = LOF(n_neighbors=30).fit(features)

probs = lof.predict_proba(features)
probs[:5]

# Create a mask that returns True if probs over threshold
is_outlier = probs[:, 1] > THRESHOLD
outliers_X_probs = features[is_outlier]

outlier_indices_lof = features.index[is_outlier]
print("These samples are outliers: ", outlier_indices_lof)

# Count up the outliers
num_outliers = len(outliers_X_probs)
print(f"The number of outliers with LOF: {num_outliers}")
print(f"Percentage of outliers: {num_outliers / len(features):.4f}")
console.print(
    "These are outlier samples using LOF: \n", features_labels.iloc[outlier_indices_lof]
)


# WITH ISOLATION FOREST
########################################
iforest = IForest(n_estimators=500).fit(features)
probs = iforest.predict_proba(features)
probs[:5]

# Create a mask that returns True if probs over threshold
is_outlier = probs[:, 1] > THRESHOLD
outliers_X_probs = features[is_outlier]

# Count up the outliers
num_outliers = len(outliers_X_probs)
print(f"The number of outliers with Isolation forest: {num_outliers}")
print(f"Percentage of outliers: {num_outliers / len(features):.4f}")

# Create a mask that returns True if probs are below the threshold
non_outlier_mask = probs[:, 1] <= THRESHOLD
outlier_indices_isolation_forest = features.index[is_outlier]
print("These samples are outliers: ", outlier_indices_isolation_forest)
console.print(
    "These are outlier samples using Isolation Forest: \n",
    features_labels.iloc[outlier_indices_isolation_forest],
)

# CHANGE DATASET
########################################

match OUTLIER_REMOVAL_METHOD:
    case "lof":
        df = df.drop(outlier_indices_lof)
    case "isolation_forest":
        df = df.drop(outlier_indices_isolation_forest)
    case "hard_values":
        outlier_limits = {
            "Number of holes": 70,
            "Grouting time": 35,
            "Total grout take": 70000,
            "Prev. grouting time": 35,
            "Prev. grout take": 70000,
        }
        print("Num features before manual outlier removal: ", df.shape)

        for key, val in outlier_limits.items():
            mask = df[key] < val
            df = df.loc[mask, :]
        print("Num features after manual outlier removal: ", df.shape)
    case _:
        raise ValueError("method is not implemented")


df.to_excel(path_model_ready_data_outlier)
