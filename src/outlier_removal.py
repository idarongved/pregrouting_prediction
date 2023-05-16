"""Functionality to detect, visualize and remove multivariate outliers"""

from src.utility import train_features_chosen

from pyod.models.lof import LOF
from pyod.models.iforest import IForest
import pandas as pd
from pathlib import Path

# TODO: check out other methods in pyod. Ecod etc. https://towardsdatascience.com/how-to-perform-multivariate-outlier-detection-in-python-pyod-for-machine-learning-b0a9c557a21c

path_model_ready_data = Path("./data/model_ready/model_ready_dataset_blasting.xlsx")
LABEL = "Total grout take"  # "Total grout take" "Grouting time"
TRAINING_FEATURES = train_features_chosen

df = pd.read_excel(path_model_ready_data, header=0, index_col=0)
df = df.sample(
    frac=1, random_state=42
)  # to break sequences in data for crossvalidation

# choose features and label
features = df[TRAINING_FEATURES]
features = pd.get_dummies(features) #one-hot-encode categorical variables
labels = df[LABEL]

# WITH LOF
########################################
# Initialize
lof = LOF(n_neighbors=30).fit(features)

probs = lof.predict_proba(features)
probs[:5]

# Set a confidence threshold
threshold = 0.8

# Create a mask that returns True if probs over threshold
is_outlier = probs[:, 1] > threshold
outliers_X_probs = features[is_outlier]

# Count up the outliers
num_outliers = len(outliers_X_probs)
print(f"The number of outliers with LOF: {num_outliers}")
print(f"Percentage of outliers: {num_outliers / len(features):.4f}")


# WITH ISOLATION FOREST
########################################
iforest = IForest(n_estimators=500).fit(features)

probs = iforest.predict_proba(features)
probs[:5]

# Set a confidence threshold
threshold = 0.8

# Create a mask that returns True if probs over threshold
is_outlier = probs[:, 1] > threshold
outliers_X_probs = features[is_outlier]

# Count up the outliers
num_outliers = len(outliers_X_probs)
print(f"The number of outliers with Isolation forest: {num_outliers}")
print(f"Percentage of outliers: {num_outliers / len(features):.4f}")

# Create a mask that returns True if probs are below the threshold
non_outlier_mask = probs[:, 1] <= threshold
outlier_indices = features.index[is_outlier]

print("These samples are outliers: ", outlier_indices)
