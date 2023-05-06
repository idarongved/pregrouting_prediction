"""Preprocessing fra raw data to model ready data"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.plotting import plot_barplots, plot_correlation_matrix, plot_histograms
from src.utility import barplot_features, correlation_features, hist_features

# CONSTANTS
###############################################
MERGE_TOLERANCE = 2

path_geology = Path("./data/raw/UDK01_mwd_Q_rocktype.csv")
path_grouting = Path("./data/raw/SikrReport32003_0_980.csv")
path_feature_histograms = Path("./plots/feature_histograms.png")
path_feature_barplots = Path("./plots/feature_barplots.png")
path_correlation_plot = Path("./plots/correlation_matrix.png")

# READ IN DATA CONTAINING GEOLOGY AND PREPROCESS
###############################################
df_geology = pd.read_csv(path_geology, delimiter=";")
df_geology = df_geology.sort_values("PegStart")
df_geology["PegStart"] = df_geology["PegStart"].round(1)


# change types of features
cols_to_convert = [
    "Q",
    "Jr",
    "Ja",
    "Jw",
    "Jn",
    "ContourWidth",
    "SRF",
]
for col in cols_to_convert:
    df_geology[col] = df_geology[col].apply(lambda x: float(str(x).replace(",", ".")))

df_geology["logQ"] = np.log(df_geology["Q"])

print(df_geology.head())
print(df_geology.describe())
print(df_geology.info())

# READ IN DATA CONTAINING GROUTING DATA AND PREPROCESS
###############################################
df_grouting = pd.read_csv(path_grouting, delimiter=";")
df_grouting = df_grouting[df_grouting.columns[0:14]]
df_grouting = df_grouting.drop(["Status", "Driveretning", "Type injeksjon"], axis=1)
df_grouting = df_grouting.sort_values("Pel")

# change types of features
cols_to_convert = [
    "Pel",
    "Stikning [m]",
    "Bormeter [m]",
    "Sement mengde [kg]",
    "Injeksjonstid [time]",
]


for col in cols_to_convert:
    df_grouting[col] = df_grouting[col].apply(lambda x: float(str(x).replace(",", ".")))

df_grouting["Dato"] = pd.to_datetime(df_grouting["Dato"])
df_grouting = df_grouting.dropna().reset_index()

print(df_grouting.head())
print(df_grouting.info())

# OPERATIONS ON TOTAL DATASET
###############################################

# merge datasets
df = pd.merge_asof(
    left=df_grouting,
    right=df_geology,
    left_on="Pel",
    right_on="PegStart",
    tolerance=MERGE_TOLERANCE,
)

# make correlation plot before shifting values
# plotting correlation matrix of some selected numerical value
plot_correlation_matrix(
    path_correlation_plot,
    df,
    correlation_features,
    highlight_features=["Injeksjonstid [time]", "Sement mengde [kg]"],
)

# shift some features due to timing of data for prediction
# TODO: shift Q-system values as well
df[["Injeksjonstid [time]", "Sement mengde [kg]", "Slutt trykk [bar]"]] = df[
    ["Injeksjonstid [time]", "Sement mengde [kg]", "Slutt trykk [bar]"]
].shift(periods=-1)

# Anonymize the people
le = LabelEncoder()
for person in ["Author", "User"]:
    df[person] = le.fit_transform(df[person])
    df[person] = df[person].astype(str).apply(lambda x: f"Engineer_{x}")

# cleaning up
df = df.drop_duplicates()
df = df.dropna().reset_index(drop=True)

print(df.head())
print(df.info())


# get temperature and precipitation data and merge with date
# https://frost.met.no/howto.html

# TODO: check outliers

# rename and save csv-file
path_model_ready = Path("./data/model_ready/model_ready_dataset.xlsx")
df.to_excel(path_model_ready)


# plotting histograms of some chose feature values
plot_histograms(df, hist_features, path_feature_histograms, binsize=20)

# plotting barplot of some chosen categorical values
plot_barplots(df, barplot_features, path_feature_barplots)
