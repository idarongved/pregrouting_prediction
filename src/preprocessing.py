"""Preprocessing fra raw data to model ready data"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.plotting import (
    plot_barplots,
    plot_correlation_matrix,
    plot_histograms,
    plot_scatter,
    plot_scatter2,
)
from src.utility import (
    barplot_features,
    correlation_features,
    hist_features,
    print_df_info,
)

# CONSTANTS
###############################################
MERGE_TOLERANCE = 1

path_model_ready_data = Path("./data/model_ready/model_ready_dataset.xlsx")
path_geology = Path("./data/raw/UDK01_mwd_Q_rocktype.csv")
path_grouting = Path("./data/raw/SikrReport32003_0_980.csv")
path_feature_histograms = Path("./plots/feature_histograms.png")
path_feature_barplots = Path("./plots/feature_barplots.png")
path_correlation_plot = Path("./plots/correlation_matrix.png")
path_temperature_data = Path("./data/raw/mean_daily_temperature.csv")
path_precipitation_data = Path("./data/raw/daily_precipitation.csv")
# path_scatter_plot = Path("./plots/scatter_plot.png")


# READ IN DATA WITH TEMPERATURE AND PRECIPITATION AND PREPROCESS
###############################################

df_precipitation = pd.read_csv(path_precipitation_data, delimiter=",", parse_dates=[1])
df_temperature = pd.read_csv(path_temperature_data, delimiter=",", parse_dates=[1])
df_climate = pd.merge(df_precipitation, df_temperature, on="date").drop(
    ["Unnamed: 0_x", "Unnamed: 0_y"], axis=1
)
df_climate = df_climate.set_index("date")

print(df_climate.info())
print(df_climate.head())

df_climate.plot()
plt.savefig("./plots/climate.png")
plt.close()


# READ IN DATA CONTAINING GEOLOGY AND PREPROCESS
###############################################
df_geology = pd.read_csv(path_geology, delimiter=";")
print_df_info(df_geology, "After read")
df_geology = df_geology.drop(["Tunnel"], axis=1)
df_geology = df_geology.sort_values("PegEnd")
df_geology["PegEnd"] = df_geology["PegEnd"].round(1)


# change types of features
df_geology["Date"] = pd.to_datetime(df_geology["Date"])
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


# feature engineering
df_geology["logQ"] = np.log(df_geology["Q"])
print_df_info(df_geology, "After all preprocessing", info=True)


# READ IN DATA CONTAINING GROUTING DATA AND PREPROCESS
###############################################
df_grouting = pd.read_csv(path_grouting, delimiter=";")
print_df_info(df_grouting, "After read")
df_grouting = df_grouting[df_grouting.columns[0:14]]
# drop obvious columns not needed
df_grouting = df_grouting.drop(["Status", "Driveretning", "Type injeksjon"], axis=1)
df_grouting = df_grouting.sort_values("Pel")
print_df_info(df_grouting, "After dropping")

# change types of features
df_grouting["Dato"] = pd.to_datetime(df_grouting["Dato"])
df_grouting = df_grouting.rename(columns={"Dato": "date"})

cols_to_convert = [
    "Pel",
    "Stikning [m]",
    "Bormeter [m]",
    "Sement mengde [kg]",
    "Injeksjonstid [time]",
]
for col in cols_to_convert:
    df_grouting[col] = df_grouting[col].apply(lambda x: float(str(x).replace(",", ".")))

# feature engineering
df_grouting["Distance last station"] = df_grouting["Pel"] - df_grouting["Pel"].shift(1)

df_grouting = df_grouting.dropna().reset_index()

df_grouting = df_grouting.set_index("date")

print_df_info(df_grouting, "After all preprocessing", info=True)

df_grouting = pd.merge(df_grouting, df_climate, on="date")


print_df_info(df_grouting, "After adding climate", info=True)

#TODO: check out label transform. The labels have not a gaussian distribution: https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py

# OPERATIONS ON TOTAL DATASET
###############################################

# merge datasets
df = pd.merge_asof(
    left=df_grouting.sort_values("Pel"),
    right=df_geology,
    left_on="Pel",
    right_on="PegEnd",
    direction="nearest",
    tolerance=MERGE_TOLERANCE,
)

print_df_info(df, "After mergasof")

# renaming and make all norwegian feature names english
df = df.rename(
    columns={
        "Antall borehull": "Number of holes",
        "Injeksjonstid [time]": "Grouting time",
        "Sement mengde [kg]": "Total grout take",
        "Slutt trykk [bar]": "Stop pressure",
        "Sementtype": "Cement type",
        "Author": "Control engineer grouting",
        "User": "Mapping geologist",
        "RockClass": "Q-class",
        "Rock": "Rocktype",
        "ContourWidth": "Tunnel width",
        "Date": "Date mapping",
        "Dato": "Date pregrouting",
        "Skjermlengde [m]": "Grouting length",
        "Stikning [m]": "Drilling inclination",
        "Bormeter [m]": "Drilling meters",
        "sum(precipitation_amount P1D)": "precipitation",
        "mean(air_temperature P1D)": "temperature",
    }
)

# shift some features due to timing of data for prediction
# NOTE: from merging process, all data in df_geology is already shifted 1 round behind
# the grouting face
df[["Prev. grouting time", "Prev. grout take", "Prev. stop pressure"]] = df[
    ["Grouting time", "Total grout take", "Stop pressure"]
].shift(periods=-1)

# Anonymize the people
le = LabelEncoder()
for person in ["Control engineer grouting", "Mapping geologist"]:
    df[person] = le.fit_transform(df[person])
    df[person] = df[person].astype(str).apply(lambda x: f"Engineer_{x}")

# cleaning up
df = df.drop_duplicates()
df = df.dropna().reset_index(drop=True)

print_df_info(df, "After all preprocessing on whole dataset", info=True)


# save excel-file
df.to_excel(path_model_ready_data)


# VISUALIZING DATASET
####################################################


plot_scatter(
    df,
    "Total grout take",
    "TerrainHeight",
    Path("./plots/scatter_terrainheight.png"),
    (0, 70000),
    (0, 250),
)
plot_scatter(df, "Total grout take", "Jn", Path("./plots/scatter_Jn.png"))
plot_scatter(
    df,
    "Stop pressure",
    "TerrainHeight",
    Path("./plots/scatter_stop_pressure_terrainheight.png"),
)
# plot_scatter2(df, "Total grout take", "TerrainHeight", "Cement type", path_scatter_plot)

# plotting correlation matrix of some selected numerical value
plot_correlation_matrix(
    path_correlation_plot,
    df,
    correlation_features,
    highlight_features=["Grouting time", "Total grout take"],
)

# plotting histograms of some chose feature values
plot_histograms(df, hist_features, path_feature_histograms, binsize=20)

# plotting barplot of some chosen categorical values
plot_barplots(df, barplot_features, path_feature_barplots)
