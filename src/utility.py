"""Utility functions"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as pprint
from sklearn.preprocessing import OneHotEncoder

# from auto feature selection with featurewiz
auto_features_take = [
    "Prev. grout take",
    "Number of holes",
    "RotaPressNormStandardDeviation",
    "precip_week",
    "temperature",
    "WaterFlowNormMean",
    "PenetrNormStandardDeviation",
    "Control engineer grouting",
    "HammerPressNormSkewness",
    "Ja",
    "Rocktype",
    "Jn",
    "Distance last station",
    "PenetrNormKurtosis",
    "Prev. stop pressure",
    "HammerPressNormMean",
    "FeedPressNormMean",
    "RotaPressRMSMean",
    "SRF",
    "Drilling inclination",
    "Mapping geologist",
    "Q-class",
    "Cement type",
]

auto_features_time = [
    "Prev. grouting time",
    "Number of holes",
    "precip_week",
    "TerrainHeight",
    "WaterFlowRMSMean",
    "WaterFlowNormMean",
    "temp_week",
    "RQD",
    "WaterFlowNormSkewness",
    "PenetrNormSkewness",
    "FeedPressNormMean",
    "PenetrRMSMean",
    "RotaPressNormMean",
    "Control engineer grouting",
    "HammerPressNormVariance",
    "HammerPressNormMean",
    "Grouting length",
    "Rocktype",
    "Mapping geologist",
    "Q-class",
    "Cement type",
]


all_relevant_features = [
    "temperature",
    "precipitation",
    "temp_week",
    "precip_week",
    "Control engineer grouting",
    # "Date pregrouting",
    # "Pel",
    "Distance last station",
    "Grouting length",
    "Drilling inclination",
    "Number of holes",
    "Drilling meters",
    "Cement type",
    # "Grouting time",
    # "Total grout take",
    # "Stop pressure",
    # "PegStart",
    # "PegEnd",
    "PenetrNormMean",
    "PenetrNormVariance",
    "PenetrNormStandardDeviation",
    "PenetrNormSkewness",
    "PenetrNormKurtosis",
    "PenetrRMSMean",
    "PenetrRMSVariance",
    "PenetrRMSStandardDeviation",
    "PenetrRMSSkewness",
    "PenetrRMSKurtosis",
    "RotaPressNormMean",
    "RotaPressNormVariance",
    "RotaPressNormStandardDeviation",
    "RotaPressNormSkewness",
    "RotaPressNormKurtosis",
    "RotaPressRMSMean",
    "RotaPressRMSVariance",
    "RotaPressRMSStandardDeviation",
    "RotaPressRMSSkewness",
    "RotaPressRMSKurtosis",
    "FeedPressNormMean",
    "FeedPressNormVariance",
    "FeedPressNormStandardDeviation",
    "FeedPressNormSkewness",
    "FeedPressNormKurtosis",
    "HammerPressNormMean",
    "HammerPressNormVariance",
    "HammerPressNormStandardDeviation",
    "HammerPressNormSkewness",
    "HammerPressNormKurtosis",
    "WaterFlowNormMean",
    "WaterFlowNormVariance",
    "WaterFlowNormStandardDeviation",
    "WaterFlowNormSkewness",
    "WaterFlowNormKurtosis",
    "WaterFlowRMSMean",
    "WaterFlowRMSVariance",
    "WaterFlowRMSStandardDeviation",
    "WaterFlowRMSSkewness",
    "WaterFlowRMSKurtosis",
    "Q-class",
    "Rocktype",
    "Q",
    "RQD",
    "Jr",
    "Jw",
    "Jn",
    # "JnMult",
    "Ja",
    "SRF",
    "Mapping geologist",
    # "Date mapping",
    "TerrainHeight",
    "Tunnel width",
    "logQ",
    "Prev. grouting time",
    "Prev. grout take",
    "Prev. stop pressure",
]

all_dataset_variables = [
    "Date mapping",
    "JnMult",
    "Grouting time",
    "Total grout take",
    "Stop pressure",
    "PegStart",
    "PegEnd",
    "Date pregrouting",
    "Pel",
    "Total grout take",
    "Grouting time",
    "temperature",
    "precipitation",
    "temp_week",
    "precip_week",
    "Mapping geologist",
    "Control engineer grouting",
    "Grouting length",
    "Distance last station",
    "Drilling inclination",
    "Number of holes",
    "Drilling meters",
    "Cement type",
    "PenetrNormMean",
    "PenetrNormVariance",
    "PenetrNormStandardDeviation",
    "PenetrNormSkewness",
    "PenetrNormKurtosis",
    "PenetrRMSMean",
    "PenetrRMSVariance",
    "PenetrRMSStandardDeviation",
    "PenetrRMSSkewness",
    "PenetrRMSKurtosis",
    "RotaPressNormMean",
    "RotaPressNormVariance",
    "RotaPressNormStandardDeviation",
    "RotaPressNormSkewness",
    "RotaPressNormKurtosis",
    "RotaPressRMSMean",
    "RotaPressRMSVariance",
    "RotaPressRMSStandardDeviation",
    "RotaPressRMSSkewness",
    "RotaPressRMSKurtosis",
    "FeedPressNormMean",
    "FeedPressNormVariance",
    "FeedPressNormStandardDeviation",
    "FeedPressNormSkewness",
    "FeedPressNormKurtosis",
    "HammerPressNormMean",
    "HammerPressNormVariance",
    "HammerPressNormStandardDeviation",
    "HammerPressNormSkewness",
    "HammerPressNormKurtosis",
    "WaterFlowNormMean",
    "WaterFlowNormVariance",
    "WaterFlowNormStandardDeviation",
    "WaterFlowNormSkewness",
    "WaterFlowNormKurtosis",
    "WaterFlowRMSMean",
    "WaterFlowRMSVariance",
    "WaterFlowRMSStandardDeviation",
    "WaterFlowRMSSkewness",
    "WaterFlowRMSKurtosis",
    "Q-class",
    "Rocktype",
    "Q",
    "logQ",
    "RQD",
    "Jr",
    "Jw",
    "Jn",
    "Ja",
    "SRF",
    "TerrainHeight",
    "Tunnel width",
    "Prev. grouting time",
    "Prev. grout take",
    "Prev. stop pressure",
]

train_features_manual_domain = [
    "Control engineer grouting",
    "Mapping geologist",
    # "Date pregrouting",
    # "temperature",
    # "precipitation",
    # "temp_week",
    "precip_week",
    "Grouting length",
    "Number of holes",
    "Drilling meters",
    # "Distance last station",
    "Cement type",
    "RotaPressNormMean",
    # "RotaPressNormVariance",
    # "RotaPressNormSkewness",
    "HammerPressNormMean",
    "HammerPressNormVariance",
    "HammerPressNormSkewness",
    "PenetrNormMean",
    # "PenetrRMSMean",
    # "RotaPressRMSMean",
    # "FeedPressNormMean",
    # "WaterFlowNormMean",
    # "WaterFlowRMSMean",
    "Rocktype",
    # "Q-class",
    "RQD",
    # "Jr",
    # "Jw",
    "Jn",
    "Ja",
    "SRF",
    "TerrainHeight",
    "Prev. grouting time",
    "Prev. grout take",
    "Prev. stop pressure",
]


train_features_small = [
    "Prev. grout take",
    "Prev. grouting time",
    "TerrainHeight",
]

train_features_no_previous = [  # train_feature_domain without previous
    "precip_week",
    "Grouting length",
    "Number of holes",
    "Drilling meters",
    "Cement type",
    "RotaPressNormMean",
    "HammerPressNormMean",
    "HammerPressNormVariance",
    "HammerPressNormSkewness",
    "PenetrNormMean",
    "Rocktype",
    "RQD",
    "Jn",
    "Ja",
    "SRF",
    "TerrainHeight",
]

MWD = [
    "PegStart",
    "PenetrNormMean",
    "PenetrNormVariance",
    "PenetrNormStandardDeviation",
    "PenetrNormSkewness",
    "PenetrNormKurtosis",
    "PenetrRMSMean",
    "PenetrRMSVariance",
    "PenetrRMSStandardDeviation",
    "PenetrRMSSkewness",
    "PenetrRMSKurtosis",
    "RotaPressNormMean",
    "RotaPressNormVariance",
    "RotaPressNormStandardDeviation",
    "RotaPressNormSkewness",
    "RotaPressNormKurtosis",
    "RotaPressRMSMean",
    "RotaPressRMSVariance",
    "RotaPressRMSStandardDeviation",
    "RotaPressRMSSkewness",
    "RotaPressRMSKurtosis",
    "FeedPressNormMean",
    "FeedPressNormVariance",
    "FeedPressNormStandardDeviation",
    "FeedPressNormSkewness",
    "FeedPressNormKurtosis",
    "HammerPressNormMean",
    "HammerPressNormVariance",
    "HammerPressNormStandardDeviation",
    "HammerPressNormSkewness",
    "HammerPressNormKurtosis",
    "WaterFlowNormMean",
    "WaterFlowNormVariance",
    "WaterFlowNormStandardDeviation",
    "WaterFlowNormSkewness",
    "WaterFlowNormKurtosis",
    "WaterFlowRMSMean",
    "WaterFlowRMSVariance",
    "WaterFlowRMSStandardDeviation",
    "WaterFlowRMSSkewness",
    "WaterFlowRMSKurtosis",
]

hist_features = [
    "Grouting length",
    # "Drilling inclination",
    "Number of holes",
    "Drilling meters",
    "Grouting time",
    "Total grout take",
    "Stop pressure",
    "Q",
    # "logQ",
    "TerrainHeight",
    # "Distance last station",
    "RotaPressNormMean",
    "HammerPressNormMean",
    "PenetrNormMean",
    "RQD",
    # "WaterFlowNormMean",
]

barplot_features = [
    "Control engineer grouting",
    "Cement type",
    "Q-class",
    "Rocktype",
    # "Mapping geologist",
]

feature_units = {
    "Grouting length":"[m]",
    "Number of holes":"",
    "Drilling meters":"[m]",
    "Grouting time":'[h]',
    "Total grout take":'[kg]',
    "Stop pressure":'[bar]',
    "Q": 'value',
    "TerrainHeight":'[m]',
    "RotaPressNormMean":'[bar/min]',
    "HammerPressNormMean":'[bar/min]',
    "PenetrNormMean":'[m/min]',
    "RQD":'[%]',
}


correlation_features = [
    "Grouting time",
    "Total grout take",
    "Stop pressure",
    "temperature",
    "precipitation",
    "Grouting length",
    "Distance last station",
    "Drilling inclination",
    "Number of holes",
    "Drilling meters",
    "Prev. grouting time",
    "Prev. grout take",
    "Prev. stop pressure",
    "PenetrNormMean",
    "PenetrRMSMean",
    "RotaPressNormMean",
    "RotaPressRMSMean",
    "FeedPressNormMean",
    "HammerPressNormMean",
    "WaterFlowNormMean",
    "WaterFlowRMSMean",
    "Q",
    "logQ",
    "RQD",
    "Jr",
    "Jw",
    "Jn",
    "Ja",
    "SRF",
    "TerrainHeight",
    "Tunnel width",
]


# UTILITY FUNCTIONS
####################################################


def print_df_info(df: pd.DataFrame, message: str = "start", info: bool = False) -> None:
    """Used to print status of dataframe.
    eg. message: after dropna"""
    samples, cols = df.shape
    na_counts = df.select_dtypes(exclude="datetime64").isna().sum(axis=1).sum()
    na_col_counts = df.isna().sum()
    pprint(f"-------------{message}----------------")
    if info:
        pprint(df.head())
        pprint(df.info())
    pprint(f"Num rows: {samples}. Num cols: {cols}. Num NA: {na_counts}")
    pprint(f"NA feature counts: {na_col_counts}")
    print("---------------------------------------")


def encode_categorical_features(features: pd.DataFrame) -> pd.DataFrame:
    # Identify categorical features in features DataFrame
    cat_cols = features.select_dtypes(include=["object"]).columns

    # Create an instance of OneHotEncoder with sparse=False to return a dense array
    encoder = OneHotEncoder(sparse=False)

    # Fit and transform the categorical features in features DataFrame
    cat_encoded = encoder.fit_transform(features[cat_cols])

    # Create a new DataFrame with the encoded features and concatenate with original
    # features DataFrame
    cat_encoded_df = pd.DataFrame(
        cat_encoded, columns=encoder.get_feature_names_out(cat_cols)
    )

    features_encoded = pd.concat(
        [features.drop(cat_cols, axis=1), cat_encoded_df], axis=1
    )

    print(features_encoded.head())
    print(features_encoded.info())

    return features_encoded


def process_geology_blastholes_csv(savepath: Path) -> pd.DataFrame:
    # read csv
    df_geology = pd.read_csv(savepath, delimiter=";")

    # drop column and sort
    df_geology = df_geology.drop(["Tunnel"], axis=1)
    df_geology = df_geology.sort_values("PegEnd")

    # round column value
    df_geology["PegEnd"] = df_geology["PegEnd"].round(1).astype("int")

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
        df_geology[col] = df_geology[col].apply(
            lambda x: float(str(x).replace(",", "."))
        )

    # feature engineering
    df_geology["logQ"] = np.log(df_geology["Q"])

    # return dataframe
    return df_geology


def process_geology_longholes(inputpath: Path) -> pd.DataFrame:
    df_geology = pd.read_csv(inputpath, delimiter=";")
    df_geology = df_geology.drop(["Tunnel"], axis=1)

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
        df_geology[col] = df_geology[col].apply(
            lambda x: float(str(x).replace(",", "."))
        )

    df_geology["PegStart"] = df_geology["PegStart"].round(0).astype("int")

    # Drop duplicates based on "PegStart" column in df_geology
    df_geology = df_geology.drop_duplicates(subset="PegStart")

    # Set "PegStart" as the index in df_geology
    df_geology.set_index("PegStart", inplace=True)

    # Create expanded_index with unique values
    expanded_index = pd.RangeIndex(
        start=df_geology.index.min(), stop=df_geology.index.max() + 1, name="PegStart"
    )

    # Reindex df_geology using expanded_index
    expanded_df = df_geology.reindex(expanded_index)

    # Interpolate numeric features
    numeric_cols = expanded_df.select_dtypes(include=np.number).columns
    expanded_df[numeric_cols] = expanded_df[numeric_cols].ffill().abs()

    # Forward fill categorical features
    categorical_cols = expanded_df.select_dtypes(include="object").columns
    expanded_df[categorical_cols] = expanded_df[categorical_cols].ffill()

    expanded_df["Date"] = pd.to_datetime(expanded_df["Date"]).dt.strftime("%Y-%m-%d")
    # feature engineering
    expanded_df["logQ"] = np.log(expanded_df["Q"])

    # Reset index
    expanded_df = expanded_df.reset_index()

    return expanded_df


def align_geology_for_longholes(
    df_geology: pd.DataFrame, df_grouting: pd.DataFrame, df_total: pd.DataFrame
) -> pd.DataFrame:
    # 1. Merge df_geology with df_grouting[["Pel","Skjermlengde [m]"]]
    df = pd.merge(
        df_geology,
        df_grouting[["Pel", "Skjermlengde [m]"]],
        how="left",
        left_on="PegStart",
        right_on="Pel",
    )

    grouped_df = pd.DataFrame()

    for idx, row in df.iterrows():
        if not pd.isnull(row["Skjermlengde [m]"]):
            length_fan = int(row["Skjermlengde [m]"])
            group_mean = df.loc[idx : idx + length_fan, MWD[1:]].mean().round(3)
            group_mean["PegStart"] = df.loc[idx, "PegStart"]
            grouped_df = pd.concat([grouped_df, group_mean], ignore_index=True, axis=1)

    grouped_df = grouped_df.transpose()
    grouped_df["PegStart"] = grouped_df["PegStart"].round(0).astype(int)

    # only replace MWD-data. All other data is ok in df_total
    grouped_df = grouped_df[MWD]
    df_total = df_total.drop(MWD, axis=1)

    # 2. Replace old blasting MWD with longhole MWD
    df = pd.merge(
        df_total,
        grouped_df,
        how="inner",
        left_on="Pel",
        right_on="PegStart",
    )
    df = df.dropna()

    df = df.sort_values("Pel")

    return df
