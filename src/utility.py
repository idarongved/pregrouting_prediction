"""Utility functions"""

import pandas as pd
from rich import print as pprint
from sklearn.preprocessing import OneHotEncoder

all_features = [
    "index",
    "Control engineer grouting",
    "Date pregrouting",
    "Pel",
    "Distance last station",
    "Grouting length",
    "Drilling inclination",
    "Number of holes",
    "Drilling meters",
    "Grouting time",
    "Cement type",
    "Total grout take",
    "Stop pressure",
    "PegStart",
    "PegEnd",
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
    "JnMult",
    "Ja",
    "SRF",
    "Mapping geologist",
    "Date mapping",
    "TerrainHeight",
    "Tunnel width",
    "logQ",
    "Prev. grouting time",
    "Prev. grout take",
    "Prev. stop pressure",
]

train_features_max = [
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

train_features_chosen = [
    # "Control engineer grouting",
    # "Date pregrouting",
    # "temperature",
    # "precipitation",
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
    "TerrainHeight",
    "HammerPressNormVariance",
]

train_features_no_previous = [
    # "Control engineer grouting",
    # "Date pregrouting",
    "Grouting length",
    "Distance last station",
    "Number of holes",
    "Drilling meters",
    "Cement type",
    "PenetrNormMean",
    "RotaPressNormMean",
    "HammerPressNormMean",
    "HammerPressNormVariance",
    "HammerPressNormSkewness",
    "Rocktype",
    "RQD",
    "Jr",
    "Jw",
    "Jn",
    "Ja",
    "SRF",
    "TerrainHeight",
]

hist_features = [
    "Grouting length",
    "Drilling inclination",
    "Number of holes",
    "Drilling meters",
    "Grouting time",
    "Total grout take",
    "Stop pressure",
    "Q",
    # "logQ",
    "TerrainHeight",
    "Distance last station",
]

barplot_features = [
    "Control engineer grouting",
    "Cement type",
    "Q-class",
    "Rocktype",
    "Mapping geologist",
]

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
    # na_sum = df.loc[na_counts > 0, df.select_dtypes(exclude="datetime64").columns].sum(
    # axis=1
    # )
    pprint(f"[Green]-------------{message}----------------")
    if info:
        pprint(df.head())
        pprint(df.info())
    pprint(f"Num rows: {samples}. Num cols: {cols}. Num NA: {na_counts}")
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
