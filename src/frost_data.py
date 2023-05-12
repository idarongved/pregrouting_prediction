from pathlib import Path

import pandas as pd
import requests  # type: ignore

client_id = "43450759-b078-41a9-96a3-e494ee027f78"

# Helper functions
###########################################


def get_api_data(
    client_id: str,
    endpoint: str,
    params: dict[str, str] | None = None,
    format: str = "df",
) -> pd.DataFrame:
    """
    Retrieves data from API using the provided endpoint and client ID.

    Parameters:
    - client_id (str): The client ID used to authorize the request to the API.
    - endpoint (str): The URL endpoint to retrieve the data from.

    Returns:
    - pd.DataFrame: The retrieved data as a pandas DataFrame.
    """
    # Make the API request
    r = requests.get(endpoint, auth=(client_id, ""), params=params)
    # Extract JSON data
    json = r.json()

    # Check if the request worked, raise an exception if not
    if r.status_code != 200:
        raise Exception(
            f"Error! Returned status code {r.status_code}. Message: {json['error']['message']}. Reason: {json['error']['reason']}"
        )

    # Convert JSON data to a pandas DataFrame
    if format == "df":
        return pd.DataFrame(json["data"])
    elif format == "json":
        return json["data"]


def get_measurements(
    client_id: str, endpoint_observation: str, params: dict[str, str], savepath: Path
) -> pd.DataFrame:
    """Returns a dataframe with measurement values for a sensor.
    Saves a csv-file"""
    data = get_api_data(
        client_id, endpoint_observation, params=parameters, format="json"
    )
    dates = []
    values = []

    for row in data:
        dates.append(row["referenceTime"])
        values.append(float(row["observations"][0]["value"]))

    df = pd.DataFrame({"date": dates, params["elements"]: values})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
    df.to_csv(savepath)
    return df


# Get a list of weather stations
################################

endpoint_weather_stations = "https://frost.met.no/sources/v0.jsonld"

df_stations = get_api_data(client_id, endpoint_weather_stations)
print(df_stations)

station_ids = df_stations[["name", "id"]].sort_values("name")

station_ids.to_csv("./data/temporary/weather_stations.csv")


# Choose available sensor methods for a location
################################

endpoint_methods = "https://frost.met.no/observations/availableTimeSeries/v0.jsonld"
parameters = dict(sources="SN27010")  # Konnerud: SN27010

df_methods = get_api_data(client_id, endpoint_methods, params=parameters)
method_names = df_methods[["sourceId", "elementId", "unit"]]
print(method_names)
method_names.to_csv("./data/temporary/methods_stations.csv")


# Get data temperature
###################################################

endpoint_observation = "https://frost.met.no/observations/v0.jsonld"

parameters = dict(
    sources="SN27010",
    elements="mean(air_temperature P1D)",
    referencetime="2020-01-01/2021-12-31",
)

path_temperature_data = Path("./data/raw/mean_daily_temperature.csv")

df = get_measurements(
    client_id, endpoint_observation, parameters, path_temperature_data
)
print(df)

# Get data precipitation
###################################################

parameters = dict(
    sources="SN27010",
    elements="sum(precipitation_amount P1D)",
    referencetime="2020-01-01/2021-12-31",
)
path_precipitation_data = Path("./data/raw/daily_precipitation.csv")

df = get_measurements(
    client_id, endpoint_observation, parameters, path_precipitation_data
)
print(df)
