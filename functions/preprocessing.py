import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

"""
Function to read all csv files in a directory and 
return a dictionary of DataFrames
"""


def read_csv_files(directory: str) -> dict:
    dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            dataframes[filename.replace(".csv", "")] = df
    return dataframes


"""
Function to check for missing values in the DataFrames
"""


def check_missing_values(dataframes: dict) -> None:
    for key in dataframes:
        print(f"Missing values for {key}:")
        print(dataframes[key].isnull().sum())
        print("\n")


"""
Function to impute missing values in the DataFrames using forward fill
"""


def forward_fill(dataframes: dict) -> dict:
    for key in dataframes:
        df = dataframes[key]
        df.fillna(method="ffill", inplace=True)
    return dataframes


"""
Function to impute missing values in the DataFrames using backward fill
"""


def backward_fill(dataframes: dict) -> dict:
    for key in dataframes:
        df = dataframes[key]
        df.fillna(method="bfill", inplace=True)
    return dataframes


"""
Function to impute missing values in the DataFrames using linear interpolation
"""


def linear_interpolation(dataframes: dict) -> dict:
    for key in dataframes:
        df = dataframes[key]
        df.interpolate(method="linear", inplace=True)
    return dataframes


"""
Function to calculate volatility
"""


def calculate_volatility(dataframes: dict) -> dict:
    for key in dataframes:
        df = dataframes[key]

        high = np.log(df["high"])
        low = np.log(df["low"])
        close = np.log(df["close"])
        open = np.log(df["open"])

        # Calculate daily volatility range
        volatility_range = (
            0.551 * (high - low) ** 2
            - 0.019
            * (
                (close - open) * (high + low - 2 * open)
                - 2 * (high - open) * (low - open)
            )
            - 0.383 * (close - open) ** 2
        )

        # Convert daily volatility to percentage annual volatility
        volatility_percentage = 100 * np.sqrt(365.25 * volatility_range)

        # Add the calculated volatility to the DataFrame
        df["volatility"] = volatility_percentage

    return dataframes


"""
Function to perform Augmented Dickey-Fuller test for stationarity
"""


def adf_test(
    df: pd.DataFrame,
    variable_name: str = "volatility",
    significance_level: float = 0.05,
) -> None:
    result = adfuller(df[variable_name])

    # Extract ADF results
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    print(f"Results of Augmented Dickey-Fuller Test for {variable_name}:")
    print(f"ADF Statistic: {adf_statistic}")
    print(f"p-value: {p_value}")
    print(f"Critical Values:")
    for key, value in critical_values.items():
        print(f"\t{key}: {value}")

    if p_value < significance_level:
        print(
            f"Reject the null hypothesis. The time series {variable_name} is stationary."
        )
    else:
        print(
            f"Fail to reject the null hypothesis. The time series {variable_name} is non-stationary."
        )
    print("\n")


"""
Function to map each DataFrame to a specified date range,
containing only business days and without any public holidays
"""


def map_date_range(dataframes: pd.DataFrame, start_date: str, end_date: str) -> dict:
    # Create datetime objects for the start and end of the year range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Dictionary to hold the result
    result = {}

    # Iterate through each DataFrame
    for name, df in dataframes.items():
        # Create an empty DataFrame with the specified date range with business days only
        date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        df_date_range = pd.DataFrame(date_range, columns=["date"])

        # Merge the original DataFrame with the date range DataFrame
        merged_df = pd.merge(df_date_range, df, on="date", how="left")

        # Update the result dictionary
        result[name] = merged_df

    return result


"""
Function to create a new DataFrame containing the volatility calculated for each DataFrame
"""


def create_volatility_df(dataframes: dict) -> pd.DataFrame:
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate through each DataFrame
    for name, df in dataframes.items():
        # Extract the date and volatility columns
        volatility_df = df[["date", "volatility"]]

        # Rename the volatility column to include the DataFrame name
        volatility_df = volatility_df.rename(columns={"volatility": name})

        # Merge the volatility data with the result DataFrame
        if result_df.empty:
            result_df = volatility_df
        else:
            result_df = pd.merge(result_df, volatility_df, on="date", how="outer")

    return result_df
