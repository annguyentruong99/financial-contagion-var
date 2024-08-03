import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

"""
This module contains a function to calculate the average spillover table.

The average spillover table measures the spillover effects between variables in a VAR model.
"""


def calculate_avg_spillover_table(
    df_volatility: pd.DataFrame, forecast_horizon: int = None, lag_order: int = None
):
    """
    Calculate the average spillover table.

    Args:
        df_volatility (pd.DataFrame): DataFrame containing the volatility data.
        forecast_horizon (int, optional): The forecast horizon. Defaults to None.
        lag_order (int, optional): The lag order for the VAR model. Defaults to None.

    Returns:
        pd.DataFrame: The average spillover table.
        int: The lag order used in the VAR model.
        int: The forecast horizon used.
    """
    forecast_horizon = 10 if forecast_horizon is None else forecast_horizon

    # Fit the VAR model
    model = VAR(df_volatility)

    # Calculate the lag order using the AIC criterion if not provided
    if lag_order is None:
        result = model.fit(maxlags=15, ic="aic")
        lag_order = result.k_ar
    else:
        result = model.fit(maxlags=lag_order)

    # Compute forecast error variance decomposition
    fevd = result.fevd(forecast_horizon)

    # Initialize the spillover table
    n = df_volatility.shape[1]
    spillover_table = np.zeros((n, n))

    # Fill the spillover table with the forecast error variance decompositions
    for i in range(n):
        for j in range(n):
            spillover_table[i, j] = fevd.decomp[j, :, i].sum()

    # Normalize the table
    spillover_table_normalized = (
        spillover_table / spillover_table.sum(axis=1)[:, None] * 100
    )

    # Create DataFrame for better readability
    countries = df_volatility.columns
    spillover_df = pd.DataFrame(
        spillover_table_normalized, columns=countries, index=countries
    )

    # Compute directional FROM and TO others
    spillover_df["Directional FROM others"] = 100 - np.diag(spillover_table_normalized)
    spillover_df.loc["Directional TO others"] = 100 - spillover_df.sum(axis=0)

    # Compute total spillover index
    total_spillover_index = (
        spillover_df.values.sum() - np.trace(spillover_table_normalized)
    ) / n

    # Adding total spillover index to the DataFrame
    spillover_df.loc["Total Spillover Index"] = [total_spillover_index] * (n + 1)

    return spillover_df, lag_order, forecast_horizon
