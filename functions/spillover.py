import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.api import VAR


"""
Function to calculate the average spillover table.

The average spillover table measures the spillover effects between variables in a VAR model.
"""


def calculate_avg_spillover_table(
    df_volatility: pd.DataFrame, forecast_horizon: int = None, lag_order: int = None
):
    """
    Calculate the average spillover table.

    Args:
        df_volatility (pd.DataFrame): DataFrame containing the volatility df_volatility.
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

    sigma_u = np.asarray(result.sigma_u)
    sd_u = np.sqrt(np.diag(sigma_u))

    # Compute forecast error variance decomposition
    fevd = result.fevd(forecast_horizon, sigma_u / sd_u)
    # Extract the forecast error variance for the last period
    fe = fevd.decomp[:, -1, :]
    # Normalize the forecast error variance decomposition
    fevd = (fe / fe.sum(1)[:, None]) * 100

    # Calculate spillover table
    directional_to = fevd.sum(0) - np.diag(fevd)
    directional_to_incl_own = fevd.sum(0)
    directional_from = fevd.sum(1) - np.diag(fevd)

    # Create DataFrame for better readability
    countries = df_volatility.columns
    spillover_df = pd.DataFrame(fevd, columns=countries, index=countries)
    spillover_df.loc["Directional TO others"] = directional_to
    spillover_df.loc["Directional TO including own"] = directional_to_incl_own
    spillover_df = pd.concat(
        [
            spillover_df,
            pd.DataFrame(
                directional_from, index=countries, columns=["Directional FROM others"]
            ),
        ],
        axis=1,
    )
    spillover_df.loc["Total spillover index"] = spillover_df.loc[
        "Directional TO others"
    ].sum() / len(countries)

    return spillover_df, lag_order, forecast_horizon


"""
Function to calculate net pairwise spillover table.

The net pairwise spillover table measures the net spillover effects between variables in a VAR model.
"""


def calculate_net_pairwise_spillover_table(spillover_table: pd.DataFrame):
    """
    Calculate the net pairwise spillover table.

    Args:
        df_volatility (pd.DataFrame): DataFrame containing the volatility df_volatility.

    Returns:
        pd.DataFrame: The net pairwise spillover table.
    """
    n = spillover_table.shape[0]
    net_pairwise_spillovers = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                net_pairwise_spillovers[i, j] = (
                    (spillover_table.iloc[j, i] - spillover_table.iloc[i, j]) / n
                ) * 100

    # Create DataFrame for better readability
    countries = spillover_table.columns
    net_pairwise_df = pd.DataFrame(
        net_pairwise_spillovers, columns=countries, index=countries
    )

    return net_pairwise_df

"""
Function to calculate rolling window spillover table.

The rolling window spillover table measures the spillover effects between variables in a VAR model over a rolling window.
"""


def calculate_rolling_spillover_table(
    df_volatility: pd.DataFrame,
    window_size: int = None,
    lag_order: int = None,
    forecast_horizon: int = None,
):
    """
    Calculate the rolling window spillover table.

    Args:
        df_volatility (pd.DataFrame): DataFrame containing the volatility df_volatility.
        window_size (int, optional): The size of the rolling window. Defaults to None.
        lag_order (int, optional): The lag order for the VAR model. Defaults to None.
        forecast_horizon (int, optional): The forecast horizon. Defaults to None.

    Returns:
        pd.DataFrame: The rolling window spillover table.
    """
    window_size = 200 if window_size is None else window_size
    forecast_horizon = 10 if forecast_horizon is None else forecast_horizon

    num_windows = len(df_volatility) - window_size + 1
    total_spillover_index = []
    dates = df_volatility.index[window_size - 1 :]

    for start in tqdm(
        range(num_windows),
        desc="Calculating rolling window spillover...",
        total=num_windows,
    ):
        end = start + window_size
        window_data = df_volatility.iloc[start:end]

        model = VAR(window_data)

        # Calculate the lag order using the AIC criterion if not provided
        if lag_order is None:
            result = model.fit(maxlags=15, ic="aic")
            lag_order = result.k_ar
        else:
            result = model.fit(maxlags=lag_order)

        sigma_u = np.asarray(result.sigma_u)
        sd_u = np.sqrt(np.diag(sigma_u))

        fevd = result.fevd(forecast_horizon, sigma_u / sd_u)
        fe = fevd.decomp[:, -1, :]
        fevd = fe / fe.sum(1)[:, None] * 100

        # Calculate spillover table
        directional_to = fevd.sum(0) - np.diag(fevd)
        total_spillover_index.append(directional_to.sum() / len(df_volatility.columns))

    return pd.DataFrame({"Total Spillover Index": total_spillover_index}, index=dates)
