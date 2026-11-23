import numpy as np
import pandas as pd


def fixed_holdings(prices, weights, initial_capital):

    weights = pd.Series(weights, dtype=float)

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("Invalid portfolio weights.")

    prices = prices.loc[:, weights.index].sort_index()

    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Invalid price history.")

    if (
        not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
    ):
        raise ValueError("Missing or invalid prices.")

    # Initial allocation
    initial_allocation = initial_capital * weights

    # Initial quantities
    quantities = initial_allocation / prices.iloc[0]

    # Daily valuation of each position
    position_values = prices.mul(quantities, axis=1)

    # Daily portfolio capital
    portfolio_value = position_values.sum(axis=1)

    # Daily portfolio weights
    dynamic_weights = position_values.div(
        portfolio_value, axis=0
    )

    # Daily portfolio returns
    portfolio_returns = portfolio_value.pct_change().dropna()

    return {
        "quantities": quantities,
        "position_values": position_values,
        "portfolio_value": portfolio_value,
        "dynamic_weights": dynamic_weights,
        "portfolio_returns": portfolio_returns
    }


def fixed_holdings_exposure(holdings, forecast_date):
    """
    Return portfolio exposures available before the forecast date.
    """

    position_values = holdings["position_values"]

    forecast_date = pd.Timestamp(forecast_date)

    available = position_values.loc[
        position_values.index < forecast_date
    ]

    if available.empty:
        raise ValueError(
            "No portfolio valuation is available before "
            "the forecast date."
        )

    latest_positions = available.iloc[-1]

    capital = float(latest_positions.sum())

    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Invalid portfolio capital.")

    weights = latest_positions / capital

    return {
        "valuation_date": available.index[-1],
        "capital": capital,
        "weights": weights
    }