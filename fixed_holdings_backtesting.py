
import numpy as np
import pandas as pd

from portfolio_strategy import fixed_holdings_exposure
from risk_models import historical_forecast, ccc_t_forecast


def run_fixed_holdings_backtest(
    returns,
    holdings,
    model,
    window,
    confidence,
    start_date,
    end_date,
    progress=None
):
    """
    Rolling one-day VaR and ES backtest for fixed holdings.

    Portfolio exposures are determined using information
    available before each forecast date.
    """

    if model not in ("Historical Simulation", "CCC-GARCH-t"):
        raise ValueError("Unsupported risk model.")

    if window < 30:
        raise ValueError("Estimation window is too short.")

    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence level.")

    if not returns.index.is_monotonic_increasing:
        raise ValueError("Returns must be sorted by date.")

    if returns.index.has_duplicates:
        raise ValueError("Duplicate return dates.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain invalid values.")

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    forecast_dates = returns.index[
        (returns.index >= start_date)
        & (returns.index <= end_date)
    ]

    if len(forecast_dates) == 0:
        raise ValueError("No forecast dates available.")

    records = []

    for step, forecast_date in enumerate(forecast_dates):

        location = returns.index.get_loc(forecast_date)

        if location < window:
            raise ValueError(
                "Insufficient historical observations "
                "before the forecast date."
            )

        exposure = fixed_holdings_exposure(
            holdings,
            forecast_date
        )

        previous_date = returns.index[location - 1]

        if exposure["valuation_date"] != previous_date:
            raise ValueError(
                "Portfolio valuation is not aligned "
                "with the previous trading date."
            )

        weights = exposure["weights"].reindex(
            returns.columns
        )

        if weights.isna().any():
            raise ValueError("Portfolio weights are misaligned.")

        capital = exposure["capital"]

        estimation_returns = returns.iloc[
            location - window:location
        ]

        if model == "Historical Simulation":

            portfolio_scenarios = estimation_returns.dot(
                weights
            )

            result = historical_forecast(
                portfolio_scenarios,
                confidence
            )

        else:

            result = ccc_t_forecast(
                estimation_returns,
                weights,
                confidence
            )

        var_pct = float(result["VaR (%)"])
        es_pct = float(result["ES (%)"])

        # Realized return with pre-forecast holdings
        realized_return = float(
            returns.iloc[location].dot(weights)
        )

        actual_loss_pct = -100.0 * realized_return

        records.append({
            "Forecast Date": forecast_date,
            "Valuation Date": previous_date,
            "Portfolio Value (USD)": capital,
            "Actual Loss (%)": actual_loss_pct,
            "VaR (%)": var_pct,
            "ES (%)": es_pct,
            "Actual Loss (USD)": (
                capital * actual_loss_pct / 100.0
            ),
            "VaR (USD)": capital * var_pct / 100.0,
            "ES (USD)": capital * es_pct / 100.0,
            "Exception": actual_loss_pct > var_pct
        })

        if progress is not None:
            progress((step + 1) / len(forecast_dates))

    return pd.DataFrame(records)