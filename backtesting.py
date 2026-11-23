import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from scipy.special import xlogy

from risk_models import historical_forecast, ccc_t_forecast
from portfolio_strategy import fixed_holdings_exposure

# ==================================================
# KUPIEC TEST
# ==================================================

def kupiec_test(exceptions, confidence):

    n = len(exceptions)
    x = int(exceptions.sum())

    p = 1 - confidence
    p_hat = x / n

    ll_null = (
        xlogy(x, p)
        + xlogy(n - x, 1 - p)
    )

    ll_alternative = (
        xlogy(x, p_hat)
        + xlogy(n - x, 1 - p_hat)
    )

    lr = max(0, 2 * (ll_alternative - ll_null))

    return lr, stats.chi2.sf(lr, df=1)


# ==================================================
# CHRISTOFFERSEN INDEPENDENCE TEST
# ==================================================

def christoffersen_test(exceptions):

    hits = np.asarray(exceptions, dtype=int)

    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))

    if n00 + n01 == 0 or n10 + n11 == 0:
        return np.nan, np.nan

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)

    pi = (n01 + n11) / (len(hits) - 1)

    ll_null = (
        xlogy(n00 + n10, 1 - pi)
        + xlogy(n01 + n11, pi)
    )

    ll_alternative = (
        xlogy(n00, 1 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11)
        + xlogy(n11, pi11)
    )

    lr = max(0, 2 * (ll_alternative - ll_null))

    return lr, stats.chi2.sf(lr, df=1)


# ==================================================
# MODEL EVALUATION
# ==================================================

def evaluate_backtest(forecasts, confidence):

    losses = forecasts["Actual Loss (%)"].to_numpy()
    var = forecasts["VaR (%)"].to_numpy()
    es = forecasts["ES (%)"].to_numpy()

    exceptions = (losses > var).astype(int)

    n = len(losses)
    p = 1 - confidence

    # VaR tests

    lr_uc, p_uc = kupiec_test(
        exceptions, confidence
    )

    lr_ind, p_ind = christoffersen_test(
        exceptions
    )

    lr_uc_cond, _ = kupiec_test(
        exceptions[1:], confidence
    )

    p_cc = stats.chi2.sf(
        lr_uc_cond + lr_ind, df=2
    )

    # Quantile Loss

    errors = losses - var

    quantile_loss = np.where(
        errors > 0,
        confidence * errors,
        p * (-errors)
    )

    # ES calibration

    excess_losses = np.maximum(losses - var, 0)

    es_gap = var + excess_losses / p - es

    es_test = sm.OLS(
        es_gap,
        np.ones((n, 1))
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 10}
    )

    p_es = float(es_test.pvalues[0])

    # Joint VaR-ES scoring function

    fz0_loss = (
        excess_losses / (p * es)
        + var / es
        + np.log(es)
        - 1
    )

    forecasts["Quantile Loss"] = quantile_loss
    forecasts["FZ0 Loss"] = fz0_loss
    forecasts["ES Calibration Gap"] = es_gap

    return {
        "Observations": n,
        "Expected Exceptions": n * p,
        "Observed Exceptions": int(exceptions.sum()),
        "Exception Rate (%)": exceptions.mean() * 100,
        "Kupiec p-value": p_uc,
        "Independence p-value": p_ind,
        "Conditional Coverage p-value": p_cc,
        "ES Calibration p-value": p_es,
        "Mean Quantile Loss": quantile_loss.mean(),
        "Mean FZ0 Loss": fz0_loss.mean(),
        "Mean ES Calibration Gap": es_gap.mean()
    }


# ==================================================
# ROLLING BACKTEST
# ==================================================

def run_backtest(
    returns,
    weights,
    model,
    window,
    confidence,
    start_date,
    end_date,
    progress=None,
    holdings=None
):

    returns = returns.sort_index()

    if not returns.index.is_unique:
        raise ValueError("Duplicate market data dates.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Invalid market data.")

    if window < 1:
        raise ValueError("Invalid estimation window.")

    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence level.")

    # ==============================================
    # CONSTANT WEIGHTS
    # ==============================================

    if holdings is None:

        weights = weights.reindex(returns.columns)

        if weights.isna().any():
            raise ValueError(
                "Portfolio weights do not match market data."
            )

        if not np.isclose(weights.sum(), 1):
            raise ValueError(
                "Portfolio weights must sum to 100%."
            )

        portfolio_returns = returns.dot(weights)

    # ==============================================
    # BACKTEST DATES
    # ==============================================

    dates = returns.index

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    positions = np.flatnonzero(
        (dates >= start_date)
        & (dates <= end_date)
    )

    if len(positions) < 30:

        raise ValueError(
            "At least 30 backtesting observations are required."
        )

    if positions[0] < window:

        raise ValueError(
            "Insufficient historical data before the backtesting start date."
        )

    if holdings is not None:

        fixed_holdings_exposure(
            holdings,
            dates[positions[0]]
        )

    records = []
    failures = []

    for count, pos in enumerate(positions, start=1):

        forecast_date = dates[pos]

        estimation_sample = returns.iloc[
            pos - window:pos
        ]

        try:

            # ======================================
            # PORTFOLIO EXPOSURE
            # ======================================

            if holdings is None:

                current_weights = weights

                estimation_portfolio_returns = (
                    portfolio_returns.iloc[pos - window:pos]
                )

                actual_return = portfolio_returns.iloc[pos]

            else:

                exposure = fixed_holdings_exposure(
                    holdings,
                    forecast_date
                )

                valuation_date = exposure["valuation_date"]

                if valuation_date != dates[pos - 1]:

                    raise ValueError(
                        "Portfolio valuation date does not match market data."
                    )

                current_weights = (
                    exposure["weights"].reindex(returns.columns)
                )

                if current_weights.isna().any():

                    raise ValueError(
                        "Portfolio weights do not match market data."
                    )

                current_capital = exposure["capital"]

                estimation_portfolio_returns = (
                    estimation_sample.dot(current_weights)
                )

                actual_return = (
                    returns.iloc[pos].dot(current_weights)
                )

            # ======================================
            # RISK FORECAST
            # ======================================

            if model == "Historical Simulation":

                result = historical_forecast(
                    estimation_portfolio_returns,
                    confidence
                )

            elif model == "CCC-GARCH-t":

                result = ccc_t_forecast(
                    estimation_sample,
                    current_weights,
                    confidence
                )

            else:

                raise ValueError("Unknown risk model.")

            actual_loss = -actual_return * 100

            var = result["VaR (%)"]
            es = result["ES (%)"]

            if not np.isfinite([var, es]).all():

                raise ValueError("Invalid risk estimates.")

            if es < var or es <= 0:

                raise ValueError("Invalid Expected Shortfall.")

            record = {
                "Forecast Date": forecast_date,
                "Estimation Start": estimation_sample.index[0],
                "Estimation End": estimation_sample.index[-1],
                "Actual Loss (%)": actual_loss,
                "VaR (%)": var,
                "ES (%)": es,
                "VaR Exception": int(actual_loss > var)
            }

            if holdings is not None:

                record.update({
                    "Valuation Date": valuation_date,
                    "Portfolio Value (USD)": current_capital,
                    "Actual Loss (USD)": (
                        current_capital * actual_loss / 100
                    ),
                    "VaR (USD)": (
                        current_capital * var / 100
                    ),
                    "ES (USD)": (
                        current_capital * es / 100
                    )
                })

            records.append(record)

        except (
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError
        ) as e:

            failures.append({
                "Forecast Date": forecast_date,
                "Error": str(e)
            })

        if progress is not None:

            progress(count / len(positions))

    forecasts = pd.DataFrame(records)
    failures = pd.DataFrame(failures)

    summary = None

    if failures.empty and len(forecasts) == len(positions):

        summary = evaluate_backtest(
            forecasts,
            confidence
        )

    return forecasts, failures, summary

def basel_traffic_light(forecasts):

    data = forecasts.sort_values("Forecast Date").copy()

    data["Exception"] = (
        data["Actual Loss (%)"] > data["VaR (%)"]
    ).astype(int)

    data["Window Start"] = data["Forecast Date"].shift(249)

    data["Rolling Exceptions"] = (
        data["Exception"]
        .rolling(window=250, min_periods=250)
        .sum()
    )

    data = data.dropna(subset=["Rolling Exceptions"]).copy()

    data["Rolling Exceptions"] = (
        data["Rolling Exceptions"].astype(int)
    )

    data["Basel Zone"] = np.select(
        [
            data["Rolling Exceptions"] <= 4,
            data["Rolling Exceptions"] <= 9
        ],
        ["Green", "Amber"],
        default="Red"
    )

    return data[
        [
            "Forecast Date",
            "Window Start",
            "Rolling Exceptions",
            "Basel Zone"
        ]
    ]