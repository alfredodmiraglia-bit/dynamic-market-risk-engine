import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SHOCKS = pd.DataFrame({
    "Equity Market Crash": [-8, -8, -6, 3, 2],
    "Technology Sell-off": [-12, -12, -5, 1, 0],
    "Interest Rate Shock": [-4, -4, -3, -6, -2],
    "Cross-Asset Liquidation": [-8, -8, -6, -4, -4]
}, index=["AAPL", "MSFT", "SPY", "TLT", "GLD"]).T


HISTORICAL_PERIODS = {
    "Global Financial Crisis": ("2008-09-01", "2009-03-31"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-31"),
    "Inflation and Rate Shock": ("2022-01-01", "2022-12-31")
}


def hypothetical_stress(shocks, weights, capital):

    shocks = shocks.loc[:, weights.index].astype(float)

    if not np.isfinite(shocks.to_numpy()).all():
        raise ValueError("Invalid scenario shocks.")

    results = shocks.copy()

    results["Portfolio Loss (%)"] = -shocks.dot(weights)

    results["Portfolio Loss (USD)"] = (
        results["Portfolio Loss (%)"] * capital / 100
    )

    return results


def historical_stress(returns, weights, capital):

    returns = returns.sort_index()
    weights = weights.reindex(returns.columns)

    records = []

    for scenario, (start, end) in HISTORICAL_PERIODS.items():

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        if (
            returns.index.min() > start
            or returns.index.max() < end
        ):

            records.append({
                "Scenario": scenario,
                "Status": "Insufficient historical coverage"
            })

            continue

        period_returns = returns.loc[start:end]

        if period_returns.empty:

            records.append({
                "Scenario": scenario,
                "Status": "No available observations"
            })

            continue

        portfolio_returns = period_returns.dot(weights)

        worst_date = portfolio_returns.idxmin()

        asset_shocks = period_returns.loc[worst_date] * 100

        portfolio_loss = -float(asset_shocks.dot(weights))

        records.append({
            "Scenario": scenario,
            "Status": "Available",
            "Worst Date": worst_date,
            **asset_shocks.to_dict(),
            "Portfolio Loss (%)": portfolio_loss,
            "Portfolio Loss (USD)": portfolio_loss * capital / 100
        })

    return pd.DataFrame(records)


def sensitivity_stress(result, weights, confidence, capital):

    covariance = result["Covariance"]

    if isinstance(covariance, pd.DataFrame):

        covariance = covariance.loc[
            weights.index, weights.index
        ].to_numpy()

    else:

        covariance = np.asarray(covariance)

    w = weights.to_numpy()

    mean = result["Mean (%)"]
    df = result["Degrees of Freedom"]

    sigma = np.sqrt(np.diag(covariance))

    R = covariance / np.outer(sigma, sigma)

    target_R = np.full(R.shape, 0.6)

    np.fill_diagonal(target_R, 1.0)

    scenarios = {
        "Baseline": (1.0, 0.0),
        "Volatility +50%": (1.5, 0.0),
        "Volatility +100%": (2.0, 0.0),
        "Correlation Stress": (1.0, 0.5),
        "Combined Stress": (1.5, 0.5)
    }

    q = stats.t.ppf(confidence, df=df)

    scale = np.sqrt((df - 2) / df)

    var_coefficient = scale * q

    es_coefficient = (
        scale
        * stats.t.pdf(q, df=df)
        * (df + q**2)
        / ((df - 1) * (1 - confidence))
    )

    records = []

    for scenario, (vol_multiplier, corr_weight) in scenarios.items():

        stressed_sigma = sigma * vol_multiplier

        stressed_R = (
            (1 - corr_weight) * R
            + corr_weight * target_R
        )

        D = np.diag(stressed_sigma)

        H = D @ stressed_R @ D

        np.linalg.cholesky(H)

        volatility = np.sqrt(w @ H @ w)

        var = -mean + var_coefficient * volatility
        es = -mean + es_coefficient * volatility

        records.append({
            "Scenario": scenario,
            "Volatility (%)": volatility,
            "VaR (%)": var,
            "ES (%)": es,
            "VaR (USD)": var * capital / 100,
            "ES (USD)": es * capital / 100
        })

    results = pd.DataFrame(records)

    baseline_es = results.loc[0, "ES (USD)"]

    results["ES Increase (%)"] = (
        (results["ES (USD)"] / baseline_es - 1) * 100
        if baseline_es > 0 else np.nan
    )

    return results