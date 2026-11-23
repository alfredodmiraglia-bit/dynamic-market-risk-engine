import numpy as np
import pandas as pd

from portfolio_strategy import fixed_holdings
from portfolio_strategy import fixed_holdings_exposure
from backtesting import run_backtest
from risk_models import ccc_t_forecast


# ==================================================
# SYNTHETIC GARCH-T MARKET DATA
# ==================================================

rng = np.random.default_rng(42)

n = 1150
df = 6

correlation = np.array([
    [1.0, 0.5],
    [0.5, 1.0]
])

z = rng.multivariate_normal(
    mean=[0, 0],
    cov=correlation,
    size=n
)

scale = np.sqrt(
    (df - 2) / rng.chisquare(df, size=n)
)

innovations = z * scale[:, None]

simulated_returns = np.zeros((n, 2))
conditional_variance = np.zeros((n, 2))

alpha = 0.10
beta = 0.85
omega = 0.000005

conditional_variance[0] = 0.0001

for t in range(n):

    if t > 0:

        conditional_variance[t] = (
            omega
            + alpha * simulated_returns[t - 1] ** 2
            + beta * conditional_variance[t - 1]
        )

    simulated_returns[t] = (
        np.sqrt(conditional_variance[t])
        * innovations[t]
    )

# Remove burn-in

simulated_returns = simulated_returns[300:]

dates = pd.bdate_range(
    start="2020-01-01",
    periods=len(simulated_returns)
)

prices = pd.DataFrame(
    100 * np.cumprod(
        1 + simulated_returns,
        axis=0
    ),
    index=dates,
    columns=["A", "B"]
)

returns = prices.pct_change().dropna()

weights = pd.Series({
    "A": 0.6,
    "B": 0.4
})

window = 756
confidence = 0.975
initial_capital = 100_000

start_date = dates[757]
end_date = dates[791]

holdings = fixed_holdings(
    prices=prices.iloc[756:],
    weights=weights,
    initial_capital=initial_capital
)


# ==================================================
# TEST BOTH STRATEGIES
# ==================================================

summaries = {}

for strategy, strategy_holdings in [
    ("Constant Weights", None),
    ("Fixed Holdings", holdings)
]:

    print(f"\nRunning {strategy}...")

    forecasts, failures, summary = run_backtest(
        returns=returns,
        weights=weights,
        model="CCC-GARCH-t",
        window=window,
        confidence=confidence,
        start_date=start_date,
        end_date=end_date,
        holdings=strategy_holdings
    )

    if not failures.empty:

        print(failures.to_string(index=False))
        raise AssertionError("GARCH backtest failed.")

    assert summary is not None
    assert len(forecasts) == 35

    assert np.isfinite(
        forecasts[["VaR (%)", "ES (%)"]].to_numpy()
    ).all()

    assert (
        forecasts["ES (%)"] >= forecasts["VaR (%)"]
    ).all()

    # Verify first and last forecast

    for row in [
        forecasts.iloc[0],
        forecasts.iloc[-1]
    ]:

        forecast_date = row["Forecast Date"]

        estimation_sample = returns.loc[
            returns.index < forecast_date
        ].tail(window)

        if strategy_holdings is None:

            current_weights = weights

        else:

            exposure = fixed_holdings_exposure(
                holdings,
                forecast_date
            )

            current_weights = exposure["weights"]

            assert np.isclose(
                row["Portfolio Value (USD)"],
                exposure["capital"]
            )

        expected = ccc_t_forecast(
            estimation_sample,
            current_weights,
            confidence
        )

        assert np.isclose(
            row["VaR (%)"],
            expected["VaR (%)"]
        )

        assert np.isclose(
            row["ES (%)"],
            expected["ES (%)"]
        )

        expected_loss = (
            -returns.loc[forecast_date].dot(
                current_weights
            ) * 100
        )

        assert np.isclose(
            row["Actual Loss (%)"],
            expected_loss
        )

    summaries[strategy] = summary

    print(f"PASSED - {strategy}")


# ==================================================
# RESULTS
# ==================================================

comparison = pd.DataFrame(summaries)

print("\nGARCH BACKTEST COMPARISON")
print(comparison.to_string())

print("\nALL GARCH BACKTEST TESTS PASSED")