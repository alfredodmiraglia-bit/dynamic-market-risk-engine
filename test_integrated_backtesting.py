import numpy as np
import pandas as pd

from portfolio_strategy import fixed_holdings
from backtesting import run_backtest, basel_traffic_light
from fixed_holdings_backtesting import run_fixed_holdings_backtest
from risk_models import historical_forecast


# ==================================================
# SYNTHETIC MARKET DATA
# ==================================================

rng = np.random.default_rng(42)

dates = pd.bdate_range(
    start="2024-01-01",
    periods=520
)

prices = pd.DataFrame(
    100 * np.cumprod(
        1 + rng.normal(0, 0.01, size=(520, 2)),
        axis=0
    ),
    index=dates,
    columns=["A", "B"]
)

returns = prices.pct_change().dropna()

weights = pd.Series({
    "A": 0.5,
    "B": 0.5
})

initial_capital = 100_000
window = 252
confidence = 0.99

start_date = dates[253]
end_date = dates[512]


# ==================================================
# TEST 1 - CONSTANT WEIGHTS
# ==================================================

cw_forecasts, cw_failures, cw_summary = run_backtest(
    returns=returns,
    weights=weights,
    model="Historical Simulation",
    window=window,
    confidence=confidence,
    start_date=start_date,
    end_date=end_date
)

assert cw_failures.empty
assert cw_summary is not None
assert len(cw_forecasts) == 260

expected_losses = (
    -returns.loc[start_date:end_date].dot(weights) * 100
)

np.testing.assert_allclose(
    cw_forecasts["Actual Loss (%)"],
    expected_losses
)

first_sample = returns.loc[:dates[252]].tail(window)

expected_forecast = historical_forecast(
    first_sample.dot(weights),
    confidence
)

assert np.isclose(
    cw_forecasts.iloc[0]["VaR (%)"],
    expected_forecast["VaR (%)"]
)

assert np.isclose(
    cw_forecasts.iloc[0]["ES (%)"],
    expected_forecast["ES (%)"]
)

print("TEST 1 PASSED - Constant Weights")


# ==================================================
# TEST 2 - FIXED HOLDINGS
# ==================================================

holdings = fixed_holdings(
    prices=prices.iloc[252:],
    weights=weights,
    initial_capital=initial_capital
)

fh_forecasts, fh_failures, fh_summary = run_backtest(
    returns=returns,
    weights=weights,
    model="Historical Simulation",
    window=window,
    confidence=confidence,
    start_date=start_date,
    end_date=end_date,
    holdings=holdings
)

assert fh_failures.empty
assert fh_summary is not None
assert len(fh_forecasts) == 260

assert np.isclose(
    fh_forecasts.iloc[0]["Portfolio Value (USD)"],
    initial_capital
)

assert fh_forecasts.iloc[0]["Valuation Date"] == dates[252]

expected_capital = holdings["portfolio_value"].loc[
    dates[252]:dates[511]
]

np.testing.assert_allclose(
    fh_forecasts["Portfolio Value (USD)"],
    expected_capital
)

expected_fh_losses = (
    -holdings["portfolio_returns"].loc[
        start_date:end_date
    ] * 100
)

np.testing.assert_allclose(
    fh_forecasts["Actual Loss (%)"],
    expected_fh_losses,
    atol=1e-10
)

print("TEST 2 PASSED - Fixed Holdings")


# ==================================================
# TEST 3 - COMPARISON WITH PREVIOUS MODULE
# ==================================================

reference = run_fixed_holdings_backtest(
    returns=returns,
    holdings=holdings,
    model="Historical Simulation",
    window=window,
    confidence=confidence,
    start_date=start_date,
    end_date=end_date
)

columns = [
    "Portfolio Value (USD)",
    "Actual Loss (%)",
    "VaR (%)",
    "ES (%)",
    "Actual Loss (USD)",
    "VaR (USD)",
    "ES (USD)"
]

for column in columns:

    np.testing.assert_allclose(
        fh_forecasts[column],
        reference[column]
    )

print("TEST 3 PASSED - Reference Module")


# ==================================================
# TEST 4 - STATISTICAL EVALUATION
# ==================================================

for forecasts, summary in [
    (cw_forecasts, cw_summary),
    (fh_forecasts, fh_summary)
]:

    assert summary["Observations"] == 260

    assert summary["Observed Exceptions"] == (
        forecasts["VaR Exception"].sum()
    )

    assert np.isclose(
        summary["Expected Exceptions"],
        2.6
    )

    assert "Quantile Loss" in forecasts.columns
    assert "FZ0 Loss" in forecasts.columns
    assert "ES Calibration Gap" in forecasts.columns

print("TEST 4 PASSED - Statistical Evaluation")


# ==================================================
# TEST 5 - BASEL TRAFFIC LIGHT
# ==================================================

cw_basel = basel_traffic_light(cw_forecasts)
fh_basel = basel_traffic_light(fh_forecasts)

assert len(cw_basel) == 11
assert len(fh_basel) == 11

for forecasts, basel in [
    (cw_forecasts, cw_basel),
    (fh_forecasts, fh_basel)
]:

    expected_exceptions = (
        forecasts["VaR Exception"].iloc[:250].sum()
    )

    assert (
        basel.iloc[0]["Rolling Exceptions"]
        == expected_exceptions
    )

print("TEST 5 PASSED - Basel Traffic Light")


# ==================================================
# RESULTS
# ==================================================

comparison = pd.DataFrame({
    "Constant Weights": cw_summary,
    "Fixed Holdings": fh_summary
})

print("\nBACKTEST COMPARISON")
print(comparison.to_string())

print("\nALL INTEGRATED BACKTEST TESTS PASSED")