import numpy as np
import pandas as pd

from portfolio_strategy import fixed_holdings
from fixed_holdings_backtesting import run_fixed_holdings_backtest
from risk_models import historical_forecast

# CONFIGURATION
capital = 100_000
window = 252
confidence = 0.975

weights = pd.Series({
    "A": 0.50,
    "B": 0.50
})

dates = pd.bdate_range(
    start="2024-01-01",
    periods=255
)

# Synthetic historical prices
t = np.arange(253)

history = pd.DataFrame({
    "A": 100 + 5 * np.sin(t / 10),
    "B": 100 + 3 * np.cos(t / 8)
}, index=dates[:253])

# Set both prices to 100 at portfolio inception
history = history.div(history.iloc[-1]) * 100

# Subsequent market movements
future = pd.DataFrame({
    "A": [120, 100],
    "B": [90, 100]
}, index=dates[253:])

prices = pd.concat([history, future])

returns = prices.pct_change().dropna()

# Portfolio inception
holdings = fixed_holdings(
    prices.loc[dates[252]:],
    weights,
    capital
)

# Rolling backtesting
forecasts = run_fixed_holdings_backtest(
    returns=returns,
    holdings=holdings,
    model="Historical Simulation",
    window=window,
    confidence=confidence,
    start_date=dates[253],
    end_date=dates[254]
)

print("\nBACKTEST RESULTS")
print(forecasts.to_string(index=False))

# TEST 1: Correct valuation dates
assert len(forecasts) == 2

assert forecasts.iloc[0]["Valuation Date"] == dates[252]
assert forecasts.iloc[1]["Valuation Date"] == dates[253]

# TEST 2: Correct portfolio capital
assert np.isclose(
    forecasts.iloc[0]["Portfolio Value (USD)"],
    100_000
)

assert np.isclose(
    forecasts.iloc[1]["Portfolio Value (USD)"],
    105_000
)

# TEST 3: Correct realized losses
assert np.isclose(
    forecasts.iloc[0]["Actual Loss (USD)"],
    -5_000
)

assert np.isclose(
    forecasts.iloc[1]["Actual Loss (USD)"],
    5_000
)

# TEST 4: Correct VaR and ES
dynamic_weights = pd.Series({
    "A": 60_000 / 105_000,
    "B": 45_000 / 105_000
})

historical_window = returns.loc[
    :dates[253]
].tail(window)

reference = historical_forecast(
    historical_window.dot(dynamic_weights),
    confidence
)

assert np.isclose(
    forecasts.iloc[1]["VaR (%)"],
    reference["VaR (%)"]
)

assert np.isclose(
    forecasts.iloc[1]["ES (%)"],
    reference["ES (%)"]
)

# TEST 5: No look-ahead
modified_prices = prices.copy()

modified_prices.loc[dates[254], "A"] = 150

modified_returns = (
    modified_prices.pct_change().dropna()
)

modified_holdings = fixed_holdings(
    modified_prices.loc[dates[252]:],
    weights,
    capital
)

modified_forecasts = run_fixed_holdings_backtest(
    returns=modified_returns,
    holdings=modified_holdings,
    model="Historical Simulation",
    window=window,
    confidence=confidence,
    start_date=dates[254],
    end_date=dates[254]
)

assert np.isclose(
    forecasts.iloc[1]["VaR (%)"],
    modified_forecasts.iloc[0]["VaR (%)"]
)

assert np.isclose(
    forecasts.iloc[1]["ES (%)"],
    modified_forecasts.iloc[0]["ES (%)"]
)

assert not np.isclose(
    forecasts.iloc[1]["Actual Loss (USD)"],
    modified_forecasts.iloc[0]["Actual Loss (USD)"]
)

print("\nALL FIXED HOLDINGS BACKTEST TESTS PASSED")