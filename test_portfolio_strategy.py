
import numpy as np
import pandas as pd

from portfolio_strategy import fixed_holdings, fixed_holdings_exposure


# Portafoglio iniziale: 100.000 EUR
capital = 100_000

weights = pd.Series({
    "A": 0.50,
    "B": 0.50
})

# Prezzi ipotetici in tre giornate
prices = pd.DataFrame(
    {
        "A": [100, 120, 100],
        "B": [100, 90, 100]
    },
    index=pd.to_datetime([
        "2026-01-02",
        "2026-01-05",
        "2026-01-06"
    ])
)

result = fixed_holdings(prices, weights, capital)

print("\nINITIAL QUANTITIES")
print(result["quantities"])

print("\nPORTFOLIO VALUE")
print(result["portfolio_value"])

print("\nDYNAMIC WEIGHTS (%)")
print(result["dynamic_weights"] * 100)

print("\nDAILY RETURNS (%)")
print(result["portfolio_returns"] * 100)


# Controlli automatici
assert np.isclose(result["portfolio_value"].iloc[0], 100_000)
assert np.isclose(result["portfolio_value"].iloc[1], 105_000)
assert np.isclose(result["portfolio_value"].iloc[2], 100_000)

assert np.isclose(
    result["dynamic_weights"]["A"].iloc[1],
    60000 / 105000
)

print("\nALL TESTS PASSED")



# Test as-of exposures without look-ahead
exposure = fixed_holdings_exposure(
    result,
    "2026-01-06"
)

assert exposure["valuation_date"] == pd.Timestamp("2026-01-05")

assert np.isclose(exposure["capital"], 105_000)

assert np.isclose(
    exposure["weights"]["A"],
    60_000 / 105_000
)

assert np.isclose(
    exposure["weights"]["B"],
    45_000 / 105_000
)

print("\nBACKTEST EXPOSURE TEST PASSED")
print("Valuation date:", exposure["valuation_date"])
print("Capital:", exposure["capital"])
print("Weights:")
print(exposure["weights"])