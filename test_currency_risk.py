"""Offline synthetic checks for the market-data currency conversion."""

import numpy as np
import pandas as pd

from currency_risk import convert_prices


def run_tests():
    dates = pd.bdate_range("2024-01-02", periods=4)
    local = pd.DataFrame(
        {"US_EQUITY": [100., 102., 103., 101.],
         "EU_EQUITY": [50., 49., 51., 51.]},
        index=dates,
    )
    asset_ccy = {"US_EQUITY": "USD", "EU_EQUITY": "EUR"}
    # EUR per USD. Positive values, i.e. EUR is the reporting currency.
    fx = pd.DataFrame({"USD": [0.90, 0.88, 0.92, 0.91]}, index=dates)

    converted = convert_prices(local, asset_ccy, fx, "EUR")
    np.testing.assert_allclose(converted["US_EQUITY"],
                               local["US_EQUITY"] * fx["USD"])
    np.testing.assert_allclose(converted["EU_EQUITY"], local["EU_EQUITY"])

    rets = converted.pct_change(fill_method=None).iloc[1:]
    local_usd = local["US_EQUITY"].pct_change(fill_method=None).iloc[1:]
    fx_ret = fx["USD"].pct_change(fill_method=None).iloc[1:]
    np.testing.assert_allclose(
        1 + rets["US_EQUITY"],
        (1 + local_usd) * (1 + fx_ret),
    )

    # A fixed FX rate reproduces native-currency returns.
    fixed_fx = pd.DataFrame({"USD": 0.9}, index=dates)
    frozen = convert_prices(local, asset_ccy, fixed_fx, "EUR")
    np.testing.assert_allclose(
        frozen.pct_change(fill_method=None).iloc[1:].to_numpy(),
        local.pct_change(fill_method=None).iloc[1:].to_numpy(),
    )

    print("CURRENCY CONVERSION TESTS PASSED")


if __name__ == "__main__":
    run_tests()