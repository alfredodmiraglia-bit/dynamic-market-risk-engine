import numpy as np
import pandas as pd


def convert_prices(
    prices,
    asset_currencies,
    fx_rates,
    base_currency="USD"
):

    converted = prices.astype(float).copy()

    if converted.empty:
        raise ValueError("Empty price table.")

    if not converted.index.is_unique or not converted.columns.is_unique:
        raise ValueError("Duplicate dates or assets.")

    if (
        not np.isfinite(converted.to_numpy()).all()
        or (converted <= 0).any().any()
    ):
        raise ValueError("Missing or invalid asset prices.")

    if set(asset_currencies) != set(converted.columns):
        raise ValueError("Currency mapping must match assets.")

    for ticker in converted.columns:

        currency = asset_currencies[ticker]

        if currency == base_currency:
            continue

        if currency not in fx_rates.columns:
            raise ValueError(f"Missing FX rates for {currency}.")

        rate = fx_rates[currency].reindex(converted.index)

        if (
            not np.isfinite(rate.to_numpy()).all()
            or (rate <= 0).any()
        ):
            raise ValueError(f"Invalid FX rates for {currency}.")

        converted[ticker] *= rate

    return converted  