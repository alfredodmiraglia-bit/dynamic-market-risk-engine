import yfinance as yf
import pandas as pd
import numpy as np

from currency_risk import convert_prices


SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "CHF", "JPY"]


def clean_close(data):

    close = data["Close"].astype(float).copy()

    close.index = close.index.tz_localize(None).normalize()

    close = close.where(
        np.isfinite(close) & (close > 0)
    )

    return close.sort_index()


# ==================================================
# FX DATA
# ==================================================

def load_fx_rate(base, quote, **history_params):

    if base == quote:
        raise ValueError("Identical currencies.")

    # Quote currency per unit of base currency
    candidates = [
        (f"{base}{quote}=X", False),
        (f"{quote}{base}=X", True)
    ]

    for ticker, inverse in candidates:

        try:

            data = yf.Ticker(ticker).history(
                interval="1d",
                auto_adjust=False,
                actions=False,
                raise_errors=True,
                **history_params
            )

            if data.empty:
                continue

            rate = clean_close(data).dropna()

            if rate.empty:
                continue

            if inverse:
                rate = 1.0 / rate

            return rate

        except Exception:
            continue

    raise ValueError(
        f"Exchange rate unavailable: {base}/{quote}"
    )


# ==================================================
# LATEST EXCHANGE RATES
# ==================================================

def load_latest_fx(base_currency, currencies):

    records = []

    for currency in sorted(set(currencies)):

        if currency == base_currency:
            continue

        try:

            rate = load_fx_rate(
                base_currency,
                currency,
                period="5d"
            )

            records.append({
                "Currency Pair": f"{base_currency}/{currency}",
                "Exchange Rate": float(rate.iloc[-1]),
                "Last Quote": rate.index[-1].date(),
                "Status": "Available"
            })

        except Exception:

            records.append({
                "Currency Pair": f"{base_currency}/{currency}",
                "Exchange Rate": np.nan,
                "Last Quote": None,
                "Status": "Unavailable"
            })

    return pd.DataFrame(
        records,
        columns=[
            "Currency Pair",
            "Exchange Rate",
            "Last Quote",
            "Status"
        ]
    )


# ==================================================
# MARKET DATA
# ==================================================

def load_market_data(
    tickers,
    start,
    end,
    base_currency="USD",
    return_details=False
):

    if base_currency not in SUPPORTED_CURRENCIES:
        raise ValueError("Unsupported base currency.")

    prices = {}
    asset_currencies = {}

    end_date = pd.Timestamp(end) + pd.Timedelta(days=1)

    for ticker in tickers:

        asset = yf.Ticker(ticker)

        data = asset.history(
            start=start,
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            actions=False,
            raise_errors=True
        )

        if data.empty:
            raise ValueError(f"No data available for {ticker}")

        metadata = asset.get_history_metadata()

        currency = metadata.get("currency")

        close = clean_close(data)

        # British pence to British pounds
        if currency in ["GBp", "GBX"]:

            close = close / 100

            currency = "GBP"

        if currency not in SUPPORTED_CURRENCIES:

            raise ValueError(
                f"Unsupported currency for {ticker}: {currency}"
            )

        prices[ticker] = close

        asset_currencies[ticker] = currency

    local_prices = pd.DataFrame(prices).sort_index()

    # Historical foreign exchange rates
    fx_rates = pd.DataFrame(index=local_prices.index)

    for currency in set(asset_currencies.values()):

        if currency == base_currency:
            continue

        rate = load_fx_rate(
            base_currency,
            currency,
            start=start,
            end=end_date.strftime("%Y-%m-%d")
        )

        # Base currency per unit of foreign currency
        fx_rates[currency] = (
            1.0 / rate
        ).reindex(local_prices.index)

    # Select common valid valuation dates
    valid_dates = (
        local_prices.notna().all(axis=1)
        & fx_rates.notna().all(axis=1)
    )

    excluded_dates = int((~valid_dates).sum())

    local_prices = local_prices.loc[valid_dates]

    fx_rates = fx_rates.loc[valid_dates]

    if len(local_prices) < 2:
        raise ValueError("Insufficient common observations.")

    converted_prices = convert_prices(
        local_prices,
        asset_currencies,
        fx_rates,
        base_currency
    )

    returns = converted_prices.pct_change(
        fill_method=None
    ).dropna()

    if returns.empty:
        raise ValueError("Insufficient valid returns.")

    details = {
        "base_currency": base_currency,
        "asset_currencies": asset_currencies,
        "local_prices": local_prices,
        "fx_rates": fx_rates,
        "excluded_dates": excluded_dates
    }

    if return_details:

        return converted_prices, returns, details

    return converted_prices, returns

def search_assets(query):

    results = yf.Search(
        query,
        max_results=20,
        news_count=0
    ).quotes

    assets = {}

    for item in results:

        ticker = item.get("symbol")
        asset_type = item.get("quoteType")

        if not ticker or asset_type not in ["EQUITY", "ETF"]:
            continue

        assets[ticker] = {
            "name": item.get("longname") or item.get("shortname") or ticker,
            "exchange": item.get("exchDisp") or item.get("exchange", ""),
            "type": asset_type
        }

    return assets

def load_asset_history(tickers, as_of_date):

    end_date = pd.Timestamp(as_of_date)

    start_date = end_date - pd.DateOffset(years=30)

    histories = {}
    failures = {}

    for ticker in tickers:

        try:

            data = yf.Ticker(ticker).history(
                start=start_date.date(),
                end=(end_date + pd.Timedelta(days=1)).date(),
                interval="1d",
                auto_adjust=True
            )

            if data.empty:
                raise ValueError("No historical data available.")

            prices = data["Close"].dropna()

            if prices.empty:
                raise ValueError("No valid prices available.")

            prices.index = prices.index.tz_localize(None)

            histories[ticker] = prices

        except Exception as e:

            failures[ticker] = str(e)

    return histories, failures