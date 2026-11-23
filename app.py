import streamlit as st
import pandas as pd
import numpy as np
from market_data import (
    load_market_data,
    search_assets,
    load_asset_history,
    load_latest_fx,
    load_fx_rate,
    SUPPORTED_CURRENCIES
)
from datetime import date, timedelta
from risk_models import historical_forecast, ccc_t_forecast
from backtesting import ( 
    run_backtest, 
    basel_traffic_light
)
import plotly.graph_objects as go
from stress_testing import ( 
    DEFAULT_SHOCKS,
    hypothetical_stress,
    historical_stress,
    sensitivity_stress
)
from risk_attribution import risk_attribution
from portfolio_strategy import fixed_holdings



# Configurazione della pagina
st.set_page_config(
    page_title="Dynamic Market Risk Engine",
    layout="wide"
)

st.title("Dynamic Market Risk Engine")
st.subheader("Exchange Rates")

fx_base = st.selectbox(
    "FX Reference Currency",
    SUPPORTED_CURRENCIES,
    index=SUPPORTED_CURRENCIES.index("EUR")
)

@st.cache_data(ttl=300)
def cached_latest_fx(base):
    return load_latest_fx(
        base,
        SUPPORTED_CURRENCIES
    )

if st.button("Refresh Exchange Rates"):
    cached_latest_fx.clear()

fx_quotes = cached_latest_fx(fx_base)

st.dataframe(
    fx_quotes.round(6),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Latest available market quotes. "
    "Exchange rates may be delayed."
)

st.divider()

st.subheader("Asset Search")

query = st.text_input(
    "Search by company name or ticker",
    placeholder="Example: NVIDIA, Apple, Vanguard..."
)

if st.button("Search"):

    st.session_state["search_results"] = {}

    if query.strip():

        try:
            with st.spinner("Searching financial instruments..."):

                st.session_state["search_results"] = search_assets(query)

            if not st.session_state["search_results"]:
                st.warning("No instruments found.")

        except Exception as e:
            st.error(str(e))

results = st.session_state.get("search_results", {})

add_asset = False
selected_ticker = None

if results:

    selected_ticker = st.selectbox(
        "Select instrument",
        options=list(results.keys()),
        format_func=lambda ticker: (
            f"{results[ticker]['name']} "
            f"({ticker}) | "
            f"{results[ticker]['exchange']} | "
            f"{results[ticker]['type']}"
        )
    )

    add_asset = st.button("Add to Portfolio")


# Portfolio configuration

st.divider()
st.subheader("Portfolio Configuration")

default_portfolio = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "SPY", "TLT", "GLD"],
    "Weight (%)": [20.0, 20.0, 30.0, 20.0, 10.0]
})

if "portfolio_editor_base" not in st.session_state:

    st.session_state["portfolio_editor_base"] = (
        st.session_state.get("portfolio", default_portfolio).copy()
    )

if "editor_version" not in st.session_state:
    st.session_state["editor_version"] = 0


base_currency = st.selectbox(
    "Portfolio Base Currency",
    SUPPORTED_CURRENCIES,
    index=SUPPORTED_CURRENCIES.index(
        st.session_state.get("base_currency", "USD")
    ),
    key="portfolio_base_input"
)

capital = st.number_input(
    f"Portfolio Value ({base_currency})",
    min_value=1.0,
    value=float(st.session_state.get("capital", 10000000)),
    step=100000.0,
    key="portfolio_capital_input"
)

portfolio = st.data_editor(
    st.session_state["portfolio_editor_base"],
    num_rows="dynamic",
    hide_index=True,
    key=f"portfolio_editor_{st.session_state['editor_version']}"
)

if add_asset:

    existing_tickers = (
        portfolio["Ticker"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if selected_ticker in existing_tickers.values:

        st.warning("This asset is already in the portfolio.")

    else:

        new_asset = pd.DataFrame({
            "Ticker": [selected_ticker],
            "Weight (%)": [0.0]
        })

        st.session_state["portfolio_editor_base"] = pd.concat(
            [portfolio, new_asset],
            ignore_index=True
        )

        st.session_state["editor_version"] += 1

        st.rerun()


# Weight validation

weights = pd.to_numeric(
    portfolio["Weight (%)"],
    errors="coerce"
)

st.metric("Total Portfolio Weight", f"{weights.sum():.2f}%")

if st.button("Confirm Portfolio"):

    portfolio = portfolio.copy()

    portfolio["Ticker"] = (
        portfolio["Ticker"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if portfolio.empty:

        st.error("Insert at least one asset.")

    elif portfolio["Ticker"].eq("").any():

        st.error("Asset tickers cannot be empty.")

    elif portfolio["Ticker"].duplicated().any():

        st.error("Duplicate asset tickers.")

    elif not np.isfinite(weights).all() or (weights <= 0).any():

        st.error("All weights must be positive numbers.")

    elif not np.isclose(weights.sum(), 100):

        st.error("Portfolio weights must sum to 100%.")

    else:

        portfolio["Weight (%)"] = weights

        st.session_state["portfolio"] = portfolio.copy()
        st.session_state["capital"] = capital
        st.session_state["base_currency"] = base_currency
        st.session_state.pop("market_data", None)

        st.success("Portfolio successfully configured.")
base_currency = st.session_state.get("base_currency", "USD")
def currency_labels(data):

    return data.rename(
        columns=lambda col: (
            col.replace("(USD)", f"({base_currency})")
            if isinstance(col, str)
            else col
        )
    )
if "portfolio" in st.session_state:

    st.subheader("Selected Portfolio")

    confirmed = st.session_state["portfolio"].copy()

    confirmed[f"Exposure ({base_currency})"] = (
        confirmed["Weight (%)"] / 100
        * st.session_state["capital"]
    )

    st.dataframe(confirmed, hide_index=True)
    st.subheader("Asset Historical Prices")

    history_tickers = confirmed["Ticker"].tolist()

    history_date = st.date_input(
        "Historical reference date",
        value=date.today() - timedelta(days=1),
        max_value=date.today() - timedelta(days=1)
    )

    if st.button("Load 30-year history"):

        with st.spinner("Downloading historical prices..."):

            histories, failures = load_asset_history(
                history_tickers,
                history_date
            )

        st.session_state["asset_history"] = {
            "tickers": history_tickers,
            "date": history_date,
            "histories": histories,
            "failures": failures
        }

    asset_history = st.session_state.get("asset_history")

    if (
        asset_history is not None
        and asset_history["tickers"] == history_tickers
        and asset_history["date"] == history_date
    ):

        histories = asset_history["histories"]

        if asset_history["failures"]:

            st.warning(
                "Historical data unavailable for: "
                + ", ".join(asset_history["failures"])
            )

        if histories:

            selected_asset = st.selectbox(
                "Select asset to display",
                list(histories.keys())
            )

            historical_prices = histories[selected_asset]

            st.caption(
                f"Available history: "
                f"{historical_prices.index.min().date()} — "
                f"{historical_prices.index.max().date()}"
            )

            fig_history = go.Figure()

            fig_history.add_trace(
                go.Scatter(
                    x=historical_prices.index,
                    y=historical_prices.values,
                    mode="lines",
                    name=selected_asset
                )
            )

            fig_history.update_layout(
                title=f"{selected_asset} - Historical Prices",
                xaxis_title="Date",
                yaxis_title="Adjusted Price (local currency)",
                height=450,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=[
                            dict(count=1, label="1Y",
                                 step="year", stepmode="backward"),
                            dict(count=5, label="5Y",
                                 step="year", stepmode="backward"),
                            dict(count=10, label="10Y",
                                 step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ]
                    ),
                    rangeslider=dict(visible=True)
                )
            )

            st.plotly_chart(
                fig_history,
                use_container_width=True
            )

    if "portfolio" in st.session_state:

     st.divider()
     st.subheader("Market Data Configuration")

    with st.form("market_data_form"):

        col1, col2 = st.columns(2)

        start_date = col1.date_input(
            "Start Date",
            value=date(2016, 1, 1)
        )

        end_date = col2.date_input(
            "End Date",
            value=date.today() - timedelta(days=1),
            max_value=date.today() - timedelta(days=1)
        )

        download = st.form_submit_button("Download Market Data")

    if download:

        st.session_state.pop("market_data", None)

        if start_date >= end_date:

            st.error("Start date must precede end date.")

        else:

            tickers = st.session_state["portfolio"]["Ticker"].tolist()

            try:

                with st.spinner("Downloading market data..."):

                    prices, returns, details = load_market_data(
                        tickers,
                        start_date,
                        end_date,
                        base_currency=base_currency,
                        return_details=True
                    )

                st.session_state["market_data"] = {
                    "prices": prices,
                    "returns": returns,
                    "details": details,
                    "base_currency": base_currency
                }

            except Exception as e:

                st.error(str(e))

    if "market_data" in st.session_state:

        prices = st.session_state["market_data"]["prices"]
        returns = st.session_state["market_data"]["returns"]

        st.success("Market data successfully downloaded.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Assets", returns.shape[1])
        col2.metric("Observations", returns.shape[0])
        col3.metric("First Date", str(returns.index.min().date()))
        col4.metric("Last Date", str(returns.index.max().date()))

        st.subheader("Asset Currencies")

        details = st.session_state["market_data"]["details"]

        st.dataframe(
            pd.Series(
                details["asset_currencies"],
                name="Local Currency"
            ).to_frame()
        )

        st.subheader(f"Adjusted Prices ({base_currency})")
        st.dataframe(prices, height=300)

        st.subheader("Daily Returns")
        st.dataframe(returns, height=300)

        st.subheader("Missing Price Observations")
        st.dataframe(prices.isna().sum().to_frame("Missing"))


# ==================================================
# PORTFOLIO STRATEGY
# ==================================================

strategy = "Daily Constant Weights"
holdings = None

if "market_data" in st.session_state:

    st.divider()
    st.header("Portfolio Strategy")

    strategy = st.selectbox(
        "Portfolio Management Strategy",
        ["Daily Constant Weights", "Fixed Holdings (Preview)"],
        index=0
    )

    if strategy == "Daily Constant Weights":
        st.info(
            "Daily rebalancing maintains the target portfolio weights."
        )

    else:
        prices_strategy = st.session_state["market_data"]["prices"]

        weights_strategy = (
            st.session_state["portfolio"]
            .set_index("Ticker")["Weight (%)"] / 100
        ).reindex(prices_strategy.columns)

        inception_date = st.date_input(
            "Portfolio Inception Date",
            value=prices_strategy.index[0].date(),
            min_value=prices_strategy.index[0].date(),
            max_value=prices_strategy.index[-1].date()
        )

        initial_investment = st.number_input(
            f"Initial Investment at Inception ({base_currency})",
            min_value=1.0,
            value=float(st.session_state["capital"]),
            key="fixed_holdings_initial_investment"
        )

        selected_prices = prices_strategy.loc[
            prices_strategy.index >= pd.Timestamp(inception_date)
        ]

        if selected_prices.empty:
            st.warning("No prices available after the selected date.")
            st.stop()

        holdings = fixed_holdings(
            selected_prices,
            weights_strategy,
            initial_investment
        )

        col1, col2 = st.columns(2)

        col1.metric(
            f"Initial Capital ({base_currency})",
            f"{holdings['portfolio_value'].iloc[0]:,.2f} {base_currency}"
        )

        col2.metric(
            f"Latest Portfolio Value ({base_currency})",
            f"{holdings['portfolio_value'].iloc[-1]:,.2f} {base_currency}"
        )

        st.subheader(f"Portfolio Value Evolution ({base_currency})")
        st.line_chart(holdings["portfolio_value"])

        st.subheader("Portfolio Weight Evolution (%)")
        st.line_chart(holdings["dynamic_weights"] * 100)

        st.subheader("Current Holdings")

        positions = pd.DataFrame({
            "Quantity": holdings["quantities"],
            f"Market Value ({base_currency})": holdings["position_values"].iloc[-1],
            "Current Weight (%)": (
                holdings["dynamic_weights"].iloc[-1] * 100
            )
        })

        st.dataframe(positions.round(4))



# ==================================================
# RISK FORECAST
# ==================================================

if "market_data" in st.session_state:

    st.divider()
    st.subheader("Risk Forecast")


    market_data = st.session_state["market_data"]
    returns = market_data["returns"]

    weights = (
        st.session_state["portfolio"]
        .set_index("Ticker")["Weight (%)"] / 100
    ).reindex(returns.columns)

    capital = float(st.session_state["capital"])

    # Use current exposures for the selected portfolio strategy
    if strategy == "Fixed Holdings (Preview)":

        weights = (
            holdings["dynamic_weights"]
            .iloc[-1]
            .reindex(returns.columns)
        )

        capital = float(holdings["portfolio_value"].iloc[-1])

    # Invalidate forecasts when the portfolio strategy changes
    strategy_config = (
        strategy,
        str(inception_date) if strategy == "Fixed Holdings (Preview)" else None,
        float(initial_investment) if strategy == "Fixed Holdings (Preview)" else None
    )

    if market_data.get("strategy_config") != strategy_config:

        market_data.pop("risk_forecast", None)
        market_data.pop("backtest", None)

        market_data["strategy_config"] = strategy_config

    if len(returns) < 252:

        st.warning("At least 252 observations are required.")

    else:

        with st.form("risk_forecast_form"):

            col1, col2, col3 = st.columns(3)

            model = col1.selectbox(
                "Risk Model",
                ["CCC-GARCH-t", "Historical Simulation"]
            )

            window = col2.number_input(
                "Estimation Window",
                min_value=252,
                max_value=len(returns),
                value=min(756, len(returns)),
                step=1
            )

            confidence = col3.selectbox(
                "Confidence Level",
                [0.95, 0.975, 0.99],
                index=1,
                format_func=lambda x: f"{x * 100:g}%"
            )

            calculate = st.form_submit_button(
                "Calculate Risk"
            )

        if calculate:

            market_data.pop("risk_forecast", None)

            returns_window = returns.tail(window)

            try:

                with st.spinner("Estimating risk model..."):

                    if model == "Historical Simulation":

                        portfolio_window = returns_window.dot(weights)

                        result = historical_forecast(
                            portfolio_window,
                            confidence
                        )

                    else:

                        result = ccc_t_forecast(
                            returns_window,
                            weights,
                            confidence
                        )

                market_data["risk_forecast"] = {
                    "strategy": strategy,
                    "capital": capital,
                    "model": model,
                    "window": window,
                    "confidence": confidence,
                    "start": returns_window.index.min(),
                    "end": returns_window.index.max(),
                    "result": result
                }

            except Exception as e:

                st.error(str(e))

        if "risk_forecast" in market_data:

            forecast = market_data["risk_forecast"]
            result = forecast["result"]

            var_pct = result["VaR (%)"]
            es_pct = result["ES (%)"]

            var_usd = forecast["capital"] * var_pct / 100
            es_usd = forecast["capital"] * es_pct / 100

            st.caption(
                f"Model: {forecast['model']} | "
                f"Confidence: {forecast['confidence'] * 100:g}% | "
                f"Window: {forecast['window']} observations | "
                f"Horizon: 1 trading day"
            )

            st.write(
                "Estimation period:",
                forecast["start"].date(),
                "–",
                forecast["end"].date()
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("VaR (%)", f"{var_pct:.4f}%")
            col2.metric("ES (%)", f"{es_pct:.4f}%")
            col3.metric(
                f"VaR ({base_currency})",
                f"{var_usd:,.2f} {base_currency}"
            )

            col4.metric(
                f"ES ({base_currency})",
                f"{es_usd:,.2f} {base_currency}"
            )

            st.subheader("Model Estimates")

            estimates = {
                "Mean (%)": result["Mean (%)"],
                "Volatility (%)": result["Volatility (%)"]
            }

            if "Degrees of Freedom" in result:

                estimates["Degrees of Freedom"] = (
                    result["Degrees of Freedom"]
                )

            st.dataframe(
                pd.Series(estimates).to_frame("Estimate")
            )

            if "Correlation" in result:

                st.subheader("Conditional Correlation Matrix")

                st.dataframe(
                    result["Correlation"].round(4)
                )

                st.subheader("Conditional Covariance Matrix")

                st.dataframe(
                    result["Covariance"].round(4)
                )

                st.caption("Covariance expressed in squared percentage units.")

            if forecast["model"] == "Historical Simulation":

                tail_observations = (
                    forecast["window"]
                    * (1 - forecast["confidence"])
                )

                if tail_observations < 10:

                    st.warning(
                        "Limited effective tail observations. "
                        "Historical ES may be unstable."
                    )

# ==================================================
# CURRENCY RISK COMPARISON
# ==================================================

if "market_data" in st.session_state:

    st.divider()
    st.header("Currency Risk Comparison")

    comparison_data = st.session_state["market_data"]
    comparison_forecast = comparison_data.get("risk_forecast")

    if comparison_forecast is None:
        st.info("Calculate a risk forecast before comparing currencies.")

    else:
        alternative_currency = st.selectbox(
            "Alternative Reporting Currency",
            [c for c in SUPPORTED_CURRENCIES if c != base_currency],
            key="alternative_reporting_currency"
        )

        st.caption(
            "FX-frozen is a theoretical benchmark, not a hedging strategy. "
            "All scenarios use the same portfolio and estimation window."
        )

        if st.button("Compare Currency Risk"):

            try:
                n_window = comparison_forecast["window"]
                model_fx = comparison_forecast["model"]
                confidence_fx = comparison_forecast["confidence"]

                market_returns = comparison_data["returns"].tail(n_window)

                price_window = comparison_data["prices"].iloc[
                    -(n_window + 1):
                ]

                local_window = comparison_data["details"]["local_prices"].reindex(
                    price_window.index
                )

                if (
                    len(price_window) != n_window + 1
                    or not market_returns.index.equals(price_window.index[1:])
                ):
                    raise ValueError(
                        "Price and return windows are not aligned."
                    )

                # Current portfolio weights
                current_weights = (
                    st.session_state["portfolio"]
                    .set_index("Ticker")["Weight (%)"] / 100
                ).reindex(market_returns.columns)

                if strategy == "Fixed Holdings (Preview)":
                    current_weights = (
                        holdings["dynamic_weights"]
                        .iloc[-1]
                        .reindex(market_returns.columns)
                    )

                if current_weights.isna().any():
                    raise ValueError("Portfolio weights are not aligned.")

                # Scenario 1: Freeze all exchange rates
                frozen_returns = (
                    local_window
                    .pct_change(fill_method=None)
                    .iloc[1:]
                )

                # Scenario 2: Convert to an alternative currency
                available_fx = comparison_data["details"]["fx_rates"]

                if alternative_currency in available_fx.columns:

                    alternative_fx = (
                        1.0 / available_fx[alternative_currency]
                    ).reindex(price_window.index)

                else:

                    # Download FX history with an additional lookback buffer
                    fx_start = (
                        price_window.index[0] - pd.Timedelta(days=7)
                    ).strftime("%Y-%m-%d")

                    fx_end = (
                        price_window.index[-1] + pd.Timedelta(days=1)
                    ).strftime("%Y-%m-%d")

                    fx_history = load_fx_rate(
                        base_currency,
                        alternative_currency,
                        start=fx_start,
                        end=fx_end
                    ).sort_index()

                    # Use only the latest available historical quotation
                    alternative_fx = fx_history.reindex(
                        price_window.index,
                        method="ffill"
                    )

                    # Record the actual date of each FX quotation
                    quote_dates = pd.Series(
                        fx_history.index,
                        index=fx_history.index
                    ).reindex(
                        price_window.index,
                        method="ffill"
                    )

                    quote_age = (
                        price_window.index.to_series() - quote_dates
                    ).dt.days

                    # Reject quotations older than four calendar days
                    alternative_fx = alternative_fx.mask(
                        quote_age > 4
                    )

                invalid_fx = (
                    alternative_fx.isna()
                    | ~np.isfinite(alternative_fx)
                    | (alternative_fx <= 0)
                )

                if invalid_fx.any():
                    missing_dates = alternative_fx.index[invalid_fx]

                    raise ValueError(
                        f"FX data unavailable for {base_currency}/"
                        f"{alternative_currency}. "
                        f"First affected date: {missing_dates[0].date()}. "
                        f"Missing observations: {len(missing_dates)}."
                    )

                alternative_prices = price_window.mul(
                    alternative_fx,
                    axis=0
                )

                alternative_returns = (
                    alternative_prices
                    .pct_change(fill_method=None)
                    .iloc[1:]
                )

                # Use the same risk model for every scenario
                def estimate_fx_scenario(asset_returns):

                    if model_fx == "Historical Simulation":
                        return historical_forecast(
                            asset_returns.dot(current_weights),
                            confidence_fx
                        )

                    return ccc_t_forecast(
                        asset_returns,
                        current_weights,
                        confidence_fx
                    )

                base_result = comparison_forecast["result"]
                frozen_result = estimate_fx_scenario(frozen_returns)
                alternative_result = estimate_fx_scenario(
                    alternative_returns
                )

                scenarios = [
                    (f"Actual ({base_currency})", base_result),
                    (f"FX Frozen ({base_currency})", frozen_result),
                    (
                        f"Alternative ({alternative_currency})",
                        alternative_result
                    )
                ]

                comparison_table = pd.DataFrame([
                    {
                        "Scenario": scenario_name,
                        "VaR (%)": float(estimate["VaR (%)"]),
                        "ES (%)": float(estimate["ES (%)"]),
                        "Volatility (%)": float(
                            estimate["Volatility (%)"]
                        )
                    }
                    for scenario_name, estimate in scenarios
                ])

                st.dataframe(
                    comparison_table.round(4),
                    hide_index=True,
                    use_container_width=True
                )

                var_fx_effect = (
                    float(base_result["VaR (%)"])
                    - float(frozen_result["VaR (%)"])
                )

                es_fx_effect = (
                    float(base_result["ES (%)"])
                    - float(frozen_result["ES (%)"])
                )

                st.write(
                    f"FX effect (actual minus frozen): "
                    f"VaR {var_fx_effect:+.4f} pp; "
                    f"ES {es_fx_effect:+.4f} pp."
                )

                st.caption(
                    "Differences are net effects including diversification "
                    "and correlation, not additive FX risk contributions. "
                    "The alternative currency is a reporting perspective "
                    "for the same holdings."
                )

            except Exception as exc:
                st.error(str(exc))


# Reports currently available only for constant weights
if "market_data" not in st.session_state:
    st.stop()



summary = None
forecasts = None

st.divider()

st.header("Backtesting")

market_data = st.session_state.get("market_data")

if market_data is not None :

    risk_forecast = market_data.get("risk_forecast")

    if risk_forecast is None:

        st.info("Calculate a risk forecast before running the backtest.")

    else:
        model = risk_forecast["model"]
        window = risk_forecast["window"]
        confidence = risk_forecast["confidence"]

        returns = market_data["returns"]

        weights = (
            st.session_state["portfolio"]
            .set_index("Ticker")["Weight (%)"] / 100
        )

        st.write(f"**Model:** {model}")
        st.write(f"**Estimation window:** {window} trading days")
        st.write(f"**Confidence level:** {confidence:.1%}")
        st.write("**Forecast horizon:** 1 trading day")

        eligible_dates = returns.index[window:]

        if holdings is not None:

            actual_inception = (
                holdings["portfolio_value"].index[0]
            )

            eligible_dates = eligible_dates[
                eligible_dates > actual_inception
            ]

        if len(eligible_dates) < 30:

            st.warning(
                "Insufficient eligible observations "
                "for backtesting. At least 30 are required."
            )

        else:

            min_date = eligible_dates[0].date()
            max_date = eligible_dates[-1].date()

            default_start = eligible_dates[
                max(0, len(eligible_dates) - 252)
            ].date()

            with st.form("backtesting_form"):

                col1, col2 = st.columns(2)

                with col1:

                    bt_start = st.date_input(
                        "Backtesting start date",
                        value=default_start,
                        min_value=min_date,
                        max_value=max_date
                    )

                with col2:

                    bt_end = st.date_input(
                        "Backtesting end date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )

                submitted = st.form_submit_button(
                    "Run backtest"
                )

            config = (
                model,
                window,
                confidence,
                strategy_config
            )

            if submitted:

                market_data.pop("backtest", None)

                if bt_start > bt_end:

                    st.error("Invalid backtesting period.")

                else:

                    progress_bar = st.progress(0.0)

                    try:

                        forecasts, failures, summary = run_backtest(
                            returns=returns,
                            weights=weights,
                            model=model,
                            window=window,
                            confidence=confidence,
                            start_date=bt_start,
                            end_date=bt_end,
                            progress=lambda x: progress_bar.progress(
                                float(x)
                            ),
                            holdings=holdings
                        )

                        market_data["backtest"] = {
                            "config": config,
                            "start": bt_start,
                            "end": bt_end,
                            "forecasts": forecasts,
                            "failures": failures,
                            "summary": summary
                        }

                    except Exception as e:

                        st.error(str(e))

                    finally:

                        progress_bar.empty()

            backtest = market_data.get("backtest")

            if (
                backtest is not None
                and backtest["config"] == config
            ):

                forecasts = backtest["forecasts"]
                failures = backtest["failures"]
                summary = backtest["summary"]

                st.subheader("Backtesting results")

                st.caption(
                    f'{backtest["start"]} — {backtest["end"]}'
                )
                st.subheader("Basel Traffic Light")

                st.caption(
                    "Rolling assessment over 250 trading days."
                )

                if not np.isclose(confidence, 0.99):

                    st.warning(
                        "Basel Traffic Light requires a 99% "
                        "confidence level."
                    )

                    st.caption(
                        "Classification thresholds are calibrated "
                        "for the 99% one-day VaR."
                    )

                elif summary is None:

                    st.warning(
                        "A complete backtest without failures "
                        "is required."
                    )

                elif len(forecasts) < 250:

                    st.warning(
                        "At least 250 observations are required."
                    )

                else:

                    basel = basel_traffic_light(forecasts)

                    latest = basel.iloc[-1]

                    zone = latest["Basel Zone"]
                    count = int(latest["Rolling Exceptions"])

                    col1, col2,col3,col4 = st.columns(4)

                    col1.metric("Latest Exceptions", count)
                    col2.metric("Expected Exceptions", 2.5)
                    col3.metric("Maximum Exceptions", int(basel["Rolling Exceptions"].max()))
                    col4.metric("Green Zone Windows", f'{(basel["Basel Zone"] == "Green").mean():.1%}')
                    
                    st.caption(
                    "Latest exceptions refer to the most recent 250 trading days. Maximum exceptions and green zone "
                    "percentage refer to the entire backtesting period "
                    )
                    
                    
                    if zone == "Green":
                        st.success("Green Zone")

                    elif zone == "Amber":
                        st.warning("Amber Zone")

                    else:
                        st.error("Red Zone")

                    upper_limit = max(
                        12,
                        basel["Rolling Exceptions"].max() + 1
                    )

                    fig_basel = go.Figure()

                    fig_basel.add_hrect(
                        y0=-0.5, y1=4.5,
                        fillcolor="green",
                        opacity=0.15,
                        line_width=0
                    )

                    fig_basel.add_hrect(
                        y0=4.5, y1=9.5,
                        fillcolor="orange",
                        opacity=0.15,
                        line_width=0
                    )

                    fig_basel.add_hrect(
                        y0=9.5, y1=upper_limit,
                        fillcolor="red",
                        opacity=0.15,
                        line_width=0
                    )

                    fig_basel.add_trace(
                        go.Scatter(
                            x=basel["Forecast Date"],
                            y=basel["Rolling Exceptions"],
                            mode="lines+markers",
                            name="Rolling Exceptions",
                            line=dict(
                                color="#0068C9",
                                width=2
                            )
                        )
                    )

                    fig_basel.update_layout(
                        title="Basel Traffic Light Zones",
                        xaxis_title="Date",
                        yaxis_title="Number of Exceptions",
                        yaxis=dict(
                            range=[-0.5, upper_limit],
                            dtick=1
                        )
                    )

                    st.plotly_chart(
                        fig_basel,
                        use_container_width=True
                    )

                
                    with st.expander("Detailed Basel Traffic Light Results"):
                        st.dataframe(
                            basel.sort_values(
                                "Forecast Date",
                                ascending=False
                            ),
                            hide_index=True,
                            use_container_width=True
                        )

                     
                if len(failures) > 0:

                    st.error(
                        f"{len(failures)} forecasts failed. "
                        "Statistical validation is unavailable."
                    )

                    st.dataframe(failures)

if summary is not None:

    significance = 0.05

    test_specs = {
        "Kupiec p-value": (
            "Kupiec",
            "Correct exception frequency"
        ),
        "Independence p-value": (
            "Christoffersen Independence",
            "Independent VaR exceptions"
        ),
        "Conditional Coverage p-value": (
            "Conditional Coverage",
            "Correct frequency and independence"
        ),
        "ES Calibration p-value": (
            "ES Calibration",
            "Correct ES calibration"
        )
    }

    # Descriptive statistics
    descriptive_stats = {
        key.replace("(USD)", f"({base_currency})"): value
        for key, value in summary.items()
        if key not in test_specs
    }

    st.dataframe(
        pd.DataFrame(
            descriptive_stats.items(),
            columns=["Metric", "Value"]
        ),
        hide_index=True
    )

    # Statistical tests
    st.subheader("Statistical validation")

    st.caption("Significance level: 5%")

    test_rows = []

    for key, (name, hypothesis) in test_specs.items():

        p_value = summary.get(key)

        if pd.isna(p_value):
            decision = "Not available"

        elif p_value < significance:
            decision = "Reject H0"

        else:
            decision = "Do not reject H0"

        test_rows.append({
            "Test": name,
            "Null hypothesis": hypothesis,
            "p-value": p_value,
            "Decision": decision
        })

    tests = pd.DataFrame(test_rows)

    st.dataframe(
        tests.round(4),
        hide_index=True,
        use_container_width=True
    )

    if (tests["Decision"] == "Reject H0").any():

        st.error("At least one statistical test rejects H0.")

    elif (tests["Decision"] == "Not available").any():

        st.warning("Statistical validation is incomplete.")

    else:

        st.success(
            "None of the statistical tests reject H0."
        )

    if summary["Expected Exceptions"] < 10:

        st.warning(
            "Low expected number of exceptions. "
            "Statistical tests may have limited power."
        )

    exceptions = forecasts[
        forecasts["Actual Loss (%)"] > forecasts["VaR (%)"]
    ].copy()

    fig = go.Figure()

    colors = {
        "Actual Loss (%)": "#0068C9",
        "VaR (%)": "#FF3030",
        "ES (%)": "#83C9FF"
    }

    for column, color in colors.items():

        fig.add_trace(
            go.Scatter(
                x=forecasts["Forecast Date"],
                y=forecasts[column],
                mode="lines",
                name=column,
                line=dict(color=color, width=1.8)
            )
        )

    # Vertical lines for VaR exceptions
    for date in exceptions["Forecast Date"]:

        fig.add_shape(
            type="line",
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            layer="below",
            line=dict(
                color="#0A0A0A",
                width=1.5,
                dash="dash"
            )
        )

    fig.update_layout(
        title=f"{model} — Confidence level {confidence:.1%}",
        xaxis_title="Date",
        yaxis_title="Loss (%)",
        hovermode="x unified",
        height=550,
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Black dashed vertical lines indicate VaR exceptions."
    )

    # VaR exceptions table
    if not exceptions.empty:

        exceptions["Excess (pp)"] = (
            exceptions["Actual Loss (%)"] -
            exceptions["VaR (%)"]
        )

        if holdings is not None:

            exceptions["Excess (USD)"] = (
                exceptions["Actual Loss (USD)"]
                - exceptions["VaR (USD)"]
            )

        else:

            exceptions["Excess (USD)"] = (
                exceptions["Excess (pp)"] / 100
                * st.session_state["capital"]
            )

        with st.expander(
            f"VaR exceptions ({len(exceptions)})",
            expanded=False
        ):

            st.dataframe(
                currency_labels(
                    exceptions[
                        [
                            "Forecast Date",
                            "Actual Loss (%)",
                            "VaR (%)",
                            "ES (%)",
                            "Excess (pp)",
                            "Excess (USD)"
                        ]
                    ].round(4)
                ),
                hide_index=True
            )


    else:
        st.info("No VaR exceptions observed.")

    st.subheader("Daily forecasts")
    st.dataframe(
        currency_labels(forecasts),
        hide_index=True
    )



st.divider()

st.header("Stress Testing")

if market_data is not None:

    returns = market_data["returns"]

    capital = float(st.session_state["capital"])

    weights = (
        st.session_state["portfolio"]
        .set_index("Ticker")["Weight (%)"] / 100
    ).reindex(returns.columns)

    if strategy == "Fixed Holdings (Preview)":

        weights = (
            holdings["dynamic_weights"]
            .iloc[-1]
            .reindex(returns.columns)
        )

        capital = float(
            holdings["portfolio_value"].iloc[-1]
        )

    st.caption(
        f"Strategy: {strategy} | "
        f"Current portfolio value: {capital:,.2f} {base_currency}"
    )

    tab1, tab2, tab3 = st.tabs([
        "Hypothetical Stress",
        "Historical Stress",
        "Sensitivity Stress"
    ])

    # 1. Hypothetical stress
    with tab1:

        st.subheader("Hypothetical market scenarios")

        st.caption(
            "Enter one-day asset price shocks in percentage points."
        )

        default_shocks = DEFAULT_SHOCKS.reindex(
            columns=weights.index
        )

        edited_shocks = st.data_editor(
            default_shocks,
            num_rows="fixed",
            disabled=["_index"],
            key="stress_shocks_" + "_".join(weights.index)
        )

        if edited_shocks.isna().any().any():

            st.warning(
                "Enter the missing shocks for all selected assets."
            )

        else:

            try:

                hypothetical_results = hypothetical_stress(
                    edited_shocks,
                    weights,
                    capital
                )

                st.dataframe(
                    currency_labels(hypothetical_results.round(2)),
                    use_container_width=True
                )

                st.bar_chart(
                    hypothetical_results["Portfolio Loss (USD)"]
                    .rename(f"Portfolio Loss ({base_currency})")
                )

            except Exception as e:

                st.error(str(e))

    # 2. Historical stress
    with tab2:

        st.subheader("Historical market crises")

        st.caption(
            "Worst daily portfolio loss during each crisis, "
            "using current portfolio weights."
        )

        historical_results = historical_stress(
            returns,
            weights,
            capital
        )

        st.dataframe(
            currency_labels(historical_results.round(2)),
            hide_index=True,
            use_container_width=True
        )

        available = historical_results[
            historical_results["Status"] == "Available"
        ]

        if not available.empty:

            st.bar_chart(
                available.set_index("Scenario")[
                    "Portfolio Loss (USD)"
                ].rename(f"Portfolio Loss ({base_currency})")
            )
    # 3. Sensitivity stress
    with tab3:

        st.subheader("Volatility and correlation stress")

        risk_forecast = market_data.get("risk_forecast")

        if (
            risk_forecast is None
            or risk_forecast["model"] != "CCC-GARCH-t"
        ):

            st.info(
                "Calculate a CCC-GARCH-t risk forecast "
                "to perform sensitivity stress testing."
            )

        else:

            confidence = risk_forecast["confidence"]

            st.caption(
                f"Confidence level: {confidence:.1%}"
            )

            try:

                sensitivity_results = sensitivity_stress(
                    risk_forecast["result"],
                    weights,
                    confidence,
                    capital
                )

                st.dataframe(
                    currency_labels(sensitivity_results.round(2)),
                    hide_index=True,
                    use_container_width=True
                )

                st.bar_chart(
                    currency_labels(
                        sensitivity_results.set_index("Scenario")[
                            ["VaR (USD)", "ES (USD)"]
                        ]
                    )
                )

            except Exception as e:

                st.error(str(e))

                st.divider()

if strategy == "Fixed Holdings (Preview)":

    st.info(
        "Risk Attribution is not yet available "
        "for Fixed Holdings."
    )

    st.stop()                
st.header("Risk Attribution")

market_data = st.session_state.get("market_data")

if market_data is not None and strategy == "Daily Constant Weights":

    risk_forecast = market_data.get("risk_forecast")

    if (
        risk_forecast is None
        or risk_forecast["model"] != "CCC-GARCH-t"
    ):

        st.info(
            "Calculate a CCC-GARCH-t risk forecast "
            "to perform risk attribution."
        )

    else:

        capital = st.session_state["capital"]

        confidence = risk_forecast["confidence"]

        weights = (
            st.session_state["portfolio"]
            .set_index("Ticker")["Weight (%)"] / 100
        ).reindex(market_data["returns"].columns)

        st.caption(
            f"CCC-GARCH-t | Confidence level: {confidence:.1%}"
        )

        try:

            attribution = risk_attribution(
                risk_forecast["result"],
                weights,
                confidence,
                capital
            )

            st.dataframe(
                currency_labels(attribution.round(2)),
                hide_index=True,
                use_container_width=True
            )

            fig_attribution = go.Figure()

            fig_attribution.add_trace(
                go.Bar(
                    x=attribution["Asset"],
                    y=attribution["Portfolio Weight (%)"],
                    name="Portfolio Weight",
                    marker_color="#83C9FF"
                )
            )

            fig_attribution.add_trace(
                go.Bar(
                    x=attribution["Asset"],
                    y=attribution["ES Contribution (%)"],
                    name="ES Contribution",
                    marker_color="#F59E0B"
                )
            )

            fig_attribution.update_layout(
                title="Portfolio Weight vs ES Contribution",
                barmode="group",
                yaxis_title="Percentage (%)",
                xaxis_title="Asset",
                height=450
            )

            st.plotly_chart(
                fig_attribution,
                use_container_width=True
            )

            st.caption(
                "Negative risk contributions indicate "
                "a risk-reducing effect within the portfolio."
            )

        except Exception as e:

            st.error(str(e))