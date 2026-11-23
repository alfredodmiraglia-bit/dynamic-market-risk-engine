# Dynamic Market Risk Engine

**Python / Streamlit portfolio risk analytics application** with one-day Value at Risk (VaR), Expected Shortfall (ES), rolling backtesting and multi-currency risk comparisons.

> **Status:** Research/portfolio project. Not a production risk platform and not investment advice.

## Overview

The application supports user-defined equity and ETF portfolios with a selectable reporting currency (USD, EUR, GBP, CHF and JPY). Market prices and historical FX rates are downloaded from Yahoo Finance using `yfinance`. The selected base currency is used for the portfolio return series and risk calculations.

The engine combines market data processing, portfolio strategies, risk forecasting, model validation, stress testing and attribution in one interactive application.

## Main features

| Area | Implementation |
|---|---|
| Risk forecasting | 1-day Historical Simulation and CCC-GARCH with Student-t portfolio innovations |
| Risk metrics | VaR and ES at 95%, 97.5% and 99% |
| Backtesting | Rolling estimation windows; actual losses versus ex-ante risk forecasts |
| Validation | Kupiec unconditional coverage, Christoffersen independence/conditional coverage, ES calibration with HAC standard errors, quantile and FZ0 scores |
| Basel diagnostic | Rolling 250-observation traffic-light check for 99% one-day VaR |
| Portfolio strategies | Constant target weights and fixed holdings with changing exposures |
| Stress testing | Editable hypothetical price shocks, historical scenarios, CCC-GARCH-t volatility/correlation sensitivity |
| Attribution | Euler VaR/ES risk contributions for CCC-GARCH-t under constant weights |
| Currency analysis | Historical FX conversion; risk comparison across reporting currencies; FX-frozen counterfactual benchmark |

## Installation

Recommended: Python 3.11 or 3.12 and a clean virtual environment.

**Windows / PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Data is retrieved at runtime; no historical market data is included in the repository.

## Typical workflow

1. Search for equities/ETFs and configure tickers, allocation weights, nominal capital and portfolio base currency.
2. Confirm the portfolio and select the historical data interval.
3. Choose constant-weight rebalancing or fixed holdings.
4. Select Historical Simulation or CCC-GARCH-t, a rolling estimation window and a confidence level, then calculate risk.
5. Optionally run backtesting, inspect exception diagnostics, apply stress scenarios, or compare currency perspectives.

## Methodology notes

- **Historical Simulation:** VaR/ES are computed on the empirical loss distribution of the estimation sample; ES uses fractional weighting at the quantile boundary.
- **CCC-GARCH-t:** Univariate GARCH(1,1) normal-innovation marginal models provide conditional volatilities. A constant conditional correlation matrix is estimated from standardized residuals and a shared Student-t tail parameter is fitted for portfolio loss quantiles. This is a practical two-stage model, **not** a full jointly estimated multivariate Student-t GARCH likelihood.
- **Currency conversion:** prices are converted using historical FX observations prior to computing returns. The *FX Frozen* view is a counterfactual with exchange rates held constant, **not** the realized performance of a currency hedge. Its difference from actual VaR/ES is a net comparison, not a separately additive FX risk contribution.
- **Fixed Holdings:** adjusted historical price series are used to represent synthetic holdings. Quantities should not be interpreted as broker trade quantities.

## Validation / tests

The repository includes small synthetic-data regression scripts:

```bash
python test_portfolio_strategy.py
python test_fixed_holdings_backtesting.py
python test_integrated_backtesting.py
python test_garch_backtesting.py
python test_currency_risk.py
```

The tests that call GARCH require the `arch` dependency and may take longer. Tests are not a substitute for an independent production-grade model validation.

## Limitations

- Historical quotes, ticker metadata and FX rates depend on the availability and quality of Yahoo Finance data. Trading holidays and mixed market calendars can exclude dates; a return between consecutive *common observations* may span more than one calendar/trading day.
- No assumed stale-forward-filled prices in the main market-data loader; a missing asset/FX observation can exclude the entire date.
- Transaction costs, slippage, funding and live trade execution are not part of the risk forecasts.
- Historical stress scenarios require sufficient historical coverage.
- Sensitivity stress testing and Euler attribution currently use CCC-GARCH-t outputs; historical risk attribution is not implemented.
- Individual bonds are **not** supported in this release.

## Future work

- Individual government and corporate bonds with verified historical total-return series, interest-rate curves and credit-spread sensitivities.
- Improved non-synchronous market/FX calendar handling and data-quality reporting.
- Additional distributional and multivariate risk models, with independent validation.
- Extended attribution and hedged/unhedged currency scenarios.

## Project structure

```text
app.py                          Streamlit application
market_data.py                  Yahoo Finance price/FX acquisition and alignment
currency_risk.py               Currency conversion and checks
risk_models.py                 Historical Simulation and CCC-GARCH-t
backtesting.py                 Rolling backtesting and statistical diagnostics
portfolio_strategy.py          Portfolio holdings and exposures
fixed_holdings_backtesting.py  Reference fixed-holdings backtest
stress_testing.py              Hypothetical/historical/sensitivity stress
risk_attribution.py            CCC-GARCH-t Euler attribution
test_*.py                       Synthetic regression tests
```

## License

No license has been selected. All rights are reserved until an explicit license is added.