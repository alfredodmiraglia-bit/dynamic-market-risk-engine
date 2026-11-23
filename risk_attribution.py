import numpy as np
import pandas as pd
from scipy import stats


def risk_attribution(result, weights, confidence, capital):

    covariance = result["Covariance"]

    if isinstance(covariance, pd.DataFrame):

        covariance = covariance.loc[
            weights.index, weights.index
        ].to_numpy()

    else:

        covariance = np.asarray(covariance)

    w = weights.to_numpy()

    means = result["Asset Means (%)"]

    if isinstance(means, pd.Series):

        mu = means.reindex(weights.index).to_numpy()

    else:

        mu = np.asarray(means)

    df = result["Degrees of Freedom"]

    # Portfolio risk
    portfolio_mean = w @ mu

    portfolio_volatility = np.sqrt(
        w @ covariance @ w
    )

    # Marginal volatility
    marginal_volatility = (
        covariance @ w
    ) / portfolio_volatility

    variance_contributions = w * (covariance @ w)

    # Standardized Student-t coefficients
    scale = np.sqrt((df - 2) / df)

    q = stats.t.ppf(confidence, df=df)

    var_coefficient = scale * q

    es_coefficient = (
        scale
        * stats.t.pdf(q, df=df)
        * (df + q**2)
        / ((df - 1) * (1 - confidence))
    )

    # Marginal risk
    marginal_var = (
        -mu + var_coefficient * marginal_volatility
    )

    marginal_es = (
        -mu + es_coefficient * marginal_volatility
    )

    # Component risk
    component_var = w * marginal_var

    component_es = w * marginal_es

    portfolio_var = result["VaR (%)"]

    portfolio_es = result["ES (%)"]

    # Euler decomposition checks
    if not np.isclose(component_var.sum(), portfolio_var):

        raise ValueError("VaR decomposition failed.")

    if not np.isclose(component_es.sum(), portfolio_es):

        raise ValueError("ES decomposition failed.")

    # Attribution results
    attribution = pd.DataFrame({

        "Asset": weights.index,

        "Portfolio Weight (%)": w * 100,

        "Variance Contribution (%)": (
            variance_contributions
            / portfolio_volatility**2 * 100
        ),

        "Component VaR (USD)": (
            component_var * capital / 100
        ),

        "Component ES (USD)": (
            component_es * capital / 100
        ),

        "VaR Contribution (%)": (
            component_var / portfolio_var * 100
            if portfolio_var != 0 else np.nan
        ),

        "ES Contribution (%)": (
            component_es / portfolio_es * 100
            if portfolio_es != 0 else np.nan
        )
    })

    return attribution