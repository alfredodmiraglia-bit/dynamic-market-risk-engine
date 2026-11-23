import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize_scalar
from arch import arch_model


# ==================================================
# HISTORICAL SIMULATION
# ==================================================

def historical_forecast(returns, confidence):

    returns = np.asarray(returns, dtype=float) * 100

    sorted_losses = np.sort(-returns)
    n = len(sorted_losses)

    position = int(np.floor(n * confidence))

    var = sorted_losses[position]

    fractional_weight = position + 1 - n * confidence

    es = (
        fractional_weight * sorted_losses[position]
        + sorted_losses[position + 1:].sum()
    ) / (n * (1 - confidence))

    return {
        "Mean (%)": returns.mean(),
        "Volatility (%)": returns.std(ddof=1),
        "VaR (%)": var,
        "ES (%)": es
    }


# ==================================================
# GARCH MARGINAL MODELS
# ==================================================

def fit_multivariate_margins(returns_window):

    standardized_residuals = {}
    forecast_means = {}
    forecast_volatilities = {}

    for asset in returns_window.columns:

        returns_pct = returns_window[asset] * 100

        model = arch_model(
            returns_pct,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False
        )

        result = model.fit(
            disp="off",
            show_warning=False
        )

        if result.convergence_flag != 0:
            raise RuntimeError(
                f"Convergence failure: {asset}"
            )

        forecast = result.forecast(
            horizon=1,
            reindex=False
        )

        variance = float(forecast.variance.iloc[-1, 0])

        if not np.isfinite(variance) or variance <= 0:
            raise RuntimeError(
                f"Invalid forecast variance: {asset}"
            )

        standardized_residuals[asset] = result.std_resid

        forecast_means[asset] = float(
            forecast.mean.iloc[-1, 0]
        )

        forecast_volatilities[asset] = np.sqrt(variance)

    residuals = pd.DataFrame(standardized_residuals)

    if not np.isfinite(residuals.to_numpy()).all():
        raise RuntimeError("Invalid standardized residuals.")

    return (
        residuals,
        pd.Series(forecast_means),
        pd.Series(forecast_volatilities)
    )


# ==================================================
# CONSTANT CONDITIONAL CORRELATION
# ==================================================

def compute_ccc_correlation(residuals):

    z = residuals.to_numpy()

    q_bar = z.T @ z / len(z)

    std = np.sqrt(np.diag(q_bar))

    correlation = q_bar / np.outer(std, std)

    return pd.DataFrame(
        correlation,
        index=residuals.columns,
        columns=residuals.columns
    )


# ==================================================
# STUDENT-T DEGREES OF FREEDOM
# ==================================================

def estimate_common_df(residuals):

    z = residuals.to_numpy().ravel()

    def negative_loglikelihood(nu):

        scale = np.sqrt((nu - 2) / nu)

        log_density = (
            stats.t.logpdf(z / scale, df=nu)
            - np.log(scale)
        )

        return -np.mean(log_density)

    result = minimize_scalar(
        negative_loglikelihood,
        bounds=(2.05, 200),
        method="bounded"
    )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Student-t estimation failure.")

    return float(result.x)


# ==================================================
# CCC-GARCH-T
# ==================================================

def ccc_t_forecast(returns_window, weights, confidence):

    residuals, means, volatilities = (
        fit_multivariate_margins(returns_window)
    )

    correlation = compute_ccc_correlation(residuals)

    nu = estimate_common_df(residuals)

    assets = correlation.columns

    w = weights.reindex(assets).to_numpy()
    mu = means.reindex(assets).to_numpy()
    sigma = volatilities.reindex(assets).to_numpy()

    D = np.diag(sigma)

    H = D @ correlation.to_numpy() @ D

    np.linalg.cholesky(H)

    portfolio_mean = float(w @ mu)

    portfolio_variance = float(w @ H @ w)

    portfolio_volatility = np.sqrt(portfolio_variance)

    scale = np.sqrt((nu - 2) / nu)

    q = stats.t.ppf(confidence, df=nu)

    var = (
        -portfolio_mean
        + portfolio_volatility * scale * q
    )

    es = (
        -portfolio_mean
        + portfolio_volatility
        * scale
        * (nu + q**2) / (nu - 1)
        * stats.t.pdf(q, df=nu)
        / (1 - confidence)
    )

    return {
        "Asset Means (%)": means,
        "Mean (%)": portfolio_mean,
        "Volatility (%)": portfolio_volatility,
        "VaR (%)": var,
        "ES (%)": es,
        "Degrees of Freedom": nu,
        "Correlation": correlation,
        "Covariance": pd.DataFrame(
            H,
            index=assets,
            columns=assets
        )
    }