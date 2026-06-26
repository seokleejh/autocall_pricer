"""
Black-Scholes closed-form pricer and implied vol inverter.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True,
) -> float:
    """Black-Scholes price for a European call or put."""
    if T <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if sigma <= 0:
        df = np.exp(-r * T)
        F = S * np.exp((r - q) * T)
        return df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)

    F = S * np.exp((r - q) * T)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = np.exp(-r * T)

    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool = True,
    bounds: tuple[float, float] = (1e-6, 5.0),
) -> float:
    """
    Invert the BS formula to find the implied vol for a given option price.
    Returns NaN if no solution exists in `bounds` or the price is below intrinsic.
    """
    if T <= 0:
        return float("nan")

    df = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    if price <= intrinsic + 1e-10:
        return float("nan")

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, q, sigma, is_call) - price

    try:
        lo, hi = bounds
        if objective(lo) * objective(hi) >= 0:
            return float("nan")
        return float(brentq(objective, lo, hi, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")
