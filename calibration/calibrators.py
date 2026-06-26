"""
Model calibration routines.

Each calibrator takes a ImpliedVolSurface and returns calibrated model parameters.
Calibration minimises the sum of squared errors between model-implied vols
and market-implied vols across all (T, K) grid points.

The local vol model does not require explicit calibration — it reads sigma_loc
directly from the surface via Dupire's formula.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, Bounds
from calibration.vol_surface import ImpliedVolSurface
from engine.heston_cf import heston_price_batch
from engine.black_scholes import implied_vol as bs_implied_vol
from models.heston import HestonParams
from models.sabr import SABRParams


# ---------------------------------------------------------------------------
# Heston calibration
# ---------------------------------------------------------------------------

def calibrate_heston(
    surface: ImpliedVolSurface,
    n_calibration_points: int = 36,
    initial_params: HestonParams | None = None,
) -> HestonParams:
    """
    Calibrate Heston to the implied vol surface via full CF-based pricing.

    Minimises RMSE of implied vols across a (T, K) grid that covers the
    full smile — not just ATM — so rho and xi are properly identified.

    Parameters
    ----------
    surface : ImpliedVolSurface
    n_calibration_points : total grid points (~sqrt × sqrt grid)
    initial_params : optional warm start; sensible defaults used if None
    """
    T_min, T_max = surface.maturity_range
    S = surface.spot
    r = surface.rate
    q = surface.div_yield

    # Build calibration grid: maturities × strikes
    n_side = max(3, int(np.sqrt(n_calibration_points)))
    T_cal = np.linspace(max(T_min, 0.25), T_max, n_side)
    # Near-the-money strikes (±30% around mid-surface forward)
    F_mid = float(surface.forward(float(np.mean(T_cal))))
    K_lo = max(surface.strike_range[0], F_mid * 0.70)
    K_hi = min(surface.strike_range[1], F_mid * 1.30)
    K_cal = np.linspace(K_lo, K_hi, n_side)

    T_grid, K_grid = np.meshgrid(T_cal, K_cal, indexing="ij")
    T_flat = T_grid.ravel()
    K_flat = K_grid.ravel()
    target_ivs = np.array([float(surface.implied_vol(t, k))
                           for t, k in zip(T_flat, K_flat)])

    # Initial parameters: anchor v0 / theta to short / long ATM vol
    atm_short = float(surface.implied_vol(T_cal[0],  float(surface.forward(T_cal[0]))))
    atm_long  = float(surface.implied_vol(T_cal[-1], float(surface.forward(T_cal[-1]))))

    if initial_params is None:
        x0 = [2.0, atm_long**2, 0.5, -0.5, atm_short**2]
    else:
        p = initial_params
        x0 = [p.kappa, p.theta, p.xi, p.rho, p.v0]

    bounds = Bounds(
        lb=[0.01, 1e-4, 0.01, -0.999, 1e-4],
        ub=[20.0,  1.0,  5.0,  0.999,  2.0],
    )

    def objective(x: np.ndarray) -> float:
        kappa, theta, xi, rho, v0 = x
        sse, n = 0.0, 0
        for i, T in enumerate(T_cal):
            # One CF evaluation per maturity covers all strikes
            prices = heston_price_batch(S, K_cal, float(T), r, q,
                                        kappa, theta, xi, rho, v0)
            for j, (price, K) in enumerate(zip(prices, K_cal)):
                iv_mkt = target_ivs[i * n_side + j]
                iv_mdl = bs_implied_vol(float(price), S, float(K), float(T), r, q,
                                        is_call=True)
                sse += (iv_mkt * 5.0) ** 2 if np.isnan(iv_mdl) else (iv_mdl - iv_mkt) ** 2
                n += 1
        return sse / max(n, 1)

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-7})

    kappa, theta, xi, rho, v0 = res.x
    return HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


# ---------------------------------------------------------------------------
# SABR calibration
# ---------------------------------------------------------------------------

def calibrate_sabr(
    surface: ImpliedVolSurface,
    beta: float = 1.0,
    calibration_maturity: float | None = None,
    n_strikes: int = 10,
) -> SABRParams:
    """
    Calibrate SABR (alpha, rho, nu) for fixed beta to the implied vol surface.

    SABR is typically calibrated slice-by-slice per maturity.
    Here we calibrate to the median maturity slice by default.

    Parameters
    ----------
    beta : float
        Fixed CEV exponent. Default 1.0 (lognormal backbone).
    calibration_maturity : float | None
        Maturity slice to calibrate to. Defaults to median maturity.
    n_strikes : int
        Number of strike points to use from that maturity slice.
    """
    T_min, T_max = surface.maturity_range
    T_cal = calibration_maturity or 0.5 * (T_min + T_max)
    T_cal = float(np.clip(T_cal, T_min, T_max))

    F = float(surface.forward(T_cal))

    # Restrict to near-ATM strikes where the Hagan formula is reliable
    K_cal = np.linspace(F * 0.80, F * 1.20, n_strikes)
    target_vols = np.array([surface.implied_vol(float(T_cal), float(k)) for k in K_cal])

    def sabr_vol(alpha, rho, nu, K):
        from models.sabr import SABRParams
        p = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
        v = p.hagan_implied_vol(F, K, T_cal)
        return max(v, 1e-6)

    atm_vol = float(surface.implied_vol(float(T_cal), float(F)))
    x0 = [atm_vol, -0.3, 0.4]  # alpha, rho, nu

    bounds = Bounds(lb=[1e-4, -0.999, 1e-4], ub=[5.0, 0.999, 10.0])

    def objective(x):
        alpha, rho, nu = x
        errors = []
        for K, tv in zip(K_cal, target_vols):
            try:
                mv = sabr_vol(alpha, rho, nu, K)
            except Exception:
                mv = atm_vol
            errors.append((mv - tv) ** 2)
        return np.mean(errors)

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 300, "ftol": 1e-10})

    alpha, rho, nu = res.x
    return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
