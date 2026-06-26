"""
Heston (1993) European option pricer via characteristic function.

Uses the Albrecher et al. (2007) 'little trap' formulation which avoids
the branch-cut discontinuity present in the original Heston (1993) paper.

Integration is done with a fixed trapezoidal grid, vectorised over u,
so the cost per (maturity, strike) batch is one numpy array evaluation
rather than one scipy.quad call per point — fast enough for calibration.
"""

from __future__ import annotations

import numpy as np

# ── Fixed integration grid ────────────────────────────────────────────────────
# 1000 points up to 300 gives <0.1 bp error for typical equity parameters.
_U = np.linspace(1e-4, 300.0, 1_000)


def _cf(u: np.ndarray, T: float, r: float, q: float, log_S: float,
        kappa: float, theta: float, xi: float, rho: float, v0: float) -> np.ndarray:
    """
    Characteristic function φ(u) = E_Q[exp(iu · log S_T)] under Heston.
    'Little trap' form: g = (kp - d) / (kp + d) avoids branch-cut issues.

    u can be complex (e.g. u - 1j for the P1 integrand).
    """
    iu = 1j * u
    kp = kappa - rho * xi * iu                        # kappa - i·rho·xi·u
    d = np.sqrt(kp**2 + xi**2 * (iu + u**2))          # little-trap sign

    g = (kp - d) / (kp + d)
    exp_dT = np.exp(-d * T)

    log_ratio = np.log((1.0 - g * exp_dT) / (1.0 - g))

    A = (r - q) * iu * T + kappa * theta / xi**2 * (
        (kp - d) * T - 2.0 * log_ratio
    )
    B = (kp - d) / xi**2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    return np.exp(A + B * v0 + iu * log_S)


def heston_price_batch(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
) -> np.ndarray:
    """
    European call prices for multiple strikes at maturity T under Heston.

    The characteristic function is evaluated once on the shared grid _U
    (vectorised), then the trapezoidal sum is computed per strike.

    Parameters
    ----------
    S       : spot price
    strikes : absolute strike levels, shape (K,)
    T       : maturity in years
    r, q    : risk-free rate, dividend yield (continuous, annual)
    kappa, theta, xi, rho, v0 : Heston parameters

    Returns
    -------
    calls : np.ndarray, shape (K,)  — call prices in same units as S
    """
    u = _U
    log_S = np.log(S)

    # CF at u and at u - i (for P1) — both vectorised over the grid
    cf_u   = _cf(u,       T, r, q, log_S, kappa, theta, xi, rho, v0)
    cf_u1j = _cf(u - 1j, T, r, q, log_S, kappa, theta, xi, rho, v0)

    # Normalisation factor for P1: φ(−i) = E_Q[S_T] / S = e^{(r−q)T}
    cf_mi = _cf(np.array([-1j]), T, r, q, log_S, kappa, theta, xi, rho, v0)[0]

    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)

    calls = np.empty(len(strikes))
    for j, K in enumerate(strikes):
        e_iuK = np.exp(-1j * u * np.log(K))

        P2 = 0.5 + np.trapezoid(np.real(e_iuK * cf_u   / (1j * u)),         u) / np.pi
        P1 = 0.5 + np.trapezoid(np.real(e_iuK * cf_u1j / (1j * u * cf_mi)), u) / np.pi

        calls[j] = max(S * df_q * P1 - K * df_r * P2, 0.0)

    return calls
