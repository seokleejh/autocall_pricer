"""
Heston model: closed-form European option prices via characteristic function.

Uses the Heston (1993) two-probability formulation with Gauss-Legendre
quadrature for the CF integrals. The Re(d) >= 0 branch convention from
Lord & Kahl (2006) is applied to avoid complex-log branch-cut errors.
"""

from __future__ import annotations

import numpy as np


def _heston_cf(
    phi: np.ndarray,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> np.ndarray:
    """
    Heston characteristic function of log(S_T / F) where F = S*exp((r-q)*T).

    phi  : complex-valued integration frequencies, shape (N,)
    Returns complex array of same shape.
    """
    d = np.sqrt((kappa - 1j * rho * xi * phi) ** 2 + xi**2 * (phi**2 + 1j * phi))
    # Lord & Kahl (2006): choose the root with Re(d) >= 0
    d = np.where(d.real >= 0, d, -d)

    g = (kappa - 1j * rho * xi * phi - d) / (kappa - 1j * rho * xi * phi + d)
    exp_dT = np.exp(-d * T)

    C = (
        kappa * theta / xi**2
        * ((kappa - 1j * rho * xi * phi - d) * T
           - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))
    )
    D = (
        (kappa - 1j * rho * xi * phi - d) / xi**2
        * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    )

    return np.exp(C + D * v0)


# Pre-compute Gauss-Legendre nodes/weights at common sizes
_gl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl_nodes(N: int) -> tuple[np.ndarray, np.ndarray]:
    if N not in _gl_cache:
        _gl_cache[N] = np.polynomial.legendre.leggauss(N)
    return _gl_cache[N]


def heston_call(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    N: int = 100,
    cutoff: float = 200.0,
) -> float:
    """
    Heston European call price via Gauss-Legendre quadrature.

    Parameters
    ----------
    S, K, T, r, q : standard option parameters (spot, strike, maturity, rates)
    v0, kappa, theta, xi, rho : Heston parameters
    N      : number of quadrature nodes (100 is accurate for most cases)
    cutoff : upper truncation of the CF integration range

    Returns
    -------
    Call price as a float.
    """
    if T <= 0.0:
        return float(max(S - K, 0.0))

    F = S * np.exp((r - q) * T)
    x = np.log(F / K)       # log-moneyness
    df = np.exp(-r * T)

    # Map Gauss-Legendre nodes from [-1, 1] to [0, cutoff]
    gl_nodes, gl_weights = _gl_nodes(N)
    phi = (gl_nodes + 1.0) * 0.5 * cutoff          # real, shape (N,)
    w   = gl_weights * 0.5 * cutoff
    phi_c = phi.astype(complex)

    # P2: risk-neutral probability that S_T > K
    cf2 = _heston_cf(phi_c, T, v0, kappa, theta, xi, rho)
    P2 = 0.5 + np.sum(w * np.real(np.exp(1j * phi * x) * cf2 / (1j * phi_c))) / np.pi

    # P1: stock-measure probability that S_T > K
    # Obtained by shifting phi → phi - i and normalising by CF(-i)
    cf_norm = _heston_cf(np.array([-1j], dtype=complex), T, v0, kappa, theta, xi, rho)[0]
    cf1 = _heston_cf(phi_c - 1j, T, v0, kappa, theta, xi, rho)
    P1 = 0.5 + np.sum(w * np.real(np.exp(1j * phi * x) * cf1 / (cf_norm * 1j * phi_c))) / np.pi

    return float(max(df * (F * P1 - K * P2), 0.0))
