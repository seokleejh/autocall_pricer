"""
Implied volatility surface: storage, interpolation, and synthetic construction.

The surface is defined on a (T, K/S0) grid of Black-Scholes implied vols.
Internally we store log-moneyness x = ln(K/F) for better interpolation behaviour.

For worst-of basket pricing (future): each underlying will carry its own
ImpliedVolSurface, and correlations are handled separately.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from typing import Sequence


class ImpliedVolSurface:
    """
    Bivariate spline interpolation over a (maturity, log-moneyness) grid.

    Parameters
    ----------
    maturities : array-like, shape (M,)
        Option maturities in years, strictly increasing.
    strikes : array-like, shape (K,)
        Absolute strike levels, strictly increasing.
    vols : array-like, shape (M, K)
        Black-Scholes implied volatilities (annualised, e.g. 0.20 for 20%).
    spot : float
        Current spot price (used to compute log-moneyness).
    rate : float
        Continuously compounded risk-free rate.
    div_yield : float
        Continuous dividend yield.
    """

    def __init__(
        self,
        maturities: Sequence[float],
        strikes: Sequence[float],
        vols: Sequence[Sequence[float]],
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
    ) -> None:
        self.spot = float(spot)
        self.rate = float(rate)
        self.div_yield = float(div_yield)

        self._T = np.array(maturities, dtype=float)
        self._K = np.array(strikes, dtype=float)
        self._vols = np.array(vols, dtype=float)  # shape (M, K)

        assert self._vols.shape == (len(self._T), len(self._K)), (
            "vols must be shape (len(maturities), len(strikes))"
        )

        # Interpolate in log-moneyness space for numerical stability
        self._log_m = np.log(self._K / self.spot)
        self._spline = RectBivariateSpline(
            self._T, self._log_m, self._vols, kx=3, ky=3
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def implied_vol(self, T: float | np.ndarray, K: float | np.ndarray) -> np.ndarray:
        """Return interpolated implied vol for maturity T and strike K."""
        scalar = np.isscalar(T) and np.isscalar(K)
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        K_arr = np.atleast_1d(np.asarray(K, dtype=float))
        log_m = np.log(K_arr / self.spot)
        result = np.maximum(self._spline(T_arr, log_m, grid=False), 1e-6)
        return float(result[0]) if scalar else result

    def forward(self, T: float | np.ndarray) -> float | np.ndarray:
        """Risk-neutral forward price F(0,T)."""
        T_arr = np.asarray(T, dtype=float)
        result = self.spot * np.exp((self.rate - self.div_yield) * T_arr)
        return float(result) if T_arr.ndim == 0 else result

    def total_variance(self, T: float | np.ndarray, K: float | np.ndarray) -> float | np.ndarray:
        """w(T,K) = sigma_imp^2 * T"""
        return self.implied_vol(T, K) ** 2 * np.asarray(T, dtype=float)

    @property
    def maturity_range(self) -> tuple[float, float]:
        return float(self._T[0]), float(self._T[-1])

    @property
    def strike_range(self) -> tuple[float, float]:
        return float(self._K[0]), float(self._K[-1])

    # ------------------------------------------------------------------
    # Dupire local variance: sigma_loc^2(T,K)
    # Using Dupire's formula in the (T, K) parameterisation.
    # ------------------------------------------------------------------

    def local_variance(self, T: float, K: float, dT: float = 1e-4, dK: float = None) -> float:
        """
        Dupire local variance at (T, K) via finite differences on the
        implied vol surface.

            sigma_loc^2 = (dw/dT) / (1 - y/w * dw/dy + 0.25*(-0.25 - 1/w + y^2/w^2)*(dw/dy)^2 + 0.5*d2w/dy2)

        where w = sigma_imp^2 * T and y = ln(K/F(T)).
        """
        if dK is None:
            dK = K * 1e-3

        F = self.forward(T)
        y = np.log(K / F)

        w = self.total_variance(T, K)
        w_T_up = self.total_variance(T + dT, K)
        w_T_dn = self.total_variance(T - dT, K)
        dw_dT = (w_T_up - w_T_dn) / (2 * dT)

        w_K_up = self.total_variance(T, K + dK)
        w_K_dn = self.total_variance(T, K - dK)
        dw_dK = (w_K_up - w_K_dn) / (2 * dK)
        d2w_dK2 = (w_K_up - 2 * w + w_K_dn) / dK**2

        # Convert dw/dK -> dw/dy using y = ln(K/F), dy/dK = 1/K
        dw_dy = dw_dK * K
        d2w_dy2 = d2w_dK2 * K**2 + dw_dK * K  # chain rule second order

        denom = 1.0 - (y / w) * dw_dy + 0.25 * (-0.25 - 1.0 / w + y**2 / w**2) * dw_dy**2 + 0.5 * d2w_dy2
        local_var = dw_dT / max(denom, 1e-8)
        return max(float(local_var), 1e-8)

    def local_vol(self, T: float, K: float, **kwargs) -> float:
        return float(np.sqrt(self.local_variance(T, K, **kwargs)))

    def local_vol_batch(self, T: float, K: np.ndarray, dT: float = 1e-4) -> np.ndarray:
        """
        Vectorised Dupire local vol for an array of spot values at a fixed time T.
        Avoids the Python loop overhead of calling local_vol() element-wise.
        """
        K = np.asarray(K, dtype=float)
        dK = np.maximum(K * 1e-3, 1e-6)
        F = self.forward(T)
        y = np.log(K / F)

        log_m     = np.log(K       / self.spot)
        log_m_up  = np.log((K + dK) / self.spot)
        log_m_dn  = np.log((K - dK) / self.spot)

        n = len(K)
        T_arr  = np.full(n, T,       dtype=float)
        T_up   = np.full(n, T + dT,  dtype=float)
        T_dn   = np.full(n, T - dT,  dtype=float)

        def _w(t_arr, lm):
            iv = np.maximum(self._spline(t_arr, lm, grid=False), 1e-6)
            return iv ** 2 * t_arr

        w       = _w(T_arr, log_m)
        w_T_up  = _w(T_up,  log_m)
        w_T_dn  = _w(T_dn,  log_m)
        w_K_up  = _w(T_arr, log_m_up)
        w_K_dn  = _w(T_arr, log_m_dn)

        dw_dT   = (w_T_up - w_T_dn) / (2.0 * dT)
        dw_dK   = (w_K_up - w_K_dn) / (2.0 * dK)
        d2w_dK2 = (w_K_up - 2.0 * w + w_K_dn) / dK ** 2

        dw_dy   = dw_dK * K
        d2w_dy2 = d2w_dK2 * K ** 2 + dw_dK * K

        denom = (1.0 - (y / w) * dw_dy
                 + 0.25 * (-0.25 - 1.0 / w + y ** 2 / w ** 2) * dw_dy ** 2
                 + 0.5 * d2w_dy2)
        local_var = dw_dT / np.maximum(denom, 1e-8)
        return np.sqrt(np.maximum(local_var, 1e-8))

    def with_spot(self, new_spot: float) -> "ImpliedVolSurface":
        """New surface with the same vol grid but a different spot (for spot-bump Greeks)."""
        return ImpliedVolSurface(
            self._T, self._K, self._vols.copy(),
            new_spot, self.rate, self.div_yield,
        )

    def with_vol_shift(self, dvol: float) -> "ImpliedVolSurface":
        """New surface with all implied vols shifted by dvol (for vega / vanna bumps)."""
        return ImpliedVolSurface(
            self._T, self._K, np.clip(self._vols + dvol, 1e-4, None),
            self.spot, self.rate, self.div_yield,
        )


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def flat_surface(
    spot: float,
    vol: float,
    rate: float = 0.0,
    div_yield: float = 0.0,
    T_max: float = 5.0,
    n_T: int = 20,
    n_K: int = 30,
    moneyness_range: tuple[float, float] = (0.5, 1.5),
) -> ImpliedVolSurface:
    """Flat (constant) implied vol surface — useful for unit testing."""
    maturities = np.linspace(0.01, T_max, n_T)
    strikes = np.linspace(spot * moneyness_range[0], spot * moneyness_range[1], n_K)
    vols = np.full((n_T, n_K), vol)
    return ImpliedVolSurface(maturities, strikes, vols, spot, rate, div_yield)


def skewed_surface(
    spot: float,
    atm_vol: float = 0.20,
    skew: float = -0.10,
    rate: float = 0.0,
    div_yield: float = 0.0,
    T_max: float = 5.0,
    n_T: int = 20,
    n_K: int = 30,
    moneyness_range: tuple[float, float] = (0.5, 1.5),
) -> ImpliedVolSurface:
    """
    Simple parametric skewed surface: sigma(T, K) = atm_vol + skew * ln(K/F(T)).
    Skew < 0 gives downward slope (equity-like).
    """
    maturities = np.linspace(0.01, T_max, n_T)
    strikes = np.linspace(spot * moneyness_range[0], spot * moneyness_range[1], n_K)

    vols = np.zeros((n_T, n_K))
    for i, T in enumerate(maturities):
        F = spot * np.exp((rate - div_yield) * T)
        for j, K in enumerate(strikes):
            vols[i, j] = max(atm_vol + skew * np.log(K / F), 0.01)

    return ImpliedVolSurface(maturities, strikes, vols, spot, rate, div_yield)


def term_structure_surface(
    spot: float,
    vol_short: float = 0.25,
    vol_long: float = 0.18,
    kappa: float = 1.5,
    skew: float = -0.10,
    convexity: float = 0.02,
    rate: float = 0.0,
    div_yield: float = 0.0,
    T_max: float = 5.0,
    n_T: int = 30,
    n_K: int = 60,
    moneyness_range: tuple[float, float] | None = None,
) -> ImpliedVolSurface:
    """
    Realistic equity vol surface with ATM term structure, skew term structure,
    and smile convexity.

        sigma_atm(T)  = vol_long + (vol_short - vol_long) * exp(-kappa * T)
        skew(T)       = skew / sqrt(T)   [reference skew is the 1-year level]
        sigma(T, K)   = sigma_atm(T) + skew(T) * ln(K/F(T)) + convexity * ln(K/F(T))^2

    Parameters
    ----------
    vol_short : float
        ATM implied vol as T → 0.
    vol_long : float
        ATM implied vol as T → ∞.
    kappa : float
        Speed of mean-reversion for the ATM term structure.
    skew : float
        Skew at T = 1y (in ln-moneyness units). Steepens as 1/√T for short T.
        Negative → equity-like downward skew.
    convexity : float
        Symmetric smile curvature (coefficient of ln(K/F)²). Zero = pure skew.
    moneyness_range : tuple | None
        (lo, hi) as fractions of spot. If None, auto-sized to cover ±4.5σ of
        a GBM path at T_max, ensuring >99.9% of simulated paths stay within
        the surface and the local vol spline never has to extrapolate.
    """
    if moneyness_range is None:
        # ±4.5σ at T_max: covers >99.9% of paths so the spline never extrapolates
        coverage = 4.5 * vol_short * np.sqrt(T_max)
        moneyness_range = (max(0.05, np.exp(-coverage)), min(20.0, np.exp(coverage)))

    maturities = np.linspace(0.01, T_max, n_T)
    strikes = np.linspace(spot * moneyness_range[0], spot * moneyness_range[1], n_K)

    vols = np.zeros((n_T, n_K))
    for i, T in enumerate(maturities):
        sigma_atm = vol_long + (vol_short - vol_long) * np.exp(-kappa * T)
        # Skew decays as 1/√T; floor at 3m to avoid explosion near T=0
        skew_T = skew / np.sqrt(max(T, 0.25))
        F = spot * np.exp((rate - div_yield) * T)
        for j, K in enumerate(strikes):
            x = np.log(K / F)   # log-moneyness
            vols[i, j] = max(sigma_atm + skew_T * x + convexity * x**2, 0.01)

    return ImpliedVolSurface(maturities, strikes, vols, spot, rate, div_yield)
