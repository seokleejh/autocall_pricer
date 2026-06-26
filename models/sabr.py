"""
SABR stochastic volatility model.

Dynamics (in the forward measure):
    dF = alpha * F^beta * dW_F
    d(alpha) = nu * alpha * dW_alpha
    dW_F dW_alpha = rho dt

For pricing equity autocallables we simulate full SABR paths rather than
relying on the Hagan et al. (2002) implied vol formula, because:
  1. The formula is only accurate near-the-money.
  2. Autocallables require path-dependent payoff evaluation.

We simulate under the risk-neutral measure by working with the spot S = F * df(t, T),
using the forward dynamics and then discounting at the end.

Euler-Maruyama on the log of alpha avoids negative vol-of-vol.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SABRParams:
    """
    alpha : initial stochastic vol level  (roughly ATM vol when beta=1)
    beta  : CEV exponent in [0, 1]. 1 → lognormal backbone, 0 → normal.
    rho   : correlation between forward and vol Brownians
    nu    : vol-of-vol
    """
    alpha: float = 0.20
    beta: float = 1.0
    rho: float = -0.30
    nu: float = 0.40

    def hagan_implied_vol(self, F: float, K: float, T: float) -> float:
        """
        Hagan et al. (2002) ATM-expanded implied vol formula.
        Useful for quick calibration sanity checks; not used for MC.
        """
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu
        FK = F * K
        mid = FK**((1 - beta) / 2)

        if abs(F - K) < 1e-8 * F:
            # ATM formula
            logFK = 0.0
            z = nu / alpha * mid * 0.0  # z -> 0 at ATM, handled below
            sigma = (alpha / mid) * (
                1
                + ((1 - beta)**2 / 24 * alpha**2 / (mid**2)
                   + 0.25 * rho * beta * nu * alpha / mid
                   + (2 - 3 * rho**2) / 24 * nu**2) * T
            )
        else:
            logFK = np.log(F / K)
            z = nu / alpha * mid * logFK
            x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            sigma = (
                alpha / (
                    mid * (
                        1
                        + (1 - beta)**2 / 24 * logFK**2
                        + (1 - beta)**4 / 1920 * logFK**4
                    )
                ) * (z / x)
                * (
                    1
                    + ((1 - beta)**2 / 24 * alpha**2 / (mid**2)
                       + 0.25 * rho * beta * nu * alpha / mid
                       + (2 - 3 * rho**2) / 24 * nu**2) * T
                )
            )
        return max(float(sigma), 1e-6)


class SABRModel:
    """
    Parameters
    ----------
    params : SABRParams
    spot : float
    rate : float
    div_yield : float
    steps_per_year : int
    seed : int | None
    """

    def __init__(
        self,
        params: SABRParams,
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
        steps_per_year: int = 52,
        seed: int | None = None,
    ) -> None:
        self.params = params
        self.spot = float(spot)
        self.rate = float(rate)
        self.div_yield = float(div_yield)
        self.steps_per_year = steps_per_year
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        n_paths: int,
        observation_times: np.ndarray,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Returns performances S(t_i)/S(0) at observation_times, shape (n_paths, n_obs)."""
        obs = np.asarray(observation_times, dtype=float)
        t_grid = self._build_time_grid(obs)

        half = n_paths // 2 if antithetic else n_paths
        perf1 = self._simulate_batch(half, obs, t_grid, sign=1)
        if antithetic:
            perf2 = self._simulate_batch(half, obs, t_grid, sign=-1)
            return np.vstack([perf1, perf2])[:n_paths]
        return perf1

    def _simulate_batch(
        self, n: int, obs: np.ndarray, t_grid: np.ndarray, sign: int
    ) -> np.ndarray:
        p = self.params
        r, q = self.rate, self.div_yield

        # Work in log-space for alpha to ensure positivity
        S = np.full(n, self.spot, dtype=float)
        log_alpha = np.full(n, np.log(p.alpha))

        performances = np.zeros((n, len(obs)))
        obs_idx = 0
        t_prev = 0.0

        for t_next in t_grid:
            dt = t_next - t_prev
            sqrt_dt = np.sqrt(dt)

            alpha = np.exp(log_alpha)

            # Correlated Brownians
            Z1 = sign * self.rng.standard_normal(n)
            Z2_ind = self.rng.standard_normal(n)
            Z2 = p.rho * Z1 + np.sqrt(1 - p.rho**2) * Z2_ind

            # SABR forward dynamics
            dF = alpha * np.abs(S)**p.beta * sqrt_dt * Z1
            S_new = S + (r - q) * S * dt + dF
            S_new = np.maximum(S_new, 1e-6)

            # Log-alpha Euler (Ito correction: d ln alpha = -0.5 nu^2 dt + nu dW)
            log_alpha = log_alpha - 0.5 * p.nu**2 * dt + p.nu * sqrt_dt * Z2

            S = S_new

            if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                performances[:, obs_idx] = S / self.spot
                obs_idx += 1

            t_prev = t_next

        return performances

    def _build_time_grid(self, obs: np.ndarray) -> np.ndarray:
        points = set()
        t_prev = 0.0
        for t in obs:
            n_steps = max(1, int(round((t - t_prev) * self.steps_per_year)))
            sub = np.linspace(t_prev, t, n_steps + 1)[1:]
            points.update(sub.tolist())
            t_prev = t
        return np.array(sorted(points))
