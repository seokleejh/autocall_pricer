"""
Local volatility model (Dupire).

Simulates risk-neutral paths under the local vol surface:
    dS(t) = (r - q) S dt + sigma_loc(t, S) S dW

sigma_loc(t, S) is read from the ImpliedVolSurface.local_vol() method,
which applies Dupire's formula via finite differences on the implied vol surface.

Euler-Maruyama discretisation. For better accuracy on the LV surface,
we use small time steps between observation dates (controlled by steps_per_year).
"""

from __future__ import annotations

import numpy as np
from calibration.vol_surface import ImpliedVolSurface


class LocalVolModel:
    """
    Parameters
    ----------
    surface : ImpliedVolSurface
        Calibrated implied vol surface from which local vol is derived.
    steps_per_year : int
        Number of Euler steps per year between observation dates.
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        surface: ImpliedVolSurface,
        steps_per_year: int = 52,
        seed: int | None = None,
    ) -> None:
        self.surface = surface
        self.steps_per_year = steps_per_year
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        n_paths: int,
        observation_times: np.ndarray,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Simulate spot paths and return performance S(t_i)/S(0) at observation_times.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths (must be even if antithetic=True).
        observation_times : array-like
            Sorted observation times in years.
        antithetic : bool
            Use antithetic variates (halves the number of random draws needed).

        Returns
        -------
        performances : np.ndarray, shape (n_paths, len(observation_times))
        """
        obs = np.asarray(observation_times, dtype=float)
        S0 = self.surface.spot
        r = self.surface.rate
        q = self.surface.div_yield

        # Antithetic paths are the exact mirror of their partner: the second
        # half of the ensemble reuses the SAME normal draws with the sign
        # flipped, so each pair is perfectly negatively correlated. Drawing
        # fresh normals for the mirror half and negating them would produce
        # statistically independent paths -- the normal law is symmetric --
        # and would buy no variance reduction at all.
        if antithetic:
            half = n_paths // 2
            n_sim = 2 * half
        else:
            half = 0
            n_sim = n_paths

        t_grid = self._build_time_grid(obs)
        K_lo, K_hi = self.surface.strike_range

        S = np.full(n_sim, S0, dtype=float)
        performances = np.zeros((n_sim, len(obs)))

        obs_idx = 0
        t_prev = 0.0

        for t_next in t_grid:
            dt = t_next - t_prev
            sqrt_dt = np.sqrt(dt)

            t_mid = 0.5 * (t_prev + t_next)
            sig = self.surface.local_vol_batch(t_mid, np.clip(S, K_lo, K_hi))

            if antithetic:
                Z_half = self.rng.standard_normal(half)
                Z = np.concatenate([Z_half, -Z_half])
            else:
                Z = self.rng.standard_normal(n_sim)

            S = S * np.exp((r - q - 0.5 * sig**2) * dt + sig * sqrt_dt * Z)
            S = np.maximum(S, 1e-6)  # absorbing floor

            if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                performances[:, obs_idx] = S / S0
                obs_idx += 1

            t_prev = t_next

        return performances[:n_paths]

    def _build_time_grid(self, obs: np.ndarray) -> np.ndarray:
        """Dense time grid: observation dates + intermediate steps."""
        points = set()
        t_prev = 0.0
        for t in obs:
            n_steps = max(1, int(round((t - t_prev) * self.steps_per_year)))
            sub = np.linspace(t_prev, t, n_steps + 1)[1:]
            points.update(sub.tolist())
            t_prev = t
        return np.array(sorted(points))
