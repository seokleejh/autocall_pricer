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

        if antithetic:
            half = n_paths // 2
            draw_paths = half
        else:
            draw_paths = n_paths

        # Build full time grid
        t_grid = self._build_time_grid(obs)

        S = np.full(draw_paths, S0, dtype=float)
        performances = np.zeros((draw_paths, len(obs)))

        obs_idx = 0
        t_prev = 0.0

        for t_next in t_grid:
            dt = t_next - t_prev
            sqrt_dt = np.sqrt(dt)

            # Local vol at current time and spot (mid-point approximation)
            t_mid = 0.5 * (t_prev + t_next)
            sig = np.vectorize(lambda s: self.surface.local_vol(t_mid, s))(S)

            Z = self.rng.standard_normal(draw_paths)
            S = S * np.exp((r - q - 0.5 * sig**2) * dt + sig * sqrt_dt * Z)
            S = np.maximum(S, 1e-6)  # absorbing floor

            # Record at observation dates
            if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                performances[:, obs_idx] = S / S0
                obs_idx += 1

            t_prev = t_next

        if antithetic:
            # Mirror paths: repeat simulation with negated normals implicitly by
            # noting S_anti = S0 * exp((r-q-0.5sig^2)*dt - sig*sqrt_dt*Z)
            # We re-simulate to keep code simple and memory-efficient.
            S_anti = np.full(draw_paths, S0, dtype=float)
            perf_anti = np.zeros((draw_paths, len(obs)))
            obs_idx = 0
            t_prev = 0.0
            self.rng  # reuse same rng but we need the same Z — regenerate deterministically
            # Regenerate by seeding a local rng with a fixed offset (simpler: just run again)
            rng2 = np.random.default_rng(self.rng.integers(1 << 31))

            t_grid2 = self._build_time_grid(obs)
            for t_next in t_grid2:
                dt = t_next - t_prev
                sqrt_dt = np.sqrt(dt)
                t_mid = 0.5 * (t_prev + t_next)
                sig = np.vectorize(lambda s: self.surface.local_vol(t_mid, s))(S_anti)
                Z = rng2.standard_normal(draw_paths)
                S_anti = S_anti * np.exp((r - q - 0.5 * sig**2) * dt + sig * sqrt_dt * (-Z))
                S_anti = np.maximum(S_anti, 1e-6)
                if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                    perf_anti[:, obs_idx] = S_anti / S0
                    obs_idx += 1
                t_prev = t_next

            performances = np.vstack([performances, perf_anti])

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
