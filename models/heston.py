"""
Heston stochastic volatility model.

Dynamics:
    dS = (r - q) S dt + sqrt(V) S dW_S
    dV = kappa (theta - V) dt + xi sqrt(V) dW_V
    dW_S dW_V = rho dt

Simulation uses the Broadie-Kaya exact scheme for V and Euler for log(S),
with the QE (Quadratic-Exponential) discretisation scheme for V
(Andersen 2007), which avoids negative variance and is far more accurate
than plain Euler at coarse step sizes.

Default parameters are roughly calibrated to a typical equity surface.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class HestonParams:
    """
    kappa : mean-reversion speed
    theta : long-run variance
    xi    : vol-of-vol
    rho   : correlation between spot and variance Brownians
    v0    : initial variance
    """
    kappa: float = 2.0
    theta: float = 0.04    # long-run vol ~ 20%
    xi: float = 0.5
    rho: float = -0.7
    v0: float = 0.04       # initial vol ~ 20%

    def validate(self) -> None:
        if not 2 * self.kappa * self.theta > self.xi**2:
            raise ValueError(
                "Feller condition 2*kappa*theta > xi^2 violated; variance may reach zero."
            )


class HestonModel:
    """
    Parameters
    ----------
    params : HestonParams
    spot : float
    rate : float
    div_yield : float
    steps_per_year : int
        Number of QE steps per year between observation dates.
    seed : int | None
    """

    def __init__(
        self,
        params: HestonParams,
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
        """
        Returns performances S(t_i)/S(0) at observation_times, shape (n_paths, n_obs).
        """
        obs = np.asarray(observation_times, dtype=float)
        t_grid = self._build_time_grid(obs)

        half = n_paths // 2 if antithetic else n_paths
        perf1 = self._simulate_batch(half, obs, t_grid, anti_sign=1)
        if antithetic:
            perf2 = self._simulate_batch(half, obs, t_grid, anti_sign=-1)
            return np.vstack([perf1, perf2])[:n_paths]
        return perf1

    def _simulate_batch(
        self, n: int, obs: np.ndarray, t_grid: np.ndarray, anti_sign: int
    ) -> np.ndarray:
        p = self.params
        r, q = self.rate, self.div_yield
        S0 = self.spot

        log_S = np.zeros(n)           # log(S/S0)
        V = np.full(n, p.v0)

        performances = np.zeros((n, len(obs)))
        obs_idx = 0
        t_prev = 0.0

        for t_next in t_grid:
            dt = t_next - t_prev

            # Draw both Brownians with anti_sign so the antithetic path fully
            # negates Z_S = rho*Z_V + sqrt(1-rho^2)*Z_perp.
            Z_V    = anti_sign * self.rng.standard_normal(n)
            Z_perp = anti_sign * self.rng.standard_normal(n)
            Z_S    = p.rho * Z_V + np.sqrt(1.0 - p.rho ** 2) * Z_perp

            # --- QE scheme for variance (uses Z_V and anti_sign) ---
            V_new = self._qe_step(V, dt, Z_V, anti_sign)

            # --- Euler step for log S ---
            log_S += (r - q - 0.5 * V) * dt + np.sqrt(V * dt) * Z_S

            V = V_new

            if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                performances[:, obs_idx] = np.exp(log_S)
                obs_idx += 1

            t_prev = t_next

        return performances

    def _qe_step(self, V: np.ndarray, dt: float, Z_V: np.ndarray,
                 anti_sign: int = 1) -> np.ndarray:
        """
        Andersen (2007) QE discretisation of CIR variance process.

        Z_V      : pre-drawn N(0,1) array (already multiplied by anti_sign),
                   shared with the spot equation to enforce rho correlation
                   in the exponential branch.
        anti_sign: +1 for original paths, -1 for antithetic. Used to flip
                   the uniform draw U → (1-U) in the chi-squared branch so
                   that both branches are fully antitheticised.
        """
        p = self.params
        kappa, theta, xi = p.kappa, p.theta, p.xi
        n = len(V)

        e_kdt = np.exp(-kappa * dt)
        m = theta + (V - theta) * e_kdt
        s2 = (
            V * xi**2 * e_kdt / kappa * (1 - e_kdt)
            + theta * xi**2 / (2 * kappa) * (1 - e_kdt)**2
        )
        psi = s2 / (m**2 + 1e-12)

        V_new = np.empty_like(V)

        # QE: use exponential for psi <= psi_c, and shifted chi-squared for psi > psi_c
        psi_c = 1.5
        mask_exp = psi <= psi_c
        mask_chi = ~mask_exp

        # Exponential branch — uses the shared Z_V to preserve rho correlation
        if mask_exp.any():
            b2 = 2 / psi[mask_exp] - 1 + np.sqrt(2 / psi[mask_exp]) * np.sqrt(2 / psi[mask_exp] - 1)
            a = m[mask_exp] / (1 + b2)
            V_new[mask_exp] = a * (np.sqrt(b2) + Z_V[mask_exp]) ** 2

        # Chi-squared branch — antithetic: flip U → (1-U) for anti_sign=-1
        if mask_chi.any():
            p_prob = (psi[mask_chi] - 1) / (psi[mask_chi] + 1)
            beta = (1 - p_prob) / m[mask_chi]
            U_raw = self.rng.uniform(size=mask_chi.sum())
            U = U_raw if anti_sign == 1 else 1.0 - U_raw
            V_new[mask_chi] = np.where(U <= p_prob, 0.0, np.log((1 - p_prob) / (1 - U)) / beta)

        return np.maximum(V_new, 0.0)

    def _build_time_grid(self, obs: np.ndarray) -> np.ndarray:
        points = set()
        t_prev = 0.0
        for t in obs:
            n_steps = max(1, int(round((t - t_prev) * self.steps_per_year)))
            sub = np.linspace(t_prev, t, n_steps + 1)[1:]
            points.update(sub.tolist())
            t_prev = t
        return np.array(sorted(points))
