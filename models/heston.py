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

        if antithetic:
            half = n_paths // 2
            n_sim = 2 * half
        else:
            half = 0
            n_sim = n_paths

        perf = self._simulate_ensemble(n_sim, half, obs, t_grid, antithetic)
        return perf[:n_paths]

    def _simulate_ensemble(
        self, n: int, half: int, obs: np.ndarray, t_grid: np.ndarray,
        antithetic: bool,
    ) -> np.ndarray:
        """
        Simulate the whole ensemble in a single pass.

        When `antithetic` is set, paths [half:] are the exact mirror of paths
        [:half]: the SAME normal draws with the sign flipped, and the SAME
        uniform draws reflected as U -> 1-U so the QE chi-squared branch is
        antitheticised too.

        Both halves must be advanced together for this to work. Drawing fresh
        randoms for the mirror half and negating them yields statistically
        independent paths -- the normal law is symmetric about 0 and the
        uniform about 1/2 -- and therefore no variance reduction at all.
        """
        p = self.params
        r, q = self.rate, self.div_yield

        log_S = np.zeros(n)           # log(S/S0)
        V = np.full(n, p.v0)

        performances = np.zeros((n, len(obs)))
        obs_idx = 0
        t_prev = 0.0

        for t_next in t_grid:
            dt = t_next - t_prev

            # Mirror both Brownians so Z_S = rho*Z_V + sqrt(1-rho^2)*Z_perp
            # is itself exactly negated on the antithetic half, and reflect
            # the uniforms for the QE chi-squared branch.
            if antithetic:
                z_v    = self.rng.standard_normal(half)
                z_perp = self.rng.standard_normal(half)
                u      = self.rng.uniform(size=half)
                Z_V    = np.concatenate([z_v, -z_v])
                Z_perp = np.concatenate([z_perp, -z_perp])
                U      = np.concatenate([u, 1.0 - u])
            else:
                Z_V    = self.rng.standard_normal(n)
                Z_perp = self.rng.standard_normal(n)
                U      = self.rng.uniform(size=n)

            Z_S = p.rho * Z_V + np.sqrt(1.0 - p.rho ** 2) * Z_perp

            # --- QE scheme for variance ---
            V_new = self._qe_step(V, dt, Z_V, U)

            # --- Euler step for log S ---
            log_S += (r - q - 0.5 * V) * dt + np.sqrt(V * dt) * Z_S

            V = V_new

            if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                performances[:, obs_idx] = np.exp(log_S)
                obs_idx += 1

            t_prev = t_next

        return performances

    def _qe_step(self, V: np.ndarray, dt: float, Z_V: np.ndarray,
                 U: np.ndarray) -> np.ndarray:
        """Andersen QE step using this model's own parameters."""
        return qe_step(V, dt, Z_V, U, self.params)

    def _build_time_grid(self, obs: np.ndarray) -> np.ndarray:
        points = set()
        t_prev = 0.0
        for t in obs:
            n_steps = max(1, int(round((t - t_prev) * self.steps_per_year)))
            sub = np.linspace(t_prev, t, n_steps + 1)[1:]
            points.update(sub.tolist())
            t_prev = t
        return np.array(sorted(points))


def qe_step(V: np.ndarray, dt: float, Z_V: np.ndarray,
            U: np.ndarray, params: HestonParams) -> np.ndarray:
    """
    Andersen (2007) QE discretisation of the CIR variance process.

    Module-level so the LSV model can reuse it verbatim: LSV shares Heston's
    variance dynamics exactly and differs only in the spot equation, which
    gains the leverage factor L(t,S).

    Z_V : pre-drawn N(0,1) array, shared with the spot equation so the
          exponential branch carries the rho correlation.
    U   : pre-drawn U(0,1) array, one per path, consumed by the shifted
          chi-squared branch. Supplied by the caller rather than drawn here
          so the antithetic half can be handed the reflected uniforms 1-U;
          drawing internally would also break the pairing, since the two
          halves select different subsets of paths into this branch.
    """
    kappa, theta, xi = params.kappa, params.theta, params.xi

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

    # Chi-squared branch — uses the caller's uniforms, already reflected
    # to 1-U on the antithetic half.
    if mask_chi.any():
        p_prob = (psi[mask_chi] - 1) / (psi[mask_chi] + 1)
        beta = (1 - p_prob) / m[mask_chi]
        U_chi = U[mask_chi]
        V_new[mask_chi] = np.where(U_chi <= p_prob, 0.0,
                                   np.log((1 - p_prob) / (1 - U_chi)) / beta)

    return np.maximum(V_new, 0.0)
