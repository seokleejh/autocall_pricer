"""
Worst-of basket simulators (correlated multi-asset).

Both models below satisfy the same `.simulate(n_paths, observation_times,
antithetic) -> np.ndarray (n_paths, n_obs)` contract as the single-asset
models in local_vol.py / heston.py, EXCEPT the returned array already holds
worst-of (element-wise min across assets) performance -- the aggregation
happens inside simulate(), not in the caller. This preserves the
MCPricer / AutocallableNote contract with zero changes to
engine/mc_pricer.py or products/autocallable.py.

Correlation model
------------------
Assets are correlated via Cholesky decomposition of an n_assets x n_assets
correlation matrix, applied to independently-drawn standard normals at
every time step. All assets share one time grid (one scalar steps_per_year
for the whole basket), so a single Cholesky application per step suffices.

Antithetic variates simulate both halves of the ensemble in a single pass,
so paths [half:] are the literal negation of paths [:half] -- the same
normal draws with the sign flipped, and the same uniforms reflected to 1-U
for the QE chi-squared branch. This matters more here than in the
single-asset case: correlating draws via Cholesky requires the antithetic
half to be the literal negation of the original half's draws, or the
realized cross-asset correlation of the two halves would not mirror
correctly. Negation commutes with the Cholesky map, so mirroring the
independent drivers mirrors the correlated ones exactly.

Drawing fresh normals for the mirror half and negating those would NOT
achieve this: the normal law is symmetric, so a negated fresh draw is just
another independent draw, and the pairing -- along with all of its variance
reduction -- is lost.
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass

from calibration.vol_surface import ImpliedVolSurface
from models.heston import HestonParams


def _cholesky_corr(corr: np.ndarray) -> np.ndarray:
    """Validate an n x n correlation matrix and return its Cholesky factor."""
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    if corr.ndim != 2 or corr.shape != (n, n):
        raise ValueError("correlation matrix must be square")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise ValueError("correlation matrix diagonal must be 1.0")
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"correlation matrix is not positive semi-definite: {e}")


def _effective_market_corr(correlation: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    """
    Convert a TARGET spot-spot correlation matrix into the correlation matrix
    that must be applied to the "market factor" draws so that the REALIZED
    spot-spot correlation of the basket Heston log-returns matches the target.

    Each asset's spot shock is Z_S_a = rho_a*Z_V_a + sqrt(1-rho_a^2)*Z_market_a,
    with Z_V independent across assets and independent of Z_market. Since
    Var(Z_market_a) = 1 and Z_V_a is independent of everything else,
        Cov(Z_S_a, Z_S_b) = sqrt(1-rho_a^2) * sqrt(1-rho_b^2) * Cov(Z_market_a, Z_market_b).
    So feeding Cholesky the matrix
        C_market[a,b] = correlation[a,b] / (sqrt(1-rho_a^2) * sqrt(1-rho_b^2))
    (diagonal fixed at 1) makes the realized correlation equal the target.

    Feasibility ceiling: because Z_V_a is required to be independent of
    Z_S_b for a != b (no cross-asset vol-vol or vol-spot correlation, per
    this model's simplifying assumption), the FULL correlation matrix over
    all (Z_S_1..N, Z_V_1..N) can only stay positive semi-definite if
        |correlation[a,b]| <= sqrt(1-rho_a^2) * sqrt(1-rho_b^2).
    This is a hard mathematical limit of the simplification (confirmed via
    the eigenvalues of the block correlation matrix going negative exactly
    past this boundary), not an artifact of this particular construction --
    for realistic equity rho (-0.6 to -0.8), the ceiling is often only
    ~0.45-0.65, well below correlations commonly used for worst-of baskets
    on similar-sector equities. Rather than fail on this common case,
    targets exceeding the ceiling are clipped to it (with a warning) --
    use BasketLocalVolModel instead if the exact target correlation matters
    more than stochastic vol dynamics.
    """
    scale = np.sqrt(1.0 - np.asarray(rhos, dtype=float) ** 2)
    denom = np.outer(scale, scale)
    # Tiny safety margin: the boundary is only PSD up to floating-point
    # noise (a target exactly at the ceiling can otherwise fail Cholesky).
    ceiling = denom * (1.0 - 1e-8)
    np.fill_diagonal(ceiling, 1.0)  # the ceiling formula only applies to cross terms
    correlation = np.asarray(correlation, dtype=float)

    clipped = np.clip(correlation, -ceiling, ceiling)
    if not np.allclose(clipped, correlation, atol=1e-12):
        bad = np.argwhere(~np.isclose(clipped, correlation, atol=1e-12))
        pairs = ", ".join(f"({i},{j}): {correlation[i, j]:+.3f} -> {clipped[i, j]:+.3f}"
                           for i, j in bad if i < j)
        warnings.warn(
            "Basket Heston: requested correlation exceeds the feasible "
            f"ceiling implied by each asset's own rho and was clipped: {pairs}. "
            "Use BasketLocalVolModel for exact correlation control.",
            stacklevel=3,
        )

    C_market = clipped / denom
    np.fill_diagonal(C_market, 1.0)
    C_market = np.clip(C_market, -1.0, 1.0)
    return C_market


def _build_time_grid(obs: np.ndarray, steps_per_year: int) -> np.ndarray:
    """Dense time grid: observation dates + intermediate steps."""
    points = set()
    t_prev = 0.0
    for t in obs:
        n_steps = max(1, int(round((t - t_prev) * steps_per_year)))
        sub = np.linspace(t_prev, t, n_steps + 1)[1:]
        points.update(sub.tolist())
        t_prev = t
    return np.array(sorted(points))


class BasketLocalVolModel:
    """
    Worst-of basket under correlated Dupire local vol, one surface per asset.

    Parameters
    ----------
    surfaces : list[ImpliedVolSurface]
        One surface per asset. Each asset's own local_vol_batch() drives its
        own drift/vol -- only the driving Brownians are correlated.
    correlation : np.ndarray, shape (n_assets, n_assets)
        Asset-asset correlation matrix for the spot Brownians.
    steps_per_year : int
    seed : int | None
    """

    def __init__(
        self,
        surfaces: list[ImpliedVolSurface],
        correlation: np.ndarray,
        steps_per_year: int = 52,
        seed: int | None = None,
    ) -> None:
        self.surfaces = surfaces
        self.n_assets = len(surfaces)
        self.correlation = np.asarray(correlation, dtype=float)
        self.L = _cholesky_corr(self.correlation)
        self.steps_per_year = steps_per_year
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        n_paths: int,
        observation_times: np.ndarray,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Returns WORST-OF performance, shape (n_paths, n_obs)."""
        return self.simulate_assets(n_paths, observation_times, antithetic).min(axis=2)

    def simulate_assets(
        self,
        n_paths: int,
        observation_times: np.ndarray,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Returns per-asset performance, shape (n_paths, n_obs, n_assets)."""
        obs = np.asarray(observation_times, dtype=float)
        t_grid = _build_time_grid(obs, self.steps_per_year)

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
        Single-pass ensemble simulation.

        Antithetic paths [half:] mirror paths [:half] using the SAME normal
        draws with the sign flipped. Negation commutes with the Cholesky
        map, so mirroring the independent drivers mirrors the correlated
        ones and preserves the correlation structure exactly. Drawing fresh
        normals for the mirror half instead would give independent paths and
        no variance reduction.
        """
        n_assets = self.n_assets
        S0 = np.array([s.spot for s in self.surfaces])
        r = np.array([s.rate for s in self.surfaces])
        q = np.array([s.div_yield for s in self.surfaces])

        S = np.tile(S0, (n, 1))                        # (n, n_assets)
        performances = np.zeros((n, len(obs), n_assets))

        obs_idx = 0
        t_prev = 0.0
        for t_next in t_grid:
            dt = t_next - t_prev
            sqrt_dt = np.sqrt(dt)
            t_mid = 0.5 * (t_prev + t_next)

            if antithetic:
                z = self.rng.standard_normal((half, n_assets))
                Z_indep = np.vstack([z, -z])
            else:
                Z_indep = self.rng.standard_normal((n, n_assets))
            Z_corr = Z_indep @ self.L.T                 # (n, n_assets)

            sig = np.empty((n, n_assets))
            for a in range(n_assets):
                K_lo, K_hi = self.surfaces[a].strike_range
                S_clamped = np.clip(S[:, a], K_lo, K_hi)
                sig[:, a] = self.surfaces[a].local_vol_batch(t_mid, S_clamped)

            S = S * np.exp((r - q - 0.5 * sig**2) * dt + sig * sqrt_dt * Z_corr)
            S = np.maximum(S, 1e-6)

            if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                performances[:, obs_idx, :] = S / S0
                obs_idx += 1
            t_prev = t_next

        return performances


@dataclass
class BasketHestonAsset:
    """One asset's Heston market data + calibrated params, for the basket."""
    name: str
    params: HestonParams
    spot: float
    rate: float
    div_yield: float


class BasketHestonModel:
    """
    Worst-of basket under correlated Heston.

    Simplifying assumption: each asset keeps its OWN independent variance
    process (own kappa/theta/xi/rho/v0). Only the "market factor" driving
    each asset's non-vol-correlated spot shock is correlated across assets
    via Cholesky -- variance processes themselves are not cross-correlated.
    This mirrors the standard desk simplification: full cross-asset
    vol-vol correlation has no natural market-implied calibration target,
    and worst-of payoffs are far more sensitive to spot-spot correlation
    than vol-vol correlation.

    Note: because each asset's own rho diverts some of its spot shock's
    variance away from the cross-correlated market factor, the raw market
    factor would need to be over-correlated to compensate -- this class
    applies that correction automatically (see _effective_market_corr).
    That correction makes `correlation` exact for the INSTANTANEOUS spot
    Brownian shocks. It is NOT exact for the realized log-return
    correlation over a full path: each asset's variance process evolves
    independently (uncorrelated across assets, per the simplification
    above), so the two assets' variance paths diverge over time, further
    attenuating the realized correlation below the input `correlation` --
    more so for larger vol-of-vol (xi) and longer horizons, negligibly so
    when xi is small (variance is closer to deterministic). This is a
    genuine, expected property of the "independent variance processes"
    simplification, verified empirically in
    tests/test_basket_models.py's correlation-recovery test -- treat
    `correlation` as a best-effort target for Heston baskets, not an exact
    one; prefer BasketLocalVolModel (exact correlation recovery) when tight
    correlation control matters more than stochastic vol dynamics.

    Parameters
    ----------
    assets : list[BasketHestonAsset]
    correlation : np.ndarray, shape (n_assets, n_assets)
        Correlation of the spot Brownians dW_S_i across assets.
    steps_per_year : int
    seed : int | None
    """

    def __init__(
        self,
        assets: list[BasketHestonAsset],
        correlation: np.ndarray,
        steps_per_year: int = 52,
        seed: int | None = None,
    ) -> None:
        self.assets = assets
        self.n_assets = len(assets)
        self.correlation = np.asarray(correlation, dtype=float)
        rhos = np.array([a.params.rho for a in assets])
        self.L = _cholesky_corr(_effective_market_corr(self.correlation, rhos))
        self.steps_per_year = steps_per_year
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        n_paths: int,
        observation_times: np.ndarray,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Returns WORST-OF performance, shape (n_paths, n_obs)."""
        return self.simulate_assets(n_paths, observation_times, antithetic).min(axis=2)

    def simulate_assets(
        self,
        n_paths: int,
        observation_times: np.ndarray,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Returns per-asset performance, shape (n_paths, n_obs, n_assets)."""
        obs = np.asarray(observation_times, dtype=float)
        t_grid = _build_time_grid(obs, self.steps_per_year)

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
        Single-pass ensemble simulation.

        Antithetic paths [half:] mirror paths [:half]: the SAME normals with
        the sign flipped, and the SAME uniforms reflected as U -> 1-U for the
        QE chi-squared branch. Negation commutes with the Cholesky map, so
        the cross-asset correlation survives the reflection intact.
        """
        n_assets = self.n_assets
        log_S = np.zeros((n, n_assets))
        V = np.tile(np.array([a.params.v0 for a in self.assets]), (n, 1))
        performances = np.zeros((n, len(obs), n_assets))

        obs_idx = 0
        t_prev = 0.0
        for t_next in t_grid:
            dt = t_next - t_prev

            # Each asset draws its own variance driver Z_V; the "market
            # factor" Z_market is correlated across assets via Cholesky and
            # plays the role Z_perp plays in the single-asset HestonModel.
            if antithetic:
                zv = self.rng.standard_normal((half, n_assets))
                zp = self.rng.standard_normal((half, n_assets))
                u = self.rng.uniform(size=(half, n_assets))
                Z_V = np.vstack([zv, -zv])
                Z_indep_perp = np.vstack([zp, -zp])
                U = np.vstack([u, 1.0 - u])
            else:
                Z_V = self.rng.standard_normal((n, n_assets))
                Z_indep_perp = self.rng.standard_normal((n, n_assets))
                U = self.rng.uniform(size=(n, n_assets))
            Z_market = Z_indep_perp @ self.L.T

            V_new = np.empty_like(V)
            for a, asset in enumerate(self.assets):
                p = asset.params
                Z_S_a = p.rho * Z_V[:, a] + np.sqrt(1.0 - p.rho**2) * Z_market[:, a]
                V_new[:, a] = self._qe_step(V[:, a], dt, Z_V[:, a], U[:, a], p)
                log_S[:, a] += (
                    (asset.rate - asset.div_yield - 0.5 * V[:, a]) * dt
                    + np.sqrt(np.maximum(V[:, a], 0.0) * dt) * Z_S_a
                )
            V = V_new

            if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                performances[:, obs_idx, :] = np.exp(log_S)
                obs_idx += 1
            t_prev = t_next

        return performances

    def _qe_step(
        self, V: np.ndarray, dt: float, Z_V: np.ndarray, U: np.ndarray, params: HestonParams
    ) -> np.ndarray:
        """
        Andersen (2007) QE discretisation of CIR variance, per-asset params.

        `U` is a caller-supplied U(0,1) array, one per path, already
        reflected to 1-U on the antithetic half.
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

        psi_c = 1.5
        mask_exp = psi <= psi_c
        mask_chi = ~mask_exp

        if mask_exp.any():
            b2 = 2 / psi[mask_exp] - 1 + np.sqrt(2 / psi[mask_exp]) * np.sqrt(2 / psi[mask_exp] - 1)
            a = m[mask_exp] / (1 + b2)
            V_new[mask_exp] = a * (np.sqrt(b2) + Z_V[mask_exp]) ** 2

        if mask_chi.any():
            p_prob = (psi[mask_chi] - 1) / (psi[mask_chi] + 1)
            beta = (1 - p_prob) / m[mask_chi]
            U = U[mask_chi]
            V_new[mask_chi] = np.where(U <= p_prob, 0.0, np.log((1 - p_prob) / (1 - U)) / beta)

        return np.maximum(V_new, 0.0)
