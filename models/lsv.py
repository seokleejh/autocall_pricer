"""
Heston stochastic local volatility (LSV) model.

Dynamics:
    dS/S = (r - q) dt + L(t, S) * sqrt(V) dW_S
    dV   = kappa (theta - V) dt + xi sqrt(V) dW_V
    dW_S dW_V = rho dt

This is exactly models/heston.py with one extra factor in the spot equation:
the deterministic *leverage function* L(t, S). The variance process is
untouched, and the QE discretisation is imported from heston.py rather than
re-implemented.

Why this model exists
---------------------
LocalVolModel reprices every vanilla on the surface by construction but has
degenerate forward dynamics (the forward smile flattens, vol-of-vol is zero).
HestonModel has realistic dynamics but only five parameters, so it cannot fit
a whole equity surface -- whatever smile residual is left over is an
uncontrolled mispricing of the barrier levels an autocallable is written on.

LSV gets both. The leverage function is chosen so that the model's *effective*
local variance matches Dupire's, which by Gyongy's theorem makes the model
reprice the entire vanilla surface while leaving (kappa, theta, xi, rho, v0)
free to carry the forward dynamics. The calibration condition is

    L(t, S)^2 = sigma_LV(t, S)^2 / E[ V_t | S_t = S ]

The numerator is Dupire local variance, already available from
ImpliedVolSurface.local_vol_batch(). The denominator has no closed form and is
estimated from a particle ensemble -- see calibration/lsv_calibration.py.

Two degenerate limits are worth remembering, and both are covered by tests:
  * xi -> 0    : variance is deterministic, so E[V|S] = V(t) and L absorbs the
                 whole surface. LSV collapses to LocalVolModel.
  * L  == 1    : the model IS Heston. This is what a calibration on a surface
                 that Heston already fits perfectly should produce.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from models.heston import HestonParams, qe_step


DEFAULT_LEVERAGE_CAP: tuple[float, float] = (0.1, 10.0)


def _interp_slice(
    nodes: np.ndarray,
    values: np.ndarray,
    S: np.ndarray,
    cap: tuple[float, float],
    spline: CubicSpline | None = None,
) -> np.ndarray:
    """
    Interpolate one time slice of the leverage function onto spot values `S`.

    Cubic in spot where there are enough nodes to support it, linear below
    that. Spot is clamped to the node range before evaluation rather than
    extrapolated: a cubic spline run past its outermost knots will happily
    return negative or enormous values, and this quantity multiplies the
    diffusion of every path.
    """
    if len(nodes) == 1:
        out = np.full_like(np.asarray(S, dtype=float), values[0])
    else:
        S_clamped = np.clip(S, nodes[0], nodes[-1])
        if spline is not None:
            out = spline(S_clamped)
        else:
            out = np.interp(S_clamped, nodes, values)
    return np.clip(out, cap[0], cap[1])


class LeverageFunction:
    """
    Calibrated leverage surface L(t, S), stored on a grid.

    Piecewise-constant in time and cubic-spline in spot. The time treatment is
    deliberately not smoothed: the particle calibration produces one
    independent estimate per time slice, and interpolating across slices would
    imply a smoothness the estimates do not have.

    Parameters
    ----------
    times : array (M,)
        Slice start times. Slice k applies on [times[k], times[k+1]).
    spot_grids : list of M arrays
        Spot nodes for each slice. Node counts may differ between slices, and
        the grid moves outward over time as the ensemble disperses.
    values : list of M arrays
        L at each node of the corresponding slice.
    cap : (lo, hi)
        Hard bounds applied on every evaluation. See calibrate_leverage().
    """

    def __init__(
        self,
        times,
        spot_grids,
        values,
        cap: tuple[float, float] = DEFAULT_LEVERAGE_CAP,
    ) -> None:
        self.times = np.asarray(times, dtype=float)
        self.spot_grids = [np.asarray(g, dtype=float) for g in spot_grids]
        self.values = [np.asarray(v, dtype=float) for v in values]
        self.cap = (float(cap[0]), float(cap[1]))

        assert len(self.times) == len(self.spot_grids) == len(self.values), (
            "times, spot_grids and values must have the same length"
        )

        # Pre-build one interpolator per slice; None means fall back to linear.
        self._splines: list[CubicSpline | None] = []
        for g, v in zip(self.spot_grids, self.values):
            if len(g) >= 4:
                self._splines.append(CubicSpline(g, v, extrapolate=False))
            else:
                self._splines.append(None)

    def slice_index(self, t: float) -> int:
        """Index of the slice in force at time t (piecewise-constant in time)."""
        k = int(np.searchsorted(self.times, t, side="right")) - 1
        return int(np.clip(k, 0, len(self.times) - 1))

    def __call__(self, t: float, S: np.ndarray) -> np.ndarray:
        """L(t, S) for an array of spot values."""
        k = self.slice_index(t)
        return _interp_slice(
            self.spot_grids[k], self.values[k], S, self.cap, self._splines[k]
        )

    # -- diagnostics -------------------------------------------------------

    @property
    def n_slices(self) -> int:
        return len(self.times)

    def value_range(self) -> tuple[float, float]:
        """(min, max) of L across the whole calibrated surface."""
        lo = min(float(v.min()) for v in self.values)
        hi = max(float(v.max()) for v in self.values)
        return lo, hi

    def __repr__(self) -> str:
        lo, hi = self.value_range()
        return (
            f"LeverageFunction(slices={self.n_slices}, "
            f"T_max={self.times[-1]:.3f}, L in [{lo:.4f}, {hi:.4f}], "
            f"cap={self.cap})"
        )


class LSVModel:
    """
    Parameters
    ----------
    params : HestonParams
        The stochastic-vol leg. Calibrate with calibration.calibrators.calibrate_heston
        exactly as for HestonModel -- LSV does not change how these are fitted.
    leverage : LeverageFunction
        Calibrated by calibration.lsv_calibration.calibrate_leverage against the
        SAME params and surface. Pairing a leverage function with different
        Heston parameters than it was calibrated for silently breaks the
        surface fit.
    spot, rate, div_yield : float
    steps_per_year : int
        Should match the value used during leverage calibration; L is
        piecewise-constant in time, so a coarser simulation grid than the
        calibration grid simply reads stale slices.
    seed : int | None
    """

    def __init__(
        self,
        params: HestonParams,
        leverage: LeverageFunction,
        spot: float,
        rate: float = 0.0,
        div_yield: float = 0.0,
        steps_per_year: int = 52,
        seed: int | None = None,
    ) -> None:
        self.params = params
        self.leverage = leverage
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
        Single-pass ensemble simulation, mirroring HestonModel exactly except
        for the leverage factor on the spot diffusion.

        Antithetic paths [half:] are the exact mirror of paths [:half]: the
        same normals negated, and the same uniforms reflected to 1-U for the
        QE chi-squared branch.
        """
        p = self.params
        r, q = self.rate, self.div_yield
        S0 = self.spot

        S = np.full(n, S0, dtype=float)
        V = np.full(n, p.v0, dtype=float)

        performances = np.zeros((n, len(obs)))
        obs_idx = 0
        t_prev = 0.0

        for t_next in t_grid:
            dt = t_next - t_prev

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

            # Leverage is read at the START of the step, from the state the
            # paths are actually in -- the same convention the calibration
            # used when it measured E[V|S] at t_prev.
            L = self.leverage(t_prev, S)

            V_new = qe_step(V, dt, Z_V, U, p)

            # Effective instantaneous variance is L^2 * V, so that is what
            # enters the Ito correction -- not V.
            V_pos = np.maximum(V, 0.0)
            eff_var = (L ** 2) * V_pos
            S = S * np.exp(
                (r - q - 0.5 * eff_var) * dt + L * np.sqrt(V_pos * dt) * Z_S
            )
            S = np.maximum(S, 1e-8)

            V = V_new

            if obs_idx < len(obs) and np.isclose(t_next, obs[obs_idx]):
                performances[:, obs_idx] = S / S0
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
