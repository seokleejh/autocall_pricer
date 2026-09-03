"""
Leverage function calibration for the Heston-LSV model.

Implements the McKean particle method of Guyon & Henry-Labordere (2011),
"The Smile Calibration Problem Solved". The condition to satisfy is

    L(t, S)^2 = sigma_LV(t, S)^2 / E[ V_t | S_t = S ]

The numerator is Dupire local variance, read straight off the implied vol
surface. The denominator is the obstacle: it depends on the joint law of
(S, V), which itself depends on L, so the calibration is a fixed point.

The particle method resolves it by walking forward one time slice at a time.
At each slice the ensemble of particles IS the joint distribution, so the
conditional expectation is just an average over the particles currently near
a given spot level. L on that slice is then known, and the ensemble can be
advanced one step -- at which point the next slice's conditional expectation
becomes measurable, and so on. Nothing is iterated; one forward sweep both
calibrates and (incidentally) simulates.

Why not the forward Fokker-Planck PDE
-------------------------------------
The alternative is to solve the 2D forward equation for the joint density of
(S, V) and integrate it for the conditional expectation. That is more accurate
and has no Monte Carlo noise, and it is what QuantLib's default SLV engine
does. It is also an ADI solver on a non-uniform grid with its own boundary
conditions -- several hundred lines of numerics sharing nothing with the rest
of this repository, which is Monte Carlo end to end. The particle method
reuses the machinery that already exists here.

Conditional expectation estimator
---------------------------------
Particles are sorted by spot and split into EQUAL-COUNT bins rather than bins
on a fixed spot lattice. This is a deliberate choice: equal-count bins cannot
be empty, so the pathological case that wrecks a fixed lattice -- a tail bin
holding two particles, producing a near-zero denominator and an exploding
leverage -- cannot arise. The cost is that the node locations move from slice
to slice, which is harmless because each slice carries its own grid.

This is a rectangular-kernel special case of the Guyon & Henry-Labordere
estimator. If the round-trip accuracy in tests/test_lsv.py ever proves
insufficient, replacing it with a Gaussian kernel regression touches only
_conditional_variance() below.
"""

from __future__ import annotations

import numpy as np

from calibration.vol_surface import ImpliedVolSurface
from models.heston import HestonParams, qe_step
from models.lsv import LeverageFunction, DEFAULT_LEVERAGE_CAP, _interp_slice


DEFAULT_LSV_CONFIG: dict = {
    "n_particles": 50_000,
    "n_spot_bins": 28,
    "leverage_cap": [0.1, 10.0],
    "steps_per_year": 52,
    "sticky_leverage": True,
}


def lsv_settings(cfg: dict) -> dict:
    """Merge the `lsv:` block of a config over the defaults."""
    settings = dict(DEFAULT_LSV_CONFIG)
    settings.update(cfg.get("lsv", {}) or {})
    return settings


def leverage_from_config(
    cfg: dict,
    surface: ImpliedVolSurface,
    params: HestonParams,
    T_max: float,
    seed: int | None = None,
    fast: bool = False,
    verbose: bool = False,
) -> LeverageFunction:
    """
    Build a leverage function using the `lsv:` settings in `cfg`.

    Single place where config turns into calibration arguments, so main.py,
    run_scenarios.py and diagnostics/model_quality.py cannot drift apart.

    fast : reduce the particle count for Greek bump calibrations, mirroring
           the `fast` flag on calibrate_heston(). Accuracy plateaus well below
           the default particle count, so this costs little.
    """
    s = lsv_settings(cfg)
    n_particles = int(s["n_particles"])
    if fast:
        n_particles = max(4_000, n_particles // 3)

    cap = tuple(s["leverage_cap"])
    return calibrate_leverage(
        surface,
        params,
        T_max=T_max,
        steps_per_year=int(s["steps_per_year"]),
        n_particles=n_particles,
        n_spot_bins=int(s["n_spot_bins"]),
        leverage_cap=cap,
        seed=seed,
        verbose=verbose,
    )


def _conditional_variance(
    S: np.ndarray,
    V: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate E[V | S] by binning the particle ensemble into equal-count bins.

    Returns
    -------
    nodes : array (G,)
        Mean spot within each bin -- the spot grid for this slice.
    e_v : array (G,)
        Mean variance within each bin -- the conditional expectation estimate.

    G may be smaller than n_bins if bins collapse onto identical spot values
    (which happens at t=0, where every particle sits at S0).
    """
    n = len(S)

    # Degenerate slice: the whole ensemble is at one spot level (t = 0).
    # There is no spot dependence to resolve, so a single node is correct.
    if n == 0 or np.ptp(S) < 1e-12:
        return np.array([float(S[0])]), np.array([float(V.mean())])

    n_bins = max(1, min(n_bins, n))

    order = np.argsort(S, kind="stable")
    S_sorted = S[order]
    V_sorted = V[order]

    # Equal-count bins: edges are index positions, not spot levels.
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    starts = edges[:-1]
    counts = np.diff(edges)

    keep = counts > 0
    starts, counts = starts[keep], counts[keep]

    nodes = np.add.reduceat(S_sorted, starts) / counts
    e_v = np.add.reduceat(V_sorted, starts) / counts

    # The spline needs strictly increasing nodes. Bin means are already
    # non-decreasing (bins are contiguous in sorted order); drop any ties.
    if len(nodes) > 1:
        strictly_up = np.concatenate([[True], np.diff(nodes) > 1e-12])
        nodes, e_v = nodes[strictly_up], e_v[strictly_up]

    return nodes, e_v


def calibrate_leverage(
    surface: ImpliedVolSurface,
    params: HestonParams,
    T_max: float,
    steps_per_year: int = 52,
    n_particles: int = 50_000,
    n_spot_bins: int = 28,
    leverage_cap: tuple[float, float] = DEFAULT_LEVERAGE_CAP,
    seed: int | None = None,
    antithetic: bool = True,
    verbose: bool = False,
) -> LeverageFunction:
    """
    Calibrate the leverage function L(t, S) by the particle method.

    Parameters
    ----------
    surface : ImpliedVolSurface
        Supplies both the Dupire numerator and (r, q, S0).
    params : HestonParams
        The stochastic-vol leg, already calibrated to `surface` via
        calibrate_heston(). The returned LeverageFunction is only valid when
        paired back with THESE parameters.
    T_max : float
        Calibrate out to this maturity. Must cover the longest observation
        date that will later be priced; beyond it the last slice is reused.
    steps_per_year : int
        Calibration time resolution. Should match the LSVModel that will
        consume the result.
    n_particles : int
        Ensemble size. Conditional-expectation noise scales roughly as
        sqrt(n_spot_bins / n_particles).
    n_spot_bins : int
        Spot nodes per slice. 25-30 is the usual accuracy/smoothness tradeoff:
        fewer under-resolves the wings, more amplifies binning noise into the
        spline.
    leverage_cap : (lo, hi)
        Hard bounds on L. This is a safety rail, not a tuning knob: in the
        sparse wings E[V|S] is estimated from few particles and can collapse
        toward zero, which would send L to infinity and destroy the run. If
        the cap binds anywhere other than the extreme wings, the calibration
        is not healthy -- check `clipped_fraction` in verbose output.
    antithetic : bool
        Mirror half the ensemble. Reduces noise in the conditional
        expectations at no extra cost.

    Returns
    -------
    LeverageFunction
    """
    S0 = surface.spot
    r, q = surface.rate, surface.div_yield
    T_min_surf = surface.maturity_range[0]
    K_lo, K_hi = surface.strike_range
    cap_lo, cap_hi = leverage_cap

    rng = np.random.default_rng(seed)

    n_steps = max(1, int(round(T_max * steps_per_year)))
    t_nodes = np.linspace(0.0, T_max, n_steps + 1)

    if antithetic:
        half = n_particles // 2
        n = 2 * half
    else:
        half = 0
        n = n_particles

    S = np.full(n, S0, dtype=float)
    V = np.full(n, params.v0, dtype=float)

    times = np.empty(n_steps)
    grids: list[np.ndarray] = []
    values: list[np.ndarray] = []

    n_clipped = 0
    n_nodes_total = 0

    for k in range(n_steps):
        t_k = t_nodes[k]
        dt = t_nodes[k + 1] - t_k

        # Evaluate Dupire at the step MIDPOINT, matching LocalVolModel's own
        # convention. The leverage calibrated on this slice is applied across
        # the whole step [t_k, t_k+dt], so the midpoint is the second-order
        # accurate choice; using the left endpoint biases every step in the
        # direction the ATM term structure happens to slope, which for a
        # typical downward-sloping equity surface means pricing too high.
        # E[V|S] is still measured from the ensemble at t_k -- it is only
        # observable there -- but kappa*dt is small enough over one step that
        # the mismatch is of higher order than the bias it removes.
        # Dupire is also singular as T -> 0, so stay inside the surface domain.
        t_eval = max(0.5 * (t_k + t_nodes[k + 1]), T_min_surf)

        # --- the fixed point, resolved one slice at a time ----------------
        nodes, e_v = _conditional_variance(S, V, n_spot_bins)
        sigma_lv = surface.local_vol_batch(t_eval, np.clip(nodes, K_lo, K_hi))
        L_raw = sigma_lv / np.sqrt(np.maximum(e_v, 1e-12))
        L_nodes = np.clip(L_raw, cap_lo, cap_hi)

        n_clipped += int(np.count_nonzero(L_raw != L_nodes))
        n_nodes_total += len(L_nodes)

        times[k] = t_k
        grids.append(nodes)
        values.append(L_nodes)

        # --- advance the ensemble one step --------------------------------
        L_p = _interp_slice(nodes, L_nodes, S, leverage_cap)

        if antithetic:
            z_v    = rng.standard_normal(half)
            z_perp = rng.standard_normal(half)
            u      = rng.uniform(size=half)
            Z_V    = np.concatenate([z_v, -z_v])
            Z_perp = np.concatenate([z_perp, -z_perp])
            U      = np.concatenate([u, 1.0 - u])
        else:
            Z_V    = rng.standard_normal(n)
            Z_perp = rng.standard_normal(n)
            U      = rng.uniform(size=n)

        Z_S = params.rho * Z_V + np.sqrt(1.0 - params.rho ** 2) * Z_perp

        V_new = qe_step(V, dt, Z_V, U, params)

        V_pos = np.maximum(V, 0.0)
        eff_var = (L_p ** 2) * V_pos
        S = S * np.exp((r - q - 0.5 * eff_var) * dt
                       + L_p * np.sqrt(V_pos * dt) * Z_S)
        S = np.maximum(S, 1e-8)

        V = V_new

    leverage = LeverageFunction(times, grids, values, leverage_cap)

    if verbose:
        lo, hi = leverage.value_range()
        frac = n_clipped / max(n_nodes_total, 1)
        print(f"    LSV leverage: {n_steps} slices, {n_particles:,} particles, "
              f"L in [{lo:.4f}, {hi:.4f}], clipped_fraction={frac:.4%}")

    return leverage
