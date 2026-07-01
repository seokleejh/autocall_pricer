"""
Tests for basket Greek computation (scenarios/run_scenarios.py's
compute_basket_greeks and the per-asset performance rescale it applies).

Design philosophy (mirrors tests/test_greeks.py):
- Delta sign checks reuse the SAME simulated per-asset paths for the
  base/up/down comparisons where possible, so the only difference is the
  rescale factor / barrier -- noise-free.
- One test is a direct regression guard for a real bug found during
  development: Heston's simulated performance is scale-invariant to the
  spot level (log S(t)/S(0) never depends on S(0) in the SDE), so without
  the per-asset performance rescale, bumping an asset's spot leaves Heston's
  simulated performance (and hence delta/gamma) EXACTLY zero, not just
  noisy. That regression is asserted directly.
"""
import numpy as np
import pytest

from calibration.vol_surface import flat_surface
from models.basket import BasketHestonModel, BasketHestonAsset, BasketLocalVolModel
from models.heston import HestonParams
from products.autocallable import AutocallableNote
from scenarios.run_scenarios import compute_basket_greeks, build_basket_pricers


SEED = 42
N_PATHS = 40_000
RATE = 0.03
DIV_YIELD = 0.0


def make_worst_of_note(**kw):
    defaults = dict(
        notional=1.0, spot=1.0, maturity=3.0,
        observation_dates=[1.0, 2.0, 3.0],
        autocall_barriers=0.95, coupon_rate=0.05,
        conditional_coupon=False, coupon_barrier=0.80,
        capital_barrier=0.60, capital_barrier_active=True,
        discount_rate=0.03,
    )
    defaults.update(kw)
    return AutocallableNote(**defaults)


def two_heston_assets():
    params_a = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)
    params_b = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04)
    return [
        BasketHestonAsset("A", params_a, spot=100.0, rate=RATE, div_yield=DIV_YIELD),
        BasketHestonAsset("B", params_b, spot=100.0, rate=RATE, div_yield=DIV_YIELD),
    ]


def two_flat_surfaces():
    return [
        flat_surface(spot=100.0, vol=0.20, rate=RATE, div_yield=DIV_YIELD, T_max=3.0),
        flat_surface(spot=100.0, vol=0.20, rate=RATE, div_yield=DIV_YIELD, T_max=3.0),
    ]


# ── regression guard: rescale must be applied, or Heston delta/gamma vanish ────

def test_basket_heston_delta_not_trivially_zero():
    """
    Without the per-asset performance rescale, bumping an asset's spot for
    Heston leaves its simulated performance completely unchanged (scale
    invariance of the Heston SDE for S(t)/S(0)), making delta/gamma exactly
    zero rather than merely small. This exercises the real compute_basket_greeks
    path end-to-end and checks the result is NOT the degenerate all-zero case.
    """
    correlation = [[1.0, 0.3], [0.3, 1.0]]
    assets = two_heston_assets()
    note = make_worst_of_note(autocall_barriers=1.0)
    model = BasketHestonModel(assets, correlation=correlation, seed=SEED)

    base_perf = model.simulate(N_PATHS, note.observation_dates, antithetic=True)
    base_price = float(np.mean(note.evaluate_payoff(base_perf)))

    cfg = {"models": {"local_vol": False, "heston": True, "sabr": False}}
    asset_surfaces = [
        ("A", flat_surface(spot=100.0, vol=np.sqrt(0.04), rate=RATE, div_yield=DIV_YIELD, T_max=3.0)),
        ("B", flat_surface(spot=100.0, vol=np.sqrt(0.04), rate=RATE, div_yield=DIV_YIELD, T_max=3.0)),
    ]
    base_heston = {"A": assets[0].params, "B": assets[1].params}

    greeks = compute_basket_greeks(
        cfg, asset_surfaces, np.array(correlation), base_heston, note,
        {"Basket Heston": base_price}, SEED, True, N_PATHS,
    )
    gr = greeks["Basket Heston"]
    assert abs(gr.delta["A"]) > 1e-6, "Heston delta for asset A should not be exactly zero"
    assert abs(gr.delta["B"]) > 1e-6, "Heston delta for asset B should not be exactly zero"
    assert abs(gr.gamma["A"]) > 1e-6, "Heston gamma for asset A should not be exactly zero"


# ── deterministic (same-path) delta sign, mirroring tests/test_greeks.py ───────

def test_basket_heston_delta_sign_atm_barrier():
    """
    Same reasoning as test_heston_delta_positive_atm_barrier in
    tests/test_greeks.py: an ATM worst-of barrier means a higher spot for
    asset A makes autocall (of the worst-of note) easier from asset A's
    side, paying the coupon sooner -> price should increase -> delta_A > 0.
    Uses the SAME simulated per-asset paths for the base/up/down comparison
    (only the rescale factor differs), so this is noise-free.
    """
    correlation = [[1.0, 0.3], [0.3, 1.0]]
    assets = two_heston_assets()
    note = make_worst_of_note(autocall_barriers=1.0, coupon_rate=0.05, capital_barrier=0.60)

    model = BasketHestonModel(assets, correlation=correlation, seed=SEED)
    per_asset = model.simulate_assets(N_PATHS, note.observation_dates, antithetic=True)

    h_S_pct = 0.01
    S0 = 100.0
    h_S = S0 * h_S_pct
    scale_up = (S0 + h_S) / S0
    scale_dn = (S0 - h_S) / S0

    a_idx = 0  # asset A
    perf_up = per_asset.copy()
    perf_up[:, :, a_idx] *= scale_up
    perf_dn = per_asset.copy()
    perf_dn[:, :, a_idx] *= scale_dn

    p_up = float(np.mean(note.evaluate_payoff(perf_up.min(axis=2))))
    p_dn = float(np.mean(note.evaluate_payoff(perf_dn.min(axis=2))))
    delta_a = (p_up - p_dn) / (2 * h_S)
    assert delta_a > 0, f"Expected positive delta for asset A on ATM worst-of barrier, got {delta_a:.6f}"


# ── no barrier rescaling in basket mode ────────────────────────────────────────

def test_basket_greeks_do_not_mutate_product_barriers():
    """
    Unlike the single-asset compute_greeks() (which builds NEW rescaled
    product instances via _rescale_product_barriers), compute_basket_greeks
    must reuse the SAME product/barriers throughout -- rescaling instead
    happens on the bumped asset's performance array. Verify the product's
    barrier schedule is bit-identical before and after computing Greeks.
    """
    correlation = [[1.0, 0.4], [0.4, 1.0]]
    note = make_worst_of_note(autocall_barriers=[0.9, 0.85, 0.8])
    barriers_before = note.autocall_barrier_schedule.copy()
    capital_barrier_before = note.capital_barrier

    cfg = {"models": {"local_vol": True, "heston": False, "sabr": False}}
    asset_surfaces = [("A", two_flat_surfaces()[0]), ("B", two_flat_surfaces()[1])]

    model = BasketLocalVolModel([s for _, s in asset_surfaces], correlation=correlation, seed=SEED)
    base_perf = model.simulate(5000, note.observation_dates, antithetic=True)
    base_price = float(np.mean(note.evaluate_payoff(base_perf)))

    compute_basket_greeks(
        cfg, asset_surfaces, np.array(correlation), {}, note,
        {"Basket Local Vol": base_price}, SEED, True, 5000,
    )

    np.testing.assert_array_equal(note.autocall_barrier_schedule, barriers_before)
    assert note.capital_barrier == capital_barrier_before


# ── bumping one asset must hold the other's surface fixed (wiring) ─────────────

def test_bumping_one_asset_reuses_others_base_surface_object():
    """
    compute_basket_greeks builds each bumped surface dict as
    {**surfaces, asset_name: <bumped>} -- every OTHER asset's surface must be
    the exact same object passed in (not a copy, not independently bumped).
    Patch BasketLocalVolModel to record the surfaces it's constructed with
    and check asset B's surface is always the original base instance,
    whichever asset is being bumped.
    """
    correlation = [[1.0, 0.3], [0.3, 1.0]]
    note = make_worst_of_note(autocall_barriers=1.0)
    cfg = {"models": {"local_vol": True, "heston": False, "sabr": False}}

    surf_a, surf_b = two_flat_surfaces()
    asset_surfaces = [("A", surf_a), ("B", surf_b)]

    seen_surfaces_by_asset: dict[str, list] = {"A": [], "B": []}
    real_init = BasketLocalVolModel.__init__

    def spy_init(self, surfaces, correlation, steps_per_year=52, seed=None):
        seen_surfaces_by_asset["A"].append(surfaces[0])
        seen_surfaces_by_asset["B"].append(surfaces[1])
        real_init(self, surfaces, correlation, steps_per_year=steps_per_year, seed=seed)

    model = BasketLocalVolModel([surf_a, surf_b], correlation=correlation, seed=SEED)
    base_perf = model.simulate(5000, note.observation_dates, antithetic=True)
    base_price = float(np.mean(note.evaluate_payoff(base_perf)))

    BasketLocalVolModel.__init__ = spy_init
    try:
        compute_basket_greeks(
            cfg, asset_surfaces, np.array(correlation), {}, note,
            {"Basket Local Vol": base_price}, SEED, True, 5000,
        )
    finally:
        BasketLocalVolModel.__init__ = real_init

    # While bumping A, every recorded B surface must be the original object.
    # While bumping B, every recorded A surface must be the original object.
    # (compute_basket_greeks processes assets in order [A, B]; the first half
    # of calls bump A, the second half bump B -- check both halves.)
    n_calls = len(seen_surfaces_by_asset["A"])
    half = n_calls // 2
    assert all(s is surf_b for s in seen_surfaces_by_asset["B"][:half]), \
        "asset B's surface should be untouched while bumping asset A"
    assert all(s is surf_a for s in seen_surfaces_by_asset["A"][half:]), \
        "asset A's surface should be untouched while bumping asset B"
