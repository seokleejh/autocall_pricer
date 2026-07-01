"""
Deterministic worst-of aggregation tests.

Worst-of basket pricing relies on AutocallableNote.evaluate_payoff() being
completely aggregation-agnostic (see products/autocallable.py) -- the
"worst-of" step is just an element-wise min across a (n_paths, n_obs,
n_assets) per-asset performance array, taken once per observation date,
before handing the (n_paths, n_obs) result to evaluate_payoff(). These tests
hand-craft per-asset arrays, apply that same aggregation, and check the
resulting payoff against hand-computed expectations -- no Monte Carlo noise.
"""
import numpy as np
from products.autocallable import AutocallableNote


def make_note(**overrides):
    """Standard 3-year note with sensible defaults (mirrors test_payoff.py)."""
    defaults = dict(
        notional=1.0,
        spot=1.0,           # unused placeholder in basket mode
        maturity=3.0,
        observation_dates=[1.0, 2.0, 3.0],
        autocall_barriers=0.95,
        coupon_rate=0.05,
        conditional_coupon=False,
        coupon_barrier=0.80,
        capital_barrier=0.60,
        capital_barrier_active=True,
        discount_rate=0.03,
    )
    defaults.update(overrides)
    return AutocallableNote(**defaults)


def worst_of(performances_per_asset: np.ndarray) -> np.ndarray:
    """Reference aggregation: element-wise min across the asset axis."""
    return performances_per_asset.min(axis=2)


# ── worst-of aggregation drives the payoff ─────────────────────────────────────

def test_worst_of_triggers_ki_when_only_one_asset_breaches():
    """
    Two assets: asset A stays strong throughout, asset B breaches the capital
    barrier at maturity. The worst-of note must take the capital loss even
    though asset A alone would have protected the note.
    """
    note = make_note(discount_rate=0.0, autocall_barriers=1.10)  # never autocalls
    # shape (n_paths=1, n_obs=3, n_assets=2)
    perf_per_asset = np.array([[
        [1.05, 0.95],   # t=1: A=1.05, B=0.95
        [1.10, 0.80],   # t=2: A=1.10, B=0.80
        [1.20, 0.50],   # t=3: A=1.20, B=0.50 (< capital_barrier=0.60)
    ]])
    worst = worst_of(perf_per_asset)
    np.testing.assert_array_equal(worst, np.array([[0.95, 0.80, 0.50]]))
    pv = note.evaluate_payoff(worst)
    np.testing.assert_allclose(pv[0], 0.50, rtol=1e-12)  # fractional notional = worst final perf


def test_worst_of_autocalls_only_when_both_assets_clear_barrier():
    """Autocall requires the WORST asset to clear the barrier, not just one."""
    note = make_note(discount_rate=0.0, autocall_barriers=0.95, coupon_rate=0.05,
                      capital_barrier_active=False)
    perf_per_asset = np.array([
        [[0.99, 0.90], [0.99, 0.99], [0.99, 0.99]],   # t1: worst=0.90 misses; t2: worst=0.99 clears
        [[0.99, 0.99], [0.99, 0.99], [0.99, 0.99]],   # t1: worst=0.99 clears immediately
    ])
    worst = worst_of(perf_per_asset)
    pv = note.evaluate_payoff(worst)
    # path 0: misses at t=1 (worst=0.90), autocalls at t=2 (worst=0.99)
    np.testing.assert_allclose(pv[0], 1.0 + 0.05, rtol=1e-12)
    # path 1: autocalls at t=1
    np.testing.assert_allclose(pv[1], 1.0 + 0.05, rtol=1e-12)


def test_worst_asset_differs_across_observation_dates():
    """
    The "worst" asset is allowed to be a different asset at each observation
    date -- aggregation must be per-date, not a single fixed worst performer
    over the whole path.
    """
    note = make_note(discount_rate=0.0, autocall_barriers=1.10,  # never autocalls
                      capital_barrier_active=False, conditional_coupon=False, coupon_rate=0.0)
    perf_per_asset = np.array([[
        [0.80, 0.99],   # t=1: B is worse (but doesn't matter, no coupon/autocall here)
        [0.99, 0.70],   # t=2: A is worse
        [0.65, 0.99],   # t=3 (maturity): B is worse -> final worst-of = 0.65
    ]])
    worst = worst_of(perf_per_asset)
    np.testing.assert_array_equal(worst, np.array([[0.80, 0.70, 0.65]]))
    pv = note.evaluate_payoff(worst)
    np.testing.assert_allclose(pv[0], 1.0, rtol=1e-12)  # 0.65 >= capital_barrier(0.60, default)


def test_single_asset_basket_matches_direct_evaluate_payoff():
    """
    A 1-asset "basket" (worst-of of one asset) must be a no-op: aggregating
    over a single asset axis reproduces the identical performance array,
    hence identical PV to calling evaluate_payoff directly.
    """
    note = make_note(discount_rate=0.03, autocall_barriers=0.95, coupon_rate=0.07)
    single_asset_perfs = np.array([
        [0.90, 0.97, 0.99],
        [0.80, 0.70, 0.50],
    ])
    perf_per_asset = single_asset_perfs[:, :, np.newaxis]  # (n_paths, n_obs, 1)
    worst = worst_of(perf_per_asset)

    np.testing.assert_array_equal(worst, single_asset_perfs)
    pv_direct = note.evaluate_payoff(single_asset_perfs)
    pv_via_worst_of = note.evaluate_payoff(worst)
    np.testing.assert_array_equal(pv_direct, pv_via_worst_of)
