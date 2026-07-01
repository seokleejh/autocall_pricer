"""
Stochastic tests for the worst-of basket simulators (models/basket.py).

Mirrors tests/test_models.py's conventions (fixed seed + antithetic for
tight MC tolerances). Two tests here are decision-driving rather than pure
regression checks: the correlation-recovery test confirms the Cholesky
correlation construction (and, for Heston, the _effective_market_corr
correction) actually reproduces the target correlation; the monotonicity
test confirms the basic worst-of economics (lower correlation -> higher
dispersion -> lower note price).
"""
import numpy as np
import pytest

from calibration.vol_surface import flat_surface
from models.basket import BasketLocalVolModel, BasketHestonModel, BasketHestonAsset
from models.heston import HestonParams
from products.autocallable import AutocallableNote
from engine.mc_pricer import MCPricer


SEED = 2025
RATE = 0.03
DIV_YIELD = 0.01
T = 2.0
EXPECTED_DRIFT = np.exp((RATE - DIV_YIELD) * T)
DRIFT_TOL = 0.01


def two_flat_surfaces(vol_a=0.20, vol_b=0.25):
    surf_a = flat_surface(spot=100.0, vol=vol_a, rate=RATE, div_yield=DIV_YIELD, T_max=3.0)
    surf_b = flat_surface(spot=100.0, vol=vol_b, rate=RATE, div_yield=DIV_YIELD, T_max=3.0)
    return [surf_a, surf_b]


def two_heston_assets(rho_a=-0.7, rho_b=-0.5):
    params_a = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=rho_a, v0=0.04)
    params_b = HestonParams(kappa=2.0, theta=0.0625, xi=0.5, rho=rho_b, v0=0.0625)
    return [
        BasketHestonAsset("A", params_a, spot=100.0, rate=RATE, div_yield=DIV_YIELD),
        BasketHestonAsset("B", params_b, spot=100.0, rate=RATE, div_yield=DIV_YIELD),
    ]


# ── shape ──────────────────────────────────────────────────────────────────────

def test_basket_local_vol_shape():
    model = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.5], [0.5, 1.0]], seed=SEED)
    perfs = model.simulate(200, [1.0, 2.0, 3.0], antithetic=False)
    assert perfs.shape == (200, 3)


def test_basket_heston_shape():
    model = BasketHestonModel(two_heston_assets(), correlation=[[1.0, 0.5], [0.5, 1.0]], seed=SEED)
    perfs = model.simulate(200, [1.0, 2.0, 3.0], antithetic=False)
    assert perfs.shape == (200, 3)


def test_simulate_assets_shape_before_aggregation():
    model = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.3], [0.3, 1.0]], seed=SEED)
    per_asset = model.simulate_assets(200, [1.0, 2.0], antithetic=False)
    assert per_asset.shape == (200, 2, 2)

    # simulate() must be the min of simulate_assets() over the asset axis.
    # Use a fresh instance for each call since simulate_assets() above already
    # advanced `model`'s rng state.
    model2 = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.3], [0.3, 1.0]], seed=SEED)
    aggregated = model2.simulate(200, [1.0, 2.0], antithetic=False)
    model3 = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.3], [0.3, 1.0]], seed=SEED)
    per_asset3 = model3.simulate_assets(200, [1.0, 2.0], antithetic=False)
    np.testing.assert_array_equal(per_asset3.min(axis=2), aggregated)


# ── worst-of <= best individual asset ──────────────────────────────────────────

def test_worst_of_never_exceeds_either_asset():
    model = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.4], [0.4, 1.0]], seed=SEED)
    per_asset = model.simulate_assets(2000, [1.0, 2.0], antithetic=True)
    worst = per_asset.min(axis=2)
    assert np.all(worst <= per_asset[:, :, 0] + 1e-12)
    assert np.all(worst <= per_asset[:, :, 1] + 1e-12)


# ── reproducibility ────────────────────────────────────────────────────────────

def test_basket_local_vol_same_seed_same_paths():
    m1 = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.5], [0.5, 1.0]], seed=42)
    m2 = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.5], [0.5, 1.0]], seed=42)
    p1 = m1.simulate(500, [1.0, 2.0], antithetic=False)
    p2 = m2.simulate(500, [1.0, 2.0], antithetic=False)
    np.testing.assert_array_equal(p1, p2)


def test_basket_heston_same_seed_same_paths():
    m1 = BasketHestonModel(two_heston_assets(), correlation=[[1.0, 0.5], [0.5, 1.0]], seed=42)
    m2 = BasketHestonModel(two_heston_assets(), correlation=[[1.0, 0.5], [0.5, 1.0]], seed=42)
    p1 = m1.simulate(500, [1.0, 2.0], antithetic=False)
    p2 = m2.simulate(500, [1.0, 2.0], antithetic=False)
    np.testing.assert_array_equal(p1, p2)


# ── correlation recovery (decision-driving) ────────────────────────────────────

def _empirical_log_return_corr(per_asset_perf_final: np.ndarray) -> float:
    log_ret = np.log(per_asset_perf_final)   # (n_paths, n_assets)
    return float(np.corrcoef(log_ret[:, 0], log_ret[:, 1])[0, 1])


@pytest.mark.parametrize("target_corr", [0.6, -0.4])
def test_local_vol_correlation_recovery(target_corr):
    """Local vol: driving Brownians are correlated directly, so recovered
    log-return correlation should match the target closely."""
    corr = [[1.0, target_corr], [target_corr, 1.0]]
    model = BasketLocalVolModel(two_flat_surfaces(), correlation=corr, seed=SEED)
    per_asset = model.simulate_assets(200_000, [T], antithetic=True)
    recovered = _empirical_log_return_corr(per_asset[:, 0, :])
    assert abs(recovered - target_corr) < 0.02, \
        f"recovered corr {recovered:.4f} vs target {target_corr}"


def test_heston_correlation_recovery_low_vol_of_vol():
    """
    With small vol-of-vol (xi), each asset's variance process is nearly
    deterministic, so the _effective_market_corr rho-diversion correction
    should recover the target spot-spot correlation almost exactly.
    """
    params_a = HestonParams(kappa=2.0, theta=0.04, xi=0.05, rho=-0.7, v0=0.04)
    params_b = HestonParams(kappa=2.0, theta=0.0625, xi=0.05, rho=-0.5, v0=0.0625)
    assets = [
        BasketHestonAsset("A", params_a, spot=100.0, rate=RATE, div_yield=DIV_YIELD),
        BasketHestonAsset("B", params_b, spot=100.0, rate=RATE, div_yield=DIV_YIELD),
    ]
    target_corr = 0.5
    corr = [[1.0, target_corr], [target_corr, 1.0]]
    model = BasketHestonModel(assets, correlation=corr, seed=SEED)
    per_asset = model.simulate_assets(200_000, [T], antithetic=True)
    recovered = _empirical_log_return_corr(per_asset[:, 0, :])
    assert abs(recovered - target_corr) < 0.03, \
        f"recovered corr {recovered:.4f} vs target {target_corr}"


@pytest.mark.parametrize("target_corr", [0.15, 0.35, 0.55])
def test_heston_correlation_recovery_monotonic_with_realistic_vol_of_vol(target_corr):
    """
    With realistic vol-of-vol, each asset's variance process evolves
    independently, which further attenuates realized correlation below the
    target beyond the rho-diversion effect (see BasketHestonModel's
    docstring) -- this is expected, not a bug. The correlation control
    should still behave correctly in DIRECTION: higher target -> higher
    realized correlation, same sign, well below the feasible ceiling
    (~0.618 for rho=-0.7/-0.5) but clearly nonzero and increasing.
    """
    corr = [[1.0, target_corr], [target_corr, 1.0]]
    model = BasketHestonModel(two_heston_assets(), correlation=corr, seed=SEED)
    per_asset = model.simulate_assets(200_000, [T], antithetic=True)
    recovered = _empirical_log_return_corr(per_asset[:, 0, :])
    assert recovered > 0, f"expected positive recovered correlation, got {recovered:.4f}"
    assert recovered < target_corr, (
        "expected some attenuation vs. the target due to independent variance "
        f"processes, got recovered={recovered:.4f} >= target={target_corr}"
    )


def test_heston_correlation_recovery_monotonic_increasing():
    """Recovered correlation must increase monotonically with the target."""
    targets = [0.15, 0.35, 0.55]
    recovered = []
    for target_corr in targets:
        corr = [[1.0, target_corr], [target_corr, 1.0]]
        model = BasketHestonModel(two_heston_assets(), correlation=corr, seed=SEED)
        per_asset = model.simulate_assets(100_000, [T], antithetic=True)
        recovered.append(_empirical_log_return_corr(per_asset[:, 0, :]))
    assert recovered[0] < recovered[1] < recovered[2], \
        f"recovered correlations not monotonic: {recovered} for targets {targets}"


# ── monotonicity: lower correlation -> lower worst-of price ────────────────────

def _price_worst_of_note(model) -> float:
    note = AutocallableNote(
        notional=1.0,
        spot=1.0,
        maturity=T,
        observation_dates=[T],
        autocall_barriers=1.05,
        coupon_rate=0.05,
        capital_barrier=0.80,
        capital_barrier_active=True,
        discount_rate=RATE,
    )
    pricer = MCPricer(model, antithetic=True)
    return pricer.price(note, n_paths=100_000).price


def test_local_vol_price_decreases_as_correlation_decreases():
    model_high = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.9], [0.9, 1.0]], seed=SEED)
    model_low = BasketLocalVolModel(two_flat_surfaces(), correlation=[[1.0, 0.1], [0.1, 1.0]], seed=SEED)
    price_high = _price_worst_of_note(model_high)
    price_low = _price_worst_of_note(model_low)
    assert price_high > price_low, \
        f"expected higher-correlation worst-of price ({price_high:.4f}) > lower-correlation ({price_low:.4f})"


def test_heston_price_decreases_as_correlation_decreases():
    # NB: with rho_a=-0.7, rho_b=-0.5, the feasible ceiling for the corrected
    # market-factor correlation is sqrt(1-0.7^2)*sqrt(1-0.5^2) ~= 0.618, so
    # target correlations must stay well below that (unlike the Local Vol
    # case, which has no such ceiling).
    model_high = BasketHestonModel(two_heston_assets(), correlation=[[1.0, 0.5], [0.5, 1.0]], seed=SEED)
    model_low = BasketHestonModel(two_heston_assets(), correlation=[[1.0, 0.05], [0.05, 1.0]], seed=SEED)
    price_high = _price_worst_of_note(model_high)
    price_low = _price_worst_of_note(model_low)
    assert price_high > price_low, \
        f"expected higher-correlation worst-of price ({price_high:.4f}) > lower-correlation ({price_low:.4f})"


# ── degeneracy: 1-asset basket matches single-asset model ──────────────────────

def test_one_asset_basket_matches_single_asset_local_vol():
    from models.local_vol import LocalVolModel
    surf = flat_surface(spot=100.0, vol=0.20, rate=RATE, div_yield=DIV_YIELD, T_max=3.0)

    single = LocalVolModel(surface=surf, seed=SEED)
    basket = BasketLocalVolModel([surf], correlation=[[1.0]], seed=SEED)

    p_single = single.simulate(1000, [1.0, 2.0], antithetic=False)
    p_basket = basket.simulate(1000, [1.0, 2.0], antithetic=False)
    np.testing.assert_allclose(p_single, p_basket, rtol=1e-10)


# ── risk-neutral drift per asset (correlation shouldn't distort marginals) ─────

def test_basket_local_vol_drift_per_asset_unaffected_by_correlation():
    model = BasketLocalVolModel(two_flat_surfaces(vol_a=0.20, vol_b=0.20),
                                 correlation=[[1.0, 0.7], [0.7, 1.0]], seed=SEED)
    per_asset = model.simulate_assets(60_000, [T], antithetic=True)
    for a in range(2):
        mean_perf = float(np.mean(per_asset[:, 0, a]))
        assert abs(mean_perf - EXPECTED_DRIFT) < DRIFT_TOL, \
            f"asset {a} drift {mean_perf:.5f} vs expected {EXPECTED_DRIFT:.5f}"
