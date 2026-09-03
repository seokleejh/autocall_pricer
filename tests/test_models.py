"""
Stochastic tests for model simulation correctness.

Each test uses a fixed seed and antithetic variates so the Monte Carlo error
is small enough for a reliable assertion, without requiring a huge path count.

Key property tested: risk-neutral drift.
Under the risk-neutral measure, E[S(T) / S(0)] = exp((r - q) * T).
"""
import numpy as np
import pytest
from calibration.vol_surface import flat_surface
from models.local_vol import LocalVolModel
from models.heston import HestonModel, HestonParams
from models.sabr import SABRModel, SABRParams


N_PATHS = 40_000   # enough for ~0.5% accuracy with antithetic
SEED = 2025
RATE = 0.03
DIV_YIELD = 0.01
T = 2.0
EXPECTED_DRIFT = np.exp((RATE - DIV_YIELD) * T)
DRIFT_TOL = 0.005   # ±0.5% absolute tolerance


@pytest.fixture
def flat_surf():
    return flat_surface(spot=100.0, vol=0.20, rate=RATE, div_yield=DIV_YIELD, T_max=3.0)


# ── risk-neutral drift ─────────────────────────────────────────────────────────

def test_local_vol_risk_neutral_drift(flat_surf):
    """
    LocalVol simulation under flat vol surface should produce E[S(T)/S(0)] = exp((r-q)T).
    """
    model = LocalVolModel(surface=flat_surf, steps_per_year=52, seed=SEED)
    perfs = model.simulate(N_PATHS, [T], antithetic=True)
    mean_perf = float(np.mean(perfs[:, 0]))
    assert abs(mean_perf - EXPECTED_DRIFT) < DRIFT_TOL, \
        f"LV drift {mean_perf:.5f} vs expected {EXPECTED_DRIFT:.5f}"


def test_heston_risk_neutral_drift():
    """Heston simulation should satisfy E[S(T)/S(0)] = exp((r-q)T)."""
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    model = HestonModel(params=params, spot=100.0, rate=RATE, div_yield=DIV_YIELD,
                        steps_per_year=52, seed=SEED)
    perfs = model.simulate(N_PATHS, [T], antithetic=True)
    mean_perf = float(np.mean(perfs[:, 0]))
    assert abs(mean_perf - EXPECTED_DRIFT) < DRIFT_TOL, \
        f"Heston drift {mean_perf:.5f} vs expected {EXPECTED_DRIFT:.5f}"


def test_sabr_risk_neutral_drift():
    """SABR simulation should satisfy E[S(T)/S(0)] = exp((r-q)T)."""
    params = SABRParams(alpha=0.20, beta=1.0, rho=-0.30, nu=0.40)
    model = SABRModel(params=params, spot=100.0, rate=RATE, div_yield=DIV_YIELD,
                      steps_per_year=52, seed=SEED)
    perfs = model.simulate(N_PATHS, [T], antithetic=True)
    mean_perf = float(np.mean(perfs[:, 0]))
    assert abs(mean_perf - EXPECTED_DRIFT) < DRIFT_TOL, \
        f"SABR drift {mean_perf:.5f} vs expected {EXPECTED_DRIFT:.5f}"


# ── performance shape ─────────────────────────────────────────────────────────

def test_simulation_output_shape(flat_surf):
    """simulate() returns array of shape (n_paths, n_obs)."""
    model = LocalVolModel(surface=flat_surf, seed=SEED)
    obs = [1.0, 2.0, 3.0]
    perfs = model.simulate(200, obs, antithetic=False)
    assert perfs.shape == (200, 3)


def test_antithetic_returns_correct_shape(flat_surf):
    model = LocalVolModel(surface=flat_surf, seed=SEED)
    perfs = model.simulate(200, [1.0, 2.0], antithetic=True)
    assert perfs.shape == (200, 2)


# ── antithetic variates are genuinely antithetic ──────────────────────────────
# A true antithetic pair reuses the SAME random draws with the sign flipped, so
# paths i and i + n/2 are strongly negatively correlated. Drawing fresh normals
# for the second half and negating them yields independent paths -- the normal
# law is symmetric -- which leaves prices unbiased but silently discards all of
# the variance reduction. These tests pin the pairing down.

ANTITHETIC_MODELS = [
    ("LocalVol", lambda surf, seed: LocalVolModel(surface=surf, seed=seed)),
    ("Heston",   lambda surf, seed: HestonModel(params=HestonParams(), spot=100.0,
                                                rate=RATE, div_yield=DIV_YIELD, seed=seed)),
    ("SABR",     lambda surf, seed: SABRModel(params=SABRParams(), spot=100.0,
                                              rate=RATE, div_yield=DIV_YIELD, seed=seed)),
]


@pytest.mark.parametrize("name,build", ANTITHETIC_MODELS, ids=[m[0] for m in ANTITHETIC_MODELS])
def test_antithetic_halves_are_negatively_correlated(flat_surf, name, build):
    """Paths i and i + n/2 must be mirror images, not independent draws."""
    perfs = build(flat_surf, SEED).simulate(20_000, [1.0], antithetic=True)[:, 0]
    half = len(perfs) // 2
    corr = float(np.corrcoef(perfs[:half], perfs[half:])[0, 1])
    assert corr < -0.5, (
        f"{name}: antithetic halves correlate {corr:+.3f}; a genuine pairing is "
        f"strongly negative, ~0 means fresh draws were negated instead"
    )


@pytest.mark.parametrize("name,build", ANTITHETIC_MODELS, ids=[m[0] for m in ANTITHETIC_MODELS])
def test_antithetic_reduces_variance(flat_surf, name, build):
    """
    The pairing must actually pay off: the standard error of an ATM call
    estimated from n antithetic paths, computed the correct way (average each
    pair into one observation), should beat n independent paths.
    """
    n = 40_000
    plain = build(flat_surf, 11).simulate(n, [1.0], antithetic=False)[:, 0]
    anti = build(flat_surf, 11).simulate(n, [1.0], antithetic=True)[:, 0]

    pay_plain = np.maximum(plain - 1.0, 0.0)
    pay_anti = np.maximum(anti - 1.0, 0.0)

    se_plain = pay_plain.std(ddof=1) / np.sqrt(len(pay_plain))
    half = len(pay_anti) // 2
    pair_means = 0.5 * (pay_anti[:half] + pay_anti[half:])
    se_anti = pair_means.std(ddof=1) / np.sqrt(half)

    assert se_anti < se_plain, (
        f"{name}: antithetic SE {se_anti:.6f} should beat plain SE {se_plain:.6f}"
    )


def test_mc_pricer_standard_error_accounts_for_pairing(flat_surf):
    """
    MCPricer must use the pair-aware estimator when antithetic is on.
    Treating the paths as independent ignores the negative covariance and
    overstates the error, throwing away the variance reduction on paper.
    """
    from engine.mc_pricer import MCPricer
    from products.autocallable import AutocallableNote

    note = AutocallableNote(
        spot=100.0, maturity=2.0, observation_dates=[1.0, 2.0],
        autocall_barriers=1.0, coupon_rate=0.06,
        capital_barrier=0.6, discount_rate=RATE,
    )
    model = LocalVolModel(surface=flat_surf, seed=SEED)
    pricer = MCPricer(model, "LV", antithetic=True)
    result = pricer.price(note, 20_000)

    payoffs = note.evaluate_payoff(
        LocalVolModel(surface=flat_surf, seed=SEED).simulate(
            20_000, note.observation_dates, antithetic=True)
    )
    naive_se = float(np.std(payoffs, ddof=1) / np.sqrt(len(payoffs)))
    assert result.std_error < naive_se


def test_performances_are_positive(flat_surf):
    """All simulated performance values should be positive."""
    for ModelCls, kwargs in [
        (LocalVolModel, {"surface": flat_surf, "seed": SEED}),
        (HestonModel,   {"params": HestonParams(), "spot": 100.0,
                         "rate": RATE, "div_yield": DIV_YIELD, "seed": SEED}),
        (SABRModel,     {"params": SABRParams(), "spot": 100.0,
                         "rate": RATE, "div_yield": DIV_YIELD, "seed": SEED}),
    ]:
        model = ModelCls(**kwargs)
        perfs = model.simulate(1000, [1.0, 2.0, 3.0], antithetic=True)
        assert np.all(perfs > 0), f"{ModelCls.__name__}: non-positive performances"


# ── reproducibility ───────────────────────────────────────────────────────────

def test_same_seed_same_paths(flat_surf):
    """Two model instances with the same seed must produce identical paths."""
    m1 = LocalVolModel(surface=flat_surf, seed=42)
    m2 = LocalVolModel(surface=flat_surf, seed=42)
    p1 = m1.simulate(500, [1.0, 2.0], antithetic=False)
    p2 = m2.simulate(500, [1.0, 2.0], antithetic=False)
    np.testing.assert_array_equal(p1, p2)


def test_different_seeds_different_paths(flat_surf):
    m1 = LocalVolModel(surface=flat_surf, seed=1)
    m2 = LocalVolModel(surface=flat_surf, seed=2)
    p1 = m1.simulate(500, [1.0], antithetic=False)
    p2 = m2.simulate(500, [1.0], antithetic=False)
    assert not np.allclose(p1, p2)


# ── pure discount bond ────────────────────────────────────────────────────────

def test_capital_protected_note_prices_as_discount_bond(flat_surf):
    """
    A capital-protected note with autocall_barriers > 1 (never triggers) and
    no coupon is equivalent to a risk-free zero-coupon bond.
    Price should be exp(-r * T) regardless of model.

    This is an integration test of the full simulation → payoff pipeline.
    """
    from products.autocallable import AutocallableNote
    from engine.mc_pricer import MCPricer

    r = 0.05
    T_bond = 2.0
    surf = flat_surface(spot=100.0, vol=0.20, rate=r, div_yield=0.0, T_max=3.0)
    note = AutocallableNote(
        notional=1.0,
        spot=100.0,
        maturity=T_bond,
        observation_dates=[T_bond],
        autocall_barriers=2.0,       # never autocalls
        coupon_rate=0.0,
        capital_barrier_active=False, # always returns notional
        discount_rate=r,
    )
    expected = np.exp(-r * T_bond)

    for ModelCls, kwargs in [
        (LocalVolModel, {"surface": surf, "seed": SEED}),
        (HestonModel,   {"params": HestonParams(), "spot": 100.0, "rate": r,
                         "div_yield": 0.0, "seed": SEED}),
        (SABRModel,     {"params": SABRParams(), "spot": 100.0, "rate": r,
                         "div_yield": 0.0, "seed": SEED}),
    ]:
        model = ModelCls(**kwargs)
        pricer = MCPricer(model, antithetic=True)
        result = pricer.price(note, n_paths=20_000)
        assert abs(result.price - expected) < 0.002, \
            f"{ModelCls.__name__}: bond price {result.price:.5f} vs {expected:.5f}"
