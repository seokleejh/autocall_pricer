"""
Tests for the Heston-LSV model and its particle-method leverage calibration.

The primary criterion is the vanilla round-trip: LSV's entire claim is that it
reprices the input vanilla surface while keeping Heston's forward dynamics. If
it does not reproduce the surface, it is not calibrated -- there is no
judgement call involved, which makes this an unusually clean acceptance test.

Path counts and particle counts here are deliberately smaller than the values
recommended for production runs, to keep the suite fast. Tolerances are set
accordingly and are looser than the ~6bp RMS achievable at 50k particles /
200k paths.
"""
import numpy as np
import pytest

from calibration.vol_surface import term_structure_surface, flat_surface
from calibration.calibrators import calibrate_heston
from calibration.lsv_calibration import calibrate_leverage, _conditional_variance
from models.lsv import LSVModel, LeverageFunction
from models.heston import HestonModel, HestonParams
from models.local_vol import LocalVolModel
from products.autocallable import AutocallableNote
from engine.mc_pricer import MCPricer
from engine.black_scholes import implied_vol as bs_implied_vol


SPOT = 100.0
RATE = 0.03
DIV_YIELD = 0.01
T_MAX = 2.0
SEED = 4242

N_PARTICLES = 20_000
N_PATHS = 100_000

MONEYNESS = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
TENORS = (0.5, 1.0, 2.0)


# ── fixtures ───────────────────────────────────────────────────────────────────
# Calibration is the expensive part, so it happens once per module.

@pytest.fixture(scope="module")
def surface():
    return term_structure_surface(
        SPOT, rate=RATE, div_yield=DIV_YIELD, T_max=5.0
    )


@pytest.fixture(scope="module")
def heston_params(surface):
    return calibrate_heston(surface)


@pytest.fixture(scope="module")
def leverage(surface, heston_params):
    return calibrate_leverage(
        surface, heston_params, T_MAX,
        n_particles=N_PARTICLES, n_spot_bins=28, seed=SEED,
    )


def _smile_errors_bp(model, surface, T, n_paths=N_PATHS):
    """
    Model-implied vol minus surface implied vol, in basis points, across
    MONEYNESS. Uses the OTM convention (puts below spot, calls above) so
    implied vol is never extracted from a deep-ITM option where vega is tiny
    and MC noise dominates.
    """
    perf = model.simulate(n_paths, [T], antithetic=True)[:, 0]
    df = np.exp(-RATE * T)
    errors = []
    for m in MONEYNESS:
        K = SPOT * m
        k = K / SPOT
        is_call = K >= SPOT
        payoff = np.maximum(perf - k, 0.0) if is_call else np.maximum(k - perf, 0.0)
        mc_price = SPOT * df * float(payoff.mean())
        iv = bs_implied_vol(mc_price, SPOT, K, T, RATE, DIV_YIELD, is_call=is_call)
        errors.append((iv - float(surface.implied_vol(T, K))) * 1e4)
    return np.array(errors)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


# ── primary acceptance criterion ───────────────────────────────────────────────

def test_vanilla_roundtrip(surface, heston_params, leverage):
    """
    THE test. LSV must reproduce the input vanilla surface it was calibrated
    to. At 20k particles / 100k paths we require 15bp RMS; production settings
    (50k / 200k) reach roughly 6bp.
    """
    all_errors = []
    for T in TENORS:
        model = LSVModel(heston_params, leverage, SPOT, RATE, DIV_YIELD, seed=SEED + 1)
        errors = _smile_errors_bp(model, surface, T)
        assert np.all(np.isfinite(errors)), f"non-finite implied vol at T={T}"
        all_errors.append(errors)

    rms = _rms(np.concatenate(all_errors))
    assert rms < 15.0, f"LSV round-trip RMS {rms:.1f}bp exceeds 15bp tolerance"


def test_lsv_fits_surface_better_than_heston(surface, heston_params, leverage):
    """
    The whole point of the leverage function: it should remove most of the
    smile residual that plain Heston leaves behind. Same Heston parameters in
    both models, so this isolates the leverage function's contribution.
    """
    lsv_err, heston_err = [], []
    for T in TENORS:
        lsv_err.append(_smile_errors_bp(
            LSVModel(heston_params, leverage, SPOT, RATE, DIV_YIELD, seed=SEED + 2),
            surface, T))
        heston_err.append(_smile_errors_bp(
            HestonModel(heston_params, SPOT, RATE, DIV_YIELD, seed=SEED + 2),
            surface, T))

    rms_lsv = _rms(np.concatenate(lsv_err))
    rms_heston = _rms(np.concatenate(heston_err))
    assert rms_lsv < rms_heston, (
        f"LSV ({rms_lsv:.1f}bp) should fit better than Heston ({rms_heston:.1f}bp)"
    )


# ── degenerate limits: LSV interpolates between the two existing models ────────

def test_degenerate_zero_volvol_matches_local_vol(surface):
    """
    With xi -> 0 the variance process is deterministic, so E[V|S] = V(t) and
    the leverage function absorbs the entire surface: LSV collapses to Dupire
    local vol. Prices must then agree with LocalVolModel within MC error.
    """
    atm_var = float(surface.implied_vol(1.0, float(surface.forward(1.0)))) ** 2
    params = HestonParams(kappa=1.0, theta=atm_var, xi=1e-6, rho=0.0, v0=atm_var)

    lev = calibrate_leverage(surface, params, T_MAX,
                             n_particles=N_PARTICLES, seed=SEED)
    lsv = LSVModel(params, lev, SPOT, RATE, DIV_YIELD, seed=SEED + 3)
    lv = LocalVolModel(surface, seed=SEED + 3)

    T = 1.0
    perf_lsv = lsv.simulate(N_PATHS, [T], antithetic=True)[:, 0]
    perf_lv = lv.simulate(N_PATHS, [T], antithetic=True)[:, 0]

    # Compare ATM call prices; both are unbiased estimates of the same number.
    c_lsv = float(np.maximum(perf_lsv - 1.0, 0.0).mean())
    c_lv = float(np.maximum(perf_lv - 1.0, 0.0).mean())
    assert abs(c_lsv - c_lv) < 0.004, (
        f"xi->0 LSV ATM call {c_lsv:.5f} should match LocalVol {c_lv:.5f}"
    )


def test_leverage_near_unity_when_heston_already_fits(heston_params):
    """
    On a flat surface, a Heston process with xi -> 0 and v0 = theta = sigma^2
    already reproduces the surface exactly. The leverage function then has
    nothing left to correct and must come out close to 1 -- it should not
    invent structure that is not in the data.
    """
    vol = 0.20
    flat = flat_surface(SPOT, vol, rate=RATE, div_yield=DIV_YIELD, T_max=3.0)
    params = HestonParams(kappa=1.0, theta=vol**2, xi=1e-6, rho=0.0, v0=vol**2)

    lev = calibrate_leverage(flat, params, T_MAX,
                             n_particles=N_PARTICLES, seed=SEED)
    lo, hi = lev.value_range()
    assert 0.95 < lo < 1.05, f"leverage min {lo:.4f} should sit near 1"
    assert 0.95 < hi < 1.05, f"leverage max {hi:.4f} should sit near 1"


# ── numerical hygiene ──────────────────────────────────────────────────────────

def test_leverage_is_finite_and_within_cap(surface, heston_params):
    """
    The leverage cap is a safety rail against a near-zero E[V|S] in a sparse
    wing bin. Nothing may escape it, and nothing may be NaN or inf.
    """
    cap = (0.1, 10.0)
    lev = calibrate_leverage(surface, heston_params, T_MAX,
                             n_particles=N_PARTICLES, leverage_cap=cap, seed=SEED)
    for grid, vals in zip(lev.spot_grids, lev.values):
        assert np.all(np.isfinite(vals)), "leverage contains NaN or inf"
        assert np.all(vals >= cap[0] - 1e-12) and np.all(vals <= cap[1] + 1e-12)
        assert np.all(np.diff(grid) > 0), "spot nodes must be strictly increasing"

    # Evaluation far outside the calibrated spot range must clamp, not explode.
    for t in (0.0, 0.5, T_MAX, T_MAX * 5):
        out = lev(t, np.array([1e-3, SPOT, 1e5]))
        assert np.all(np.isfinite(out))
        assert np.all(out >= cap[0] - 1e-12) and np.all(out <= cap[1] + 1e-12)


def test_conditional_variance_degenerate_ensemble():
    """At t=0 every particle sits at S0, so there is no spot dependence yet."""
    S = np.full(1000, SPOT)
    V = np.full(1000, 0.04)
    nodes, e_v = _conditional_variance(S, V, n_bins=28)
    assert len(nodes) == 1 and len(e_v) == 1
    assert nodes[0] == pytest.approx(SPOT)
    assert e_v[0] == pytest.approx(0.04)


def test_conditional_variance_recovers_known_relationship():
    """
    With V a deterministic increasing function of S, equal-count binning must
    recover that relationship at the bin means.
    """
    rng = np.random.default_rng(0)
    S = rng.lognormal(np.log(SPOT), 0.25, size=50_000)
    V = 0.01 + 0.0002 * S
    nodes, e_v = _conditional_variance(S, V, n_bins=28)
    assert len(nodes) > 20
    assert np.all(np.diff(nodes) > 0)
    np.testing.assert_allclose(e_v, 0.01 + 0.0002 * nodes, rtol=2e-3)


def test_leverage_function_piecewise_constant_in_time():
    """Slice lookup: t in [times[k], times[k+1]) must select slice k."""
    times = np.array([0.0, 0.5, 1.0])
    grids = [np.array([90.0, 100.0, 110.0])] * 3
    values = [np.full(3, v) for v in (1.0, 2.0, 3.0)]
    lev = LeverageFunction(times, grids, values, cap=(0.1, 10.0))

    assert lev(0.0, np.array([100.0]))[0] == pytest.approx(1.0)
    assert lev(0.4, np.array([100.0]))[0] == pytest.approx(1.0)
    assert lev(0.5, np.array([100.0]))[0] == pytest.approx(2.0)
    assert lev(0.9, np.array([100.0]))[0] == pytest.approx(2.0)
    assert lev(1.0, np.array([100.0]))[0] == pytest.approx(3.0)
    assert lev(99.0, np.array([100.0]))[0] == pytest.approx(3.0)   # clamps to last
    assert lev(-1.0, np.array([100.0]))[0] == pytest.approx(1.0)   # clamps to first


# ── reproducibility and integration ────────────────────────────────────────────

def test_reproducible(heston_params, leverage):
    """Same seed, identical paths; different seed, different paths."""
    a = LSVModel(heston_params, leverage, SPOT, RATE, DIV_YIELD, seed=7)
    b = LSVModel(heston_params, leverage, SPOT, RATE, DIV_YIELD, seed=7)
    c = LSVModel(heston_params, leverage, SPOT, RATE, DIV_YIELD, seed=8)

    pa = a.simulate(2_000, [1.0, 2.0], antithetic=True)
    pb = b.simulate(2_000, [1.0, 2.0], antithetic=True)
    pc = c.simulate(2_000, [1.0, 2.0], antithetic=True)

    np.testing.assert_array_equal(pa, pb)
    assert not np.allclose(pa, pc)


def test_calibration_reproducible(surface, heston_params):
    """The particle calibration itself must be deterministic given a seed."""
    a = calibrate_leverage(surface, heston_params, 1.0, n_particles=4_000, seed=3)
    b = calibrate_leverage(surface, heston_params, 1.0, n_particles=4_000, seed=3)
    for va, vb in zip(a.values, b.values):
        np.testing.assert_array_equal(va, vb)


def test_risk_neutral_drift(heston_params, leverage):
    """E[S(T)/S(0)] = exp((r-q)T) -- the leverage factor must not bias the drift."""
    model = LSVModel(heston_params, leverage, SPOT, RATE, DIV_YIELD, seed=SEED)
    T = 1.0
    perf = model.simulate(N_PATHS, [T], antithetic=True)[:, 0]
    expected = np.exp((RATE - DIV_YIELD) * T)
    assert abs(float(perf.mean()) - expected) < 0.005


def test_prices_autocallable_between_local_vol_and_heston(surface, heston_params, leverage):
    """
    Integration check through the full pricing stack. LSV shares Heston's
    dynamics and LocalVol's surface fit, so its price on a real note should
    land between the two. Not guaranteed by theory -- a violation is a smell
    worth surfacing rather than a proof of error, hence the generous margin.
    """
    note = AutocallableNote(
        spot=SPOT, maturity=T_MAX, observation_dates=[1.0, T_MAX],
        autocall_barriers=1.0, coupon_rate=0.07,
        capital_barrier=0.60, discount_rate=RATE,
    )
    n = 40_000
    p_lsv = MCPricer(LSVModel(heston_params, leverage, SPOT, RATE, DIV_YIELD, seed=SEED),
                     "LSV").price(note, n).price
    p_lv = MCPricer(LocalVolModel(surface, seed=SEED), "LV").price(note, n).price
    p_h = MCPricer(HestonModel(heston_params, SPOT, RATE, DIV_YIELD, seed=SEED),
                   "Heston").price(note, n).price

    lo, hi = min(p_lv, p_h), max(p_lv, p_h)
    margin = 0.02 * abs(hi - lo) + 0.01
    assert lo - margin <= p_lsv <= hi + margin, (
        f"LSV {p_lsv:.5f} outside [LV {p_lv:.5f}, Heston {p_h:.5f}] by more than margin"
    )
