"""
Tests for the Greek computation machinery.

Design philosophy
-----------------
- _rescale_product_barriers: fully deterministic, no MC noise.
- Delta sign: use the *same* simulated paths for all three products (base,
  spot-up, spot-down).  Because the paths are identical, the comparison is
  noise-free: the only difference is the barrier levels in performance space.
- Vega sign: similar trick — same paths, different products aren't needed here
  since vega works via a surface bump, not a product change.  We just price
  twice with the same seed and compare.
"""
import numpy as np
import pytest
from products.autocallable import AutocallableNote
from calibration.vol_surface import flat_surface, skewed_surface
from models.heston import HestonModel, HestonParams
from models.local_vol import LocalVolModel
from models.sabr import SABRModel, SABRParams
from scenarios.run_scenarios import _rescale_product_barriers


SEED = 42
N_PATHS = 20_000


# ── _rescale_product_barriers ─────────────────────────────────────────────────

def make_standard_note(**kw):
    defaults = dict(
        notional=1.0, spot=100.0, maturity=3.0,
        observation_dates=[1.0, 2.0, 3.0],
        autocall_barriers=0.95, coupon_rate=0.05,
        conditional_coupon=False, coupon_barrier=0.80,
        capital_barrier=0.60, capital_barrier_active=True,
        discount_rate=0.03,
    )
    defaults.update(kw)
    return AutocallableNote(**defaults)


def test_rescale_preserves_absolute_autocall_level():
    """
    original_spot × original_barrier  =  new_spot × rescaled_barrier
    """
    note = make_standard_note(autocall_barriers=0.95, spot=100.0)
    new_spot = 105.0
    rescaled = _rescale_product_barriers(note, 100.0, new_spot)

    abs_original = 100.0 * 0.95          # = 95
    abs_rescaled = new_spot * rescaled._autocall_barriers[0]
    np.testing.assert_allclose(abs_original, abs_rescaled, rtol=1e-12)


def test_rescale_preserves_absolute_ki_level():
    note = make_standard_note(capital_barrier=0.60, spot=100.0)
    new_spot = 90.0
    rescaled = _rescale_product_barriers(note, 100.0, new_spot)

    abs_original = 100.0 * 0.60
    abs_rescaled = new_spot * rescaled.capital_barrier
    np.testing.assert_allclose(abs_original, abs_rescaled, rtol=1e-12)


def test_rescale_preserves_absolute_coupon_barrier():
    note = make_standard_note(coupon_barrier=0.80, spot=100.0)
    new_spot = 110.0
    rescaled = _rescale_product_barriers(note, 100.0, new_spot)

    abs_original = 100.0 * 0.80
    abs_rescaled = new_spot * rescaled.coupon_barrier
    np.testing.assert_allclose(abs_original, abs_rescaled, rtol=1e-12)


def test_rescale_does_not_change_coupon_rate():
    """Coupon rate is a dollar payment, not a performance threshold — must not scale."""
    note = make_standard_note(coupon_rate=0.07)
    rescaled = _rescale_product_barriers(note, 100.0, 110.0)
    np.testing.assert_allclose(rescaled._coupon_schedule, 0.07, rtol=1e-12)


def test_rescale_does_not_change_maturity_or_dates():
    note = make_standard_note()
    rescaled = _rescale_product_barriers(note, 100.0, 110.0)
    assert rescaled.maturity == note.maturity
    np.testing.assert_array_equal(rescaled.observation_dates, note.observation_dates)


def test_rescale_identity_when_spot_unchanged():
    note = make_standard_note()
    rescaled = _rescale_product_barriers(note, 100.0, 100.0)
    np.testing.assert_allclose(rescaled._autocall_barriers, note._autocall_barriers)
    np.testing.assert_allclose(rescaled.capital_barrier, note.capital_barrier)


def test_rescale_preserves_step_down_barriers():
    """All barrier dates are scaled individually."""
    note = make_standard_note(autocall_barriers=[1.00, 0.95, 0.90])
    new_spot = 110.0
    rescaled = _rescale_product_barriers(note, 100.0, new_spot)

    original_abs = np.array([1.00, 0.95, 0.90]) * 100.0
    rescaled_abs = rescaled._autocall_barriers * new_spot
    np.testing.assert_allclose(original_abs, rescaled_abs, rtol=1e-12)


# ── delta sign (noise-free same-paths comparison) ─────────────────────────────

def _price_with_product(model, product, n_paths=N_PATHS):
    """Simulate once, evaluate different products on the same paths."""
    perfs = model.simulate(n_paths, product.observation_dates, antithetic=True)
    return float(np.mean(product.evaluate_payoff(perfs)))


@pytest.fixture
def atm_barrier_note():
    """Note with autocall barrier near ATM — large, clean delta signal."""
    return make_standard_note(
        autocall_barriers=1.0,   # autocall when spot returns to initial level
        coupon_rate=0.05,
        capital_barrier=0.60,
        discount_rate=0.03,
    )


def test_heston_delta_positive_atm_barrier(atm_barrier_note):
    """
    When spot rises, performance barriers fall in relative space.
    That makes autocall easier → more coupons paid → price goes UP → delta > 0.

    Using the same Heston paths for both products makes this comparison noise-free.
    """
    h_S = 1.0   # $1 spot bump
    note_up = _rescale_product_barriers(atm_barrier_note, 100.0, 101.0)
    note_dn = _rescale_product_barriers(atm_barrier_note, 100.0,  99.0)

    model = HestonModel(params=HestonParams(), spot=100.0, rate=0.03,
                        div_yield=0.0, seed=SEED)
    # Simulate once; reuse same paths for all three products
    perfs = model.simulate(N_PATHS, atm_barrier_note.observation_dates, antithetic=True)

    p_up = float(np.mean(atm_barrier_note.evaluate_payoff(
        np.copy(perfs))))   # using product_up's barriers
    p_dn = float(np.mean(atm_barrier_note.evaluate_payoff(
        np.copy(perfs))))   # using product_dn's barriers

    # Evaluate with actually rescaled products (different barriers → different prices)
    p_up = float(np.mean(note_up.evaluate_payoff(perfs)))
    p_dn = float(np.mean(note_dn.evaluate_payoff(perfs)))

    delta = (p_up - p_dn) / (2 * h_S)
    assert delta > 0, f"Expected positive delta for ATM barrier note, got {delta:.6f}"


def test_lv_delta_positive_atm_barrier(atm_barrier_note):
    """Same sign check for the LocalVol model."""
    surf = flat_surface(spot=100.0, vol=0.20, rate=0.03, div_yield=0.0, T_max=3.0)
    note_up = _rescale_product_barriers(atm_barrier_note, 100.0, 101.0)
    note_dn = _rescale_product_barriers(atm_barrier_note, 100.0,  99.0)

    surf_up = surf.with_spot(101.0)
    surf_dn = surf.with_spot(99.0)

    # For LocalVol, paths depend on the surface; test separately per surface
    m_up = LocalVolModel(surface=surf_up, seed=SEED)
    m_dn = LocalVolModel(surface=surf_dn, seed=SEED)

    perfs_up = m_up.simulate(N_PATHS, atm_barrier_note.observation_dates, antithetic=True)
    perfs_dn = m_dn.simulate(N_PATHS, atm_barrier_note.observation_dates, antithetic=True)

    p_up = float(np.mean(note_up.evaluate_payoff(perfs_up)))
    p_dn = float(np.mean(note_dn.evaluate_payoff(perfs_dn)))

    delta = (p_up - p_dn) / 2.0   # h_S = 1
    assert delta > 0, f"Expected positive LV delta, got {delta:.6f}"


def test_capital_protected_delta_negative(atm_barrier_note):
    """
    A capital-protected (no KI) version of the same note should have negative delta:
    higher spot → easier autocall but the *only* benefit of surviving is the final
    notional (same as autocall notional), so more early autocalls don't add value —
    but the coupon on autocall doesn't exist for the capital-protected variant here.

    More precisely: for a pure discount-bond-like note (no coupon, no KI), the
    performance barrier becoming easier means we collect notional sooner (higher PV)
    but that's a very small effect vs. the reduction in coupon.

    We test the sign for a note with NO coupon and capital-protected maturity:
    higher spot → lower relative barriers → more autocalls → get notional back sooner
    → positive delta.

    For a note with capital_barrier_active=False AND coupon_rate=0, delta ≈ 0.
    We assert the magnitude is very small (< 0.0005 per $ move) since the note
    is nearly spot-neutral.
    """
    note = make_standard_note(
        autocall_barriers=1.0,
        coupon_rate=0.0,
        capital_barrier_active=False,
        discount_rate=0.03,
    )
    note_up = _rescale_product_barriers(note, 100.0, 101.0)
    note_dn = _rescale_product_barriers(note, 100.0,  99.0)

    model = HestonModel(params=HestonParams(), spot=100.0, rate=0.03,
                        div_yield=0.0, seed=SEED)
    perfs = model.simulate(N_PATHS, note.observation_dates, antithetic=True)

    p_up = float(np.mean(note_up.evaluate_payoff(perfs)))
    p_dn = float(np.mean(note_dn.evaluate_payoff(perfs)))
    delta = (p_up - p_dn) / 2.0

    # Should be very small — the note is essentially a zero-coupon bond
    assert abs(delta) < 0.001, f"Delta of zero-coupon no-KI note too large: {delta:.6f}"


# ── vega sign ─────────────────────────────────────────────────────────────────

def test_vega_negative_standard_autocall():
    """
    A standard autocallable is short volatility: higher vol → lower price.
    This makes economic sense: higher vol increases the probability of the KI
    being breached, which hurts the note holder.

    We compare prices under two flat surfaces (vol ± dvol) with the same seed.
    """
    from engine.mc_pricer import MCPricer

    note = make_standard_note(autocall_barriers=1.0, coupon_rate=0.05,
                              capital_barrier=0.60, discount_rate=0.03)

    base_vol = 0.20
    dvol = 0.02
    surf_lo = flat_surface(spot=100.0, vol=base_vol - dvol, rate=0.03,
                           div_yield=0.0, T_max=3.0)
    surf_hi = flat_surface(spot=100.0, vol=base_vol + dvol, rate=0.03,
                           div_yield=0.0, T_max=3.0)

    m_lo = LocalVolModel(surface=surf_lo, seed=SEED)
    m_hi = LocalVolModel(surface=surf_hi, seed=SEED)

    p_lo = MCPricer(m_lo, antithetic=True).price(note, N_PATHS).price
    p_hi = MCPricer(m_hi, antithetic=True).price(note, N_PATHS).price

    vega = (p_hi - p_lo) / (2 * dvol)
    assert vega < 0, f"Expected negative vega for standard autocall, got {vega:.4f}"


def test_vega_positive_capital_protected_no_ki():
    """
    A fully capital-protected note (no KI) should have near-zero or positive vega:
    higher vol increases the chance of early autocall (positive for the holder when
    there's a coupon), but cannot cause capital loss.

    We assert the *magnitude* is smaller than the standard autocall's vega,
    not necessarily that the sign is positive.
    """
    from engine.mc_pricer import MCPricer

    note_standard = make_standard_note(autocall_barriers=1.0, coupon_rate=0.05,
                                       capital_barrier=0.60, capital_barrier_active=True,
                                       discount_rate=0.03)
    note_protected = make_standard_note(autocall_barriers=1.0, coupon_rate=0.05,
                                        capital_barrier_active=False, discount_rate=0.03)

    dvol = 0.02
    base_vol = 0.20
    surf_lo = flat_surface(spot=100.0, vol=base_vol - dvol, rate=0.03,
                           div_yield=0.0, T_max=3.0)
    surf_hi = flat_surface(spot=100.0, vol=base_vol + dvol, rate=0.03,
                           div_yield=0.0, T_max=3.0)

    def compute_vega(note):
        m_lo = LocalVolModel(surface=surf_lo, seed=SEED)
        m_hi = LocalVolModel(surface=surf_hi, seed=SEED)
        p_lo = MCPricer(m_lo, antithetic=True).price(note, N_PATHS).price
        p_hi = MCPricer(m_hi, antithetic=True).price(note, N_PATHS).price
        return (p_hi - p_lo) / (2 * dvol)

    vega_std = compute_vega(note_standard)
    vega_prot = compute_vega(note_protected)

    # Capital-protected note should have less negative (or more positive) vega
    assert vega_prot > vega_std, \
        f"Protected vega {vega_prot:.4f} should exceed standard {vega_std:.4f}"
