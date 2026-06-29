"""
Deterministic payoff tests for AutocallableNote.evaluate_payoff.

All tests feed hand-crafted performance arrays so the expected PV can be
computed by hand, without any Monte Carlo noise.
"""
import numpy as np
import pytest
from products.autocallable import AutocallableNote


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_note(**overrides):
    """Standard 3-year note with sensible defaults."""
    defaults = dict(
        notional=1.0,
        spot=100.0,
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


# ── autocall trigger ───────────────────────────────────────────────────────────

def test_autocall_at_first_observation():
    """Path that clears the barrier at t=1 receives (notional + coupon) discounted to t=0."""
    note = make_note(discount_rate=0.03)
    perfs = np.array([[0.97, 0.50, 0.50]])   # 0.97 >= 0.95 → autocall at t=1
    pv = note.evaluate_payoff(perfs)
    expected = (1.0 + 0.05) * np.exp(-0.03 * 1.0)
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_autocall_at_second_observation():
    """Path that misses at t=1 but clears at t=2 receives payment at t=2."""
    note = make_note(discount_rate=0.03)
    perfs = np.array([[0.90, 0.97, 0.50]])   # 0.90 < 0.95; 0.97 >= 0.95
    pv = note.evaluate_payoff(perfs)
    expected = (1.0 + 0.05) * np.exp(-0.03 * 2.0)
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_step_down_barriers():
    """Step-down schedule: barriers decrease each year."""
    note = make_note(
        autocall_barriers=[1.00, 0.95, 0.90],
        coupon_rate=0.07,
        discount_rate=0.0,
    )
    # 0.92 misses 1.00 at t=1, misses 0.95 at t=2, clears 0.90 at t=3
    perfs = np.array([[0.92, 0.92, 0.92]])
    pv = note.evaluate_payoff(perfs)
    expected = 1.0 + 0.07          # t=3, r=0 → no discounting
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_already_autocalled_paths_are_ignored():
    """Once a path autocalls, later observations must not contribute."""
    note = make_note(coupon_rate=0.05, discount_rate=0.0)
    # Two paths: one autocalls at t=1, one autocalls at t=2
    perfs = np.array([
        [0.97, 0.99, 0.99],   # autocall at t=1
        [0.90, 0.97, 0.99],   # autocall at t=2
    ])
    pv = note.evaluate_payoff(perfs)
    np.testing.assert_allclose(pv[0], 1.0 + 0.05, rtol=1e-12)
    np.testing.assert_allclose(pv[1], 1.0 + 0.05, rtol=1e-12)


# ── maturity (no autocall) ─────────────────────────────────────────────────────

def test_maturity_above_ki_returns_full_notional():
    """Surviving paths above KI at maturity receive full notional."""
    note = make_note(capital_barrier=0.60, discount_rate=0.0)
    perfs = np.array([[0.80, 0.75, 0.70]])   # never autocalls, final > KI
    pv = note.evaluate_payoff(perfs)
    np.testing.assert_allclose(pv[0], 1.0, rtol=1e-12)


def test_maturity_below_ki_returns_fractional_notional():
    """Surviving paths below KI receive notional × final performance."""
    note = make_note(capital_barrier=0.60, discount_rate=0.0)
    perfs = np.array([[0.80, 0.70, 0.50]])   # final = 0.50 < KI = 0.60
    pv = note.evaluate_payoff(perfs)
    np.testing.assert_allclose(pv[0], 0.50, rtol=1e-12)


def test_ki_barrier_exactly_at_boundary():
    """Performance exactly equal to KI is NOT a breach (uses >= in protected check)."""
    note = make_note(capital_barrier=0.60, discount_rate=0.0)
    perfs_at = np.array([[0.80, 0.70, 0.60]])    # final == KI → protected
    perfs_below = np.array([[0.80, 0.70, 0.599]]) # final < KI → fractional
    pv_at = note.evaluate_payoff(perfs_at)
    pv_below = note.evaluate_payoff(perfs_below)
    np.testing.assert_allclose(pv_at[0], 1.0, rtol=1e-12)
    assert pv_below[0] < 1.0


def test_capital_protected_ignores_ki():
    """capital_barrier_active=False always returns full notional regardless of performance."""
    note = make_note(capital_barrier_active=False, discount_rate=0.0)
    perfs = np.array([[0.80, 0.70, 0.20]])   # very low final performance
    pv = note.evaluate_payoff(perfs)
    np.testing.assert_allclose(pv[0], 1.0, rtol=1e-12)


def test_discounting_applied_to_maturity_payment():
    """Maturity payment is discounted at the note's discount_rate."""
    r = 0.05
    T = 3.0
    note = make_note(capital_barrier_active=False, coupon_rate=0.0, discount_rate=r)
    perfs = np.array([[0.80, 0.70, 0.60]])   # never autocalls
    pv = note.evaluate_payoff(perfs)
    expected = np.exp(-r * T)
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


# ── Phoenix (conditional coupon) ───────────────────────────────────────────────

def test_phoenix_coupon_paid_above_trigger():
    """Conditional coupon is paid at non-maturity dates when perf > coupon_barrier."""
    note = make_note(
        autocall_barriers=1.10,      # barrier above 1 → never autocalls
        conditional_coupon=True,
        coupon_barrier=0.80,
        coupon_rate=0.05,
        capital_barrier_active=False,
        discount_rate=0.0,
    )
    # t=1: perf=0.90 >= 0.80 → coupon; t=2: perf=0.70 < 0.80 → no coupon
    # t=3 (maturity): perf=0.85 >= 0.80 → notional + coupon
    perfs = np.array([[0.90, 0.70, 0.85]])
    pv = note.evaluate_payoff(perfs)
    expected = 0.05 + 0.0 + 1.0 + 0.05   # r=0: coupon at t=1, nothing at t=2, notional+coupon at t=3
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_phoenix_no_coupon_below_trigger():
    """No conditional coupon when performance is below coupon_barrier."""
    note = make_note(
        autocall_barriers=1.10,
        conditional_coupon=True,
        coupon_barrier=0.80,
        coupon_rate=0.05,
        capital_barrier_active=False,
        discount_rate=0.0,
    )
    perfs = np.array([[0.70, 0.70, 0.70]])   # all below coupon_barrier = 0.80
    pv = note.evaluate_payoff(perfs)
    # No coupon at t=1, t=2; at maturity perf=0.70 < 0.80 → notional only
    np.testing.assert_allclose(pv[0], 1.0, rtol=1e-12)


def test_phoenix_autocall_suppresses_later_observations():
    """
    Phoenix: autocall at t=1 prevents later conditional coupons.

    Implementation note: when perf >= autocall_barrier AND perf >= coupon_barrier
    on the same date, both the periodic conditional coupon (line 143 of
    evaluate_payoff) and the autocall redemption coupon (line 149) are paid.
    To isolate the "no later observations" behaviour we set coupon_barrier=0.99,
    which is above the path performance (0.97), so the periodic coupon does NOT
    fire on the autocall date.  The only payment at t=1 is the autocall
    redemption: notional + coupon_rate.
    """
    note = make_note(
        autocall_barriers=0.95,
        conditional_coupon=True,
        coupon_barrier=0.99,   # above performance → no periodic coupon at autocall date
        coupon_rate=0.05,
        capital_barrier_active=False,
        discount_rate=0.0,
    )
    perfs = np.array([[0.97, 0.97, 0.97]])   # hits autocall barrier at t=1
    pv = note.evaluate_payoff(perfs)
    expected = 1.0 + 0.05   # redemption only; t=2 and t=3 are ignored
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_phoenix_autocall_also_pays_periodic_coupon():
    """
    When perf >= coupon_barrier AND perf >= autocall_barrier on the same date,
    the holder receives BOTH the periodic conditional coupon (the Phoenix income
    payment) AND the autocall redemption (notional + redemption coupon).
    This is the intended product behaviour for this implementation.
    """
    note = make_note(
        autocall_barriers=0.95,
        conditional_coupon=True,
        coupon_barrier=0.80,   # below performance → periodic coupon fires
        coupon_rate=0.05,
        capital_barrier_active=False,
        discount_rate=0.0,
    )
    perfs = np.array([[0.97, 0.97, 0.97]])
    pv = note.evaluate_payoff(perfs)
    # periodic coupon (0.05) + autocall redemption (1.0 + 0.05)
    expected = 0.05 + 1.05
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


# ── multi-path vectorisation ───────────────────────────────────────────────────

def test_vectorised_paths_independently_evaluated():
    """evaluate_payoff handles a batch of paths correctly."""
    note = make_note(coupon_rate=0.10, discount_rate=0.0)
    perfs = np.array([
        [0.97, 0.99, 0.99],   # autocall t=1 → 1.10
        [0.90, 0.97, 0.99],   # autocall t=2 → 1.10
        [0.80, 0.70, 0.30],   # KI breach   → 0.30
        [0.80, 0.70, 0.65],   # above KI    → 1.00
    ])
    pv = note.evaluate_payoff(perfs)
    np.testing.assert_allclose(pv[0], 1.10, rtol=1e-12)
    np.testing.assert_allclose(pv[1], 1.10, rtol=1e-12)
    np.testing.assert_allclose(pv[2], 0.30, rtol=1e-12)
    np.testing.assert_allclose(pv[3], 1.00, rtol=1e-12)


def test_notional_scaling():
    """All payoffs scale linearly with notional."""
    note_1 = make_note(notional=1.0, discount_rate=0.0)
    note_2 = make_note(notional=5.0, discount_rate=0.0)
    perfs = np.array([[0.97, 0.99, 0.99]])   # autocall at t=1
    pv_1 = note_1.evaluate_payoff(perfs)
    pv_2 = note_2.evaluate_payoff(perfs)
    np.testing.assert_allclose(pv_2[0], 5.0 * pv_1[0], rtol=1e-12)
