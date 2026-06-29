"""
Deterministic tests for ImpliedVolSurface and its helper methods.
"""
import numpy as np
import pytest
from calibration.vol_surface import flat_surface, skewed_surface, ImpliedVolSurface


# ── flat_surface baseline ──────────────────────────────────────────────────────

def make_flat(vol=0.20, spot=100.0, rate=0.03):
    return flat_surface(spot=spot, vol=vol, rate=rate, div_yield=0.0, T_max=3.0,
                        n_T=20, n_K=30)


def test_flat_surface_implied_vol_at_grid_points():
    """implied_vol should return the input vol everywhere on a flat surface."""
    sigma = 0.20
    surf = make_flat(vol=sigma)
    for T in [0.5, 1.0, 2.0]:
        for K in [85.0, 100.0, 115.0]:
            iv = surf.implied_vol(T, K)
            assert abs(iv - sigma) < 1e-4, f"T={T} K={K}: expected {sigma}, got {iv}"


def test_flat_surface_local_vol_approx_input_vol():
    """
    Dupire local vol of a flat surface should equal the input vol.
    We allow a small tolerance for finite-difference artefacts in the Dupire formula.
    """
    sigma = 0.20
    surf = make_flat(vol=sigma)
    for T in [0.5, 1.0, 2.0]:
        for K in [85.0, 100.0, 115.0]:
            lv = surf.local_vol(T, K)
            assert abs(lv - sigma) < 0.005, f"T={T} K={K}: lv={lv:.4f}, expected≈{sigma}"


# ── with_spot ─────────────────────────────────────────────────────────────────

def test_with_spot_updates_spot_attribute():
    surf = make_flat()
    bumped = surf.with_spot(110.0)
    assert bumped.spot == 110.0
    assert surf.spot == 100.0     # original unchanged


def test_with_spot_preserves_raw_vol_grid():
    """The raw _vols array should be identical after with_spot."""
    surf = make_flat(vol=0.25)
    bumped = surf.with_spot(95.0)
    np.testing.assert_array_equal(surf._vols, bumped._vols)


def test_with_spot_preserves_rate_and_div_yield():
    surf = make_flat(rate=0.03)
    bumped = surf.with_spot(110.0)
    assert bumped.rate == surf.rate
    assert bumped.div_yield == surf.div_yield


def test_with_spot_flat_surface_vol_unchanged():
    """For a flat surface, implied_vol at any (T,K) is the same after spot shift."""
    sigma = 0.20
    surf = make_flat(vol=sigma)
    bumped = surf.with_spot(110.0)
    for T in [0.5, 1.5]:
        for K in [90.0, 110.0]:
            assert abs(bumped.implied_vol(T, K) - sigma) < 1e-4


# ── with_vol_shift ────────────────────────────────────────────────────────────

def test_with_vol_shift_increases_all_vols():
    """implied_vol at every grid point should rise by exactly dvol (flat surface)."""
    surf = make_flat(vol=0.20)
    dvol = 0.01
    bumped = surf.with_vol_shift(dvol)
    for T in [0.5, 1.0, 2.0]:
        for K in [85.0, 100.0, 115.0]:
            iv_base = surf.implied_vol(T, K)
            iv_bump = bumped.implied_vol(T, K)
            assert abs(iv_bump - iv_base - dvol) < 1e-4, \
                f"T={T} K={K}: shift={iv_bump-iv_base:.6f}, expected {dvol}"


def test_with_vol_shift_negative():
    """Negative shift lowers all vols; clipped to 1e-4 minimum."""
    surf = make_flat(vol=0.20)
    bumped = surf.with_vol_shift(-0.01)
    for T in [0.5, 1.5]:
        iv = bumped.implied_vol(T, 100.0)
        assert abs(iv - 0.19) < 1e-4


def test_with_vol_shift_preserves_spot():
    surf = make_flat()
    bumped = surf.with_vol_shift(0.02)
    assert bumped.spot == surf.spot


def test_with_vol_shift_does_not_modify_original():
    """The original surface should be unchanged after with_vol_shift."""
    sigma = 0.20
    surf = make_flat(vol=sigma)
    _ = surf.with_vol_shift(0.05)
    assert abs(surf.implied_vol(1.0, 100.0) - sigma) < 1e-4


# ── local_vol_batch vs scalar ─────────────────────────────────────────────────

def test_local_vol_batch_matches_scalar_flat():
    """local_vol_batch must return the same values as local_vol() element-wise."""
    surf = make_flat(vol=0.20)
    T = 1.0
    K_arr = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    batch = surf.local_vol_batch(T, K_arr)
    scalar = np.array([surf.local_vol(T, K) for K in K_arr])
    np.testing.assert_allclose(batch, scalar, rtol=1e-10)


def test_local_vol_batch_matches_scalar_skewed():
    """Same check on a skewed surface (non-trivial Dupire calculation)."""
    surf = skewed_surface(spot=100.0, atm_vol=0.25, skew=-0.10,
                          rate=0.03, div_yield=0.0, T_max=3.0)
    T = 1.0
    K_arr = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    batch = surf.local_vol_batch(T, K_arr)
    scalar = np.array([surf.local_vol(T, K) for K in K_arr])
    np.testing.assert_allclose(batch, scalar, rtol=1e-10)


def test_local_vol_batch_returns_positive_values():
    surf = skewed_surface(spot=100.0, atm_vol=0.25, skew=-0.10,
                          rate=0.03, div_yield=0.0, T_max=3.0)
    K_arr = np.linspace(75.0, 130.0, 20)
    for T in [0.5, 1.0, 2.0]:
        lv = surf.local_vol_batch(T, K_arr)
        assert np.all(lv > 0), f"T={T}: non-positive local vol found"


# ── combined bump ─────────────────────────────────────────────────────────────

def test_spot_and_vol_shift_commute():
    """with_spot(S).with_vol_shift(dv) and with_vol_shift(dv).with_spot(S) give same vols."""
    surf = make_flat(vol=0.20)
    path_a = surf.with_spot(110.0).with_vol_shift(0.02)
    path_b = surf.with_vol_shift(0.02).with_spot(110.0)
    for T in [0.5, 1.5]:
        for K in [95.0, 110.0]:
            assert abs(path_a.implied_vol(T, K) - path_b.implied_vol(T, K)) < 1e-6
