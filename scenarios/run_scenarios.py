"""
Scenario runner: prices the autocallable under each model for every scenario
defined in scenarios.yaml and prints a summary comparison table.

Run with:
    .venv/bin/python scenarios/run_scenarios.py
    .venv/bin/python scenarios/run_scenarios.py --scenarios scenarios/scenarios.yaml
    .venv/bin/python scenarios/run_scenarios.py --n-paths 50000 --output results.csv
    .venv/bin/python scenarios/run_scenarios.py --greeks --output results.csv
    .venv/bin/python scenarios/run_scenarios.py --greeks --n-paths-greeks 10000
"""

from __future__ import annotations

import sys
import os
import argparse
import copy
import csv
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import numpy as np

from calibration.vol_surface import skewed_surface, flat_surface, term_structure_surface
from calibration.calibrators import calibrate_heston, calibrate_sabr
from models.local_vol import LocalVolModel
from models.heston import HestonModel, HestonParams
from models.sabr import SABRModel, SABRParams
from models.basket import BasketLocalVolModel, BasketHestonModel, BasketHestonAsset
from products.autocallable import AutocallableNote
from engine.mc_pricer import MCPricer


# ── Helpers ───────────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a deep copy of `base`."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def build_surface(cfg: dict, spot: float, rate: float, div_yield: float):
    vc = cfg["vol_surface"]
    kind = vc.get("type", "skewed")
    T_max = vc.get("T_max", 5.0)
    if kind == "flat":
        return flat_surface(spot=spot, vol=vc["atm_vol"], rate=rate,
                            div_yield=div_yield, T_max=T_max)
    if kind == "term_structure":
        lo = vc.get("moneyness_lo")
        hi = vc.get("moneyness_hi")
        m_range = (float(lo), float(hi)) if (lo is not None and hi is not None) else None
        return term_structure_surface(
            spot=spot, rate=rate, div_yield=div_yield, T_max=T_max,
            vol_short=vc.get("vol_short", 0.25),
            vol_long=vc.get("vol_long", 0.18),
            kappa=vc.get("kappa", 1.5),
            skew=vc.get("skew", -0.10),
            convexity=vc.get("convexity", 0.02),
            n_T=int(vc.get("n_T", 30)),
            n_K=int(vc.get("n_K", 60)),
            moneyness_range=m_range,
        )
    return skewed_surface(spot=spot, atm_vol=vc["atm_vol"], skew=vc.get("skew", 0.0),
                          rate=rate, div_yield=div_yield, T_max=T_max)


def calibrate_all(
    cfg: dict,
    surface,
    verbose: bool = False,
    fast: bool = False,
    warm_start: dict | None = None,
) -> dict:
    """
    Calibrate Heston and SABR on `surface`.
    Returns {model_name: params} for each enabled SV model.
    LocalVol needs no calibration (reads Dupire vols directly from surface).

    fast       : use relaxed convergence tolerances (for Greek FD bumps)
    warm_start : {model_name: params} to use as starting point (speeds up fast mode)
    """
    models_cfg = cfg.get("models", {})
    cal: dict = {}

    if models_cfg.get("heston", True):
        init = warm_start.get("Heston") if warm_start else None
        try:
            params = calibrate_heston(surface, initial_params=init, fast=fast)
            if verbose:
                print(f"    Heston: kappa={params.kappa:.3f} theta={params.theta:.4f}"
                      f" xi={params.xi:.3f} rho={params.rho:.3f}")
        except Exception as e:
            if verbose:
                print(f"    Heston calibration failed ({e}), using defaults")
            params = HestonParams() if init is None else init
        cal["Heston"] = params

    if models_cfg.get("sabr", True):
        try:
            params = calibrate_sabr(surface, beta=1.0, calibration_maturity=1.0)
            if verbose:
                print(f"    SABR:   alpha={params.alpha:.4f} rho={params.rho:.3f}"
                      f" nu={params.nu:.3f}")
        except Exception as e:
            if verbose:
                print(f"    SABR calibration failed ({e}), using defaults")
            params = SABRParams() if not warm_start else warm_start.get("SABR", SABRParams())
        cal["SABR"] = params

    return cal


def build_pricers(cfg: dict, surface, spot: float, rate: float,
                  div_yield: float, seed: int, antithetic: bool,
                  calibrated_params: dict | None = None) -> list[MCPricer]:
    """
    Build one MCPricer per enabled model.

    calibrated_params : pre-calibrated {model_name: params} from calibrate_all().
        If provided, SV models skip re-calibration (important for CRN Greek bumps).
        If None, calibration happens internally (backward-compatible path).
    """
    models_cfg = cfg.get("models", {})
    pricers = []

    if models_cfg.get("local_vol", True):
        m = LocalVolModel(surface=surface, steps_per_year=52, seed=seed)
        pricers.append(MCPricer(m, "Local Vol", antithetic))

    if models_cfg.get("heston", True):
        if calibrated_params and "Heston" in calibrated_params:
            params = calibrated_params["Heston"]
        else:
            try:
                params = calibrate_heston(surface)
            except Exception:
                params = HestonParams()
        m = HestonModel(params=params, spot=spot, rate=rate, div_yield=div_yield,
                        steps_per_year=52, seed=seed)
        pricers.append(MCPricer(m, "Heston", antithetic))

    if models_cfg.get("sabr", True):
        if calibrated_params and "SABR" in calibrated_params:
            params = calibrated_params["SABR"]
        else:
            try:
                params = calibrate_sabr(surface, beta=1.0, calibration_maturity=1.0)
            except Exception:
                params = SABRParams()
        m = SABRModel(params=params, spot=spot, rate=rate, div_yield=div_yield,
                      steps_per_year=52, seed=seed)
        pricers.append(MCPricer(m, "SABR", antithetic))

    return pricers


# ── Basket helpers ──────────────────────────────────────────────────────────────

def build_basket_surfaces(cfg: dict) -> list[tuple[str, object]]:
    """Returns [(asset_name, surface), ...] built from cfg['assets']."""
    result = []
    for asset_cfg in cfg["assets"]:
        spot = float(asset_cfg["spot"])
        rate = float(asset_cfg["rate"])
        div_yield = float(asset_cfg["div_yield"])
        surf = build_surface({"vol_surface": asset_cfg["vol_surface"]}, spot, rate, div_yield)
        result.append((asset_cfg["name"], surf))
    return result


def load_correlation(cfg: dict, n_assets: int) -> np.ndarray:
    corr = np.array(cfg["correlation"], dtype=float)
    assert corr.shape == (n_assets, n_assets), \
        "correlation matrix size must match number of assets"
    return corr


def calibrate_basket_heston(
    asset_surfaces: list[tuple[str, object]],
    verbose: bool = False,
    fast: bool = False,
    warm_start: dict[str, HestonParams] | None = None,
) -> dict[str, HestonParams]:
    """Returns {asset_name: HestonParams}, one independent calibration per asset."""
    result = {}
    for name, surf in asset_surfaces:
        init = warm_start.get(name) if warm_start else None
        try:
            params = calibrate_heston(surf, initial_params=init, fast=fast)
            if verbose:
                print(f"    {name}: kappa={params.kappa:.3f} theta={params.theta:.4f}"
                      f" xi={params.xi:.3f} rho={params.rho:.3f}")
        except Exception as e:
            if verbose:
                print(f"    {name}: Heston calibration failed ({e}), using defaults")
            params = HestonParams() if init is None else init
        result[name] = params
    return result


def build_basket_pricers(
    cfg: dict,
    asset_surfaces: list[tuple[str, object]],
    correlation: np.ndarray,
    seed: int,
    antithetic: bool,
    heston_params: dict[str, HestonParams] | None = None,
) -> list[MCPricer]:
    """Builds BasketLocalVolModel / BasketHestonModel pricers, mirroring build_pricers()."""
    models_cfg = cfg.get("models", {})
    pricers = []
    surfaces = [s for _, s in asset_surfaces]

    if models_cfg.get("local_vol", True):
        m = BasketLocalVolModel(surfaces=surfaces, correlation=correlation, seed=seed)
        pricers.append(MCPricer(m, "Basket Local Vol", antithetic))

    if models_cfg.get("heston", True):
        if heston_params is None:
            heston_params = calibrate_basket_heston(asset_surfaces)
        assets = [
            BasketHestonAsset(name, heston_params[name], surf.spot, surf.rate, surf.div_yield)
            for name, surf in asset_surfaces
        ]
        m = BasketHestonModel(assets=assets, correlation=correlation, seed=seed)
        pricers.append(MCPricer(m, "Basket Heston", antithetic))

    # SABR is not supported for basket pricing (see models/basket.py).
    return pricers


def build_basket_product(cfg: dict, discount_rate: float) -> AutocallableNote:
    """Same as build_product() but spot is a fixed placeholder (unused in basket mode)."""
    pc = cfg["product"]
    return AutocallableNote(
        notional=float(pc["notional"]),
        spot=1.0,
        maturity=float(pc["maturity"]),
        observation_dates=pc["observation_dates"],
        autocall_barriers=pc["autocall_barriers"],
        coupon_rate=pc["coupon_rate"],
        conditional_coupon=bool(pc.get("conditional_coupon", False)),
        coupon_barrier=float(pc.get("coupon_barrier", 0.80)),
        capital_barrier=float(pc.get("capital_barrier", 0.60)),
        capital_barrier_active=bool(pc.get("capital_barrier_active", True)),
        discount_rate=discount_rate,
    )


def build_product(cfg: dict, spot: float, rate: float) -> AutocallableNote:
    pc = cfg["product"]
    return AutocallableNote(
        notional=float(pc["notional"]),
        spot=spot,
        maturity=float(pc["maturity"]),
        observation_dates=pc["observation_dates"],
        autocall_barriers=pc["autocall_barriers"],
        coupon_rate=pc["coupon_rate"],
        conditional_coupon=bool(pc.get("conditional_coupon", False)),
        coupon_barrier=float(pc.get("coupon_barrier", 0.80)),
        capital_barrier=float(pc.get("capital_barrier", 0.60)),
        capital_barrier_active=bool(pc.get("capital_barrier_active", True)),
        discount_rate=rate,
    )


@dataclass
class GreekResult:
    """Finite-difference Greeks for one (scenario, model) pair."""
    delta: float      # ∂P/∂S
    gamma: float      # ∂²P/∂S²
    vega:  float      # ∂P/∂σ · 0.01  (per 1 percentage-point parallel vol shift)
    vanna: float      # ∂²P / (∂S ∂σ)
    skew_sens: float  # ∂P/∂skew · 0.01 (per 0.01 shift in the skew coefficient; see with_skew_shift)


@dataclass
class BasketGreekResult:
    """Per-asset finite-difference Greeks for one (scenario, model) pair."""
    delta: dict[str, float]       # asset_name → ∂P/∂S_asset
    gamma: dict[str, float]       # asset_name → ∂²P/∂S_asset²
    vega:  dict[str, float]       # asset_name → ∂P/∂σ_asset · 0.01
    vanna: dict[str, float]       # asset_name → ∂²P/(∂S_asset ∂σ_asset)
    skew_sens: dict[str, float]   # asset_name → ∂P/∂skew_asset · 0.01


@dataclass
class ScenarioResult:
    name: str
    group: str
    description: str
    prices: dict[str, float]       # model_name → price
    std_errors: dict[str, float]   # model_name → SE
    durations: Optional[dict[str, float]] = None       # model_name → expected duration (years)
    greeks: Optional[dict[str, GreekResult]] = None   # model_name → GreekResult (single-asset)
    basket_greeks: Optional[dict[str, BasketGreekResult]] = None  # model_name → BasketGreekResult


# ── Greek computation ──────────────────────────────────────────────────────────

def _rescale_product_barriers(
    product: AutocallableNote,
    original_spot: float,
    new_spot: float,
) -> AutocallableNote:
    """
    Return a new product whose performance thresholds are rescaled so that their
    ABSOLUTE dollar levels stay fixed as the current spot moves.

    Delta measures sensitivity to today's spot while the contract terms set at
    inception (barrier levels, KI level, coupon trigger) remain unchanged in
    dollar terms.  Because the simulation returns S(t)/S0_new rather than
    S(t)/S0_original, every performance threshold must be scaled by
    S0_original / S0_new.

    Example: if the note was issued with spot=100 and an autocall barrier of
    0.95 (absolute = 95), and we bump spot to 101, the barrier in performance
    space becomes 95/101 ≈ 0.9406 — NOT the original 0.95.

    Only performance thresholds are rescaled (autocall barriers, coupon barrier,
    capital barrier).  Coupon rates are dollar payments and are not rescaled.
    """
    scale = original_spot / new_spot
    return AutocallableNote(
        notional=product.notional,
        spot=new_spot,
        maturity=product.maturity,
        observation_dates=product.observation_dates.tolist(),
        autocall_barriers=(product._autocall_barriers * scale).tolist(),
        coupon_rate=product._coupon_schedule.tolist(),
        conditional_coupon=product.conditional_coupon,
        coupon_barrier=product.coupon_barrier * scale,
        capital_barrier=product.capital_barrier * scale,
        capital_barrier_active=product.capital_barrier_active,
        discount_rate=product.discount_rate,
    )


def compute_greeks(
    cfg: dict,
    base_surface,
    base_cal: dict,
    product: AutocallableNote,
    base_prices: dict[str, float],
    SPOT: float,
    RATE: float,
    DIV_YIELD: float,
    seed: int,
    antithetic: bool,
    n_paths: int,
    h_S_pct: float = 0.01,
    h_v: float = 0.001,
    h_skew: float = 0.01,
) -> dict[str, GreekResult]:
    """
    Compute delta, gamma, vega, vanna, skew_sens via central finite differences.

    Delta convention
    ----------------
    The spot bumps shift S0 while holding all contract terms (autocall barrier
    levels, KI, coupon trigger) fixed at their ORIGINAL DOLLAR values set at
    inception.  Concretely, if the original barrier is 0.95 × 100 = 95 and we
    bump spot to 101, the performance threshold becomes 95/101 ≈ 0.9406.
    This is achieved by calling _rescale_product_barriers() for each spot bump.

    Vol bump strategy
    -----------------
    Heston and SABR are re-calibrated on the vol-shifted surface using a
    fast warm-start (relaxed tolerance) to keep runtime manageable.
    LocalVol reads the shifted surface directly — no calibration needed.

    Skew bump
    ---------
    skew_sens uses ImpliedVolSurface.with_skew_shift(), a maturity-dependent
    tilt (zero at each maturity's own ATM point, decaying as 1/sqrt(T) to
    match term_structure_surface's own `skew` parameter convention) rather
    than a flat shift — so it isolates the smile SLOPE from the level (vega)
    already captured above. Re-calibrated the same way as the vega bumps.

    Variance reduction
    ------------------
    All bumped evaluations use the same RNG seed (common random numbers),
    so noise in the FD numerators largely cancels.

    Parameters
    ----------
    h_S_pct : spot bump as a fraction of spot (default 0.01 = 1 %)
    h_v     : absolute parallel implied-vol shift (default 0.001 = 10 bp)
    h_skew  : skew-coefficient bump for with_skew_shift (default 0.01)
    vega    : reported per 1 percentage-point (0.01) parallel vol shift
    skew_sens : reported per 0.01 shift in the skew coefficient
    """
    h_S = SPOT * h_S_pct

    # ── spot-bumped products: barriers stay fixed in dollar terms ──────────────
    product_up = _rescale_product_barriers(product, SPOT, SPOT + h_S)
    product_dn = _rescale_product_barriers(product, SPOT, SPOT - h_S)

    # ── bumped surfaces ─────────────────────────────────────────────────────────
    s_up     = base_surface.with_spot(SPOT + h_S)
    s_dn     = base_surface.with_spot(SPOT - h_S)
    s_vup    = base_surface.with_vol_shift(+h_v)
    s_vdn    = base_surface.with_vol_shift(-h_v)
    s_skup   = base_surface.with_skew_shift(+h_skew)
    s_skdn   = base_surface.with_skew_shift(-h_skew)
    # Cross terms for vanna: vol shift first, then spot shift
    s_up_vup = s_vup.with_spot(SPOT + h_S)
    s_dn_vup = s_vup.with_spot(SPOT - h_S)
    s_up_vdn = s_vdn.with_spot(SPOT + h_S)
    s_dn_vdn = s_vdn.with_spot(SPOT - h_S)

    # Re-calibrate SV models for vol- and skew-bumped surfaces (fast warm-start).
    cal_vup  = calibrate_all(cfg, s_vup,  fast=True, warm_start=base_cal)
    cal_vdn  = calibrate_all(cfg, s_vdn,  fast=True, warm_start=base_cal)
    cal_skup = calibrate_all(cfg, s_skup, fast=True, warm_start=base_cal)
    cal_skdn = calibrate_all(cfg, s_skdn, fast=True, warm_start=base_cal)

    def price_all(surface, spot, cal, prod) -> dict[str, float]:
        pricers = build_pricers(cfg, surface, spot, RATE, DIV_YIELD,
                                seed, antithetic, calibrated_params=cal)
        return {p.model_name: p.price(prod, n_paths).price for p in pricers}

    # Bumped price evaluations (CRN via shared seed).
    # Spot-bumped calls use rescaled products; vol/skew-only calls use the original.
    P_up     = price_all(s_up,     SPOT + h_S, base_cal, product_up)
    P_dn     = price_all(s_dn,     SPOT - h_S, base_cal, product_dn)
    P_vup    = price_all(s_vup,    SPOT,        cal_vup,  product)
    P_vdn    = price_all(s_vdn,    SPOT,        cal_vdn,  product)
    P_skup   = price_all(s_skup,   SPOT,        cal_skup, product)
    P_skdn   = price_all(s_skdn,   SPOT,        cal_skdn, product)
    P_up_vup = price_all(s_up_vup, SPOT + h_S,  cal_vup,  product_up)
    P_dn_vup = price_all(s_dn_vup, SPOT - h_S,  cal_vup,  product_dn)
    P_up_vdn = price_all(s_up_vdn, SPOT + h_S,  cal_vdn,  product_up)
    P_dn_vdn = price_all(s_dn_vdn, SPOT - h_S,  cal_vdn,  product_dn)

    greeks: dict[str, GreekResult] = {}
    for name in P_up:
        P0    = base_prices[name]
        delta = (P_up[name]  - P_dn[name])  / (2.0 * h_S)
        gamma = (P_up[name]  - 2.0 * P0 + P_dn[name]) / (h_S ** 2)
        vega  = (P_vup[name] - P_vdn[name]) / (2.0 * h_v) * 0.01
        vanna = (  P_up_vup[name] - P_dn_vup[name]
                 - P_up_vdn[name] + P_dn_vdn[name]) / (4.0 * h_S * h_v)
        skew_sens = (P_skup[name] - P_skdn[name]) / (2.0 * h_skew) * 0.01
        greeks[name] = GreekResult(delta=delta, gamma=gamma, vega=vega, vanna=vanna,
                                   skew_sens=skew_sens)

    return greeks


def compute_basket_greeks(
    cfg: dict,
    asset_surfaces: list[tuple[str, object]],
    correlation: np.ndarray,
    base_heston: dict[str, HestonParams],
    product: AutocallableNote,
    base_prices: dict[str, float],
    seed: int,
    antithetic: bool,
    n_paths: int,
    h_S_pct: float = 0.01,
    h_v: float = 0.001,
    h_skew: float = 0.01,
) -> dict[str, BasketGreekResult]:
    """
    Per-asset delta/gamma/vega/vanna/skew_sens via central finite differences:
    bump ONE asset's spot/vol/skew at a time, holding all other assets at base.
    Cost: 10 bump points per asset (2 for delta/gamma, 2 for vega, 2 for
    skew_sens, 4 for vanna cross terms) — 10N total for N assets, vs. the flat
    10 in the single-asset compute_greeks(). Only the bumped asset's Heston
    params are recalibrated per vol/skew bump (4N total recalibrations, not
    4N × N). skew_sens uses ImpliedVolSurface.with_skew_shift() -- see
    compute_greeks()'s docstring for why this isolates slope from level (vega).

    Barrier convention — per-asset performance rescale, not barrier rescale
    ------------------------------------------------------------------------
    The single-asset compute_greeks() rescales the note's (shared, scalar)
    barriers via _rescale_product_barriers() so a spot bump doesn't silently
    change the ABSOLUTE dollar barrier level fixed at inception. A worst-of
    barrier can't be rescaled the same way: it's ONE fraction shared across
    whichever asset happens to be worst on a given date, not a per-asset
    threshold, so there's nothing to rescale per-asset on the barrier side.

    Instead, bumping asset i's ImpliedVolSurface.spot changes the reference
    S_i(0) the model normalizes asset i's own performance against (S_i(t)/
    S_i(0)_bumped). To keep the comparison against the ORIGINAL dollar
    barrier level for asset i, that asset's per-asset performance is rescaled
    by S_i(0)_bumped / S_i(0)_original BEFORE taking the worst-of min:
        S_i(t)/S_i(0)_original = [S_i(t)/S_i(0)_bumped] * [S_i(0)_bumped/S_i(0)_original]
    This is the exact mathematical equivalent of the single-asset barrier
    rescale (same correction, applied to the performance side instead of the
    barrier side, since only the performance side is asset-specific here).

    This distinction matters most for Heston/SABR-style dynamics, whose SDEs
    for S(t)/S(0) are scale-invariant (independent of the absolute spot
    level) — without this rescale, a spot bump would leave the simulated
    performance completely unchanged for those models, making delta/gamma
    trivially zero rather than just noisy. Local Vol's dynamics DO depend on
    the absolute spot level (the local vol surface is evaluated at absolute
    S), so it gets a real (if small) contribution from both channels; the
    rescale is applied uniformly to all models for consistency.
    """
    names = [n for n, _ in asset_surfaces]
    surfaces = {n: s for n, s in asset_surfaces}
    n_assets = len(names)
    print(f"           ({10 * n_assets} bump evaluations for {n_assets} assets)")

    delta = {m: {} for m in base_prices}
    gamma = {m: {} for m in base_prices}
    vega = {m: {} for m in base_prices}
    vanna = {m: {} for m in base_prices}
    skew_sens = {m: {} for m in base_prices}

    def price_all(bumped_surfaces: dict[str, object],
                  bumped_heston: dict[str, HestonParams],
                  bumped_asset_name: str | None = None,
                  scale_factor: float = 1.0) -> dict[str, float]:
        """
        Price under each basket model. If bumped_asset_name is given, that
        asset's per-asset performance is rescaled by scale_factor before
        taking the worst-of min (see docstring above); otherwise this is
        just the ordinary aggregated basket price (used for pure vol bumps,
        which don't touch any asset's spot reference and need no rescale).
        """
        ordered = [(n, bumped_surfaces[n]) for n in names]
        pricers = build_basket_pricers(cfg, ordered, correlation, seed, antithetic,
                                       heston_params=bumped_heston)
        prices = {}
        for pricer in pricers:
            if bumped_asset_name is None:
                perf = pricer.model.simulate(n_paths, product.observation_dates, antithetic=antithetic)
            else:
                per_asset = pricer.model.simulate_assets(n_paths, product.observation_dates,
                                                         antithetic=antithetic)
                a_idx = names.index(bumped_asset_name)
                per_asset = per_asset.copy()
                per_asset[:, :, a_idx] *= scale_factor
                perf = per_asset.min(axis=2)
            payoffs = product.evaluate_payoff(perf)
            prices[pricer.model_name] = float(np.mean(payoffs))
        return prices

    for asset_name in names:
        base_surf = surfaces[asset_name]
        h_S = base_surf.spot * h_S_pct
        scale_up = (base_surf.spot + h_S) / base_surf.spot
        scale_dn = (base_surf.spot - h_S) / base_surf.spot

        surf_up = {**surfaces, asset_name: base_surf.with_spot(base_surf.spot + h_S)}
        surf_dn = {**surfaces, asset_name: base_surf.with_spot(base_surf.spot - h_S)}
        P_up = price_all(surf_up, base_heston, asset_name, scale_up)
        P_dn = price_all(surf_dn, base_heston, asset_name, scale_dn)

        # Pure vol bump: no spot change, so no rescale needed.
        surf_vup = {**surfaces, asset_name: base_surf.with_vol_shift(+h_v)}
        surf_vdn = {**surfaces, asset_name: base_surf.with_vol_shift(-h_v)}
        heston_vup = {**base_heston,
                      asset_name: calibrate_heston(surf_vup[asset_name],
                                                   initial_params=base_heston.get(asset_name),
                                                   fast=True)}
        heston_vdn = {**base_heston,
                      asset_name: calibrate_heston(surf_vdn[asset_name],
                                                   initial_params=base_heston.get(asset_name),
                                                   fast=True)}
        P_vup = price_all(surf_vup, heston_vup)
        P_vdn = price_all(surf_vdn, heston_vdn)

        # Pure skew bump: no spot change, so no rescale needed (same as vega above).
        surf_skup = {**surfaces, asset_name: base_surf.with_skew_shift(+h_skew)}
        surf_skdn = {**surfaces, asset_name: base_surf.with_skew_shift(-h_skew)}
        heston_skup = {**base_heston,
                       asset_name: calibrate_heston(surf_skup[asset_name],
                                                    initial_params=base_heston.get(asset_name),
                                                    fast=True)}
        heston_skdn = {**base_heston,
                       asset_name: calibrate_heston(surf_skdn[asset_name],
                                                    initial_params=base_heston.get(asset_name),
                                                    fast=True)}
        P_skup = price_all(surf_skup, heston_skup)
        P_skdn = price_all(surf_skdn, heston_skdn)

        # Vanna cross terms combine a vol shift with a spot bump -> rescale needed.
        surf_up_vup = {**surfaces, asset_name: surf_vup[asset_name].with_spot(base_surf.spot + h_S)}
        surf_dn_vup = {**surfaces, asset_name: surf_vup[asset_name].with_spot(base_surf.spot - h_S)}
        surf_up_vdn = {**surfaces, asset_name: surf_vdn[asset_name].with_spot(base_surf.spot + h_S)}
        surf_dn_vdn = {**surfaces, asset_name: surf_vdn[asset_name].with_spot(base_surf.spot - h_S)}
        P_up_vup = price_all(surf_up_vup, heston_vup, asset_name, scale_up)
        P_dn_vup = price_all(surf_dn_vup, heston_vup, asset_name, scale_dn)
        P_up_vdn = price_all(surf_up_vdn, heston_vdn, asset_name, scale_up)
        P_dn_vdn = price_all(surf_dn_vdn, heston_vdn, asset_name, scale_dn)

        for model_name in base_prices:
            P0 = base_prices[model_name]
            delta[model_name][asset_name] = (P_up[model_name] - P_dn[model_name]) / (2.0 * h_S)
            gamma[model_name][asset_name] = (P_up[model_name] - 2.0 * P0 + P_dn[model_name]) / (h_S ** 2)
            vega[model_name][asset_name] = (P_vup[model_name] - P_vdn[model_name]) / (2.0 * h_v) * 0.01
            skew_sens[model_name][asset_name] = (
                (P_skup[model_name] - P_skdn[model_name]) / (2.0 * h_skew) * 0.01
            )
            vanna[model_name][asset_name] = (
                P_up_vup[model_name] - P_dn_vup[model_name]
                - P_up_vdn[model_name] + P_dn_vdn[model_name]
            ) / (4.0 * h_S * h_v)

    return {
        model_name: BasketGreekResult(
            delta=delta[model_name], gamma=gamma[model_name],
            vega=vega[model_name], vanna=vanna[model_name],
            skew_sens=skew_sens[model_name],
        )
        for model_name in base_prices
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(
    scenarios_path: str,
    n_paths: int,
    verbose: bool,
    compute_greeks_flag: bool = False,
    n_paths_greeks: int | None = None,
    h_S_pct: float = 0.01,
    h_v: float = 0.001,
    h_skew: float = 0.01,
) -> list[ScenarioResult]:
    with open(scenarios_path) as f:
        spec = yaml.safe_load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(scenarios_path)))
    base_path = os.path.join(project_root, spec["base_config"])
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    sc_cfg = base_cfg["simulation"]
    seed = int(sc_cfg["seed"])
    antithetic = bool(sc_cfg.get("antithetic", True))
    n_greek_paths = n_paths_greeks if n_paths_greeks is not None else n_paths

    results: list[ScenarioResult] = []

    for idx, scenario in enumerate(spec["scenarios"]):
        name = scenario["name"]
        group = scenario.get("group", "")
        description = scenario.get("description", "").strip()
        overrides = scenario.get("overrides", {})

        cfg = deep_merge(base_cfg, overrides)
        is_basket = "assets" in cfg

        print(f"\n[{idx+1:2d}/{len(spec['scenarios'])}] {group.upper():7s} | {name}")
        if verbose:
            print(f"         {description[:80]}")

        greeks = None
        basket_greeks = None

        if is_basket:
            asset_surfaces = build_basket_surfaces(cfg)
            correlation = load_correlation(cfg, len(asset_surfaces))
            rates = {s.rate for _, s in asset_surfaces}
            assert len(rates) == 1, "all basket assets must share the same discount rate (v1 constraint)"
            RATE = rates.pop()

            base_heston = calibrate_basket_heston(asset_surfaces, verbose=verbose)
            pricers = build_basket_pricers(cfg, asset_surfaces, correlation, seed, antithetic,
                                           heston_params=base_heston)
            product = build_basket_product(cfg, RATE)

            prices, ses, durations = {}, {}, {}
            for pricer in pricers:
                result = pricer.price(product, n_paths)
                prices[pricer.model_name] = result.price
                ses[pricer.model_name] = result.std_error
                durations[pricer.model_name] = result.expected_duration
                print(f"           {pricer.model_name:12s}  {result.price:.6f}  ±{result.std_error:.6f}"
                      f"  dur={result.expected_duration:.2f}y")

            if compute_greeks_flag:
                print(f"           Computing basket Greeks  (N={n_greek_paths:,}, "
                      f"h_S={h_S_pct*100:.1f}%, h_v={h_v*10000:.0f}bp, h_skew={h_skew}) ...")
                basket_greeks = compute_basket_greeks(
                    cfg, asset_surfaces, correlation, base_heston, product, prices,
                    seed, antithetic, n_greek_paths, h_S_pct=h_S_pct, h_v=h_v, h_skew=h_skew,
                )
                for model_name, gr in basket_greeks.items():
                    print(f"             {model_name}:")
                    for asset_name in gr.delta:
                        print(f"               {asset_name:12s}"
                              f"  Δ={gr.delta[asset_name]:+.6f}"
                              f"  Γ={gr.gamma[asset_name]:+.6f}"
                              f"  ν={gr.vega[asset_name]:+.6f}"
                              f"  vanna={gr.vanna[asset_name]:+.6f}"
                              f"  skew_sens={gr.skew_sens[asset_name]:+.6f}")

        else:
            mc = cfg["market"]
            SPOT = float(mc["spot"])
            RATE = float(mc["rate"])
            DIV_YIELD = float(mc["div_yield"])

            surface = build_surface(cfg, SPOT, RATE, DIV_YIELD)

            # Calibrate SV models once; reuse params for spot-bump Greeks
            base_cal = calibrate_all(cfg, surface, verbose=verbose)

            pricers = build_pricers(cfg, surface, SPOT, RATE, DIV_YIELD,
                                    seed, antithetic, calibrated_params=base_cal)
            product = build_product(cfg, SPOT, RATE)

            prices, ses, durations = {}, {}, {}
            for pricer in pricers:
                result = pricer.price(product, n_paths)
                prices[pricer.model_name] = result.price
                ses[pricer.model_name] = result.std_error
                durations[pricer.model_name] = result.expected_duration
                print(f"           {pricer.model_name:12s}  {result.price:.6f}  ±{result.std_error:.6f}"
                      f"  dur={result.expected_duration:.2f}y")

            if compute_greeks_flag:
                print(f"           Computing Greeks  (N={n_greek_paths:,}, "
                      f"h_S={h_S_pct*100:.1f}%, h_v={h_v*10000:.0f}bp, h_skew={h_skew}) ...")
                greeks = compute_greeks(
                    cfg, surface, base_cal, product, prices,
                    SPOT, RATE, DIV_YIELD, seed, antithetic, n_greek_paths,
                    h_S_pct=h_S_pct, h_v=h_v, h_skew=h_skew,
                )
                for model_name, gr in greeks.items():
                    print(f"             {model_name:12s}"
                          f"  Δ={gr.delta:+.6f}"
                          f"  Γ={gr.gamma:+.6f}"
                          f"  ν={gr.vega:+.6f}"
                          f"  vanna={gr.vanna:+.6f}"
                          f"  skew_sens={gr.skew_sens:+.6f}")

        results.append(ScenarioResult(
            name=name, group=group, description=description,
            prices=prices, std_errors=ses, durations=durations,
            greeks=greeks, basket_greeks=basket_greeks,
        ))

    return results


def print_summary(results: list[ScenarioResult]) -> None:
    if not results:
        return

    model_names = list(results[0].prices.keys())
    col = 12

    sep = "─" * (32 + col * len(model_names) + 12)
    header = f"  {'Scenario':<30}" + "".join(f"  {n:>{col}}" for n in model_names) + f"  {'Spread(bp)':>10}"

    print()
    print("=" * len(header))
    print("  Autocallable Scenario Summary  —  Price")
    print("=" * len(header))

    current_group = None
    for r in results:
        if r.group != current_group:
            current_group = r.group
            print()
            print(f"  ── {current_group.upper()} ──")
            print(header)
            print(f"  {sep}")

        prices = list(r.prices.values())
        spread_bp = (max(prices) - min(prices)) * 10_000
        row = f"  {r.name:<30}" + "".join(f"  {p:>{col}.6f}" for p in prices)
        row += f"  {spread_bp:>10.1f}"
        print(row)

    print()

    # ── Duration table ────────────────────────────────────────────────────────
    results_d = [r for r in results if r.durations]
    if results_d:
        dcol = 14
        dhdr = f"  {'Scenario':<30}" + "".join(f"  {n:>{dcol}}" for n in model_names)
        dsep = "─" * len(dhdr)

        print(f"  ── Expected Duration (years, risk-neutral MC) ──")
        print(dhdr)
        print(f"  {dsep}")

        current_group = None
        for r in results_d:
            if r.group != current_group:
                current_group = r.group
            vals = [r.durations[n] for n in model_names]
            row = f"  {r.name:<30}" + "".join(f"  {v:>{dcol}.2f}" for v in vals)
            print(row)
        print()

    # ── Greek tables ──────────────────────────────────────────────────────────
    results_g = [r for r in results if r.greeks]
    if not results_g:
        return

    greek_meta = [
        ("delta", "Delta  ∂P/∂S"),
        ("gamma", "Gamma  ∂²P/∂S²"),
        ("vega",  "Vega   ∂P/∂σ  (per 1% vol shift)"),
        ("vanna", "Vanna  ∂²P/(∂S ∂σ)"),
        ("skew_sens", "Skew Sensitivity  ∂P/∂skew  (per 0.01 skew shift)"),
    ]

    gcol = 14
    for attr, label in greek_meta:
        ghdr = f"  {'Scenario':<30}" + "".join(f"  {n:>{gcol}}" for n in model_names)
        gsep = "─" * len(ghdr)

        print(f"  ── {label} ──")
        print(ghdr)
        print(f"  {gsep}")

        current_group = None
        for r in results_g:
            if r.group != current_group:
                current_group = r.group
            vals = [getattr(r.greeks[n], attr) for n in model_names]
            row = f"  {r.name:<30}" + "".join(f"  {v:+{gcol}.6f}" for v in vals)
            print(row)
        print()

    print_basket_greeks_summary(results)


def print_basket_greeks_summary(results: list[ScenarioResult]) -> None:
    """Per-asset Greeks for basket scenarios (see compute_basket_greeks)."""
    results_bg = [r for r in results if r.basket_greeks]
    if not results_bg:
        return

    print("=" * 70)
    print("  Basket Greeks (per asset)")
    print("=" * 70)
    for r in results_bg:
        print(f"\n  {r.name} ({r.group})")
        for model_name, gr in r.basket_greeks.items():
            print(f"    {model_name}:")
            for asset_name in gr.delta:
                print(f"      {asset_name:12s}"
                      f"  Δ={gr.delta[asset_name]:+.6f}"
                      f"  Γ={gr.gamma[asset_name]:+.6f}"
                      f"  ν={gr.vega[asset_name]:+.6f}"
                      f"  vanna={gr.vanna[asset_name]:+.6f}"
                      f"  skew_sens={gr.skew_sens[asset_name]:+.6f}")
    print()


def save_csv(results: list[ScenarioResult], path: str) -> None:
    if not results:
        return
    model_names = list(results[0].prices.keys())
    has_durations = any(r.durations for r in results)
    has_greeks = any(r.greeks for r in results)
    has_basket_greeks = any(r.basket_greeks for r in results)
    greek_attrs = ["delta", "gamma", "vega", "vanna", "skew_sens"]

    # Union of (model_name, asset_name) pairs seen across basket scenarios,
    # in first-seen order -- columns are the same for every row.
    basket_keys: list[tuple[str, str]] = []
    if has_basket_greeks:
        for r in results:
            if r.basket_greeks:
                for model_name, gr in r.basket_greeks.items():
                    for asset_name in gr.delta:
                        key = (model_name, asset_name)
                        if key not in basket_keys:
                            basket_keys.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (["group", "scenario"]
                  + model_names
                  + [f"{n}_se" for n in model_names]
                  + ["spread_bp"])
        if has_durations:
            header += [f"{n}_duration" for n in model_names]
        if has_greeks:
            for attr in greek_attrs:
                header += [f"{n}_{attr}" for n in model_names]
        if has_basket_greeks:
            for model_name, asset_name in basket_keys:
                for attr in greek_attrs:
                    header += [f"{model_name}_{asset_name}_{attr}"]
        writer.writerow(header)

        for r in results:
            prices = [r.prices[n] for n in model_names]
            ses = [r.std_errors[n] for n in model_names]
            spread_bp = (max(prices) - min(prices)) * 10_000
            row = [r.group, r.name] + prices + ses + [round(spread_bp, 2)]
            if has_durations:
                if r.durations:
                    row += [round(r.durations[n], 4) for n in model_names]
                else:
                    row += [""] * len(model_names)
            if has_greeks:
                if r.greeks:
                    for attr in greek_attrs:
                        row += [getattr(r.greeks[n], attr) for n in model_names]
                else:
                    row += [""] * (len(greek_attrs) * len(model_names))
            if has_basket_greeks:
                if r.basket_greeks:
                    for model_name, asset_name in basket_keys:
                        gr = r.basket_greeks.get(model_name)
                        if gr and asset_name in gr.delta:
                            for attr in greek_attrs:
                                row.append(getattr(gr, attr)[asset_name])
                        else:
                            row += [""] * len(greek_attrs)
                else:
                    row += [""] * (len(greek_attrs) * len(basket_keys))
            writer.writerow(row)

    print(f"Results saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autocallable scenario comparison")
    parser.add_argument("--scenarios", default="scenarios/scenarios.yaml",
                        help="Path to scenarios YAML (default: scenarios/scenarios.yaml)")
    parser.add_argument("--n-paths", type=int, default=20_000,
                        help="MC paths per (scenario, model) (default: 20,000)")
    parser.add_argument("--output", default="scenarios/results.csv",
                        help="CSV output path (default: scenarios/results.csv)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print calibrated model parameters for each scenario")
    parser.add_argument("--greeks", action="store_true",
                        help="Compute delta, gamma, vega, vanna, skew_sens via finite differences")
    parser.add_argument("--n-paths-greeks", type=int, default=None,
                        help="MC paths for Greek FD bumps (default: same as --n-paths)")
    parser.add_argument("--h-spot-pct", type=float, default=0.01,
                        help="Spot bump as fraction of spot for FD (default: 0.01 = 1%%)")
    parser.add_argument("--h-vol", type=float, default=0.001,
                        help="Parallel vol bump for FD in absolute terms (default: 0.001 = 10bp)")
    parser.add_argument("--h-skew", type=float, default=0.01,
                        help="Skew-coefficient bump for FD (see ImpliedVolSurface.with_skew_shift, "
                             "default: 0.01)")
    args = parser.parse_args()

    print(f"Autocallable Scenario Runner")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  N paths:   {args.n_paths:,} per (scenario × model)")
    if args.greeks:
        ngp = args.n_paths_greeks or args.n_paths
        print(f"  Greeks:    ON  (N={ngp:,}, h_S={args.h_spot_pct*100:.1f}%, h_v={args.h_vol*10000:.0f}bp, "
              f"h_skew={args.h_skew})")

    results = run_all(
        args.scenarios,
        args.n_paths,
        verbose=args.verbose,
        compute_greeks_flag=args.greeks,
        n_paths_greeks=args.n_paths_greeks,
        h_S_pct=args.h_spot_pct,
        h_v=args.h_vol,
        h_skew=args.h_skew,
    )
    print_summary(results)
    save_csv(results, args.output)


if __name__ == "__main__":
    main()
