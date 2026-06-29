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
    delta: float   # ∂P/∂S
    gamma: float   # ∂²P/∂S²
    vega:  float   # ∂P/∂σ · 0.01  (per 1 percentage-point parallel vol shift)
    vanna: float   # ∂²P / (∂S ∂σ)


@dataclass
class ScenarioResult:
    name: str
    group: str
    description: str
    prices: dict[str, float]       # model_name → price
    std_errors: dict[str, float]   # model_name → SE
    durations: Optional[dict[str, float]] = None       # model_name → expected duration (years)
    greeks: Optional[dict[str, GreekResult]] = None   # model_name → GreekResult


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
) -> dict[str, GreekResult]:
    """
    Compute delta, gamma, vega, vanna via central finite differences.

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

    Variance reduction
    ------------------
    All 8 bumped evaluations use the same RNG seed (common random numbers),
    so noise in the FD numerators largely cancels.

    Parameters
    ----------
    h_S_pct : spot bump as a fraction of spot (default 0.01 = 1 %)
    h_v     : absolute parallel implied-vol shift (default 0.001 = 10 bp)
    vega    : reported per 1 percentage-point (0.01) parallel vol shift
    """
    h_S = SPOT * h_S_pct

    # ── spot-bumped products: barriers stay fixed in dollar terms ──────────────
    product_up = _rescale_product_barriers(product, SPOT, SPOT + h_S)
    product_dn = _rescale_product_barriers(product, SPOT, SPOT - h_S)

    # ── 8 bumped surfaces ──────────────────────────────────────────────────────
    s_up     = base_surface.with_spot(SPOT + h_S)
    s_dn     = base_surface.with_spot(SPOT - h_S)
    s_vup    = base_surface.with_vol_shift(+h_v)
    s_vdn    = base_surface.with_vol_shift(-h_v)
    # Cross terms for vanna: vol shift first, then spot shift
    s_up_vup = s_vup.with_spot(SPOT + h_S)
    s_dn_vup = s_vup.with_spot(SPOT - h_S)
    s_up_vdn = s_vdn.with_spot(SPOT + h_S)
    s_dn_vdn = s_vdn.with_spot(SPOT - h_S)

    # Re-calibrate SV models for vol-bumped surfaces (fast warm-start).
    cal_vup = calibrate_all(cfg, s_vup, fast=True, warm_start=base_cal)
    cal_vdn = calibrate_all(cfg, s_vdn, fast=True, warm_start=base_cal)

    def price_all(surface, spot, cal, prod) -> dict[str, float]:
        pricers = build_pricers(cfg, surface, spot, RATE, DIV_YIELD,
                                seed, antithetic, calibrated_params=cal)
        return {p.model_name: p.price(prod, n_paths).price for p in pricers}

    # 8 bumped price evaluations (CRN via shared seed)
    # Spot-bumped calls use rescaled products; vol-only calls use the original.
    P_up     = price_all(s_up,     SPOT + h_S, base_cal, product_up)
    P_dn     = price_all(s_dn,     SPOT - h_S, base_cal, product_dn)
    P_vup    = price_all(s_vup,    SPOT,        cal_vup,  product)
    P_vdn    = price_all(s_vdn,    SPOT,        cal_vdn,  product)
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
        greeks[name] = GreekResult(delta=delta, gamma=gamma, vega=vega, vanna=vanna)

    return greeks


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(
    scenarios_path: str,
    n_paths: int,
    verbose: bool,
    compute_greeks_flag: bool = False,
    n_paths_greeks: int | None = None,
    h_S_pct: float = 0.01,
    h_v: float = 0.001,
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

        mc = cfg["market"]
        SPOT = float(mc["spot"])
        RATE = float(mc["rate"])
        DIV_YIELD = float(mc["div_yield"])

        print(f"\n[{idx+1:2d}/{len(spec['scenarios'])}] {group.upper():7s} | {name}")
        if verbose:
            print(f"         {description[:80]}")

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

        greeks = None
        if compute_greeks_flag:
            print(f"           Computing Greeks  (N={n_greek_paths:,}, "
                  f"h_S={h_S_pct*100:.1f}%, h_v={h_v*10000:.0f}bp) ...")
            greeks = compute_greeks(
                cfg, surface, base_cal, product, prices,
                SPOT, RATE, DIV_YIELD, seed, antithetic, n_greek_paths,
                h_S_pct=h_S_pct, h_v=h_v,
            )
            for model_name, gr in greeks.items():
                print(f"             {model_name:12s}"
                      f"  Δ={gr.delta:+.6f}"
                      f"  Γ={gr.gamma:+.6f}"
                      f"  ν={gr.vega:+.6f}"
                      f"  vanna={gr.vanna:+.6f}")

        results.append(ScenarioResult(
            name=name, group=group, description=description,
            prices=prices, std_errors=ses, durations=durations, greeks=greeks,
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


def save_csv(results: list[ScenarioResult], path: str) -> None:
    if not results:
        return
    model_names = list(results[0].prices.keys())
    has_durations = any(r.durations for r in results)
    has_greeks = any(r.greeks for r in results)
    greek_attrs = ["delta", "gamma", "vega", "vanna"]

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
                        help="Compute delta, gamma, vega, vanna via finite differences")
    parser.add_argument("--n-paths-greeks", type=int, default=None,
                        help="MC paths for Greek FD bumps (default: same as --n-paths)")
    parser.add_argument("--h-spot-pct", type=float, default=0.01,
                        help="Spot bump as fraction of spot for FD (default: 0.01 = 1%%)")
    parser.add_argument("--h-vol", type=float, default=0.001,
                        help="Parallel vol bump for FD in absolute terms (default: 0.001 = 10bp)")
    args = parser.parse_args()

    print(f"Autocallable Scenario Runner")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  N paths:   {args.n_paths:,} per (scenario × model)")
    if args.greeks:
        ngp = args.n_paths_greeks or args.n_paths
        print(f"  Greeks:    ON  (N={ngp:,}, h_S={args.h_spot_pct*100:.1f}%, h_v={args.h_vol*10000:.0f}bp)")

    results = run_all(
        args.scenarios,
        args.n_paths,
        verbose=args.verbose,
        compute_greeks_flag=args.greeks,
        n_paths_greeks=args.n_paths_greeks,
        h_S_pct=args.h_spot_pct,
        h_v=args.h_vol,
    )
    print_summary(results)
    save_csv(results, args.output)


if __name__ == "__main__":
    main()
