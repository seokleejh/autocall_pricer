"""
Scenario runner: prices the autocallable under each model for every scenario
defined in scenarios.yaml and prints a summary comparison table.

Run with:
    .venv/bin/python scenarios/run_scenarios.py
    .venv/bin/python scenarios/run_scenarios.py --scenarios scenarios/scenarios.yaml
    .venv/bin/python scenarios/run_scenarios.py --n-paths 50000 --output results.csv
"""

from __future__ import annotations

import sys
import os
import argparse
import copy
import csv
from dataclasses import dataclass

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
    """
    Recursively merge `override` into a deep copy of `base`.
    Dicts are merged key-by-key; any other type (scalar, list) is replaced.
    """
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
        return term_structure_surface(
            spot=spot, rate=rate, div_yield=div_yield, T_max=T_max,
            vol_short=vc.get("vol_short", 0.25),
            vol_long=vc.get("vol_long", 0.18),
            kappa=vc.get("kappa", 1.5),
            skew=vc.get("skew", -0.10),
            convexity=vc.get("convexity", 0.02),
        )
    return skewed_surface(spot=spot, atm_vol=vc["atm_vol"], skew=vc.get("skew", 0.0),
                          rate=rate, div_yield=div_yield, T_max=T_max)


def build_pricers(cfg: dict, surface, spot: float, rate: float,
                  div_yield: float, seed: int, antithetic: bool,
                  verbose: bool = False) -> list[MCPricer]:
    models_cfg = cfg.get("models", {})
    pricers = []

    if models_cfg.get("local_vol", True):
        m = LocalVolModel(surface=surface, steps_per_year=52, seed=seed)
        pricers.append(MCPricer(m, "Local Vol", antithetic))

    if models_cfg.get("heston", True):
        try:
            params = calibrate_heston(surface)
            if verbose:
                print(f"    Heston: kappa={params.kappa:.3f} theta={params.theta:.4f}"
                      f" xi={params.xi:.3f} rho={params.rho:.3f}")
        except Exception as e:
            if verbose:
                print(f"    Heston calibration failed ({e}), using defaults")
            params = HestonParams()
        m = HestonModel(params=params, spot=spot, rate=rate, div_yield=div_yield,
                        steps_per_year=52, seed=seed)
        pricers.append(MCPricer(m, "Heston", antithetic))

    if models_cfg.get("sabr", True):
        try:
            params = calibrate_sabr(surface, beta=1.0, calibration_maturity=1.0)
            if verbose:
                print(f"    SABR:   alpha={params.alpha:.4f} rho={params.rho:.3f}"
                      f" nu={params.nu:.3f}")
        except Exception as e:
            if verbose:
                print(f"    SABR calibration failed ({e}), using defaults")
            from models.sabr import SABRParams
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
class ScenarioResult:
    name: str
    group: str
    description: str
    prices: dict[str, float]      # model_name → price
    std_errors: dict[str, float]  # model_name → SE


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(scenarios_path: str, n_paths: int, verbose: bool) -> list[ScenarioResult]:
    with open(scenarios_path) as f:
        spec = yaml.safe_load(f)

    # Base config path is relative to the project root (parent of scenarios/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(scenarios_path)))
    base_path = os.path.join(project_root, spec["base_config"])
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    sc_cfg = base_cfg["simulation"]
    seed = int(sc_cfg["seed"])
    antithetic = bool(sc_cfg.get("antithetic", True))

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
        pricers = build_pricers(cfg, surface, SPOT, RATE, DIV_YIELD, seed,
                                antithetic, verbose=verbose)
        product = build_product(cfg, SPOT, RATE)

        prices, ses = {}, {}
        for pricer in pricers:
            result = pricer.price(product, n_paths)
            prices[pricer.model_name] = result.price
            ses[pricer.model_name] = result.std_error
            print(f"           {pricer.model_name:12s}  {result.price:.6f}  ±{result.std_error:.6f}")

        results.append(ScenarioResult(
            name=name, group=group, description=description,
            prices=prices, std_errors=ses,
        ))

    return results


def print_summary(results: list[ScenarioResult]) -> None:
    if not results:
        return

    model_names = list(results[0].prices.keys())
    col = 12  # column width per model

    sep = "─" * (32 + col * len(model_names) + 12)
    header = f"  {'Scenario':<30}" + "".join(f"  {n:>{col}}" for n in model_names) + f"  {'Spread(bp)':>10}"

    print()
    print("=" * len(header))
    print("  Autocallable Scenario Summary")
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


def save_csv(results: list[ScenarioResult], path: str) -> None:
    if not results:
        return
    model_names = list(results[0].prices.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["group", "scenario"] + model_names + \
                 [f"{n}_se" for n in model_names] + ["spread_bp"]
        writer.writerow(header)
        for r in results:
            prices = [r.prices[n] for n in model_names]
            ses = [r.std_errors[n] for n in model_names]
            spread_bp = (max(prices) - min(prices)) * 10_000
            writer.writerow([r.group, r.name] + prices + ses + [round(spread_bp, 2)])
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
    args = parser.parse_args()

    print(f"Autocallable Scenario Runner")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  N paths:   {args.n_paths:,} per (scenario × model)")

    results = run_all(args.scenarios, args.n_paths, args.verbose)
    print_summary(results)
    save_csv(results, args.output)


if __name__ == "__main__":
    main()
