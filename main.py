"""
Demo: price a 3-year autocallable note under Local Vol, Heston, and SABR.

Run with:
    .venv/bin/python main.py                        # uses config.yaml
    .venv/bin/python main.py --config my_note.yaml  # custom config file

Edit config.yaml (or a copy) to change product and market parameters
without touching this file.
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

import yaml
import numpy as np
from calibration.vol_surface import skewed_surface, flat_surface, term_structure_surface
from calibration.calibrators import calibrate_heston, calibrate_sabr
from models.local_vol import LocalVolModel
from models.heston import HestonModel, HestonParams
from models.sabr import SABRModel, SABRParams
from products.autocallable import AutocallableNote
from engine.mc_pricer import MCPricer, compare_models
from engine.heston_cf import heston_price_batch
from engine.black_scholes import implied_vol as bs_implied_vol


def quick_fit_check(surface, heston_params, sabr_params, spot, rate, div_yield) -> None:
    """
    Analytical fit quality check — no MC, completes in ~1 second.

    Evaluates model-implied vols at a small (T, K) grid and reports
    RMSE and worst-case error in basis points per model.

    Heston: priced via characteristic function (exact).
    SABR:   priced via Hagan asymptotic formula (exact for beta=1).
    Local Vol: fits the surface by construction — always exact.
    """
    T_min, T_max = surface.maturity_range
    maturities = sorted({
        round(max(T_min + 0.01, T_max * 0.25), 2),
        round(T_max * 0.5, 2),
        float(T_max),
    })
    moneyness = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]

    print("Vol Surface Fit — Analytical Check")
    print("  Local Vol   exact fit (by construction)")

    checks = []
    if heston_params is not None:
        checks.append(("Heston", heston_params))
    if sabr_params is not None:
        checks.append(("SABR  ", sabr_params))

    for label, params in checks:
        errors, worst = [], (0.0, None, None)
        for T in maturities:
            F = float(surface.forward(T))
            strikes = np.array([F * m for m in moneyness])
            iv_mkt = np.array([surface.implied_vol(T, K) for K in strikes])

            if label.startswith("Heston"):
                p = params
                prices = heston_price_batch(spot, strikes, T, rate, div_yield,
                                            p.kappa, p.theta, p.xi, p.rho, p.v0)
                iv_mdl = np.array([
                    bs_implied_vol(float(pr), spot, float(K), T, rate, div_yield, is_call=True)
                    for pr, K in zip(prices, strikes)
                ])
            else:
                iv_mdl = np.array([
                    params.hagan_implied_vol(F, float(K), T) for K in strikes
                ])

            for j, (mkt, mdl, K) in enumerate(zip(iv_mkt, iv_mdl, strikes)):
                if not np.isnan(mdl):
                    err = mdl - mkt
                    errors.append(err)
                    if abs(err) > abs(worst[0]):
                        worst = (err, T, K / spot)

        if errors:
            rmse = np.sqrt(np.mean(np.array(errors) ** 2)) * 10_000
            max_err = worst[0] * 10_000
            print(f"  {label}  RMSE: {rmse:5.1f}bp  "
                  f"max: {max_err:+6.1f}bp  (T={worst[1]:.1f}y, K/S={worst[2]:.0%})")
    print()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


def main():
    parser = argparse.ArgumentParser(description="Autocallable note pricer")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")

    fit_group = parser.add_mutually_exclusive_group()
    fit_group.add_argument("--full-diagnostic", action="store_true",
                           help="Run full MC vol surface diagnostic before pricing "
                                "(slow — uses diagnostics/model_quality.py)")
    fit_group.add_argument("--no-fit-check", action="store_true",
                           help="Skip fit check entirely and go straight to pricing")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Market ────────────────────────────────────────────────────────────────
    mc = cfg["market"]
    SPOT = float(mc["spot"])
    RATE = float(mc["rate"])
    DIV_YIELD = float(mc["div_yield"])

    # ── Simulation ────────────────────────────────────────────────────────────
    sc = cfg["simulation"]
    N_PATHS = int(sc["n_paths"])
    SEED = int(sc["seed"])
    ANTITHETIC = bool(sc.get("antithetic", True))

    # ── Vol surface ───────────────────────────────────────────────────────────
    surface = build_surface(cfg, SPOT, RATE, DIV_YIELD)

    # ── Product ───────────────────────────────────────────────────────────────
    pc = cfg["product"]
    product = AutocallableNote(
        notional=float(pc["notional"]),
        spot=SPOT,
        maturity=float(pc["maturity"]),
        observation_dates=pc["observation_dates"],
        autocall_barriers=pc["autocall_barriers"],
        coupon_rate=pc["coupon_rate"],
        conditional_coupon=bool(pc.get("conditional_coupon", False)),
        coupon_barrier=float(pc.get("coupon_barrier", 0.80)),
        capital_barrier=float(pc.get("capital_barrier", 0.60)),
        capital_barrier_active=bool(pc.get("capital_barrier_active", True)),
        discount_rate=RATE,
    )

    print("=" * 65)
    print("Autocallable Note — Model Comparison")
    print("=" * 65)
    print(f"  Config:   {args.config}")
    print(f"  Spot: {SPOT}  |  Rate: {RATE:.1%}  |  Div yield: {DIV_YIELD:.1%}")
    print(f"  Maturity: {product.maturity}y  |  Obs dates: {product.observation_dates.tolist()}")
    print(f"  Autocall barrier: {product.autocall_barrier_schedule.tolist()}")
    coupon_display = [f"{c:.1%}" for c in product._coupon_schedule.tolist()]
    print(f"  Coupon schedule: {coupon_display}  |  Capital barrier: {product.capital_barrier:.0%}")
    print(f"  Phoenix coupons: {product.conditional_coupon}")
    print(f"  N paths: {N_PATHS:,}")
    print()

    # ── Build pricers for selected models ─────────────────────────────────────
    models_cfg = cfg.get("models", {})
    pricers = []
    heston_params = None
    sabr_params = None

    if models_cfg.get("local_vol", True):
        lv_model = LocalVolModel(surface=surface, steps_per_year=52, seed=SEED)
        pricers.append(MCPricer(lv_model, model_name="Local Vol (Dupire)", antithetic=ANTITHETIC))

    if models_cfg.get("heston", True):
        try:
            print("Calibrating Heston to vol surface...")
            heston_params = calibrate_heston(surface, n_calibration_points=36)
            print(f"  kappa={heston_params.kappa:.3f}  theta={heston_params.theta:.4f}"
                  f"  xi={heston_params.xi:.3f}  rho={heston_params.rho:.3f}  v0={heston_params.v0:.4f}")
        except Exception as e:
            print(f"  Calibration failed ({e}), using default params.")
            heston_params = HestonParams()
        heston_model = HestonModel(params=heston_params, spot=SPOT, rate=RATE,
                                   div_yield=DIV_YIELD, steps_per_year=52, seed=SEED)
        pricers.append(MCPricer(heston_model, model_name="Heston SV", antithetic=ANTITHETIC))

    if models_cfg.get("sabr", True):
        try:
            print("Calibrating SABR to vol surface (1y slice)...")
            sabr_params = calibrate_sabr(surface, beta=1.0, calibration_maturity=1.0)
            print(f"  alpha={sabr_params.alpha:.4f}  beta={sabr_params.beta:.1f}"
                  f"  rho={sabr_params.rho:.3f}  nu={sabr_params.nu:.3f}")
        except Exception as e:
            print(f"  Calibration failed ({e}), using default params.")
            sabr_params = SABRParams()
        sabr_model = SABRModel(params=sabr_params, spot=SPOT, rate=RATE,
                               div_yield=DIV_YIELD, steps_per_year=52, seed=SEED)
        pricers.append(MCPricer(sabr_model, model_name="SABR SV", antithetic=ANTITHETIC))

    if not pricers:
        print("No models selected in config. Enable at least one under 'models:'.")
        sys.exit(1)

    # ── Vol surface fit check ─────────────────────────────────────────────────
    if args.full_diagnostic:
        print()
        print("Running full MC diagnostic (diagnostics/model_quality.py)...")
        print("-" * 65)
        subprocess.run(
            [sys.executable, "-u", "diagnostics/model_quality.py", "--config", args.config],
            check=True,
        )
        print("-" * 65)
        print()
    elif not args.no_fit_check:
        quick_fit_check(surface,
                        heston_params if models_cfg.get("heston", True) else None,
                        sabr_params   if models_cfg.get("sabr",   True) else None,
                        SPOT, RATE, DIV_YIELD)

    # ── Run comparison ────────────────────────────────────────────────────────
    print()
    print("Pricing...")
    print("-" * 65)
    results = compare_models(pricers=pricers, product=product, n_paths=N_PATHS)
    print("-" * 65)

    prices = [r.price for r in results]
    names = [r.model_name for r in results]
    if len(prices) > 1:
        print(f"\nModel spread (max - min): {max(prices) - min(prices):.6f}")
        print(f"  High: {names[prices.index(max(prices))]}  ({max(prices):.6f})")
        print(f"  Low:  {names[prices.index(min(prices))]}  ({min(prices):.6f})")


if __name__ == "__main__":
    main()
