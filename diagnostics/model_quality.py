"""
Model quality diagnostic: compare each model's vanilla implied vols
against the input Black-Scholes surface.

For each (model, maturity) pair, paths are simulated once and reused
across all strikes — so the full grid runs in O(models × maturities)
simulations rather than O(models × maturities × strikes).

Run with:
    .venv/bin/python diagnostics/model_quality.py
    .venv/bin/python diagnostics/model_quality.py --config my_note.yaml
    .venv/bin/python diagnostics/model_quality.py --n-paths 50000 --output results/fit.png
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.vol_surface import skewed_surface, flat_surface, term_structure_surface
from calibration.calibrators import calibrate_heston, calibrate_sabr
from models.local_vol import LocalVolModel
from models.heston import HestonModel, HestonParams
from models.sabr import SABRModel, SABRParams
from engine.black_scholes import bs_price, implied_vol as bs_implied_vol

# ── Diagnostic grid defaults ───────────────────────────────────────────────────
DEFAULT_MATURITIES = [0.1, 0.2, 0.3, 0.5]
DEFAULT_MONEYNESS = np.linspace(0.70, 1.30, 13)   # K/S0
DEFAULT_N_PATHS = 30_000


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


def build_models(cfg: dict, surface, spot: float, rate: float,
                 div_yield: float, seed: int) -> list[tuple[str, object]]:
    """Return list of (name, model) for each enabled model."""
    models_cfg = cfg.get("models", {})
    models = []

    if models_cfg.get("local_vol", True):
        m = LocalVolModel(surface=surface, steps_per_year=52, seed=seed)
        models.append(("Local Vol", m))

    if models_cfg.get("heston", True):
        try:
            print("  Calibrating Heston...", end=" ", flush=True)
            params = calibrate_heston(surface, n_calibration_points=36)
            print(f"kappa={params.kappa:.3f}  theta={params.theta:.4f}"
                  f"  xi={params.xi:.3f}  rho={params.rho:.3f}")
        except Exception as e:
            print(f"failed ({e}), using defaults")
            params = HestonParams()
        m = HestonModel(params=params, spot=spot, rate=rate, div_yield=div_yield,
                        steps_per_year=52, seed=seed)
        models.append(("Heston", m))

    if models_cfg.get("sabr", True):
        try:
            print("  Calibrating SABR...", end=" ", flush=True)
            params = calibrate_sabr(surface, beta=1.0, calibration_maturity=1.0)
            print(f"alpha={params.alpha:.4f}  beta={params.beta:.1f}"
                  f"  rho={params.rho:.3f}  nu={params.nu:.3f}")
        except Exception as e:
            print(f"failed ({e}), using defaults")
            from models.sabr import SABRParams
            params = SABRParams()
        m = SABRModel(params=params, spot=spot, rate=rate, div_yield=div_yield,
                      steps_per_year=52, seed=seed)
        models.append(("SABR", m))

    return models


def price_smile(
    model,
    spot: float,
    strikes: np.ndarray,
    T: float,
    rate: float,
    div_yield: float,
    n_paths: int,
    antithetic: bool,
) -> np.ndarray:
    """
    Simulate once to maturity T, evaluate OTM option payoffs for all strikes,
    and return model-implied vols. Shape: (len(strikes),).

    Uses puts for K < spot and calls for K >= spot (OTM convention) to avoid
    extracting IV from deep-ITM options where vega is tiny and MC noise
    dominates the IV estimate.
    """
    performances = model.simulate(
        n_paths=n_paths,
        observation_times=np.array([T]),
        antithetic=antithetic,
    )
    perf = performances[:, 0]   # S_T / S_0
    df = np.exp(-rate * T)

    ivs = np.empty(len(strikes))
    for j, K in enumerate(strikes):
        k = K / spot
        is_call = K >= spot
        if is_call:
            mc_price = spot * df * float(np.mean(np.maximum(perf - k, 0.0)))
        else:
            mc_price = spot * df * float(np.mean(np.maximum(k - perf, 0.0)))
        ivs[j] = bs_implied_vol(mc_price, spot, K, T, rate, div_yield,
                                is_call=is_call)
    return ivs


def print_table(
    T: float,
    moneyness: np.ndarray,
    input_ivs: np.ndarray,
    model_names: list[str],
    model_ivs: dict[str, np.ndarray],
) -> None:
    model_cols = "  ".join(f"{'IV':>8}  {'Err(bp)':>8}" for _ in model_names)
    header = f"{'K/S0':>6}  {'Input IV':>8}  " + "  ".join(
        f"{n:>18}" for n in model_names
    )
    sub = f"{'':6}  {'':8}  " + "  ".join(f"{'IV':>8}  {'Err(bp)':>8}" for _ in model_names)
    sep = "-" * len(sub)

    print(f"\nT = {T:.1f}y")
    print(header)
    print(sub)
    print(sep)

    for j, (k_frac, iv_in) in enumerate(zip(moneyness, input_ivs)):
        row = f"{k_frac:>6.2f}  {iv_in:>7.2%}  "
        for name in model_names:
            m_iv = model_ivs[name][j]
            if np.isnan(m_iv):
                row += f"{'  N/A  ':>8}  {'':>8}  "
            else:
                err_bp = (m_iv - iv_in) * 10_000
                row += f"  {m_iv:>6.2%}  {err_bp:>+8.1f}  "
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vol surface fit diagnostic")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-paths", type=int, default=DEFAULT_N_PATHS,
                        help=f"MC paths per (model, maturity) (default {DEFAULT_N_PATHS:,})")
    parser.add_argument("--output", default="diagnostics/vol_surface_fit.png",
                        help="Path to save the smile plot")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mc_cfg = cfg["market"]
    sc_cfg = cfg["simulation"]

    SPOT = float(mc_cfg["spot"])
    RATE = float(mc_cfg["rate"])
    DIV_YIELD = float(mc_cfg["div_yield"])
    SEED = int(sc_cfg["seed"])
    ANTITHETIC = bool(sc_cfg.get("antithetic", True))
    N_PATHS = args.n_paths

    print("=" * 65)
    print("Model Quality Diagnostic — Vol Surface Fit")
    print("=" * 65)
    print(f"  Config:   {args.config}")
    print(f"  Spot: {SPOT}  |  Rate: {RATE:.1%}  |  Div yield: {DIV_YIELD:.1%}")
    print(f"  N paths per (model, maturity): {N_PATHS:,}")
    print()

    surface = build_surface(cfg, SPOT, RATE, DIV_YIELD)
    T_min, T_max_surf = surface.maturity_range

    print("Calibrating models...")
    models = build_models(cfg, surface, SPOT, RATE, DIV_YIELD, SEED)
    model_names = [name for name, _ in models]
    print()

    # Filter maturities to surface range
    maturities = [T for T in DEFAULT_MATURITIES if T_min < T <= T_max_surf]
    if not maturities:
        maturities = [T_max_surf]

    strikes = SPOT * DEFAULT_MONEYNESS

    # ── Input surface IVs ──────────────────────────────────────────────────────
    input_ivs_by_T: dict[float, np.ndarray] = {
        T: np.array([surface.implied_vol(T, K) for K in strikes])
        for T in maturities
    }

    # ── Model-implied vols ─────────────────────────────────────────────────────
    model_ivs_by_T: dict[float, dict[str, np.ndarray]] = {T: {} for T in maturities}

    for name, model in models:
        for T in maturities:
            print(f"  Pricing smile: {name:10s}  T={T:.1f}y  ({N_PATHS:,} paths)...", end=" ", flush=True)
            ivs = price_smile(model, SPOT, strikes, T, RATE, DIV_YIELD, N_PATHS, ANTITHETIC)
            model_ivs_by_T[T][name] = ivs
            n_valid = int(np.sum(~np.isnan(ivs)))
            print(f"done ({n_valid}/{len(strikes)} IVs solved)")

    # ── Print tables ───────────────────────────────────────────────────────────
    print()
    for T in maturities:
        print_table(T, DEFAULT_MONEYNESS, input_ivs_by_T[T], model_names, model_ivs_by_T[T])

    # ── Plot ───────────────────────────────────────────────────────────────────
    n_T = len(maturities)
    fig, axes = plt.subplots(1, n_T, figsize=(5 * n_T, 4), sharey=False)
    if n_T == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1))]

    for ax, T in zip(axes, maturities):
        x = DEFAULT_MONEYNESS * 100   # percentage

        ax.plot(x, input_ivs_by_T[T] * 100, "k-", lw=2.5, label="Input surface", zorder=5)

        for (name, color, ls) in zip(model_names, colors, linestyles):
            ivs = model_ivs_by_T[T][name] * 100
            mask = ~np.isnan(ivs)
            ax.plot(x[mask], ivs[mask], color=color, ls=ls, lw=1.8,
                    marker="o", markersize=3.5, label=name)

        ax.set_title(f"T = {T:.1f}y", fontsize=11, fontweight="bold")
        ax.set_xlabel("Moneyness  K / S₀  (%)", fontsize=9)
        ax.set_ylabel("Implied Vol (%)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")

    fig.suptitle("Vol Surface Fit: Input vs Model-Implied Vols", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = args.output
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
