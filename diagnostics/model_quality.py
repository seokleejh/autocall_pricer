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

Scenario mode -- fit check + raw vol surface shape for every scenario in a
scenarios YAML file (same base_config/overrides format as scenarios/run_scenarios.py):
    .venv/bin/python diagnostics/model_quality.py --scenarios scenarios/sce_std_els_by_market.yaml
    .venv/bin/python diagnostics/model_quality.py --scenarios scenarios/sce_std_els_by_market.yaml --group market
"""

from __future__ import annotations

import re
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
from models.lsv import LSVModel
from calibration.lsv_calibration import leverage_from_config, lsv_settings
from engine.black_scholes import bs_price, implied_vol as bs_implied_vol
from scenarios.run_scenarios import deep_merge

# ── Diagnostic grid defaults ───────────────────────────────────────────────────
DEFAULT_MATURITIES = [0.1, 0.2, 0.3, 0.5]
DEFAULT_MONEYNESS = np.linspace(0.70, 1.30, 13)   # K/S0
DEFAULT_N_PATHS = 30_000
# Scenario mode runs the fit check N times (once per scenario) -- default to
# fewer paths than the single-config default to keep total runtime reasonable.
SCENARIO_DEFAULT_N_PATHS = 10_000
# Raw surface shape needs no MC/calibration, so it can afford a much finer grid.
SURFACE_PLOT_N_T = 60
SURFACE_PLOT_N_K = 60


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
                 div_yield: float, seed: int,
                 t_max: float = 1.0) -> list[tuple[str, object]]:
    """
    Return list of (name, model) for each enabled model.

    t_max : longest maturity that will be priced. Only LSV needs it -- its
            leverage function is calibrated forward in time and must cover
            the whole horizon, unlike the other models which are stateless
            in time.
    """
    models_cfg = cfg.get("models", {})
    models = []
    heston_params = None   # reused by LSV, which builds on the same calibration

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
        heston_params = params
        m = HestonModel(params=params, spot=spot, rate=rate, div_yield=div_yield,
                        steps_per_year=52, seed=seed)
        models.append(("Heston", m))

    if models_cfg.get("lsv", False):
        try:
            print("  Calibrating LSV...", end=" ", flush=True)
            hp = heston_params
            if hp is None:
                hp = calibrate_heston(surface, n_calibration_points=36)
            leverage = leverage_from_config(cfg, surface, hp, T_max=t_max, seed=seed)
            lo, hi = leverage.value_range()
            print(f"L in [{lo:.4f}, {hi:.4f}] over {leverage.n_slices} slices")
            m = LSVModel(params=hp, leverage=leverage, spot=spot, rate=rate,
                         div_yield=div_yield,
                         steps_per_year=int(lsv_settings(cfg)["steps_per_year"]),
                         seed=seed)
            models.append(("LSV", m))
        except Exception as e:
            print(f"failed ({e}), skipping LSV")

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


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "scenario"


def plot_raw_surface(surface, title: str, out_path: str,
                     n_T: int = SURFACE_PLOT_N_T, n_K: int = SURFACE_PLOT_N_K,
                     moneyness_range: tuple[float, float] = (0.6, 1.4)) -> None:
    """
    Heatmap of the input implied-vol surface itself (no MC, no calibration) --
    shows exactly what shape the vol_surface: parameters actually produce.

    moneyness_range is a practical display window, NOT surface.strike_range --
    the surface's internal strike grid is auto-sized to cover +/-4.5 sigma
    (often 20-1200% moneyness) purely so the Dupire spline never has to
    extrapolate; that range is unreadable as a plot.
    """
    T_min, T_max_surf = surface.maturity_range
    T_min = max(T_min, 1e-3)
    Ts = np.linspace(T_min, T_max_surf, n_T)
    Ks = np.linspace(surface.spot * moneyness_range[0], surface.spot * moneyness_range[1], n_K)

    iv_grid = np.array([surface.implied_vol(T, Ks) for T in Ts])  # (n_T, n_K)

    fig, ax = plt.subplots(figsize=(7, 5))
    moneyness = Ks / surface.spot * 100
    mesh = ax.pcolormesh(moneyness, Ts, iv_grid * 100, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="Implied Vol (%)")
    ax.set_xlabel("Moneyness  K / S₀  (%)")
    ax.set_ylabel("Maturity T (years)")
    ax.set_title(f"Input Vol Surface — {title}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_summary_line(model_names: list[str],
                     input_ivs_by_T: dict[float, np.ndarray],
                     model_ivs_by_T: dict[float, dict[str, np.ndarray]]) -> str:
    """One line per scenario: mean/max abs IV error (bp) per model, across all maturities/strikes."""
    parts = []
    for name in model_names:
        diffs = np.concatenate([
            (model_ivs_by_T[T][name] - input_ivs_by_T[T]) * 10_000
            for T in input_ivs_by_T
        ])
        valid = diffs[~np.isnan(diffs)]
        if len(valid):
            parts.append(f"{name}: mean={valid.mean():+.1f}bp max={np.max(np.abs(valid)):.1f}bp")
        else:
            parts.append(f"{name}: N/A")
    return "  |  ".join(parts)


def run_fit_check(
    cfg: dict,
    label: str,
    n_paths: int,
    plot_path: str,
) -> str | None:
    """
    Run the fit check (calibrate + MC smile pricing vs input surface) for one
    config, print the per-strike tables, save the fit plot, and return the
    one-line error summary (or None if this is a basket config, which this
    diagnostic doesn't support).
    """
    if "assets" in cfg:
        print(f"  [{label}] basket config -- fit check not supported for multi-asset configs, skipping")
        return None

    mc_cfg = cfg["market"]
    sc_cfg = cfg["simulation"]

    SPOT = float(mc_cfg["spot"])
    RATE = float(mc_cfg["rate"])
    DIV_YIELD = float(mc_cfg["div_yield"])
    SEED = int(sc_cfg["seed"])
    ANTITHETIC = bool(sc_cfg.get("antithetic", True))

    surface = build_surface(cfg, SPOT, RATE, DIV_YIELD)
    T_min, T_max_surf = surface.maturity_range

    # Maturities are resolved BEFORE building models: LSV's leverage function
    # is calibrated forward in time, so it needs to know the horizon up front.
    maturities = [T for T in DEFAULT_MATURITIES if T_min < T <= T_max_surf]
    if not maturities:
        maturities = [T_max_surf]

    print(f"  Calibrating models for [{label}]...")
    models = build_models(cfg, surface, SPOT, RATE, DIV_YIELD, SEED,
                          t_max=float(max(maturities)))
    model_names = [name for name, _ in models]

    strikes = SPOT * DEFAULT_MONEYNESS

    input_ivs_by_T: dict[float, np.ndarray] = {
        T: np.array([surface.implied_vol(T, K) for K in strikes])
        for T in maturities
    }

    model_ivs_by_T: dict[float, dict[str, np.ndarray]] = {T: {} for T in maturities}
    for name, model in models:
        for T in maturities:
            print(f"    Pricing smile: {name:10s}  T={T:.1f}y  ({n_paths:,} paths)...", end=" ", flush=True)
            ivs = price_smile(model, SPOT, strikes, T, RATE, DIV_YIELD, n_paths, ANTITHETIC)
            model_ivs_by_T[T][name] = ivs
            n_valid = int(np.sum(~np.isnan(ivs)))
            print(f"done ({n_valid}/{len(strikes)} IVs solved)")

    print()
    for T in maturities:
        print_table(T, DEFAULT_MONEYNESS, input_ivs_by_T[T], model_names, model_ivs_by_T[T])

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

    fig.suptitle(f"Vol Surface Fit: Input vs Model-Implied Vols — {label}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {plot_path}")

    return fit_summary_line(model_names, input_ivs_by_T, model_ivs_by_T)


def run_scenarios_mode(args: argparse.Namespace) -> None:
    with open(args.scenarios) as f:
        spec = yaml.safe_load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(args.scenarios)))
    base_path = os.path.join(project_root, spec["base_config"])
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    n_paths = args.n_paths if args.n_paths is not None else SCENARIO_DEFAULT_N_PATHS
    scenarios = spec["scenarios"]
    if args.group:
        scenarios = [s for s in scenarios if s.get("group") == args.group]

    print("=" * 65)
    print("Model Quality Diagnostic — Scenario Fit Check")
    print("=" * 65)
    print(f"  Scenarios:  {args.scenarios}")
    print(f"  Base config: {spec['base_config']}")
    print(f"  N paths per (model, maturity): {n_paths:,}")
    print(f"  Output dir:  {args.output_dir}")
    print()

    summaries: list[tuple[str, str, str | None]] = []   # (name, group, summary)

    for idx, scenario in enumerate(scenarios):
        name = scenario["name"]
        group = scenario.get("group", "")
        overrides = scenario.get("overrides", {})
        cfg = deep_merge(base_cfg, overrides)
        slug = slugify(name)

        print(f"[{idx + 1}/{len(scenarios)}] {group.upper():7s} | {name}")

        if "assets" not in cfg:
            surface = build_surface(cfg, float(cfg["market"]["spot"]),
                                    float(cfg["market"]["rate"]),
                                    float(cfg["market"]["div_yield"]))
            plot_raw_surface(surface, name, os.path.join(args.output_dir, f"{slug}_surface.png"))

        summary = run_fit_check(
            cfg, name, n_paths,
            os.path.join(args.output_dir, f"{slug}_fit.png"),
        )
        summaries.append((name, group, summary))
        print()

    print("=" * 65)
    print("  Fit Summary (mean/max abs IV error across all maturities & strikes)")
    print("=" * 65)
    current_group = None
    for name, group, summary in summaries:
        if group != current_group:
            current_group = group
            print(f"\n  ── {group.upper()} ──")
        if summary is None:
            print(f"  {name:<30}  (skipped -- basket config)")
        else:
            print(f"  {name:<30}  {summary}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vol surface fit diagnostic")
    parser.add_argument("--config", default="config.yaml",
                        help="Single config to check (ignored if --scenarios is given)")
    parser.add_argument("--scenarios", default=None,
                        help="Scenario YAML (base_config + scenarios:, same format as "
                             "scenarios/run_scenarios.py) -- runs the fit check and raw "
                             "surface plot for every scenario instead of a single config")
    parser.add_argument("--group", default=None,
                        help="Only run scenarios in this group (--scenarios mode only)")
    parser.add_argument("--n-paths", type=int, default=None,
                        help=f"MC paths per (model, maturity) (default {DEFAULT_N_PATHS:,} "
                             f"single-config, {SCENARIO_DEFAULT_N_PATHS:,} scenario mode)")
    parser.add_argument("--output", default="diagnostics/vol_surface_fit.png",
                        help="Path to save the smile plot (single-config mode only)")
    parser.add_argument("--output-dir", default="diagnostics/scenario_fit",
                        help="Directory to save per-scenario plots (--scenarios mode only)")
    args = parser.parse_args()

    if args.scenarios:
        run_scenarios_mode(args)
        return

    cfg = load_config(args.config)
    n_paths = args.n_paths if args.n_paths is not None else DEFAULT_N_PATHS

    print("=" * 65)
    print("Model Quality Diagnostic — Vol Surface Fit")
    print("=" * 65)
    print(f"  Config:   {args.config}")
    print(f"  Spot: {float(cfg['market']['spot'])}  |  Rate: {float(cfg['market']['rate']):.1%}"
          f"  |  Div yield: {float(cfg['market']['div_yield']):.1%}")
    print(f"  N paths per (model, maturity): {n_paths:,}")
    print()

    run_fit_check(cfg, args.config, n_paths, args.output)


if __name__ == "__main__":
    main()
