# Autocallable Pricer — User Manual

## Table of Contents

1. [Overview](#1-overview)
2. [Setup](#2-setup)
3. [Quick Start](#3-quick-start)
4. [Configuration Reference](#4-configuration-reference)
   - [market](#41-market)
   - [simulation](#42-simulation)
   - [vol_surface](#43-vol_surface)
   - [product](#44-product)
   - [models](#45-models)
5. [Running a Single Pricing](#5-running-a-single-pricing)
   - [Fit check modes](#51-fit-check-modes)
   - [Calibration diagnostics](#52-calibration-diagnostics)
   - [Scenario fit check and surface shapes](#53-scenario-fit-check-and-surface-shapes)
6. [Scenario Analysis](#6-scenario-analysis)
   - [Basic run](#61-basic-run)
   - [Computing Greeks](#62-computing-greeks)
   - [Reading the output](#63-reading-the-output)
   - [CSV output](#64-csv-output)
7. [Writing Custom Scenarios](#7-writing-custom-scenarios)
8. [Python API](#8-python-api)
9. [Running the Tests](#9-running-the-tests)
10. [Performance Guide](#10-performance-guide)
11. [Worst-of Basket Pricing](#11-worst-of-basket-pricing)
12. [Numerical Corrections](#12-numerical-corrections)

---

> ### ⚠ Prices have shifted since the LSV release
>
> Two numerical defects were found and fixed while adding LSV. **Results produced before that
> change are not directly comparable to results produced after it.** Both are documented in
> [§12 Numerical Corrections](#12-numerical-corrections):
>
> - **Dupire local vol** was computed with `∂w/∂T` taken at fixed strike where the formula
>   requires fixed log-moneyness. This biased every skewed-surface price. Local Vol's vanilla
>   round-trip improved from ~21bp RMS to ~2bp once corrected.
> - **Antithetic variates were not antithetic** in any model — the mirror half drew fresh
>   random numbers instead of reusing the partner's. Prices were unbiased, but the advertised
>   variance reduction was absent and reported standard errors were too wide.
>
> Regenerate any saved `results.csv` you intend to rely on.

---

## 1. Overview

This tool prices autocallable structured notes (FCN, EKI, Phoenix, Snowball variants) under four
volatility models simultaneously and reports the **model spread** — the price difference arising
solely from the choice of dynamics — as a measure of model risk.

### Four models

| Model | Dynamics | Surface fit | Typical use |
|-------|----------|-------------|-------------|
| **Local Vol** (Dupire) | Deterministic σ(t, S) derived from market vols via Dupire's formula | Exact by construction | Model risk lower bound; conservative barrier pricing |
| **Heston SV** | Mean-reverting stochastic variance (CIR process); spot-vol correlation ρ | Global fit over all maturities and strikes | Captures forward skew dynamics; well-suited for long-dated autocalls |
| **Heston-LSV** | Heston, with a leverage function L(t, S) on the spot diffusion | Near-exact — matches Local Vol, ~2× better than Heston | **The production choice.** Exact surface fit *and* realistic forward dynamics |
| **SABR SV** | Log-normal stochastic vol; backbone β = 1 | Calibrated to a single maturity slice (default: 1y) | **Disabled by default.** See note below. |

All are calibrated to the same implied vol surface, so differences in price are not due to a vol
level mismatch — they reflect genuine disagreement about the joint distribution of path and final payoff.

> **Why LSV matters for autocallables**: an autocall is a sequence of barrier decisions on future
> dates, so its price depends both on today's smile (which fixes the barrier levels) and on how
> that smile evolves (which fixes the crossing probabilities). Local Vol gets the first right and
> the second wrong — its forward smile flattens and vol-of-vol is zero. Heston gets the second
> right but, with only five parameters, leaves a smile residual that is an uncontrolled
> mispricing of the barriers. LSV multiplies the Heston diffusion by a deterministic leverage
> function
>
> ```
> L(t, S)² = σ_LV(t, S)² / E[ V_t | S_t = S ]
> ```
>
> chosen so the model reprices the whole vanilla surface while (κ, θ, ξ, ρ, v₀) continue to carry
> the forward dynamics. On the default config surface, measured over 0.25–3y and ±20% moneyness:
> Local Vol 11.7bp RMS, **LSV 12.8bp**, Heston 27.4bp. LSV buys Heston's dynamics at essentially
> Local Vol's fit quality.
>
> The leverage function is calibrated by the **particle method** (Guyon & Henry-Labordère, 2011):
> an ensemble of particles is walked forward, and at each time slice the conditional expectation
> `E[V|S]` is estimated by averaging variance within equal-count spot bins. Cost is roughly one
> second per surface at 50k particles. See `calibration/lsv_calibration.py`.

> **Why SABR is disabled by default**: SABR's single `alpha` parameter produces a nearly flat ATM
> vol term structure, but equity surfaces have an exponential ATM decay (high short-term vol
> converging to a lower long-run level). On any realistic term-structure surface, SABR will
> systematically underprice vol at short maturities (≥150 bp at 0.5y is typical) and overprice
> it at long maturities, leading to mispriced barrier crossing probabilities at each observation
> date. SABR was designed for single-maturity vanilla books, not for multi-date path-dependent
> payoffs. Enable it with `sabr: true` in `config.yaml` for research or model comparison only.

### Two entry points

| Script | Purpose |
|--------|---------|
| `main.py` | Price a single note; inspect calibration quality |
| `scenarios/run_scenarios.py` | Run a battery of scenario overrides; compare model spreads across product variants or market regimes |

Both entry points also support **worst-of basket** notes (payoff depends on the worst-performing
of several correlated underlyings) — see [Section 11](#11-worst-of-basket-pricing).

---

## 2. Setup

**Prerequisites:** Python 3.10+, a Unix-like shell (Linux, macOS, WSL).

On native Windows (no WSL), see [WINDOWS_SETUP.md](WINDOWS_SETUP.md) instead —
same steps, adapted for PowerShell/CMD.

```bash
# 1. Clone / enter the project directory
cd autocall-pricer

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install numpy scipy pyyaml matplotlib pytest

# 4. Quick sanity check
python main.py --no-fit-check
```

A successful run prints calibrated model parameters followed by one pricing line per enabled model.

---

## 3. Quick Start

Price the note defined in `config.yaml` under all enabled models:

```bash
python main.py
```

Run a full scenario sweep and save results to CSV:

```bash
python scenarios/run_scenarios.py --n-paths 20000 --output results.csv
```

Add Greeks (delta, gamma, vega, vanna, expected duration is always computed):

```bash
python scenarios/run_scenarios.py --greeks --n-paths 20000 --n-paths-greeks 10000
```

---

## 4. Configuration Reference

All parameters live in a YAML file. The default is `config.yaml`. Copy and edit it freely — the
original is the baseline for scenario overrides and should not normally be modified for a single deal.

```bash
cp config.yaml my_deal.yaml
python main.py --config my_deal.yaml
```

All rates, vols, barriers and coupons are **decimals** (0.20 = 20%). All times are in **years**.

---

### 4.1 `market`

```yaml
market:
  spot: 100.0       # Current spot price S₀. All performances = S(t) / spot.
  rate: 0.03        # Continuously compounded risk-free rate.
  div_yield: 0.01   # Continuous dividend yield. Drift = rate - div_yield.
```

`spot` is the reference level for **all** performance and barrier calculations. When comparing
models across scenarios, keep `spot` fixed so that dollar-level barriers are consistent.

---

### 4.2 `simulation`

```yaml
simulation:
  n_paths: 10000    # MC paths per (scenario, model) call.
  seed: 42          # Fixed seed for reproducibility. Change to test MC stability.
  antithetic: true  # Antithetic variates — recommended; halves variance at near-zero cost.
```

Standard error scales as 1/√n_paths. With antithetic variates enabled:

| n_paths | Typical SE (price) |
|---------|-------------------|
| 5,000   | ~3–5 bp |
| 20,000  | ~1–2 bp |
| 100,000 | ~0.3–0.5 bp |

---

### 4.3 `vol_surface`

The vol surface defines σ_imp(T, K) — the implied vol grid that all three models are calibrated to.

#### Surface types

**`term_structure`** (recommended):

ATM vol follows an exponential term structure; skew steepens at short maturities (∝ 1/√T), matching
equity market empirics. All five parameters below apply.

```yaml
vol_surface:
  type: term_structure
  vol_short: 0.25   # ATM implied vol as T → 0 (short-term realized vol level).
  vol_long:  0.18   # ATM implied vol as T → ∞ (long-run equilibrium).
  kappa:     1.5    # ATM convergence speed. Half-life ≈ ln(2)/kappa years.
                    # kappa=1.5 → half-life ≈ 0.46y; kappa=0.5 → half-life ≈ 1.4y.
  skew:     -0.10   # Skew at T = 1y (vol per unit log-moneyness). Negative = put premium.
                    # At maturity T: effective skew = skew / √T.
  convexity: 0.02   # Smile curvature. Contribution: convexity × ln(K/F(T))².
  T_max:     5.0    # Surface grid extends to this maturity (must ≥ product maturity).
```

**`skewed`**: flat ATM vol across all maturities with a constant skew slope. Simpler but less
realistic — the 6-month and 3-year smile have the same shape.

```yaml
vol_surface:
  type: skewed
  atm_vol: 0.20
  skew:   -0.10
  T_max:   5.0
```

**`flat`**: constant σ everywhere. Use only for model validation — all three models should agree.

```yaml
vol_surface:
  type: flat
  atm_vol: 0.20
  T_max:   5.0
```

#### Advanced: strike grid

The surface is evaluated on an (n_T × n_K) grid. By default the strike range is auto-sized to
±4.5σ at T_max. Override only if you see spline extrapolation warnings at extreme paths:

```yaml
vol_surface:
  moneyness_lo: 0.05   # lower boundary as fraction of spot
  moneyness_hi: 5.00   # upper boundary as fraction of spot
  n_T: 30
  n_K: 60
```

---

### 4.4 `product`

```yaml
product:
  notional: 1.0
  maturity: 3.0
  observation_dates: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # must end at maturity
  autocall_barriers: [0.80, 0.80, 0.75, 0.70, 0.65, 0.60]
  coupon_rate:       [0.07, 0.14, 0.21, 0.28, 0.35, 0.42]
  conditional_coupon: false
  coupon_barrier: 0.80
  capital_barrier_active: true
  capital_barrier: 0.60
```

#### Autocall trigger

At each observation date t_i the note terminates early if:

```
S(t_i) / spot  ≥  autocall_barriers[i]
```

Payoff on autocall: `notional × (1 + coupon_rate[i])` discounted to today.

`autocall_barriers` accepts either a **scalar** (same barrier every date) or a **list** (step-down
or step-up schedule, one entry per observation date).

#### Coupon structures

| `coupon_rate` | `conditional_coupon` | Structure |
|---|---|---|
| scalar, `false` | Standard autocall — single coupon paid on early exit |
| increasing list, `false` | Snowball — missed coupons accumulate |
| scalar, `true` | Phoenix — periodic income paid at each observation where stock > `coupon_barrier`, regardless of autocall |

**Snowball example** — 7% per semi-annual period:
```yaml
coupon_rate: [0.07, 0.14, 0.21, 0.28, 0.35, 0.42]
```
Autocalling at date 3 (t = 1.5y) pays 21% of notional.

**Phoenix example** — flat 7% periodic coupon:
```yaml
conditional_coupon: true
coupon_rate: 0.07
coupon_barrier: 0.80
```
At each observation: if `S(t)/spot ≥ 0.80`, investor receives 7% of notional. If the note also
autocalls on the same date, investor receives the periodic coupon *and* the autocall redemption.

#### Capital protection at maturity

If the note survives all observation dates without autocalling:

- `capital_barrier_active: false` → investor always receives full notional (capital-guaranteed)
- `capital_barrier_active: true` → payoff depends on final performance:
  - `S(T)/spot ≥ capital_barrier` → full notional returned
  - `S(T)/spot < capital_barrier` → notional × S(T)/spot (loss proportional to drawdown)

The `capital_barrier` is expressed as a fraction of initial spot (e.g. 0.60 = investor loses
capital only if the stock finishes more than 40% below its initial level).

---

### 4.5 `models`

```yaml
models:
  local_vol: true   # recommended
  heston:    true   # recommended
  lsv:       true   # recommended — best surface fit with realistic dynamics
  sabr:      false  # disabled by default — see Section 1 for rationale
```

Set any to `false` to skip that model. Running with only one model is faster and useful for
sensitivity testing. SABR can be re-enabled for research or comparison runs, but its prices
should not be used for production autocallable risk management.

LSV builds on the Heston calibration, so enabling `lsv` triggers a Heston fit even when
`heston: false` — you can price LSV without also reporting plain Heston.

#### LSV leverage calibration settings

```yaml
lsv:
  n_particles: 50000      # ensemble size for E[V|S]
  n_spot_bins: 28         # spot nodes per time slice
  leverage_cap: [0.1, 10.0]
  steps_per_year: 52      # should match the simulation grid
  sticky_leverage: true   # see below — affects LSV delta
```

| Setting | Effect |
|---|---|
| `n_particles` | Accuracy plateaus around 10–50k. Beyond that the residual is time discretisation, not particle noise, so raising it further buys nothing. |
| `n_spot_bins` | Particles are split into equal-**count** bins, so bins can never be empty and the wings cannot produce a spurious leverage spike. 25–30 is the usual tradeoff. |
| `leverage_cap` | A safety rail, not a tuning knob: in sparse wings `E[V|S]` can collapse toward zero and send L to infinity. If the cap binds anywhere but the extreme wings, the calibration is unhealthy. Run with `--verbose` to see `clipped_fraction`. |
| `steps_per_year` | Raising both this and the simulation grid to 104 tightens the vanilla round-trip from ~6bp to ~4bp RMS, at roughly double the calibration cost. |
| `sticky_leverage` | **Changes what LSV delta means** — see below. |

> **`sticky_leverage` is a modelling choice, not just an optimisation.** When `true` (the
> default), the leverage surface is held fixed as spot is bumped for delta and gamma, so the
> spot-bumped repricings reuse the base leverage function. When `false`, the leverage is
> recalibrated on every bumped surface — six extra particle calibrations per scenario, and a
> different (larger) delta. Vega and skew sensitivity always recalibrate, in both settings,
> because a vol bump changes the surface the leverage is defined against.
>
> Turning it off does **not** affect Local Vol, Heston or SABR deltas: the re-levering path
> deliberately holds their parameters frozen so that an LSV setting cannot change another
> model's reported risk.

---

## 5. Running a Single Pricing

```bash
python main.py [--config FILE] [--no-fit-check | --full-diagnostic]
```

### 5.1 Fit check modes

Three levels of calibration quality check are available:

| Flag | Speed | Method | Use case |
|------|-------|--------|----------|
| *(default)* | ~1s | Analytical (Heston CF, SABR Hagan) | Daily use — quick sanity check |
| `--no-fit-check` | 0s | None | Scripted runs where you trust the config |
| `--full-diagnostic` | ~2–5 min | Monte Carlo smile recovery | Before pricing a new deal structure |

The **default analytical check** evaluates model-implied vols at a small (T, K) grid using closed
forms — the Heston characteristic function for Heston and the Hagan asymptotic formula for SABR —
and reports RMSE and worst-case error in basis points. Local Vol always fits exactly by construction.

```
Vol Surface Fit — Analytical Check
  Local Vol   exact fit (by construction)
  Heston  RMSE:  12.3bp  max:  +28.4bp  (T=0.5y, K/S=70%)
  SABR    RMSE:   8.1bp  max:  -19.2bp  (T=3.0y, K/S=130%)
```

This tells you whether calibration converged well, but it uses the same formula as the calibration
objective — it cannot detect systematic model mis-specification. Use `--full-diagnostic` for that.

### 5.2 Calibration diagnostics

`diagnostics/model_quality.py` is the heavy-duty check. It prices OTM vanilla options using full
Monte Carlo paths, inverts to implied vols, and compares against the input surface. This tests
whether the model dynamics actually reproduce the input smile in-sample, not just whether the
calibration objective converged.

```bash
# Run standalone (uses config.yaml by default)
python diagnostics/model_quality.py

# Custom config, more paths, save plot to a specific location
python diagnostics/model_quality.py \
    --config my_deal.yaml \
    --n-paths 50000 \
    --output results/fit_check.png

# Trigger from main.py (calibrates models once, then runs the diagnostic)
python main.py --full-diagnostic
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Config file (market, surface, models) |
| `--n-paths` | 30,000 | MC paths per (model, maturity) pair |
| `--output` | `diagnostics/vol_surface_fit.png` | PNG plot output path |

#### What it evaluates

The diagnostic uses a fixed grid regardless of your product structure:

- **Maturities**: 0.1y, 0.2y, 0.3y, 0.5y (short-end of the surface where calibration is hardest)
- **Strikes**: 70% to 130% of spot in 13 steps (moneyness grid)
- **OTM convention**: puts for K < spot, calls for K ≥ spot — avoids inverting deep-ITM options
  where vega is negligible and MC noise dominates the implied vol estimate

For each (model, maturity) pair:
1. Simulate `n_paths` paths to that maturity (paths are reused across all strikes — no extra simulation per strike)
2. Evaluate OTM payoffs for each strike
3. Invert to implied vol via Black-Scholes

#### Reading the table output

```
T = 0.3y
K/S0    Input IV   Local Vol          Heston             SABR
                    IV   Err(bp)       IV   Err(bp)       IV   Err(bp)
--------------------------------------------------------------------
  0.70   28.15%   28.12%    -3.2     27.44%   -70.8     27.91%   -24.3
  0.80   24.10%   24.08%    -2.1     23.81%   -29.4     24.01%    -9.1
  0.90   20.88%   20.87%    -1.2     20.74%   -13.5     20.86%    -1.9
  1.00   18.32%   18.31%    -0.9     18.28%    -4.1     18.30%    -1.8
  1.10   17.21%   17.20%    -0.8     17.26%    +5.2     17.19%    -1.5
  1.20   17.54%   17.52%    -1.6     17.68%   +14.1     17.51%    -3.1
  1.30   18.78%   18.76%    -1.9     18.99%   +20.8     18.74%    -3.8
```

- **Input IV**: the market surface value (ground truth)
- **IV**: model MC-implied vol (with MC noise)
- **Err(bp)**: model − market in basis points. Positive = model overprices vol at that strike.

**Interpreting errors by model:**

| Model | Expected behaviour | Warning sign |
|---|---|---|
| Local Vol | Near-zero errors everywhere (by construction) | Errors > 5 bp indicate spline interpolation artifacts or paths near the grid boundary |
| Heston | Errors grow at very short maturities and deep strikes (one-factor SV limits) | RMSE > 30 bp at T ≤ 0.3y means the model will misprice early-date barriers |
| SABR | Calibrated at 1y; may have larger errors at 0.1–0.3y and at very deep OTM | Errors > 50 bp at T < 0.3y are expected; errors > 50 bp at T = 0.5y near ATM deserve attention |

#### The smile plot

The PNG shows one subplot per maturity, each with:
- **Black solid line**: input surface (market)
- **Dashed / dash-dot / dotted lines**: Local Vol, Heston, SABR model-implied vols

A well-calibrated model's line should closely track the black line. Systematic bowing (model curves
above or below the market across all strikes at a given maturity) usually means the ATM vol level
is off. Crossing lines (model fits OTM puts but not OTM calls, or vice versa) usually means the
skew is wrong.

#### When to re-run before pricing

- Whenever you change `vol_surface` parameters in the config
- If Heston or SABR calibration printed warnings during `main.py`
- Before pricing a new product with barriers significantly different from ATM (deep OTM/ITM
  paths are where model errors matter most)

### 5.3 Scenario fit check and surface shapes

Section 5.2 checks calibration fit for one config. To check every market scenario in a scenario
file in one pass — and to see the actual vol surface shape each scenario's parameters produce —
use `--scenarios` instead of `--config`:

```bash
python diagnostics/model_quality.py --scenarios scenarios/sce_std_els_by_market.yaml

# Only the "market" group, fewer paths for a quick look
python diagnostics/model_quality.py --scenarios scenarios/sce_std_els_by_market.yaml \
    --group market --n-paths 5000
```

For every scenario, this:
1. Applies the scenario's `overrides` to `base_config` the same way `scenarios/run_scenarios.py`
   does (`deep_merge`), so scenario files are interchangeable between the two tools.
2. Saves a **raw vol surface heatmap** — implied vol evaluated directly on a (T, K) grid from the
   `vol_surface:` parameters, with no Monte Carlo and no model calibration involved. This is the
   fastest way to sanity-check that a parameter change (e.g. `skew`, `kappa`, `convexity`) produced
   the surface shape you expected, independent of how well any model fits it.
3. Runs the same fit check as [Section 5.2](#52-calibration-diagnostics) (calibrate Heston/SABR,
   MC-price the smile, compare to the input surface) and saves the usual fit plot and per-strike
   tables.
4. Prints a **one-line summary per scenario** — mean and max absolute IV error in bp per model,
   across all maturities and strikes — so you can scan every scenario for outliers without reading
   every per-strike table.

```
=================================================================
  Fit Summary (mean/max abs IV error across all maturities & strikes)
=================================================================

  ── MARKET ──
  Base case                       Local Vol: mean=-1.7bp max=273.6bp  |  Heston: mean=+28.5bp max=296.4bp
  Steep skew                      Local Vol: mean=-5.3bp max=561.1bp  |  Heston: mean=-315.5bp max=770.3bp
```

A scenario whose max/mean error is far larger than its neighbors (like "Steep skew" above) is where
model risk concentrates — worth a closer look at that scenario's saved fit plot before trusting its
price in a scenario comparison.

| Flag | Default | Description |
|------|---------|-------------|
| `--scenarios` | *(none)* | Scenario YAML (same `base_config:` + `scenarios:` format as `run_scenarios.py`). Enables scenario mode; `--config` is ignored if this is set. |
| `--group` | *(all)* | Only run scenarios whose `group:` matches this value |
| `--n-paths` | 10,000 | MC paths per (model, maturity) pair — lower than the single-config default (30,000) since this runs once per scenario |
| `--output-dir` | `diagnostics/scenario_fit` | Directory for per-scenario plots: `<scenario>_surface.png` and `<scenario>_fit.png` |

**Basket configs** (`assets:` present after merging overrides) are detected and skipped with a
console note — this diagnostic's calibration/fit-check logic is single-asset only (see
[Section 11](#11-worst-of-basket-pricing) for basket pricing).

**Surface plot display window**: the heatmap always uses a 60%–140% moneyness window, regardless
of the surface's *internal* strike grid. `term_structure_surface`'s internal grid is auto-sized to
cover ±4.5σ (often 20%–1200%+ moneyness) purely so the Dupire local-vol spline never has to
extrapolate — plotting that full range would compress the readable region into a sliver at the
left edge of the chart.

### Reading the output

```
[Local Vol (Dupire)]  price = 0.952310  ± 0.000831  ...  dur=1.83y
[Heston SV]           price = 0.967401  ± 0.000749  ...  dur=1.62y
[SABR SV]             price = 0.971205  ± 0.000812  ...  dur=1.57y

Model spread (max - min): 0.018895
  High: SABR SV   (0.971205)
  Low:  Local Vol  (0.952310)
```

- **price**: Monte Carlo estimate of the note's fair value as a fraction of notional (e.g. 0.967 = 96.7% of face).
- **± SE**: one standard error of the MC estimate.
- **dur**: risk-neutral expected time to termination in years (see [Section 6.3](#63-reading-the-output)).
- **Model spread**: the gap between the highest and lowest model prices in the same units. Quoted in bp via the scenario runner.

---

## 6. Scenario Analysis

### 6.1 Basic run

```bash
python scenarios/run_scenarios.py \
    --scenarios scenarios/scenarios.yaml \
    --n-paths 20000 \
    --output results.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scenarios` | `scenarios/scenarios.yaml` | Scenario YAML file |
| `--n-paths` | 20,000 | MC paths per (scenario, model) |
| `--output` | `scenarios/results.csv` | CSV output path |
| `--verbose` | off | Print calibrated model parameters per scenario |

### 6.2 Computing Greeks

```bash
python scenarios/run_scenarios.py \
    --greeks \
    --n-paths 20000 \
    --n-paths-greeks 10000 \
    --h-spot-pct 0.01 \
    --h-vol 0.001 \
    --h-skew 0.01
```

| Flag | Default | Description |
|------|---------|-------------|
| `--greeks` | off | Enable finite-difference Greeks |
| `--n-paths-greeks` | same as `--n-paths` | Paths for FD bumps (fewer paths → faster but noisier) |
| `--h-spot-pct` | 0.01 | Spot bump as fraction of spot (1%) |
| `--h-vol` | 0.001 | Parallel implied vol shift for vega/vanna (10 bp) |
| `--h-skew` | 0.01 | Skew-coefficient bump for skew sensitivity (see [below](#skew-sensitivity-convention)) |

**Runtime**: with 10,000 paths and all three models, Greeks add ~2–5 minutes per scenario
(10 bumped evaluations × 3 models, plus 4 vol-surface recalibrations). See [Section 10](#10-performance-guide).

#### Delta convention

Delta is the **sensitivity to the current spot price holding all contract terms fixed in dollar
terms**. If the initial spot was 100 and the autocall barrier was set at 0.95 × 100 = 95, that
dollar level of 95 remains fixed when the spot is bumped. The performance threshold in the bumped
simulation becomes 95 / (100 ± h) rather than 0.95.

This is physically correct: the autocall trigger, KI level, and coupon barrier are all fixed at
contract inception and do not move when today's spot price changes.

#### Vega convention

Vega is reported **per 1 percentage-point (100 bp) parallel shift** in implied vol. So if Heston
shows vega = −0.05, a 1 vol-point increase in implied vols lowers the note value by 5% of notional.

#### Skew sensitivity convention

Skew sensitivity is not a standard Greek — it measures the note's exposure to the **slope** of the
smile (the skew), as distinct from vega's exposure to the smile's **level**. It is computed via
`ImpliedVolSurface.with_skew_shift(h)`, which tilts the surface around each maturity's own
at-the-forward point:

```
new_vol(T, K) = vol(T, K) + [h / sqrt(max(T, 0.25))] * ln(K / F(T))
```

Two properties make this a clean, isolated slope measure:

- **Zero at the forward for every maturity** (`ln(K/F(T)) = 0` when `K = F(T)`), so the bump never
  moves the ATM level and therefore never double-counts with vega.
- **Decays as `1/√T`**, identical to `term_structure_surface`'s own `skew` parameter convention
  (quoted at the T=1y reference, floored at 3 months). This means **short maturities get a larger
  tilt than long ones** — e.g. with the default `h=0.01`, the bump at T=3 months is 4× the size of
  the bump at T=4 years. This matches the empirical fact that skew is steeper (in absolute
  ln-moneyness terms) at the short end, and it lets `skew_sens` correspond to `h` in the same units
  as the config's own `skew:` field, regardless of which factory built the surface (the bump only
  touches the stored vol grid, not the parametric generator).

Like vega, it is reported **per 0.01 shift** in the skew coefficient and requires Heston/SABR to be
re-calibrated on the bumped surface (same fast warm-start approach as the vega bumps).

### 6.3 Reading the output

**Price table**

```
  Autocallable Scenario Summary  —  Price
  ─────────────────────────────────────────────────────────────────────────
  ── MARKET ──
  Scenario                           Local Vol        Heston          SABR  Spread(bp)
  ─────────────────────────────────────────────────────────────────────────────────────
  Base case                           0.952310      0.967401      0.971205       189.0
  High vol                            0.921440      0.938210      0.941320       199.0
```

**Expected Duration table**

```
  ── Expected Duration (years, risk-neutral MC) ──
  Scenario                           Local Vol        Heston          SABR
  ─────────────────────────────────────────────────────────────────────────────────────
  Base case                               1.83          1.62          1.57
```

The expected duration is the average time (in years) until the note terminates, averaged over all
MC paths under the **risk-neutral measure**. Paths that autocall at year 1 contribute 1.0; paths
that survive all dates contribute the full maturity.

This number is **not** a real-world probability-weighted estimate of when the note will call —
the risk-neutral measure overweights paths that are consistent with current option prices, which
typically means more upward drift than the physical measure. Use it to:

- Compare products: a shorter duration means a higher effective reinvestment risk.
- Compare models: divergence in duration (e.g. Heston 1.62y vs. Local Vol 1.83y) reflects that
  the two models disagree on the probability of autocall at each date.
- Sanity-check extreme scenarios: if duration equals maturity for all models, the note essentially
  never autocalls and may be mispriced for an investor expecting frequent calls.

**Greek tables** (when `--greeks` is set)

```
  ── Delta  ∂P/∂S ──
  Scenario                           Local Vol        Heston          SABR
  Base case                          +0.003210     +0.004102     +0.004350

  ── Gamma  ∂²P/∂S² ──
  ...

  ── Vega   ∂P/∂σ  (per 1% vol shift) ──
  ...

  ── Vanna  ∂²P/(∂S ∂σ) ──
  ...

  ── Skew Sensitivity  ∂P/∂skew  (per 0.01 skew shift) ──
  ...
```

All Greeks are per unit notional in the same currency as `spot`.

### 6.4 CSV output

The CSV contains one row per scenario with columns:

```
group, scenario,
<model>_price × 3,
<model>_se × 3,
spread_bp,
<model>_duration × 3,
<model>_delta × 3,   (only if --greeks)
<model>_gamma × 3,
<model>_vega × 3,
<model>_vanna × 3,
<model>_skew_sens × 3
```

Load in Python:

```python
import pandas as pd
df = pd.read_csv("results.csv")
df[["scenario", "Local Vol", "Heston", "SABR", "spread_bp"]]
```

---

## 7. Writing Custom Scenarios

Scenarios are defined in a YAML file. Each scenario specifies only the fields that differ from the
base config — everything else is inherited.

### Structure

```yaml
base_config: config.yaml   # relative to project root

scenarios:
  - name: My scenario
    group: market           # used for grouping in the printed table
    description: >          # optional, shown with --verbose
      One-line description of what this scenario tests.
    overrides:
      vol_surface:
        vol_short: 0.30
      product:
        capital_barrier: 0.70
```

`overrides` is a deep-merge patch: only the specified keys change; all other keys keep their
base config values. You can override any subset of `market`, `simulation`, `vol_surface`,
`product`, and `models`.

### Example: product variant sweep

```yaml
base_config: config.yaml

scenarios:
  - name: Short maturity 1y
    group: maturity
    overrides:
      product:
        maturity: 1.0
        observation_dates: [0.25, 0.5, 0.75, 1.0]
        autocall_barriers: [1.00, 0.97, 0.94, 0.90]
        coupon_rate: [0.03, 0.06, 0.09, 0.12]

  - name: Long maturity 5y
    group: maturity
    overrides:
      product:
        maturity: 5.0
        observation_dates: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        autocall_barriers: [1.00, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73]
        coupon_rate: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
      vol_surface:
        T_max: 6.0   # must exceed maturity
```

### Example: market stress sweep

```yaml
base_config: config.yaml

scenarios:
  - name: Vol spike
    group: stress
    overrides:
      vol_surface:
        vol_short: 0.50
        vol_long: 0.35
        skew: -0.25

  - name: Zero rates
    group: stress
    overrides:
      market:
        rate: 0.00
        div_yield: 0.00

  - name: High div yield
    group: stress
    overrides:
      market:
        div_yield: 0.05
```

Run your custom file:

```bash
python scenarios/run_scenarios.py --scenarios my_scenarios.yaml --output my_results.csv
```

---

## 8. Python API

Use the building blocks directly when you need finer control — for example, pricing a note in a
Jupyter notebook, wiring up to a live market feed, or building a custom calibration loop.

### Price a note programmatically

```python
import numpy as np
from calibration.vol_surface import term_structure_surface
from calibration.calibrators import calibrate_heston
from models.heston import HestonModel
from products.autocallable import AutocallableNote
from engine.mc_pricer import MCPricer

SPOT, RATE, DIV = 100.0, 0.03, 0.01

surface = term_structure_surface(
    spot=SPOT, rate=RATE, div_yield=DIV, T_max=4.0,
    vol_short=0.25, vol_long=0.18, kappa=1.5, skew=-0.10, convexity=0.02,
)

params = calibrate_heston(surface)
model  = HestonModel(params=params, spot=SPOT, rate=RATE, div_yield=DIV, seed=42)

note = AutocallableNote(
    notional=1.0,
    spot=SPOT,
    maturity=3.0,
    observation_dates=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    autocall_barriers=[1.0, 1.0, 0.95, 0.90, 0.85, 0.80],
    coupon_rate=0.08,
    capital_barrier=0.60,
    capital_barrier_active=True,
    discount_rate=RATE,
)

result = MCPricer(model, "Heston", antithetic=True).price(note, n_paths=50_000)
print(f"Price:    {result.price:.4f}")
print(f"Std err:  {result.std_error:.6f}")
print(f"Duration: {result.expected_duration:.2f}y")
```

### Query the vol surface

```python
# Implied vol at T=1, K=95
surface.implied_vol(T=1.0, K=95.0)

# Dupire local vol at T=1, K=95
surface.local_vol(T=1.0, K=95.0)

# Vectorised local vol (fast — used internally in LocalVolModel)
K_arr = np.array([85., 90., 95., 100., 105., 110., 115.])
surface.local_vol_batch(T=1.0, K=K_arr)

# Create a surface with a shifted spot (for delta bumps)
s_up = surface.with_spot(SPOT + 1.0)

# Create a parallel vol-shifted surface (for vega bumps)
s_vup = surface.with_vol_shift(dvol=0.001)
```

### Run payoff evaluation directly

```python
# simulate paths (n_paths, n_obs)
model = HestonModel(params=params, spot=SPOT, rate=RATE, div_yield=DIV, seed=42)
perfs = model.simulate(n_paths=10_000, observation_times=[1.0, 2.0, 3.0], antithetic=True)
# perfs[i, j] = S(t_j) / S(0)  for path i

# evaluate payoff and duration on those paths
payoffs  = note.evaluate_payoff(perfs)     # shape (n_paths,) — discounted PV per path
duration = note.evaluate_duration(perfs)   # float — average years to termination
```

### Phoenix structure

```python
phoenix = AutocallableNote(
    notional=1.0,
    spot=SPOT,
    maturity=3.0,
    observation_dates=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    autocall_barriers=1.0,        # scalar: same barrier every date
    coupon_rate=0.07,             # scalar: same coupon every date
    conditional_coupon=True,      # Phoenix: periodic coupon even without autocall
    coupon_barrier=0.80,          # coupon paid if S(t)/spot ≥ 0.80
    capital_barrier=0.60,
    capital_barrier_active=True,
    discount_rate=RATE,
)
```

### Capital-protected (no KI)

```python
capital_protected = AutocallableNote(
    ...
    capital_barrier_active=False,  # always return full notional at maturity
)
```

---

## 9. Running the Tests

```bash
# Run the full test suite
.venv/bin/python -m pytest

# Run only the fast deterministic tests
.venv/bin/python -m pytest tests/test_payoff.py tests/test_vol_surface.py -v

# Run with timing
.venv/bin/python -m pytest --durations=10
```

### What the tests cover

| File | Tests | Notes |
|------|-------|-------|
| `test_payoff.py` | Autocall trigger, step-down barriers, KI breach, Phoenix coupons, discounting | Fully deterministic — hand-crafted performance arrays |
| `test_vol_surface.py` | `implied_vol`, `local_vol`, `local_vol_batch`, `with_spot`, `with_vol_shift` | Fully deterministic |
| `test_models.py` | Risk-neutral drift E[S(T)/S(0)] = exp((r−q)T) for all models; discount bond pricing | Stochastic; uses fixed seed + antithetic for stable assertions |
| `test_greeks.py` | `_rescale_product_barriers` absolute level preservation; delta sign; vega sign | Mix of deterministic and stochastic |
| `test_basket_payoff.py` | Worst-of aggregation (min across assets per date) feeding the unmodified `evaluate_payoff` | Fully deterministic — hand-crafted per-asset arrays |
| `test_basket_models.py` | Correlated path shape/reproducibility; correlation recovery (exact for Local Vol, approximate for Heston); price monotonicity in correlation | Stochastic; some cases use large path counts (200k) for tight correlation-recovery tolerances |
| `test_basket_greeks.py` | Per-asset delta/gamma sign and non-degeneracy; no barrier mutation; bumped-asset isolation | Mix of deterministic and stochastic |

All tests should pass in a few minutes (the basket correlation-recovery tests use large path
counts and dominate the runtime).

---

## 10. Performance Guide

### Pricing runtime

| Scenario | n_paths | Time (approx.) |
|----------|---------|----------------|
| Single note, 1 model | 10,000 | < 1s (Heston/SABR) |
| Single note, 1 model | 10,000 | ~0.7s (Local Vol) |
| Single note, 3 models | 20,000 | ~3–5s (includes calibration) |
| 12 scenarios, 3 models | 20,000 each | ~2–3 min |

### Greeks runtime

Each call to `--greeks` requires:
- 10 bumped pricing evaluations (spot ±, vol ±, skew ±, vanna cross-terms)
- 4 vol surface recalibrations for Heston (fast warm-start mode, ~0.8s each) — 2 for the vol bumps,
  2 for the skew bumps

Per scenario with 10,000 paths:

| Contribution | Time |
|---|---|
| 10 × Local Vol pricings | ~6s |
| 10 × Heston pricings | ~4s |
| 10 × SABR pricings | ~4s |
| 4 × Heston recalibrations | ~3s |
| **Total** | **~17s per scenario** |

For a 12-scenario run with Greeks: allow ~4–6 minutes.

### Reducing Greek noise

Greek estimates are noisy because they are FD differences of two MC prices. Three levers:

1. **Increase `--n-paths-greeks`**: directly reduces noise but increases runtime proportionally.
2. **Increase `--h-spot-pct`**: larger bump reduces relative noise but adds truncation error. The
   default (1%) is usually a good balance; 2% is reasonable for a first pass.
3. **Common random numbers (CRN)**: the pricer already uses this — all 10 bumped evaluations share
   the same seed, so MC noise largely cancels in the FD numerator. Do not change the seed between
   the base and bumped runs if you replicate the Greek logic in your own code.

### Turning off unneeded models

Set `sabr: false` or `heston: false` in the config (or in a scenario override) to skip a model.
Local Vol is cheapest per path but most expensive to vectorise at very high path counts due to the
Dupire grid evaluation; Heston and SABR scale well with path count.

---

## 11. Worst-of Basket Pricing

A worst-of basket note pays based on the **worst-performing** of several correlated underlyings —
the autocall, coupon, and capital barriers all apply to `min_i(S_i(t) / S_i(0))`, not to any single
asset. Supported models: **Local Vol** and **Heston**. **SABR is not supported** for basket pricing.

### 11.1 Config schema

Add a top-level `assets:` list and `correlation:` matrix to switch a config into basket mode. When
`assets:` is present, the single-asset `market:` and top-level `vol_surface:` blocks are ignored —
`product:`, `simulation:`, and `models:` are unchanged and reused as-is. See
[configs/config_worst_of_2asset.yaml](configs/config_worst_of_2asset.yaml) for a full working example.

```yaml
assets:
  - name: AssetA
    spot: 100.0
    rate: 0.03          # all assets must share the same rate (v1 constraint)
    div_yield: 0.01     # div_yield may differ per asset
    vol_surface:        # identical schema to the single-asset vol_surface: block
      type: term_structure
      vol_short: 0.25
      vol_long: 0.18
      kappa: 1.5
      skew: -0.10
      convexity: 0.02
      T_max: 5.0

  - name: AssetB
    spot: 100.0
    rate: 0.03
    div_yield: 0.02
    vol_surface:
      type: term_structure
      vol_short: 0.30
      vol_long: 0.22
      kappa: 1.3
      skew: -0.12
      convexity: 0.02
      T_max: 5.0

correlation:            # n × n, order matches the assets: list
  - [1.00, 0.60]
  - [0.60, 1.00]

product:
  # unchanged schema — barriers apply to worst-of performance, not any one asset
  autocall_barriers: [0.85, 0.85, 0.80, 0.75, 0.70, 0.65]
  ...

models:
  local_vol: true
  heston: true
  sabr: false          # ignored in basket mode — not supported
```

### 11.2 Running

```bash
python main.py --config configs/config_worst_of_2asset.yaml --no-fit-check

python scenarios/run_scenarios.py \
    --scenarios scenarios/sce_worst_of_2asset.yaml \
    --n-paths 20000 --greeks --output basket_results.csv
```

`main.py` prints "Basket Local Vol" / "Basket Heston" pricing lines instead of the single-asset
model names; `--full-diagnostic` is not yet supported in basket mode.

### 11.3 Correlation: what to expect from each model

- **Local Vol**: correlation is applied directly to the driving Brownians via Cholesky decomposition
  — the realized asset-asset correlation matches your `correlation:` input almost exactly.
- **Heston**: each asset keeps its own independent variance process (no cross-asset vol-vol
  correlation), which has two consequences worth knowing before you rely on the number:
  1. **Feasibility ceiling.** Because each asset's own spot-vol correlation `rho` "uses up" some of
     its spot shock's variance, the maximum achievable correlation between two assets is
     `sqrt(1-rho_a²) × sqrt(1-rho_b²)`. For typical calibrated equity `rho` (−0.6 to −0.8), this
     ceiling is often only **~0.45–0.65** — well below correlations commonly assumed for same-sector
     equities. Requesting a higher correlation doesn't error; it's automatically clipped to the
     feasible ceiling with a printed warning (e.g. `requested correlation ... was clipped: (0,1):
     +0.600 -> +0.521`). If you need exact control over a high target correlation, use Local Vol
     instead.
  2. **Residual attenuation.** Even within the feasible ceiling, the *realized* log-return
     correlation tends to run somewhat below the target, because each asset's variance evolves
     independently over the life of the trade (more pronounced for higher vol-of-vol and longer
     maturities). Use Basket Local Vol if you need the realized correlation to match your input
     closely; use Basket Heston if the primary concern is skew/smile dynamics rather than exact
     correlation control.

### 11.4 Basket Greeks

`--greeks` computes **per-asset** delta, gamma, vega, vanna, and skew sensitivity — one full set
per underlying, not one aggregate number:

```
Basket Local Vol:
  AssetA        Δ=+0.000223  Γ=-0.018895  ν=-0.000884  vanna=+0.090661  skew_sens=+0.000432
  AssetB        Δ=-0.000613  Γ=-0.018393  ν=-0.004090  vanna=-0.115756  skew_sens=+0.001426
```

`skew_sens` bumps that asset's own surface with `with_skew_shift()` (see
[§6.2 Skew sensitivity convention](#skew-sensitivity-convention)), holding every other asset's
surface — and the correlation matrix — at base. It requires no per-asset performance rescale (like
vega, not like delta/gamma): a pure skew tilt doesn't touch the asset's spot reference, so there's
nothing to rescale before taking the worst-of min.

Cost scales linearly with the number of assets: **10N bump evaluations** for N assets (vs. a flat 10
in the single-asset case) — 2 for delta/gamma, 2 for vega, 2 for skew_sens, 4 for vanna cross-terms,
per asset. Only the bumped asset's Heston parameters are recalibrated per vol or skew bump (4N
recalibrations total, not 4N × N), but runtime still grows with basket size — a printed line shows
the total bump count before the computation starts.

**Barrier convention**: unlike the single-asset Greeks (which rescale the note's barriers to hold
their absolute dollar level fixed under a spot bump), basket Greeks rescale the **bumped asset's own
performance** before taking the worst-of min, leaving the barrier untouched. A worst-of barrier is a
single fraction shared across whichever asset is currently worst — it can't be rescaled per-asset
the way a single-asset barrier can. The two approaches are mathematically equivalent; this
distinction matters in practice because Heston's simulated performance is scale-invariant to the
spot level, so without this rescale, Heston's basket delta/gamma would come out as exactly zero.

---

## 12. Numerical Corrections

Two defects were found while adding the LSV model. Both predate it and affected all models.
Neither was a modelling choice — both were implementation errors with measurable consequences.

### 12.1 Dupire local volatility: `∂w/∂T` taken at the wrong constant

**What was wrong.** `ImpliedVolSurface.local_variance()` implements Dupire's formula in Gatheral's
log-moneyness parameterisation:

```
σ_LV² = (∂w/∂T) / [1 − (y/w)·∂w/∂y + ¼(−¼ − 1/w + y²/w²)(∂w/∂y)² + ½·∂²w/∂y²]
```

where `w = σ_imp²·T` and `y = ln(K/F(T))`. Every derivative in that expression is with respect to
`(T, y)`. The code computed `∂w/∂T` by bumping `T` at **fixed strike K**. Because `F(T)` drifts
with `T`, holding `K` fixed does not hold `y` fixed, and the two derivatives differ by

```
∂w/∂T|_K  −  ∂w/∂T|_y  =  −(r − q) · ∂w/∂y
```

**Why it went unnoticed.** The error term is proportional to `∂w/∂y`, the skew. On a **flat**
surface it is identically zero — so every flat-surface test passed, and the flat-surface
round-trip was accurate to ~1bp. It only bites when there is skew, which is to say on every
realistic surface.

**Impact.** With `r − q = 2%` on the default config surface, local vol was overstated by ~1.3%
relative, and Local Vol's own vanilla round-trip carried a **+21bp RMS** bias — a bias that did
not shrink with more timesteps, a finer surface grid, or a smaller finite-difference step,
because it was in the formula rather than the numerics.

**The fix.** Shift the strike with the forward so `y` is held constant:

```python
K_T_up = K * self.forward(T + dT) / F
K_T_dn = K * self.forward(T - dT) / F
dw_dT  = (self.total_variance(T + dT, K_T_up)
          - self.total_variance(T - dT, K_T_dn)) / (2 * dT)
```

In the vectorised `local_vol_batch()` the same correction is a shift of the spline coordinate by
`(r − q)·dT`. Local Vol's round-trip improved from **21bp RMS to 2.1bp**; the flat-surface control
was unchanged, as expected.

### 12.2 Antithetic variates were not antithetic

**What was wrong.** A true antithetic pair reuses the **same** random draws with the sign flipped.
All five models (`local_vol`, `heston`, `sabr`, and both basket models) instead generated the
second half of the ensemble from **fresh** draws and negated those:

```python
Z = anti_sign * self.rng.standard_normal(n)   # drawn again for the mirror pass
```

Negating a fresh standard normal yields another independent standard normal — the law is
symmetric — so the "antithetic" half was statistically independent of the original.

**Impact.** Prices were **unbiased and remain correct**; nothing previously computed was wrong in
expectation. What was lost was the variance reduction: `antithetic=True` bought nothing, and
reported standard errors and confidence intervals were wider than they should have been.

**The fix.** All models now simulate both halves in a **single pass**, with paths `[half:]` the
literal mirror of paths `[:half]`: the same normals negated, and — for the QE variance scheme —
the same uniforms reflected as `U → 1−U`. `MCPricer` was also updated: the standard error is now
computed by averaging each antithetic pair into one observation, because treating the paths as
independent ignores their negative covariance and discards the improvement on paper.

Measured on the default note: standard error improved **1.11–1.23×** for single-asset and basket
models, and 1.24–1.45× on a vanilla call. (The autocall gains less because its payoff is a sum of
digital barrier events rather than monotone in terminal spot, and antithetic variates only help
for payoffs that are monotone in the driving randomness.)

Regression tests for both properties live in `tests/test_models.py`
(`test_antithetic_halves_are_negatively_correlated`, `test_antithetic_reduces_variance`,
`test_mc_pricer_standard_error_accounts_for_pairing`).
