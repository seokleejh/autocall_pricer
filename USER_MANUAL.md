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
6. [Scenario Analysis](#6-scenario-analysis)
   - [Basic run](#61-basic-run)
   - [Computing Greeks](#62-computing-greeks)
   - [Reading the output](#63-reading-the-output)
   - [CSV output](#64-csv-output)
7. [Writing Custom Scenarios](#7-writing-custom-scenarios)
8. [Python API](#8-python-api)
9. [Running the Tests](#9-running-the-tests)
10. [Performance Guide](#10-performance-guide)

---

## 1. Overview

This tool prices autocallable structured notes (FCN, EKI, Phoenix, Snowball variants) under three
volatility models simultaneously and reports the **model spread** — the price difference arising
solely from the choice of dynamics — as a measure of model risk.

### Three models

| Model | Dynamics | Surface fit | Typical use |
|-------|----------|-------------|-------------|
| **Local Vol** (Dupire) | Deterministic σ(t, S) derived from market vols via Dupire's formula | Exact by construction | Model risk lower bound; conservative barrier pricing |
| **Heston SV** | Mean-reverting stochastic variance (CIR process); spot-vol correlation ρ | Global fit over all maturities and strikes | Captures forward skew dynamics; well-suited for long-dated autocalls |
| **SABR SV** | Log-normal stochastic vol; backbone β = 1 | Calibrated to a single maturity slice (default: 1y) | Rich short-dated smile; may extrapolate poorly to maturities far from calibration |

All three are calibrated to the same implied vol surface, so differences in price are not due to a vol
level mismatch — they reflect genuine disagreement about the joint distribution of path and final payoff.

### Two entry points

| Script | Purpose |
|--------|---------|
| `main.py` | Price a single note; inspect calibration quality |
| `scenarios/run_scenarios.py` | Run a battery of scenario overrides; compare model spreads across product variants or market regimes |

---

## 2. Setup

**Prerequisites:** Python 3.10+, a Unix-like shell (Linux, macOS, WSL).

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

A successful run prints calibrated model parameters followed by three pricing lines.

---

## 3. Quick Start

Price the note defined in `config.yaml` under all three models:

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
  local_vol: true
  heston:    true
  sabr:      true
```

Set any to `false` to skip that model. Running with only one model is faster and useful for
sensitivity testing.

---

## 5. Running a Single Pricing

```bash
python main.py [--config FILE] [--no-fit-check | --full-diagnostic]
```

**Default** (`python main.py`): calibrates Heston and SABR, runs a fast analytical vol fit check,
then prices the note under all enabled models.

**`--no-fit-check`**: skip the fit quality table and go straight to pricing. Useful in scripts.

**`--full-diagnostic`**: run a Monte Carlo vol surface diagnostic via `diagnostics/model_quality.py`
before pricing. This simulates each model at a grid of maturities and strikes and recovers implied
vols from MC prices, then plots them against the market surface. Useful for verifying calibration
quality with a realistic number of paths.

```bash
python main.py --full-diagnostic
python diagnostics/model_quality.py --n-paths 50000 --output fit_check.png
```

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
    --h-vol 0.001
```

| Flag | Default | Description |
|------|---------|-------------|
| `--greeks` | off | Enable finite-difference Greeks |
| `--n-paths-greeks` | same as `--n-paths` | Paths for FD bumps (fewer paths → faster but noisier) |
| `--h-spot-pct` | 0.01 | Spot bump as fraction of spot (1%) |
| `--h-vol` | 0.001 | Parallel implied vol shift for vega/vanna (10 bp) |

**Runtime**: with 10,000 paths and all three models, Greeks add ~2–5 minutes per scenario
(8 bumped evaluations × 3 models, plus 2 vol-surface recalibrations). See [Section 10](#10-performance-guide).

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
<model>_vanna × 3
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

All 50 tests should pass in under 35 seconds.

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
- 8 bumped pricing evaluations (spot ±, vol ±, cross-terms)
- 2 vol surface recalibrations for Heston (fast warm-start mode, ~0.8s each)

Per scenario with 10,000 paths:

| Contribution | Time |
|---|---|
| 8 × Local Vol pricings | ~5s |
| 8 × Heston pricings | ~3s |
| 8 × SABR pricings | ~3s |
| 2 × Heston recalibrations | ~2s |
| **Total** | **~13s per scenario** |

For a 12-scenario run with Greeks: allow ~3–5 minutes.

### Reducing Greek noise

Greek estimates are noisy because they are FD differences of two MC prices. Three levers:

1. **Increase `--n-paths-greeks`**: directly reduces noise but increases runtime proportionally.
2. **Increase `--h-spot-pct`**: larger bump reduces relative noise but adds truncation error. The
   default (1%) is usually a good balance; 2% is reasonable for a first pass.
3. **Common random numbers (CRN)**: the pricer already uses this — all 8 bumped evaluations share
   the same seed, so MC noise largely cancels in the FD numerator. Do not change the seed between
   the base and bumped runs if you replicate the Greek logic in your own code.

### Turning off unneeded models

Set `sabr: false` or `heston: false` in the config (or in a scenario override) to skip a model.
Local Vol is cheapest per path but most expensive to vectorise at very high path counts due to the
Dupire grid evaluation; Heston and SABR scale well with path count.
