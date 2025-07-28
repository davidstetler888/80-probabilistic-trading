# Fx-mL-v69: Context-Aware Forex Trading System

> **Author:** David Stetler  
> **Last Updated:** 2025‑01‑29

---

## Quick Start

### Raw Data Setup

Place your EURUSD 5‑minute price file under `data/raw/` as `EURUSD5.csv`. The reference dataset spans from **1971‑01‑04 02:00** through **2024‑07‑12 23:55** and uses the format:

```
date,time,open,high,low,close,volume
```

`prepare.py` automatically clamps the requested start date to this range. The first and last timestamps are detected by reading only the first and last rows of `EURUSD5.csv`, so very large files load quickly.

### CSV Formatting

The loader expects two columns named `date` and `time` with values like `2021.01.01` and `00:05`. Rows where these fields cannot be combined into a valid timestamp are discarded with a warning. The first few offending rows are printed so you can inspect them. Pass `strict=True` to `utils.load_data()` (or use `--strict` with the CLI scripts) to abort instead of dropping them.

### Basic Workflow

```bash
# Set up run ID
RUN_ID=models/run_$(date +%Y%m%d_%H%M%S)
export RUN_ID

# Feature engineering
python prepare.py --config config.yaml --start_date 2023-01-01 --end_date 2024-12-31

# Add labels
python label.py --run $RUN_ID --start_date 2023-01-01 --end_date 2024-12-31

# Train models
python train_base.py --run $RUN_ID --start_date 2023-01-01 --train_end_date 2024-06-30 --end_date 2024-12-31
python train_meta.py --run $RUN_ID --start_date 2023-01-01 --train_end_date 2024-06-30 --end_date 2024-12-31
python train_ranker.py --run $RUN_ID --start_date 2023-01-01 --train_end_date 2024-06-30 --end_date 2024-12-31

# Simulate
python simulate.py --run $RUN_ID --start_date 2024-07-01 --end_date 2024-12-31

# Or run walk-forward validation
python walkforward.py --run OUTDIR --start_date 2023-01-01 --end_date 2024-12-31
```

### Walk-Forward Validation

`walkforward.py` verifies that the requested simulation end date does not extend beyond the timestamps contained in the prepared dataset (`data/prepared.csv`). `prepare.py` and `walkforward.py` both interpret `--end_date` inclusively, so `--end_date 2024-06-30` loads bars through `2024-06-30 23:55` in the configured timezone.

Each window now ends on the configured market close so the last day of data is always included, preventing spurious "exceeds available data" errors. If the simulation end falls outside the dates in `prepared.csv`, `walkforward.py` now adjusts the end date to the last available timestamp and emits a warning.

Each iteration also checks the raw CSV before any scripts run. If its simulation window would go past the last timestamp in `EURUSD5.csv` a warning like:

```
Simulation window 2024-06-23–2024-06-30 exceeds available data ending 2024-06-21. Extend EURUSD5.csv or adjust stepback settings.
```

is printed and the entire iteration is skipped.

### Temporary Outputs

Results are written under `output/` and run-specific folders like `models/run_*`. These files are regenerated each run and are not tracked in version control.

During a walk-forward run each helper script writes its full output to `models/run_*/logs/<script>.log`. The file `output/output.txt` records only important summary lines. By default it captures markers like `[prepare]` and `[label]`, as well as sections labeled `Test Simulation Results` and `Full Simulation Results`. Additional metric lines such as `Total Trades`, `Win Rate`, `Profit Factor`, and `Average RR` are also included. Pass `--verbose` to `walkforward.py` to disable this filtering and capture all output.

---

## Project Vision

Build a **context‑aware, modular Forex trading system** that:

* **Enters trades with high precision** (Class 1 signals only when we are very confident).
* **Optimises each trade's SL/TP levels dynamically** based on market regime.
* **Adapts weekly** through walk‑forward retraining.
* **Target metrics** (not strict acceptance):
  * Aim for **58 – 72 % win rate**
  * Average **1 : 2 to 1 : 3 risk‑reward**
  * **25 – 50 trades per week**
  * Seamless live execution via MT4/DWX

## Strategic Principles

* **Precision‑first mindset** – predicting "no trade" (Class 0) is cheap; entering a bad trade is deadly.
* **Rare‑event classification** – embrace imbalance; design labels, loss functions, and metrics accordingly.
* **Separation of concerns** – direction (enter?) and exits (SL/TP) are solved by different models.
* **Market regimes matter** – treat each regime independently when it helps, share signal power when data are sparse.
* **Reproducibility > tinkering** – every experiment writes to its own run‑ID; artefacts are immutable.
* **Walk‑forward discipline** – weekly retrain + sim loop to keep the model honest.

## Pipeline Overview

| Phase            | Script(s)             | Status     | Description                                                    |
| ---------------- | --------------------- | ---------- | -------------------------------------------------------------- |
| **Feature Prep** | `prepare.py`          | ✅ Stable   | Indicators, lags, regime clustering, scaling                   |
| **Labeling**     | `label.py`            | ✅ Stable  | Direction labels + SL/TP bucket + cooldown/ATR gates          |
| **Base Models**  | `train_base.py` ✚ CNN | ✅ Working | LGBM per (regime, side) + global 1‑D CNN; output per‑bar probs |
| **Meta Model**   | `train_meta.py`       | ✅ Working | Logistic GBM stacks base probabilities + rule features         |
| **Edge Ranker**  | `train_ranker.py`     | ✅ Working | Ranks signals via meta prob × (TP−SL) and auto‑thresholds to 25‑50 trades/wk |
| **Simulation**   | `simulate.py`         | ✅ Working  | Equity‑aware, single-position trade manager with per-side cooldown |
| **Walk‑Forward** | `walkforward.py`      | ✅ Working | 18‑month train window, weekly stepback loop; optional optimisation via `optimize.py` |

## Script Interface Contracts

### `prepare.py`
*Purpose* – engineer features, cluster regimes, scale.
*CLI* – `python prepare.py --config config.yaml --start_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD [--train_end_date YYYY‑MM‑DD] [--input_dir PATH] [--strict]`

### `label.py`
Generates SL/TP grid, labels `label_long/short`, enforces ATR & cooldown.
*CLI* – `python label.py --run RUN --start_date YYYY‑MM‑DD [--train_end_date YYYY‑MM‑DD] [--strict] --end_date YYYY‑MM‑DD`

### `train_base.py`
Trains **LGBM** per (regime, side) and a global **1‑D CNN** on sliding windows.
*CLI* – `python train_base.py --run RUN --start_date YYYY‑MM‑DD --train_end_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD`

### `train_meta.py`
Stacks base probabilities + contextual features (ATR_pct, hour, regime) into a logistic GBM.
*CLI* – `python train_meta.py --run RUN --start_date YYYY‑MM‑DD --train_end_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD`

### `train_ranker.py`
Computes edge scores and auto‑selects threshold for 25‑50 trades/week.
*CLI* – `python train_ranker.py --run RUN --start_date YYYY‑MM‑DD --train_end_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD [--edge_threshold THR]`

### `simulate.py`
Equity‑aware back‑test with single-position trade manager.
*CLI* – `python simulate.py --run RUN --start_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD [--test_frac 0.2]`

### `walkforward.py`
Orchestrates weekly retrain + sim loop.
*CLI* – `python walkforward.py --run OUTDIR [--stepback_weeks N] [--target_trades_per_week N] [--min_trades_per_week N] [--max_trades_per_week N]`

## Trade Logic

> **Trade = High‑confidence Directional Signal + Best SL/TP from grid**

1. `label.py` builds the SL/TP grid from `sl_tp_grid` in `config.yaml`, then labels each bar while honouring `label.cooldown_min` and the optional `--train_end_date` cutoff.
2. `train_base.py` learns to predict high‑precision direction (Class 1) **per regime & side**.
3. `train_meta.py` combines multiple base probabilities; `train_ranker.py` converts this into an **edge score** and picks trades until the weekly cap hits.
4. `simulate.py` merges signals with market data via a **left join** so every bar is processed. The simulation loops over all bars, opening new trades when `has_signal` is true and concurrent positions < `simulation.max_positions`.

## Configuration

All scripts pull grid & parameters from **config.yaml** for consistency. Key settings:

* Fixed spread: **0.00013**
* SL multipliers = 1.8 → 5.2 (step 0.2)
* TP multipliers = 2.0 → 3.2 (step 0.2)
* Default cooldown: **10 min**
* Target trades per week: **25-50**

## Testing & CI

| Stage      | pytest suite                  | Gate                      |
| ---------- | ----------------------------- | ------------------------- |
| Prepare    | schema + NaN checks           | fail if `nan_pct > 0.1 %` |
| Label      | weekly signal count           | fail if `< 5` signals/wk  |
| Train‑Base | track precision (target ≥ 0.70) | warn if recall < 0.05     |
| Meta       | ROC‑AUC ≥ 0.80                | hard fail                 |
| Simulate   | track win rate, RR, trades/wk | fail if win rate < 58 %   |

## Lessons Learned

### Performance Gaps
* Early iterations showed 70–80 % precision in the deprecated `train.py` pipeline; later refactors lost that edge.
* SL/TP modelling adds noise; only useful **after** direction is rock‑solid.

### What Matters in Class 1
* False positives are far worse than missing positives.
* It's better to skip a great trade than enter a bad one.

### Architecture Wins
* Per‑regime modelling outperforms global models.
* Separating direction and SL/TP keeps modules interpretable.
* Grid‑based labelling is transparent and debuggable.

### Failures to Avoid
* Rewriting core logic because of temporary metric dips.
* Letting model quantity outweigh signal quality.
* Ignoring lessons on imbalance and temporal leakage.

---

*For detailed technical documentation, see [project.md](project.md).* 