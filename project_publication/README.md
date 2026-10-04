# Project Publication Pipeline

This folder contains publication-ready plotting and benchmarking code for the VG-dTS paper comparison study.

## What it generates

`python -m project_publication.cli` generates:

- Single tuned environment plot (reward + regret)
- Original Raj & Kalyani environments (Slow/Fast/Abrupt) with tuned policies
- Rigorous pure-random signal tuned plot (reward + regret)
- One oracle environment plot per environment (8 suite environments + pure random signal)
- One fixed-parameter reward/regret plot per each of the 8 suite environments
- Two heatmaps over the same 8 environments x 16 policies:
  - Optimized (tuned per environment)
  - Fixed-parameter (same policy parameters across all environments)
- JSON summaries containing tuned/fixed parameters and final normalized regrets

## Policy set used everywhere

- VG-dTS
- dTS
- dOTS
- TS
- REXP3
- Dynamic TS
- SW-UCB (0805.3415)
- D-UCB (0805.3415)
- CUSUM-UCB (1711.03539)
- GLR-klUCB (1902.01575)
- AdaSwitch (1902.07010)
- SW-TS (Trovo 2020)
- gamma-SWGTS (2409.05181)
- Global-CTS (1302.3721)
- D-LinUCB one-hot (1909.09146)
- Beta-SWTS

The policy colors are fixed across all reward/regret plots, with VG-dTS as green.

## Environment set for heatmaps and per-environment figures

1. Slow Sine
2. Fast Sine
3. Abrupt Cycle
4. Mixed Regime
5. Random Breakpoints
6. Random Drift Amplitude
7. Global Switching
8. Per-Arm Switching

## Recommended run budgets

- Quick sanity check:
  - `--quick`
  - Effective: `horizon<=2500`, `tuning_runs=10`, `eval_runs=20`
- Draft-quality figures:
  - `--horizon 5000 --tuning-runs 30 --eval-runs 100`
- Paper-ready baseline:
  - `--paper`
  - Effective: `horizon>=10000`, `tuning_runs=60`, `eval_runs=180`
- Final high-confidence pass (compute-heavy):
  - `--horizon 10000 --tuning-runs 100 --eval-runs 300`

## Example commands

```bash
# Draft pass
PYTHONPATH=. .venv/bin/python -m project_publication.cli --horizon 5000 --tuning-runs 30 --eval-runs 100

# Paper-ready pass
PYTHONPATH=. .venv/bin/python -m project_publication.cli --paper --single-environment fast

# Freeze VG-dTS sweep while keeping all other outputs
PYTHONPATH=. .venv/bin/python -m project_publication.cli --paper --no-vgdts-sweep
```
