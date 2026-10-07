# Variance-Gated Discounted Thompson Sampling (VG-dTS)

Code and reproducibility materials for **Variance-Gated Discounted Thompson
Sampling for Non-Stationary Bandits**, by Mayand Gulati, Kerong Wang, and
Wei-Chen Au (ECE 270, UC Santa Barbara).

The authoritative manuscript is
[`report/ECE_270_Project_Publication/main.tex`](report/ECE_270_Project_Publication/main.tex).
This repository contains the original VG-dTS method and the six baselines in
that manuscript: TS, dTS, dOTS, REXP3, Dynamic TS, and Beta-SWTS. Later VG-dTS
variants, additional baselines, recovering-bandit experiments, and ablation
studies have been removed.

## Install

Use Python 3.12 (the pinned NumPy release supports Python 3.10–3.12). From the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` pins the dependencies used for validation. No datasets,
credentials, or external services are needed; all environments are synthetic.
Run commands below from the repository root so the reference paths resolve.

## Reproduce the paper

First check the installation with a small simulation and all figure generators:

```sh
MPLBACKEND=Agg python -m project_publication --mode smoke --output-dir artifacts/smoke
python -m pytest -q
```

The smoke test uses **120 rounds, 1 tuning run, and 2 evaluation runs**. It does
not reproduce the paper's numerical results.

Replay the paper's saved hyperparameters with **T = 5000, K = 4, seed = 0,
and 1000 evaluation runs**:

```sh
MPLBACKEND=Agg python -m project_publication --mode paper --workers 4
python -m project_publication --mode verify
```

This recomputes all **119 final normalized regrets**: 8 environments × 7 policies
for each of the tuned and fixed protocols, plus 7 negative-control results.
It also regenerates all 12 manuscript figures. It is a substantial CPU job;
`--workers 1` reduces concurrent resource use. Each completed environment is
saved immediately. Existing nonempty output directories are never overwritten.

To repeat the original hyperparameter search as well:

```sh
MPLBACKEND=Agg python -m project_publication --mode retune --workers 4 \
  --tuning-runs 70 --eval-runs 1000 --output-dir artifacts/retuned
python -m project_publication --mode verify --output-dir artifacts/retuned
```

Retuning evaluates 106 candidates per environment, including 64 VG-dTS candidates,
and takes substantially longer than replay. In `paper` mode, tuning scores are
copied from the archive and labeled as such; evaluation scores are recomputed.
In `retune` mode, both are recomputed. Verification exits nonzero on a mismatch;
retune verification also compares selected parameters and tuning scores.

For a faster exploratory run, use `--eval-runs 10 --output-dir artifacts/draft`.
Those results are not eligible for exact paper verification.

**Cleanup validation:** all 119 archived evaluation values were reproduced with
zero numerical difference; 23 tests pass; all 12 figures regenerate; the paper
builds to 17 pages. The initial audit used NumPy 2.5.2, with both breakpoint jobs
rerun under 1.26.4 after diagnosing a random-generator compatibility difference.
New runs use the pinned 1.26.4 environment throughout. The full tuning search was
not rerun during cleanup. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for provenance and paper/code caveats.

## Outputs and figures

The default output directory is `artifacts/reproduction/` (ignored by Git):

- `figures/`: the same relative PNG paths referenced by `main.tex`.
- `results/`: recomputed JSON summaries.
- `runs/`: per-environment JSON, true means, average reward curves, and regret
  curves in compressed NumPy archives; these allow plotting without resimulation.
- `metadata.json`: parameters, seed schedule, candidate grids, dependency versions,
  source hashes, reference hashes, and completion status.
- `verification.json`: numerical comparison with the archived paper results.

To regenerate plots from a completed simulation:

```sh
MPLBACKEND=Agg python -m project_publication --mode plot
```

The original paper figures remain in the manuscript directory; reruns never
replace them automatically. The original result JSON files live in
[`results/paper/`](results/paper/).

| Manuscript figure | Generated file under `figures/` |
| --- | --- |
| Environment/oracle gallery | `environment_oracle/environment_oracle_<environment>.png` (8 files) |
| Canonical benchmarks | `original_paper_envs_tuned.png` |
| Tuned eight-environment heatmap | `optimized_heatmap_8env_7policies.png` |
| Pure-random-signal control | `rigorous_pure_random_signal_tuned.png` |
| Fixed-parameter heatmap | `fixed_params_heatmap_8env_7policies.png` |

## Build the manuscript

Install a TeX distribution with pdfLaTeX and BibTeX, then run:

```sh
sh report/ECE_270_Project_Publication/build.sh
```

The output is `report/ECE_270_Project_Publication/main.pdf`. The required local
style files are included. To build a separate copy using regenerated figures:

```sh
mkdir -p artifacts/rebuilt-paper
cp -R report/ECE_270_Project_Publication/. artifacts/rebuilt-paper/
cp -R artifacts/reproduction/figures/. artifacts/rebuilt-paper/
sh artifacts/rebuilt-paper/build.sh
```

## Code layout

- `src/adts/policies.py`: the seven paper policies and their parameter classes.
- `src/adts/envs.py`, `oracle.py`: environment generators and dynamic oracle.
- `src/adts/vgdts_config.py`: original VG-dTS defaults and ordered tuning grid.
- `project_publication/pipeline.py`: candidate construction, Monte Carlo evaluation,
  and paper plotting routines.
- `project_publication/reproduce.py`: replay, retuning, persisted results, and checks.
- `tests/`: algorithm, environment, archival-integrity, and integration checks.

The cleanup preserves the manuscript text and its results. It does not resolve
scientific interpretation issues listed in `REPRODUCIBILITY.md`. No project-wide
software license was present in the inherited repository; the authors should
choose one before advertising permission to reuse the code. Bundled third-party
LaTeX files retain their own notices.
