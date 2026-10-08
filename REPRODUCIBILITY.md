# Reproducing the VG-dTS paper

[Paper overview](README.md) · [Manuscript](report/ECE_270_Project_Publication/main.tex)

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
not rerun during cleanup. The sections below describe provenance and paper/code caveats.

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

### Regret curves in the README

The README highlights the four fixed-parameter environments where VG-dTS has
the lowest final normalized regret: fast, abrupt, global switching, and per-arm
switching. The plot includes all seven methods, all 5,000 rounds, and unsmoothed
averages over 1,000 rollouts. The full-suite heatmaps provide the other cases.

After a completed reproduction, regenerate this figure with:

```sh
python -m project_publication.readme_figures
```

Use `--run-dir PATH` for a different reproduction directory. The generator checks
every displayed curve's endpoint against the archived fixed-protocol result
before writing `docs/figures/fixed_parameter_regret.png`.

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

The cleanup preserves the manuscript text and its results. Scientific
interpretation issues are listed below. Original project content is licensed
under the [Apache License 2.0](LICENSE). Bundled third-party LaTeX styles retain
their upstream terms and notices; see [NOTICE](NOTICE).

## What was preserved and recovered

The manuscript, appendix, bibliography, macros, and 12 figures referenced by the
manuscript are unchanged. Their original export hashes are retained in
`report/ECE_270_Project_Publication/OVERLEAF_MANIFEST.json`. Unused figures and
superseded report directories were removed. The three canonical numerical
summaries in `results/paper/` are byte-identical copies from the inherited
`report/project_publication_final/results/` directory. The separate canonical
three-environment summary duplicated rows in the tuned heatmap and was removed.

The fork's initial Python entry point had already switched to 16 policies and
VG-dTS v2.1. The original seven-policy pipeline was recovered from the local
upstream repository's Git commit `41a1730` ("adding project paper figures").
Only the seven relevant policy implementations and supporting environment code
are retained. Neither the public repository nor its reproduction commands depend
on that upstream checkout.

The archived JSON records T=5000, K=4, all selected parameters, and final regrets,
but **does not record the original invocation, seed, run counts, dependency
versions, or reward curves**. Seed 0, 70 tuning runs, and 1000 evaluation runs were
reconstructed and checked by rerunning archived results. These are reconstructed
settings, not an original recovered run log. The new runner records the missing
metadata and curves for every new run.

**NumPy compatibility matters.** NumPy 1.26.4 reproduces the archived random-
breakpoint realization; tested NumPy 2.2.6, 2.3.5, 2.4.2, 2.4.3, and 2.5.2 produce
a different realization for the same multinomial segment-length construction
and seed. The project pins NumPy 1.26.4 and refuses full paper runs on other
versions. No environment parameters were changed to force agreement. The
original PNG metadata identifies Matplotlib 3.9.1; the dependency file pins its
3.9.1.post1 release plus a compatible contourpy version. Regenerated PNG bytes
can still vary with fonts, operating system, or plotting dependencies; compare
numerical results and retain the original publication PNGs for archival use.

The validation performed during cleanup is recorded in `results/paper/VALIDATION.json`.
A full replay checks evaluation at the archived selected settings. It does not
establish that an independent tuning search selects the same settings; use
`--mode retune` and `--mode verify` to check that separately.

## Exact experimental conventions

All policies consume means shaped `(K, T)`; environment generators return `(T, K)`.
Rewards are Bernoulli observations. Normalized regret is the realized statistic

```
NR_t = (sum_{s<=t} max_k mu[k,s] - sum_{s<=t} reward[s]) / t
```

This can be negative on an individual run. It is not clipped, and the code does
not substitute expected chosen-arm means for the observed rewards. The final
metric averages NR_T over rollouts. The eight-environment average weights each
environment equally. Each stochastic environment has a single fixed realization;
reward/policy randomness is repeated, not the environment realization.

The ordered families are VG-dTS, dTS, dOTS, TS, REXP3, Dynamic TS, Beta-SWTS.
Candidate order and policy order affect random seeds and must remain stable.
Tuning candidates are enumerated in the insertion order of the VG-dTS grid;
other numeric grids are sorted. Ties keep the first candidate.

For zero-based environment index `e`, policy index `p`, and candidate index `c`:

| Protocol | Base seed |
| --- | --- |
| Tuned eight-environment suite | `seed + 1000000*(e+1)` |
| Fixed eight-environment suite | `seed + 4000000 + 2000000*(e+1)` |
| Pure-random-signal control | `seed + 9000000` |

Tuning uses `base + 10000*(p+1) + 100*(c+1)`; evaluation uses
`base + 500000*(p+1)`. Each uses `SeedSequence(seed).spawn(n_runs)` and
`default_rng(child_seed)` for each rollout. Environment construction uses the
original offsets in `build_publication_environment_suite`; the random control
uses `seed + 71`.

**The legacy arithmetic seed schedule reuses some streams across environment/
policy pairs.** Tuning and evaluation for a given pair are separate, but global
independence across all cells should not be claimed. The replay retains these
seeds to reproduce the original results. Changing this schedule would create a
new experiment.

## Details the manuscript should clarify before publication

These are documented rather than silently changing the manuscript or results:

1. **Optimistic action selection.** Every archived VG-dTS setting has
   `optimistic=True`: actions maximize `max(Beta sample, posterior mean)`.
   The main algorithm shows ordinary TS action selection. The appendix mentions
   optimism as optional, but the reported implementation actually enables it.
2. **Reliability gate.** The fixed protocol uses `n0=0`, so its gate opens fully
   after an arm has positive effective evidence. Most tuned settings also use
   zero. Claims that a gradual reliability gate explains these gains need
   qualification or a separate controlled comparison.
3. **Unequal search sizes.** VG-dTS has 64 candidates; dTS, dOTS, and Dynamic TS
   have 7 each; TS has 1; REXP3 has 15; Beta-SWTS has 5. The manuscript's
   "comparable breadth" phrase should not be read as an equal tuning budget.
4. **Dynamic TS is thresholded discounting.** The implementation discounts the
   played arm when its posterior mass exceeds C. It does not run an explicit
   changepoint test or hard-reset all posteriors. Some manuscript descriptions
   characterize this more strongly as reset/change detection.
5. **The tuned parameter table is abbreviated.** Use the archived `best_params`
   dictionaries for exact settings, especially `optimistic`, `vol_high`, and the
   other parameters omitted from the printed tuples. The implementation clips
   `gamma_default` into `[gamma_min, gamma_max]` before mixing.
6. **Conditional simulation evidence.** Repeated reward runs on one environment
   realization do not establish robustness over independent environment draws.
   No confidence intervals or claims of statistical significance are added by
   this cleanup.

## Scope and recovery

Removed material includes VG-dTS v2/v2.1/v3 and enhancement toggles, nine later
baseline families, recovering-bandit code, unrelated ablations, obsolete CLIs,
portfolio placeholders, old report drafts, duplicated figures/results, and TeX
build products. Applicable tests were retained or updated. Local IDE settings
are ignored. `.sty` files are no longer ignored, so a fresh clone includes the
paper's build dependencies.

Git history was not rewritten and no changes were committed or pushed. Earlier
commits still contain the removed work; a paper-only *history* would require a
separate export or history rewrite. The cleanup also made a temporary local
recovery copy of pre-existing workspace files before deleting obsolete files.
