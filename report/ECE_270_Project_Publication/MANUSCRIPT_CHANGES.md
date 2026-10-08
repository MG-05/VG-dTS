# Manuscript corrections for public release

On 2026-10-07, `main.tex` and `appendix.tex` were revised against the retained
implementation, archived parameter dictionaries, and evaluation validation record.
The original export hashes in `OVERLEAF_MANIFEST.json` remain historical provenance;
they are not checksums for the revised sources. Numerical result JSON files and
all 12 manuscript figures are unchanged.

The revisions:

- Specify optimistic scoring in every reported VG-dTS experiment, the uniform
  prior, inverse-linear mapping, shared constants, and clipping of the supplied
  default discount before mixing. Identify the reported algorithm in the pseudocode.
- Distinguish a gradual gate (`n0 > 0`) from the binary gate (`n0 = 0`); the latter
  is used in every fixed setting and seven of eight tuned suite settings.
- Describe the volatility proxy as centered surprise variation, rather than
  persistent prediction-error magnitude. Remove unsupported component-level
  explanations of the observed gains.
- Correct the expected-regret equation to average over random actions as well
  as reward noise; state that realized normalized regret may be negative.
- State the unequal search sizes, reconstructed seed/run-count provenance,
  evaluation-only replay validation, shared environment realizations, and
  cross-cell seed reuse. Distinguish point estimates from significance claims.
- Describe the implemented Dynamic TS baseline as played-arm thresholded
  discounting, without changepoint testing or hard resets.
- Supply previously omitted active VG-dTS constants and mapping thresholds;
  identify rounded baseline values and point to full-precision archived settings.
- Clarify the abrupt environment's zero-based cycle offsets and the negative
  control's independent uniform draws of arm means.
- Limit robustness and causal claims to what these experiments establish.
- Use the template's preprint mode to display the named authors, without implying
  conference acceptance. Citation metadata identifies an unpublished manuscript;
  no DOI, publication venue, or release date is asserted.

Evidence: `src/adts/policies.py`, `src/adts/vgdts_config.py`, `src/adts/envs.py`,
`project_publication/pipeline.py`, `project_publication/reproduce.py`, and the
three reference summaries plus `VALIDATION.json` in `results/paper/`.
No new experiment or full tuning sweep was run for these editorial corrections.
