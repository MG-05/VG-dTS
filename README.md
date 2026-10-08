<h1 align="center">Variance-Gated Discounted Thompson Sampling<br>for Non-Stationary Bandits</h1>

<p align="center">
  <strong>Mayand Gulati &nbsp;·&nbsp; Kerong Wang &nbsp;·&nbsp; Wei-Chen Au</strong><br>
  Department of Electrical and Computer Engineering<br>
  University of California, Santa Barbara
</p>

<p align="center">
  <a href="report/ECE_270_Project_Publication/main.tex">Manuscript</a>
  &nbsp;·&nbsp;
  <a href="REPRODUCIBILITY.md">Reproduction guide</a>
  &nbsp;·&nbsp;
  <a href="results/paper/">Reference results</a>
  &nbsp;·&nbsp;
  <a href="src/adts/policies.py">Implementation</a>
  &nbsp;·&nbsp;
  <a href="CITATION.cff">Cite this work</a>
</p>

---

## Abstract

Non-stationary bandits require a learner to retain useful evidence while
forgetting observations that no longer reflect the environment. A fixed discount
factor imposes one memory timescale on every arm. **Variance-Gated Discounted
Thompson Sampling (VG-dTS)** instead adapts each arm's forgetting rate using an
online estimate of variability in posterior prediction errors. Across eight
synthetic environments, VG-dTS achieves the lowest average normalized regret
among seven evaluated methods under both tuned and fixed-parameter protocols.
The gains are strongest in abrupt dynamics; performance remains dependent on
the environment and parameter choices.

## 1. Method

VG-dTS maintains a Beta posterior for each arm and couples Thompson-style action
selection with adaptive, prior-preserving forgetting:

1. **Select.** Draw a sample from each posterior and play the arm with the highest
   score. The reported experiments use optimistic scores: the larger of the
   sample and its posterior mean.
2. **Measure surprise.** Standardize the played arm's prediction error and update
   its exponentially weighted mean and variance. Greater estimated variability
   maps to a smaller discount factor.
3. **Adapt memory.** Mix the volatility-based discount with a default using an
   effective-sample-size gate. Discount every arm's evidence, then incorporate
   the observed reward into the played arm's posterior.

$$
\gamma_{k,t}=(1-w_{k,t})\gamma_{\mathrm{def}}+w_{k,t}\gamma^{\mathrm{vol}}_{k,t}
$$

$$
\alpha_k \leftarrow 1+\gamma_{k,t}(\alpha_k-1),
\qquad
\beta_k \leftarrow 1+\gamma_{k,t}(\beta_k-1).
$$

Smaller $\gamma_{k,t}$ means shorter memory. The gate can temper adaptation when
evidence is scarce; with $n_0=0$, as in the fixed protocol, it opens fully once
an arm has positive effective evidence.

## 2. Empirical results

We evaluate **four arms over 5,000 rounds**, averaging over **1,000 rollouts** on
each fixed environment realization. The suite spans smooth drift, abrupt shifts,
mixed dynamics, and stochastic switching. Baselines are TS, dTS, dOTS, REXP3,
Dynamic TS, and Beta-SWTS. Normalized regret is cumulative dynamic-oracle mean
reward minus realized reward, divided by the horizon; **lower is better**.

### Adaptation over time

With one fixed parameter setting per method across environments, VG-dTS attains
the lowest final regret in four of the eight environments. In abrupt changes,
its final regret is **20.2% below the next-best method, dOTS**.

<p align="center">
  <a href="docs/figures/fixed_parameter_regret.png">
    <img src="docs/figures/fixed_parameter_regret.png" width="100%" alt="Normalized regret curves for fast periodic drift, abrupt changes, global switching, and per-arm switching, with all seven policies shown over 5,000 rounds.">
  </a>
</p>

*Figure 1. Selected fixed-parameter wins. All seven methods, the full horizon,
and unsmoothed averages over 1,000 rollouts are shown. VG-dTS is highlighted in
green; these are the four environments where it finishes with the lowest regret.*

### Performance across the full suite

VG-dTS achieves the best eight-environment average in both protocols. Against
dTS and dOTS, its fixed-parameter average is **33.5%** and **24.2%** lower,
respectively.

| Evaluation protocol | **VG-dTS** | dTS | dOTS |
| :--- | ---: | ---: | ---: |
| Tuned per environment | **0.1047** | 0.1200 | 0.1067 |
| Fixed across environments | **0.1351** | 0.2031 | 0.1781 |

*Table 1. Mean final normalized regret across all eight environments, comparing
VG-dTS with the two fixed-discount TS variants. The heatmaps below include all
seven methods.*

<table>
  <tr>
    <th align="center">Tuned per environment</th>
    <th align="center">Fixed across environments</th>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="report/ECE_270_Project_Publication/optimized_heatmap_8env_7policies.png">
        <img src="report/ECE_270_Project_Publication/optimized_heatmap_8env_7policies.png" width="100%" alt="Tuned final normalized regret across eight environments and seven policies.">
      </a>
    </td>
    <td width="50%" valign="top">
      <a href="report/ECE_270_Project_Publication/fixed_params_heatmap_8env_7policies.png">
        <img src="report/ECE_270_Project_Publication/fixed_params_heatmap_8env_7policies.png" width="100%" alt="Fixed-parameter final normalized regret across the same eight environments and seven policies.">
      </a>
    </td>
  </tr>
</table>

*Figure 2. Final normalized regret across the complete benchmark suite. Select
either panel to inspect the full-resolution figure.*

**Scope of the evidence.** The average advantage does not imply a win in every
regime. Surprise is an indirect signal of change, other methods lead in several
environments, and the pure-random-signal control leaves little room for
improvement. See the [methodological notes](REPRODUCIBILITY.md#experimental-clarifications-and-limitations)
for tuning budgets, gate settings, and the limits of these comparisons.

## 3. Reproduce the results

Use **Python 3.12** and the pinned dependencies. NumPy 2.x changes the breakpoint
environment generated from the same seed.

<details>
<summary><strong>Installation and paper replay</strong></summary>

Run from the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m project_publication --mode paper --workers 4
python -m project_publication --mode verify
```

This replays archived hyperparameters with 1,000 evaluation runs and writes
figures, curves, and summaries to `artifacts/reproduction/`. Choose a new
`--output-dir` if it already contains a run. To regenerate Figure 1 afterward:

```sh
python -m project_publication.readme_figures
```

</details>

All **119 reported evaluation values** were reproduced exactly during validation;
this checks evaluation at the archived settings, not a new tuning search.
[Full instructions](REPRODUCIBILITY.md) cover retuning, smoke tests, figure
regeneration, manuscript compilation, and the validation record.

## Citation

Please cite the accompanying manuscript and identify the software version or
commit used for your experiments. [CITATION.cff](CITATION.cff) provides structured
metadata for the manuscript and software; the manuscript is listed as unpublished.

```bibtex
@unpublished{gulati_vgdts,
  author = {Gulati, Mayand and Wang, Kerong and Au, Wei-Chen},
  title = {Variance-Gated Discounted Thompson Sampling for Non-Stationary Bandits},
  note = {Unpublished manuscript. University of California, Santa Barbara},
  url = {https://github.com/MG-05/VG-dTS}
}
```

## License

Original project content is licensed under the [Apache License 2.0](LICENSE).
Bundled third-party LaTeX styles retain their upstream terms; see [NOTICE](NOTICE)
and the notices in those files.
