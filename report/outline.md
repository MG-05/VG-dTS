# Adaptive Discounted Thompson Sampling for Non-Stationary Bernoulli Bandits

## 0. Target paper shape (recommended)
- Length: 6-10 pages (excluding references/appendix), single-column class format unless instructed otherwise.
- Core sections: Abstract, Introduction, Related Work, Problem Formulation, Method, Experiments, Results, Discussion, Conclusion, References, Appendix.
- Main claim style: "Our method improves adaptation to changing reward distributions while preserving stable behavior when evidence is low."

---

## 1. Abstract (write last, keep to 150-250 words)
Include exactly these items:
1. Problem: non-stationary multi-armed bandits with Bernoulli rewards.
2. Method: volatility-gated discounted Thompson Sampling (VG-dTS).
3. Setup: slow/fast sinusoidal and abrupt-change environments; compare against TS, DTS, dTS, dOTS, and REXP3.
4. Results: summarize trend (e.g., lower normalized regret in changing regimes, competitive reward).
5. Takeaway: adaptive discounting helps balance responsiveness and stability.

Template:
"We study ... . Existing methods ... but struggle when ... . We propose ... which ... . Across ... environments and ... baselines, we find ... . These results suggest ... ."

---

## 2. Introduction

### 2.1 Motivation and story
- Explain the explore/exploit tradeoff with the classic bandit story.
- Explain why non-stationarity matters: user preferences, markets, ad response, sensor drift.
- State practical consequence: methods that over-trust old data react too slowly after shifts.

### 2.2 Applications to multi-armed bandits
- Recommender systems (content/news ranking over evolving interests).
- Online advertising (click-through rates drift with context).
- Dynamic pricing (conversion probabilities shift).
- Clinical trials with temporal drift.
- Optional tie-in: portfolio-style binary reward mapping (planned in repo configs).

### 2.3 Project objective and contributions
Use a clear bullet list:
1. Implement a non-stationary Bernoulli bandit benchmark suite.
2. Reproduce paper-style environments and dynamic-oracle regret plots.
3. Introduce VG-dTS: per-arm volatility-gated discounting with uncertainty-aware fallback.
4. Empirically compare against standard and non-stationary baselines.

End section with a one-paragraph roadmap of the paper.

---

## 3. Related Work and Prior Limits

### 3.1 Thompson Sampling background
- Briefly describe Beta-Bernoulli TS and why it is strong in stationary settings.

### 3.2 Non-stationary bandit approaches
- Fixed discounting methods (dTS, dOTS): react to change but require manually tuned gamma.
- Dynamic TS thresholding (DTS): discounts after sufficient evidence.
- Adversarial/restart baselines (REXP3): robust but may trade off stochastic efficiency.

### 3.3 Gap your project targets
- Single global discount can be too rigid.
- Different arms can have different local volatility.
- Need adaptive memory with low-evidence protection.

Write this section as comparison logic, not a long citation list.

---

## 4. Problem Formulation

### 4.1 Environment and notation
- K arms, horizon T.
- Bernoulli reward for arm k at time t: r_t in {0,1}, with mean mu_{k,t} in [0,1].
- Non-stationary setting: mu_{k,t} changes with t.

### 4.2 Dynamic oracle benchmark
- Oracle at each t chooses arm with largest true mean: k_t^* = argmax_k mu_{k,t}.
- Oracle expected reward at t: mu_t^* = max_k mu_{k,t}.

### 4.3 Evaluation metrics
- Cumulative reward: sum_{t=1}^T r_t.
- Dynamic cumulative regret:
  R_T = sum_{t=1}^T mu_t^* - sum_{t=1}^T r_t.
- Normalized regret curve:
  R_t / t.

Explicitly note that this project uses dynamic-oracle regret, matching non-stationary evaluation.

---

## 5. Method

### 5.1 Baselines implemented
- Thompson Sampling (`run_TS`)
- Dynamic TS (`run_DTS`)
- Discounted TS (`run_dTS`)
- Discounted Optimistic TS (`run_dOTS`)
- REXP3 (`run_REXP3`)

### 5.2 Proposed algorithm: VG-dTS (`run_VG_dTS`)
Describe each mechanism in the same order as implementation:
1. Per-arm Beta posterior alpha_k, beta_k.
2. Action score:
   theta_k = max(Beta sample, posterior mean) when optimistic mode is enabled.
3. Selected-arm standardized surprise:
   s_t = |x_t - p_hat_t| / sqrt(p_hat_t(1 - p_hat_t) + eps).
4. Per-arm EWMA volatility estimate from surprise statistics.
5. Volatility-to-discount mapping (inverse-linear or logistic) to get gamma_vol.
6. Uncertainty gate using effective sample size n_eff:
   w = n_eff / (n_eff + n0), then
   gamma = (1 - w) * gamma_default + w * gamma_vol.
7. Prior-preserving discount for all arms:
   alpha <- 1 + gamma * (alpha - 1),
   beta  <- 1 + gamma * (beta - 1).
8. Bernoulli conjugate update on played arm.

### 5.3 Why this should help
- High volatility => smaller gamma => faster forgetting.
- Low volatility => larger gamma => more stability.
- Low evidence => fallback toward gamma_default.

### 5.4 Complexity
- Per step O(K) for posterior sampling and discount update.
- Total O(KT), same order as standard discounted TS variants.

Add one pseudocode block for VG-dTS in the final report.

---

## 6. Experimental Setup

### 6.1 Environments (from code)
1. Slow sinusoid: horizon 5000, period 1000, K=4.
2. Fast sinusoid: horizon 1000, period 100, K=4.
3. Abrupt cycle: horizon 1000, cycle length 250, step-change arms.

### 6.2 Comparison protocol
- Use same reward model and horizon per environment.
- Multiple random seeds/runs (e.g., n_runs=100 or 200).
- Report mean reward and normalized regret curves.
- Use dynamic oracle as benchmark.

### 6.3 Hyperparameters
Document exactly what is used in plotting script:
- dTS/dOTS gamma by environment.
- DTS threshold C by environment.
- REXP3 (gamma, Delta) by environment.
- VG-dTS parameter set (gamma range, lambda_vol, n0, mapping).

### 6.4 Reproducibility details (must include for class quality)
- Runtime environment (Python version, package versions).
- Seed handling strategy.
- Script entry points (`main.py`, plotting functions).
- Paths to generated figures in `report/figures/`.

---

## 7. Results

### 7.1 Main figure
- Include the 2x3 panel reproduction:
  - Top row: average reward over time.
  - Bottom row: normalized regret over time.
- Explain each environment separately; do not average all settings into one claim.

### 7.2 Quantitative summary table
Add a table with one row per policy per environment:
- Final cumulative reward.
- Final cumulative regret.
- Final normalized regret.
- (Optional) mean +/- std over seeds.

### 7.3 Observed trends to discuss
- Which methods react fastest in fast-changing regimes?
- Which methods are most stable in slow-changing regimes?
- Where does VG-dTS improve over fixed-gamma methods?
- Any setting where VG-dTS is not best (important for credibility).

### 7.4 Ablations (strongly recommended for grading)
1. VG-dTS optimistic on vs off.
2. Inverse-linear vs logistic gamma mapping.
3. Sensitivity to n0 and gamma range.
4. (Optional) surprise clipping sensitivity.

---

## 8. Discussion, Limitations, and Threats to Validity
- Synthetic environments may not capture all real non-stationarity patterns.
- Hyperparameter fairness is important; tuned baselines can change conclusions.
- Bernoulli reward assumption may limit transfer to continuous rewards.
- Statistical uncertainty: report variance/error bars when possible.

Include at least one paragraph on negative or mixed findings.

---

## 9. Conclusion
- Re-state problem and what was built.
- Re-state empirical takeaway without overclaiming.
- Mention concrete next steps:
  - contextual/non-Bernoulli extensions,
  - real-data evaluation,
  - principled hyperparameter adaptation.

Keep conclusion short (1-2 paragraphs).

---

## 10. Figure and table checklist
- Figure 1: Dynamic oracle visualization (`slow_varying_sinusoid_oracle.png`).
- Figure 2: Main 2x3 comparison (`figure1_reproduction*.png`).
- Table 1: Final metrics by environment/policy.
- Table 2 (optional): Ablation summary.

Every figure must have:
- clear axis labels,
- legend with policy names,
- caption stating environment, runs, and metric definitions.

---

## 11. Suggested section-level writing targets (quality rubric helper)
- Clarity of problem statement: exact notation and objective.
- Method novelty: clearly isolate what is new vs baseline.
- Experimental rigor: fair comparisons, repeated seeds, documented settings.
- Analysis depth: explain why results happen, not just what happened.
- Limitations: explicit and technically honest.
- Reproducibility: enough detail that another student can rerun plots.

---

## 12. Optional appendix contents
- Full VG-dTS pseudocode.
- Additional plots (per-seed variance bands).
- Hyperparameter sweep grids.
- Sanity checks/unit test summary.
