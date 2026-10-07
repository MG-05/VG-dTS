from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.adts.envs import (
    BernoulliNonStationaryEnv,
    global_switching_means,
    mixed_regime_challenge_means,
    per_arm_switching_means,
    random_breakpoint_means,
    random_drift_amplitude_means,
    slow_varying_sinusoid_means,
)
from src.adts.envs import abrupt_means
from src.adts.oracle import run_dynamic_oracle
from src.adts.policies import (
    BetaSWTSParams,
    DTSParams,
    REXP3Params,
    TSParams,
    VGdTSParams,
    dOTSParams,
    dTSParams,
    run_DTS,
    run_REXP3,
    run_TS,
    run_VG_dTS,
    run_beta_swts,
    run_dOTS,
    run_dTS,
)
from src.adts.vgdts_config import make_benchmark_vgdts_params, make_default_vgdts_tuning_grid

PolicyScalar = bool | int | float | str
PolicyRunner = Callable[[np.ndarray, Any, np.random.Generator], np.ndarray]

ALGORITHM_ORDER: tuple[str, ...] = (
    "VG-dTS",
    "dTS",
    "dOTS",
    "TS",
    "REXP3",
    "Dynamic TS",
    "Beta-SWTS",
)

# VG-dTS is intentionally green; all other algorithms avoid green-family colors.
ALGORITHM_COLORS: dict[str, str] = {
    "VG-dTS": "#2E7D32",
    "dTS": "#1565C0",
    "dOTS": "#EF6C00",
    "TS": "#455A64",
    "REXP3": "#C62828",
    "Dynamic TS": "#8D6E63",
    "Beta-SWTS": "#6A1B9A",
}

ORACLE_COLOR = "#111111"
FIG_DPI = 300


@dataclass(frozen=True)
class EnvironmentSpec:
    key: str
    name: str
    means_tk: np.ndarray


@dataclass(frozen=True)
class PolicyCandidate:
    runner: PolicyRunner
    params: Any
    best_params: dict[str, PolicyScalar]


@dataclass(frozen=True)
class TunedPolicyEvaluation:
    best_params: dict[str, PolicyScalar]
    tuning_final_normalized_regret: float
    eval_final_normalized_regret: float
    avg_reward_t: np.ndarray
    avg_norm_regret_t: np.ndarray


@dataclass(frozen=True)
class EnvironmentEvaluation:
    environment: EnvironmentSpec
    policies: dict[str, TunedPolicyEvaluation]


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise ModuleNotFoundError(
                "matplotlib is required for project_publication plotting functions."
            ) from exc
        raise
    return plt


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
    return path


def _pure_random_signal_means(
    horizon: int,
    n_arms: int,
    seed: int,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_arms <= 0:
        raise ValueError("n_arms must be > 0.")
    if not (0.0 <= min_mean < max_mean <= 1.0):
        raise ValueError("Require 0 <= min_mean < max_mean <= 1.")

    rng = np.random.default_rng(seed)
    return rng.uniform(min_mean, max_mean, size=(horizon, n_arms))


def build_publication_environment_suite(
    horizon: int = 5000,
    n_arms: int = 4,
    seed: int = 0,
) -> list[EnvironmentSpec]:
    """
    Eight-environment suite:
      1) Slow sine
      2) Fast sine
      3) Abrupt cycle
      4) Mixed regime
      5) Random breakpoints
      6) Random drift amplitude
      7) Global switching
      8) Per-arm switching
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_arms != 4:
        raise ValueError("The paper suite requires exactly four arms.")

    breakpoint_min_segment = max(1, min(max(10, horizon // 150), horizon))
    breakpoint_max = min(16, max(1, horizon // breakpoint_min_segment - 1))
    breakpoint_min = min(4, breakpoint_max)

    return [
        EnvironmentSpec(
            key="slow",
            name="Slow Sine",
            means_tk=slow_varying_sinusoid_means(horizon=horizon, n_arms=n_arms, period=1000),
        ),
        EnvironmentSpec(
            key="fast",
            name="Fast Sine",
            means_tk=slow_varying_sinusoid_means(horizon=horizon, n_arms=n_arms, period=100),
        ),
        EnvironmentSpec(
            key="abrupt",
            name="Abrupt Cycle",
            means_tk=abrupt_means(horizon=horizon, n_arms=n_arms, cycle_len=250),
        ),
        EnvironmentSpec(
            key="mixed",
            name="Mixed Regime",
            means_tk=mixed_regime_challenge_means(
                horizon=horizon,
                n_arms=n_arms,
                burst_prob=0.05,
                burst_duration=20,
                switch_interval=max(80, horizon // 8),
                seed=seed + 11,
            ),
        ),
        EnvironmentSpec(
            key="random_breakpoints",
            name="Random Breakpoints",
            means_tk=random_breakpoint_means(
                horizon=horizon,
                n_arms=n_arms,
                min_breakpoints=breakpoint_min,
                max_breakpoints=breakpoint_max,
                min_segment=breakpoint_min_segment,
                min_jump=0.10,
                seed=seed + 21,
            ),
        ),
        EnvironmentSpec(
            key="random_drift",
            name="Random Drift Amplitude",
            means_tk=random_drift_amplitude_means(
                horizon=horizon,
                n_arms=n_arms,
                amplitude_min=0.05,
                amplitude_max=0.30,
                amplitude_step_std=0.015,
                period_min=80,
                period_max=max(150, horizon // 5),
                drift_step_std=0.006,
                seed=seed + 31,
            ),
        ),
        EnvironmentSpec(
            key="global_switching",
            name="Global Switching",
            means_tk=global_switching_means(
                horizon=horizon,
                n_arms=n_arms,
                switch_prob=0.02,
                seed=seed + 41,
            ),
        ),
        EnvironmentSpec(
            key="per_arm_switching",
            name="Per-Arm Switching",
            means_tk=per_arm_switching_means(
                horizon=horizon,
                n_arms=n_arms,
                switch_prob=0.02,
                seed=seed + 51,
            ),
        ),
    ]


def _build_single_environment(
    key: str,
    horizon: int,
    n_arms: int,
    seed: int,
) -> EnvironmentSpec:
    key_norm = key.strip().lower().replace("-", "_").replace(" ", "_")

    for env in build_publication_environment_suite(horizon=horizon, n_arms=n_arms, seed=seed):
        if env.key == key_norm:
            return env

    if key_norm in {"random_signal", "pure_random", "random_noise"}:
        return EnvironmentSpec(
            key="pure_random_signal",
            name="Pure Random Signal",
            means_tk=_pure_random_signal_means(horizon=horizon, n_arms=n_arms, seed=seed + 71),
        )

    raise ValueError(
        "Unknown environment key. Use one of: "
        "slow, fast, abrupt, mixed, random_breakpoints, random_drift, "
        "global_switching, per_arm_switching, random_signal."
    )


def _coerce_sorted_unique_floats(values: list[float] | tuple[float, ...]) -> list[float]:
    output = sorted({float(v) for v in values})
    if not output:
        raise ValueError("Float grid cannot be empty.")
    return output


def _coerce_sorted_unique_ints(values: list[int] | tuple[int, ...]) -> list[int]:
    output = sorted({int(v) for v in values})
    if not output:
        raise ValueError("Integer grid cannot be empty.")
    return output


def _build_policy_candidates_tuned(
    horizon: int,
    lambda_grid: list[float] | tuple[float, ...] | None = None,
    rexp3_gamma_grid: list[float] | tuple[float, ...] | None = None,
    rexp3_delta_grid: list[int] | tuple[int, ...] | None = None,
    beta_swts_tau_grid: list[int] | tuple[int, ...] | None = None,
    vgdts_params: VGdTSParams | None = None,
    vgdts_grid: dict[str, list[PolicyScalar] | tuple[PolicyScalar, ...]] | None = None,
) -> dict[str, list[PolicyCandidate]]:
    lam_values = _coerce_sorted_unique_floats(
        lambda_grid or [0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.97]
    )
    rexp3_gamma_values = _coerce_sorted_unique_floats(
        rexp3_gamma_grid or [0.05, 0.10, 0.20, 0.30, 0.40]
    )
    rexp3_delta_values = _coerce_sorted_unique_ints(
        rexp3_delta_grid
        or [
            max(50, horizon // 40),
            max(100, horizon // 20),
            max(200, horizon // 10),
        ]
    )
    beta_tau_values = _coerce_sorted_unique_ints(
        beta_swts_tau_grid
        or [
            max(50, horizon // 200),
            max(100, horizon // 100),
            max(200, horizon // 50),
            max(500, horizon // 20),
            max(1000, horizon // 10),
        ]
    )

    base_vgdts = make_benchmark_vgdts_params() if vgdts_params is None else vgdts_params

    if vgdts_grid is None:
        raw_vgdts_grid = make_default_vgdts_tuning_grid()
    else:
        raw_vgdts_grid = {key: list(values) for key, values in vgdts_grid.items()}

    allowed_vgdts_fields = {field.name for field in fields(VGdTSParams)}
    unknown_fields = sorted(set(raw_vgdts_grid) - allowed_vgdts_fields)
    if unknown_fields:
        raise ValueError(f"Unknown VG-dTS grid field(s): {', '.join(unknown_fields)}")

    vgdts_keys = tuple(raw_vgdts_grid.keys())
    vgdts_values: list[list[PolicyScalar]] = []
    for key in vgdts_keys:
        values = list(raw_vgdts_grid[key])
        if not values:
            raise ValueError(f"VG-dTS grid '{key}' must contain at least one value.")
        vgdts_values.append(values)

    vgdts_candidates: list[PolicyCandidate] = []
    for combo in product(*vgdts_values):
        overrides = dict(zip(vgdts_keys, combo))
        params = replace(base_vgdts, **overrides)
        vgdts_candidates.append(
            PolicyCandidate(
                runner=run_VG_dTS,
                params=params,
                best_params=asdict(params),
            )
        )

    candidates: dict[str, list[PolicyCandidate]] = {
        "VG-dTS": vgdts_candidates,
        "dTS": [
            PolicyCandidate(
                runner=run_dTS,
                params=dTSParams(alpha0=1.0, beta0=1.0, gamma=lam),
                best_params={"lambda": lam, "gamma": lam},
            )
            for lam in lam_values
        ],
        "dOTS": [
            PolicyCandidate(
                runner=run_dOTS,
                params=dOTSParams(alpha0=1.0, beta0=1.0, gamma=lam),
                best_params={"lambda": lam, "gamma": lam},
            )
            for lam in lam_values
        ],
        "TS": [
            PolicyCandidate(
                runner=run_TS,
                params=TSParams(alpha0=1.0, beta0=1.0),
                best_params={},
            )
        ],
        "REXP3": [
            PolicyCandidate(
                runner=run_REXP3,
                params=REXP3Params(gamma=gamma, Delta=delta),
                best_params={"gamma": gamma, "Delta": delta},
            )
            for gamma in rexp3_gamma_values
            for delta in rexp3_delta_values
        ],
        "Dynamic TS": [
            PolicyCandidate(
                runner=run_DTS,
                params=DTSParams(alpha0=1.0, beta0=1.0, C=lam / max(1.0 - lam, 1e-12)),
                best_params={"lambda": lam, "C": lam / max(1.0 - lam, 1e-12)},
            )
            for lam in lam_values
        ],
        "Beta-SWTS": [
            PolicyCandidate(
                runner=run_beta_swts,
                params=BetaSWTSParams(alpha0=1.0, beta0=1.0, tau=tau),
                best_params={"tau": tau},
            )
            for tau in beta_tau_values
        ],
    }

    missing = [name for name in ALGORITHM_ORDER if name not in candidates]
    if missing:
        raise RuntimeError(f"Missing policy families: {', '.join(missing)}")

    return candidates


def _build_policy_candidates_fixed(
    horizon: int,
    *,
    vgdts_params: VGdTSParams | None = None,
    dts_gamma: float = 0.75,
    dots_gamma: float = 0.75,
    rexp3_gamma: float = 0.1136,
    rexp3_delta: int | None = None,
    dynamic_c: float = 250.0,
    beta_swts_tau: int = 200,
) -> dict[str, list[PolicyCandidate]]:
    _ = horizon
    delta = int(rexp3_delta if rexp3_delta is not None else 250)
    vgdts_eff = make_benchmark_vgdts_params() if vgdts_params is None else vgdts_params

    return {
        "VG-dTS": [
            PolicyCandidate(
                runner=run_VG_dTS,
                params=vgdts_eff,
                best_params=asdict(vgdts_eff),
            )
        ],
        "dTS": [
            PolicyCandidate(
                runner=run_dTS,
                params=dTSParams(alpha0=1.0, beta0=1.0, gamma=dts_gamma),
                best_params={"gamma": dts_gamma},
            )
        ],
        "dOTS": [
            PolicyCandidate(
                runner=run_dOTS,
                params=dOTSParams(alpha0=1.0, beta0=1.0, gamma=dots_gamma),
                best_params={"gamma": dots_gamma},
            )
        ],
        "TS": [
            PolicyCandidate(
                runner=run_TS,
                params=TSParams(alpha0=1.0, beta0=1.0),
                best_params={},
            )
        ],
        "REXP3": [
            PolicyCandidate(
                runner=run_REXP3,
                params=REXP3Params(gamma=rexp3_gamma, Delta=delta),
                best_params={"gamma": rexp3_gamma, "Delta": delta},
            )
        ],
        "Dynamic TS": [
            PolicyCandidate(
                runner=run_DTS,
                params=DTSParams(alpha0=1.0, beta0=1.0, C=dynamic_c),
                best_params={"C": dynamic_c},
            )
        ],
        "Beta-SWTS": [
            PolicyCandidate(
                runner=run_beta_swts,
                params=BetaSWTSParams(alpha0=1.0, beta0=1.0, tau=beta_swts_tau),
                best_params={"tau": beta_swts_tau},
            )
        ],
    }


def _compute_avg_curves_and_final_norm_regret(
    mu: np.ndarray,
    runner: PolicyRunner,
    params: Any,
    n_runs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")

    _, horizon = mu.shape
    oracle_mean = mu.max(axis=0)
    oracle_cum = np.cumsum(oracle_mean)
    denom = np.arange(horizon, dtype=float) + 1.0

    sum_reward_t = np.zeros(horizon, dtype=float)
    sum_norm_regret_t = np.zeros(horizon, dtype=float)
    sum_final_norm_regret = 0.0

    run_seeds = np.random.SeedSequence(seed).spawn(n_runs)
    for run_seed in run_seeds:
        rng = np.random.default_rng(run_seed)
        rewards = runner(mu, params, rng).astype(float)
        if rewards.shape != (horizon,):
            raise ValueError("Policy runner must return rewards with shape (T,).")

        sum_reward_t += rewards
        alg_cum = np.cumsum(rewards)
        norm_regret_t = (oracle_cum - alg_cum) / denom
        sum_norm_regret_t += norm_regret_t
        sum_final_norm_regret += float(norm_regret_t[-1])

    return (
        sum_reward_t / float(n_runs),
        sum_norm_regret_t / float(n_runs),
        sum_final_norm_regret / float(n_runs),
    )


def _evaluate_environment_with_candidates(
    env: EnvironmentSpec,
    candidates_by_policy: dict[str, list[PolicyCandidate]],
    eval_runs: int,
    seed: int,
    tuning_runs: int | None,
) -> EnvironmentEvaluation:
    if eval_runs <= 0:
        raise ValueError("eval_runs must be > 0.")
    if tuning_runs is not None and tuning_runs <= 0:
        raise ValueError("tuning_runs must be > 0 when provided.")

    mu = env.means_tk.T
    policy_results: dict[str, TunedPolicyEvaluation] = {}

    for policy_idx, policy_name in enumerate(ALGORITHM_ORDER):
        candidates = candidates_by_policy[policy_name]
        if not candidates:
            raise ValueError(f"No candidates provided for {policy_name}.")

        if tuning_runs is None:
            best_candidate = candidates[0]
            tuning_regret = np.nan
        else:
            best_candidate = candidates[0]
            best_tuning_regret = np.inf
            for candidate_idx, candidate in enumerate(candidates):
                _, _, candidate_regret = _compute_avg_curves_and_final_norm_regret(
                    mu=mu,
                    runner=candidate.runner,
                    params=candidate.params,
                    n_runs=tuning_runs,
                    seed=seed + 10_000 * (policy_idx + 1) + 100 * (candidate_idx + 1),
                )
                if candidate_regret < best_tuning_regret:
                    best_tuning_regret = candidate_regret
                    best_candidate = candidate
            tuning_regret = float(best_tuning_regret)

        avg_reward_t, avg_norm_regret_t, eval_final_regret = _compute_avg_curves_and_final_norm_regret(
            mu=mu,
            runner=best_candidate.runner,
            params=best_candidate.params,
            n_runs=eval_runs,
            seed=seed + 500_000 * (policy_idx + 1),
        )

        if tuning_runs is None:
            tuning_regret = float(eval_final_regret)

        policy_results[policy_name] = TunedPolicyEvaluation(
            best_params=best_candidate.best_params,
            tuning_final_normalized_regret=float(tuning_regret),
            eval_final_normalized_regret=float(eval_final_regret),
            avg_reward_t=avg_reward_t,
            avg_norm_regret_t=avg_norm_regret_t,
        )

    return EnvironmentEvaluation(environment=env, policies=policy_results)


def evaluate_environment_suite(
    env_suite: list[EnvironmentSpec],
    tuning_runs: int,
    eval_runs: int,
    seed: int,
    lambda_grid: list[float] | tuple[float, ...] | None = None,
    rexp3_gamma_grid: list[float] | tuple[float, ...] | None = None,
    rexp3_delta_grid: list[int] | tuple[int, ...] | None = None,
    beta_swts_tau_grid: list[int] | tuple[int, ...] | None = None,
    vgdts_params: VGdTSParams | None = None,
    vgdts_grid: dict[str, list[PolicyScalar] | tuple[PolicyScalar, ...]] | None = None,
) -> dict[str, EnvironmentEvaluation]:
    evaluations: dict[str, EnvironmentEvaluation] = {}

    for env_idx, env in enumerate(env_suite):
        candidates = _build_policy_candidates_tuned(
            horizon=int(env.means_tk.shape[0]),
            lambda_grid=lambda_grid,
            rexp3_gamma_grid=rexp3_gamma_grid,
            rexp3_delta_grid=rexp3_delta_grid,
            beta_swts_tau_grid=beta_swts_tau_grid,
            vgdts_params=vgdts_params,
            vgdts_grid=vgdts_grid,
        )
        evaluations[env.key] = _evaluate_environment_with_candidates(
            env=env,
            candidates_by_policy=candidates,
            eval_runs=eval_runs,
            seed=seed + 1_000_000 * (env_idx + 1),
            tuning_runs=tuning_runs,
        )

    return evaluations


def evaluate_environment_suite_fixed(
    env_suite: list[EnvironmentSpec],
    eval_runs: int,
    seed: int,
    *,
    vgdts_params: VGdTSParams | None = None,
    dts_gamma: float = 0.75,
    dots_gamma: float = 0.75,
    rexp3_gamma: float = 0.1136,
    rexp3_delta: int | None = None,
    dynamic_c: float = 250.0,
    beta_swts_tau: int = 200,
) -> dict[str, EnvironmentEvaluation]:
    evaluations: dict[str, EnvironmentEvaluation] = {}

    for env_idx, env in enumerate(env_suite):
        candidates = _build_policy_candidates_fixed(
            horizon=int(env.means_tk.shape[0]),
            vgdts_params=vgdts_params,
            dts_gamma=dts_gamma,
            dots_gamma=dots_gamma,
            rexp3_gamma=rexp3_gamma,
            rexp3_delta=rexp3_delta,
            dynamic_c=dynamic_c,
            beta_swts_tau=beta_swts_tau,
        )
        evaluations[env.key] = _evaluate_environment_with_candidates(
            env=env,
            candidates_by_policy=candidates,
            eval_runs=eval_runs,
            seed=seed + 2_000_000 * (env_idx + 1),
            tuning_runs=None,
        )

    return evaluations


def _plot_reward_regret_pair(
    ax_reward,
    ax_regret,
    env_eval: EnvironmentEvaluation,
    *,
    legend: bool,
) -> None:
    mu = env_eval.environment.means_tk.T
    _, horizon = mu.shape
    t = np.arange(horizon)

    ax_reward.plot(
        t,
        mu.max(axis=0),
        linestyle="--",
        color=ORACLE_COLOR,
        linewidth=2.0,
        label="Oracle",
    )

    for algo in ALGORITHM_ORDER:
        result = env_eval.policies[algo]
        label = f"{algo} (R_T/T={result.eval_final_normalized_regret:.3f})"
        color = ALGORITHM_COLORS[algo]
        ax_reward.plot(t, result.avg_reward_t, linewidth=1.2, color=color, label=label)
        ax_regret.plot(t, result.avg_norm_regret_t, linewidth=1.2, color=color, label=label)

    ax_reward.set_ylabel("Average Reward")
    ax_reward.grid(alpha=0.25)
    if legend:
        ax_reward.legend(fontsize=8, ncol=2)

    ax_regret.set_xlabel("Timestep")
    ax_regret.set_ylabel("Normalized Regret")
    ax_regret.grid(alpha=0.25)
    if legend:
        ax_regret.legend(fontsize=8, ncol=2)


def _plot_environment_reward_regret(
    env_eval: EnvironmentEvaluation,
    output_path: str | Path,
    *,
    title_prefix: str,
    show: bool = False,
) -> Path:
    plt = _import_matplotlib_pyplot()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)
    _plot_reward_regret_pair(axes[0], axes[1], env_eval, legend=True)

    axes[0].set_title(f"{env_eval.environment.name}: {title_prefix} (Reward)")
    axes[1].set_title(f"{env_eval.environment.name}: {title_prefix} (Normalized Regret)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_original_paper_environments(
    evaluations: dict[str, EnvironmentEvaluation],
    output_path: str | Path,
    show: bool = False,
) -> Path:
    plt = _import_matplotlib_pyplot()

    keys = ["slow", "fast", "abrupt"]
    missing = [key for key in keys if key not in evaluations]
    if missing:
        raise ValueError(f"Missing original-paper environments in evaluation set: {', '.join(missing)}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)

    for col, key in enumerate(keys):
        env_eval = evaluations[key]
        _plot_reward_regret_pair(axes[0, col], axes[1, col], env_eval, legend=True)
        axes[0, col].set_title(f"{env_eval.environment.name}: Tuned Policies (Reward)")
        axes[1, col].set_title(f"{env_eval.environment.name}: Tuned Policies (Normalized Regret)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_random_signal_failure(
    env_eval: EnvironmentEvaluation,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    return _plot_environment_reward_regret(
        env_eval=env_eval,
        output_path=output_path,
        title_prefix="Tuned Policies",
        show=show,
    )


def plot_environment_with_oracle(
    env_spec: EnvironmentSpec,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    plt = _import_matplotlib_pyplot()

    env = BernoulliNonStationaryEnv(mean_rewards=env_spec.means_tk, seed=123)
    oracle = run_dynamic_oracle(env)
    t = np.arange(env.horizon)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    for arm in range(env.n_arms):
        ax.plot(t, env.mean_rewards[:, arm], linewidth=1.1, label=f"Arm {arm + 1}")
    ax.plot(
        t,
        oracle.expected_rewards,
        linestyle="--",
        color=ORACLE_COLOR,
        linewidth=2.0,
        label="Oracle mean",
    )

    ax.set_title(f"{env_spec.name}: Arm Means + Oracle")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Expected Reward")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_environment_oracle_gallery(
    env_suite: list[EnvironmentSpec],
    output_dir: str | Path,
    show: bool = False,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for env in env_suite:
        outputs[env.key] = plot_environment_with_oracle(
            env_spec=env,
            output_path=output_root / f"environment_oracle_{env.key}.png",
            show=show,
        )
    return outputs


def plot_optimized_heatmap(
    evaluations: dict[str, EnvironmentEvaluation],
    output_path: str | Path,
    *,
    title: str,
    show: bool = False,
) -> Path:
    plt = _import_matplotlib_pyplot()

    env_keys = [
        "slow",
        "fast",
        "abrupt",
        "mixed",
        "random_breakpoints",
        "random_drift",
        "global_switching",
        "per_arm_switching",
    ]

    missing = [key for key in env_keys if key not in evaluations]
    if missing:
        raise ValueError(f"Missing environments for heatmap: {', '.join(missing)}")

    env_names = [evaluations[key].environment.name for key in env_keys]
    matrix = np.zeros((len(env_keys), len(ALGORITHM_ORDER)), dtype=float)

    for i, key in enumerate(env_keys):
        env_eval = evaluations[key]
        for j, algo in enumerate(ALGORITHM_ORDER):
            matrix[i, j] = env_eval.policies[algo].eval_final_normalized_regret

    fig, ax = plt.subplots(
        figsize=(max(10, 1.6 * len(ALGORITHM_ORDER)), max(6, 0.9 * len(env_keys))),
        constrained_layout=True,
    )

    im = ax.imshow(matrix, aspect="auto", cmap=plt.get_cmap("RdYlGn_r"))

    ax.set_xticks(np.arange(len(ALGORITHM_ORDER)))
    ax.set_xticklabels(ALGORITHM_ORDER, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(env_names)))
    ax.set_yticklabels(env_names)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        row_best = int(np.argmin(matrix[i]))
        for j in range(matrix.shape[1]):
            weight = "bold" if j == row_best else "normal"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                fontweight=weight,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("R_T / T")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def _evaluation_to_json_dict(env_eval: EnvironmentEvaluation) -> dict[str, Any]:
    return {
        "environment": {
            "key": env_eval.environment.key,
            "name": env_eval.environment.name,
            "horizon": int(env_eval.environment.means_tk.shape[0]),
            "n_arms": int(env_eval.environment.means_tk.shape[1]),
        },
        "policies": {
            policy_name: {
                "best_params": policy_eval.best_params,
                "tuning_final_normalized_regret": policy_eval.tuning_final_normalized_regret,
                "eval_final_normalized_regret": policy_eval.eval_final_normalized_regret,
            }
            for policy_name, policy_eval in env_eval.policies.items()
        },
    }
