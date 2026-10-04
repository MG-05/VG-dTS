from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .envs import (
    BernoulliNonStationaryEnv,
    global_switching_means,
    make_mixed_regime_challenge_env,
    make_recovering_reward_table,
    make_random_breakpoint_env,
    make_random_drift_amplitude_env,
    make_slow_varying_sinusoid_env,
    mixed_regime_challenge_means,
    per_arm_switching_means,
    random_breakpoint_means,
    random_drift_amplitude_means,
    slow_varying_sinusoid_means,
)
from .oracle import run_dynamic_oracle
from .policies import (
    AdaSwitchParams,
    AFFTSParams,
    BetaSWTSParams,
    CUSUMUCBParams,
    DTSParams,
    DUCBParams,
    DLinUCBParams,
    GLRklUCBParams,
    GammaSWGTSParams,
    GlobalCTSParams,
    REXP3Params,
    RGPParams,
    SWTSParams,
    SWUCBParams,
    TSParams,
    VGdTSParams,
    dOTSParams,
    dTSParams,
    run_DTS,
    run_REXP3,
    run_TS,
    run_VG_dTS,
    run_adaswitch,
    run_aff_ts,
    run_beta_swts,
    run_cusum_ucb,
    run_d_ucb,
    run_dlinucb_onehot,
    run_drpg_ts,
    run_drpg_ucb,
    run_gamma_swgts,
    run_glr_klucb,
    run_global_cts,
    run_sw_ts,
    run_sw_ucb,
    run_vgdts_recovering,
    run_dOTS,
    run_dTS,
)
from .vgdts_config import VGDTSScalar, make_benchmark_vgdts_params


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise ModuleNotFoundError(
                "matplotlib is required for plotting. Install it or use make_plots=False where supported."
            ) from exc
        raise
    return plt


def run_slow_varying_sinusoid_oracle_experiment(
    horizon: int = 5000,
    period: int = 1000,
    n_arms: int = 4,
    seed: int | None = 0,
    output_path: str | Path = "report/figures/slow_varying_sinusoid_oracle.png",
    show: bool = False,
) -> Path:
    """
    Build the slow-varying sinusoidal environment from the paper and
    graph the dynamic oracle benchmark.
    """

    plt = _import_matplotlib_pyplot()

    env = make_slow_varying_sinusoid_env(
        horizon=horizon,
        n_arms=n_arms,
        period=period,
        seed=seed,
    )
    oracle = run_dynamic_oracle(env)

    t = np.arange(env.horizon)
    means = env.mean_rewards

    # avg_expected = oracle.cumulative_expected_reward / (t + 1)
    # avg_sampled = oracle.cumulative_sampled_reward / (t + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    for arm in range(env.n_arms):
        axes[0].plot(t, means[:, arm], linewidth=1.5, label=f"Arm {arm + 1}")

    axes[0].plot(
        t,
        oracle.expected_rewards,
        linestyle="--",
        color="black",
        linewidth=2.0,
        label="Dynamic oracle mean",
    )
    axes[0].set_title("Slow-Varying Sinusoidal Environment (Period = 1000)")
    axes[0].set_ylabel("Expected reward")
    axes[0].legend(ncol=3, fontsize=9)

    axes[1].step(t, oracle.chosen_arms + 1, where="post", color="tab:blue")
    axes[1].set_title("Dynamic Oracle Arm Choice (argmax mean at each t)")
    axes[1].set_ylabel("Arm index")
    axes[1].set_yticks(np.arange(1, env.n_arms + 1))

    # axes[2].plot(t, avg_expected, color="black", label="Oracle cumulative expected / t")
    # axes[2].plot(t, avg_sampled, color="tab:green", alpha=0.9, label="Oracle cumulative sampled / t")
    # axes[2].set_title("Dynamic Oracle Performance")
    # axes[2].set_xlabel("Time step")
    # axes[2].set_ylabel("Cumulative average reward")
    # axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)

    if show:
        plt.show()
    plt.close(fig)

    return output_path


def _build_rigorous_env_suite(
    horizon: int,
    n_arms: int,
    seed: int | None,
) -> list[tuple[str, BernoulliNonStationaryEnv]]:
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_arms < 4:
        raise ValueError("n_arms must be >= 4 for the mixed-regime challenge environment.")

    if seed is None:
        means_seeds: list[int | None] = [None, None, None]
        env_seeds: list[int | None] = [None, None, None]
    else:
        seed_state = np.random.SeedSequence(seed).generate_state(6)
        means_seeds = [int(x) for x in seed_state[:3]]
        env_seeds = [int(x) for x in seed_state[3:]]

    breakpoint_min_segment = max(1, min(max(5, horizon // 100), horizon))
    breakpoint_max = min(12, max(0, horizon // breakpoint_min_segment - 1))
    breakpoint_min = min(3, breakpoint_max)

    suite = [
        (
            "Mixed Regime (stable / drift / burst / switch)",
            make_mixed_regime_challenge_env(
                horizon=horizon,
                n_arms=n_arms,
                burst_prob=0.05,
                burst_duration=20,
                switch_interval=max(50, horizon // 8),
                means_seed=means_seeds[0],
                seed=env_seeds[0],
            ),
        ),
        (
            "Random Breakpoints",
            make_random_breakpoint_env(
                horizon=horizon,
                n_arms=n_arms,
                min_breakpoints=breakpoint_min,
                max_breakpoints=breakpoint_max,
                min_segment=breakpoint_min_segment,
                min_jump=0.10,
                means_seed=means_seeds[1],
                seed=env_seeds[1],
            ),
        ),
        (
            "Random Drift Amplitude",
            make_random_drift_amplitude_env(
                horizon=horizon,
                n_arms=n_arms,
                amplitude_min=0.05,
                amplitude_max=0.30,
                amplitude_step_std=0.015,
                period_min=80,
                period_max=max(120, horizon // 4),
                drift_step_std=0.006,
                means_seed=means_seeds[2],
                seed=env_seeds[2],
            ),
        ),
    ]
    return suite


def run_rigorous_env_suite_oracle_experiment(
    horizon: int = 2000,
    n_arms: int = 4,
    seed: int | None = 0,
    output_path: str | Path = "report/figures/rigorous_env_suite_oracle.png",
    show: bool = False,
) -> Path:
    """
    Build a stronger set of non-stationary environments and visualize:
      - per-arm means with dynamic-oracle expected reward
      - dynamic-oracle arm choices over time
    """

    plt = _import_matplotlib_pyplot()

    env_specs = _build_rigorous_env_suite(horizon=horizon, n_arms=n_arms, seed=seed)
    n_rows = len(env_specs)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(15, 4 * n_rows),
        constrained_layout=True,
        sharex="col",
    )
    axes_arr = np.atleast_2d(axes)

    for row, (name, env) in enumerate(env_specs):
        oracle = run_dynamic_oracle(env)
        t = np.arange(env.horizon)

        ax_means = axes_arr[row, 0]
        ax_oracle = axes_arr[row, 1]

        for arm in range(env.n_arms):
            ax_means.plot(t, env.mean_rewards[:, arm], linewidth=1.2, label=f"Arm {arm + 1}")
        ax_means.plot(
            t,
            oracle.expected_rewards,
            linestyle="--",
            color="black",
            linewidth=2.0,
            label="Dynamic oracle mean",
        )
        ax_means.set_title(name)
        ax_means.set_ylabel("Expected reward")
        ax_means.grid(alpha=0.25)
        ax_means.legend(ncol=3, fontsize=8)

        ax_oracle.step(t, oracle.chosen_arms + 1, where="post", color="tab:blue")
        ax_oracle.set_title(f"{name} - Oracle arm choice")
        ax_oracle.set_ylabel("Arm index")
        ax_oracle.set_yticks(np.arange(1, env.n_arms + 1))
        ax_oracle.grid(alpha=0.25)

    axes_arr[-1, 0].set_xlabel("Time step")
    axes_arr[-1, 1].set_xlabel("Time step")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


PolicyParamValue = bool | float | int | str


@dataclass(frozen=True)
class TunedPolicyResult:
    policy: str
    best_params: dict[str, PolicyParamValue]
    final_normalized_regret: float


@dataclass(frozen=True)
class _PolicyCandidate:
    policy: str
    label: str
    runner: Callable[[np.ndarray, Any, np.random.Generator], np.ndarray]
    params: Any
    best_params: dict[str, PolicyParamValue]


def _abrupt_means(
    horizon: int = 1000,
    n_arms: int = 4,
    cycle_len: int = 250,
    change_times: list[int] | None = None,
    levels: list[float] | None = None,
) -> np.ndarray:
    if change_times is None:
        change_times = [50, 100, 150, 200]
    if levels is None:
        levels = [0.10, 0.37, 0.63, 0.90]
    if len(change_times) != n_arms or len(levels) != n_arms:
        raise ValueError("change_times and levels must have length n_arms.")
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if cycle_len <= 0:
        raise ValueError("cycle_len must be > 0.")

    means = np.zeros((horizon, n_arms), dtype=float)
    for t in range(horizon):
        tau = t % cycle_len
        for arm in range(n_arms):
            if tau >= change_times[arm]:
                means[t, arm] = levels[arm]
    return means


def _normalize_env_key(environment: str) -> str:
    return environment.strip().lower().replace("-", "_").replace(" ", "_")


def _build_selected_environment_means(
    environment: str,
    seed: int,
    environment_kwargs: dict[str, int | float] | None,
) -> tuple[str, np.ndarray]:
    kwargs = dict(environment_kwargs or {})
    env_key = _normalize_env_key(environment)

    if env_key in {"slow", "slow_sine", "slow_varying_sinusoid"}:
        kwargs.setdefault("horizon", 5000)
        kwargs.setdefault("n_arms", 4)
        kwargs.setdefault("period", 1000)
        return "Slow Varying Environment", slow_varying_sinusoid_means(**kwargs)

    if env_key in {"fast", "fast_sine", "fast_varying_sinusoid"}:
        kwargs.setdefault("horizon", 1000)
        kwargs.setdefault("n_arms", 4)
        kwargs.setdefault("period", 100)
        return "Fast Varying Environment", slow_varying_sinusoid_means(**kwargs)

    if env_key in {"abrupt", "step_cycle"}:
        kwargs.setdefault("horizon", 1000)
        kwargs.setdefault("n_arms", 4)
        kwargs.setdefault("cycle_len", 250)
        return "Abrupt Varying Environment", _abrupt_means(**kwargs)

    if env_key in {"mixed", "mixed_regime"}:
        kwargs.setdefault("horizon", 2000)
        kwargs.setdefault("n_arms", 4)
        kwargs.setdefault("burst_prob", 0.05)
        kwargs.setdefault("burst_duration", 20)
        kwargs.setdefault("switch_interval", max(50, int(kwargs["horizon"]) // 8))
        kwargs.setdefault("seed", seed + 11)
        return "Mixed Regime Environment", mixed_regime_challenge_means(**kwargs)

    if env_key in {"random_breakpoints", "breakpoints", "random_breakpoint"}:
        kwargs.setdefault("horizon", 2000)
        kwargs.setdefault("n_arms", 4)
        horizon = int(kwargs["horizon"])
        breakpoint_min_segment = max(1, min(max(5, horizon // 100), horizon))
        breakpoint_max = min(12, max(0, horizon // breakpoint_min_segment - 1))
        breakpoint_min = min(3, breakpoint_max)
        kwargs.setdefault("min_breakpoints", breakpoint_min)
        kwargs.setdefault("max_breakpoints", breakpoint_max)
        kwargs.setdefault("min_segment", breakpoint_min_segment)
        kwargs.setdefault("min_jump", 0.10)
        kwargs.setdefault("seed", seed + 29)
        return "Random Breakpoints Environment", random_breakpoint_means(**kwargs)

    if env_key in {"random_drift", "drift", "random_drift_amplitude"}:
        kwargs.setdefault("horizon", 2000)
        kwargs.setdefault("n_arms", 4)
        horizon = int(kwargs["horizon"])
        kwargs.setdefault("amplitude_min", 0.05)
        kwargs.setdefault("amplitude_max", 0.30)
        kwargs.setdefault("amplitude_step_std", 0.015)
        kwargs.setdefault("period_min", 80)
        kwargs.setdefault("period_max", max(120, horizon // 4))
        kwargs.setdefault("drift_step_std", 0.006)
        kwargs.setdefault("seed", seed + 47)
        return "Random Drift Amplitude Environment", random_drift_amplitude_means(**kwargs)

    raise ValueError(
        "Unknown environment. Use one of: "
        "'slow', 'fast', 'abrupt', 'mixed', 'random_breakpoints', 'random_drift'."
    )


def _coerce_float_grid(
    values: list[float] | tuple[float, ...] | None,
    *,
    default: list[float],
    name: str,
    min_value: float,
    max_value: float,
    include_max: bool = False,
) -> list[float]:
    source = default if values is None else list(values)
    if not source:
        raise ValueError(f"{name} must contain at least one value.")

    parsed: list[float] = []
    for raw in source:
        value = float(raw)
        if value <= min_value or (value > max_value if include_max else value >= max_value):
            bound = f"({min_value}, {max_value}{']' if include_max else ')'}"
            raise ValueError(f"{name} values must be in {bound}.")
        parsed.append(value)
    return sorted(set(parsed))


def _coerce_int_grid(
    values: list[int] | tuple[int, ...] | None,
    *,
    default: list[int],
    name: str,
    min_value: int,
) -> list[int]:
    source = default if values is None else list(values)
    if not source:
        raise ValueError(f"{name} must contain at least one value.")
    parsed = sorted({int(v) for v in source})
    if any(v < min_value for v in parsed):
        raise ValueError(f"{name} values must be >= {min_value}.")
    return parsed


def _estimate_final_normalized_regret(
    mu: np.ndarray,
    runner: Callable[[np.ndarray, Any, np.random.Generator], np.ndarray],
    params: Any,
    n_runs: int,
    seed: int,
) -> float:
    _, horizon = mu.shape
    oracle_total = float(np.sum(mu.max(axis=0)))
    total_norm_regret = 0.0

    run_seeds = np.random.SeedSequence(seed).spawn(n_runs)
    for run_seed in run_seeds:
        rng = np.random.default_rng(run_seed)
        rewards = runner(mu, params, rng).astype(float)
        if rewards.shape != (horizon,):
            raise ValueError("Policy runner must return rewards with shape (T,).")
        total_norm_regret += (oracle_total - float(np.sum(rewards))) / float(horizon)

    return total_norm_regret / float(n_runs)


def _compute_avg_curves_and_final_normalized_regret(
    mu: np.ndarray,
    runner: Callable[[np.ndarray, Any, np.random.Generator], np.ndarray],
    params: Any,
    n_runs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
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

    avg_reward_t = sum_reward_t / float(n_runs)
    avg_norm_regret_t = sum_norm_regret_t / float(n_runs)
    avg_final_norm_regret = sum_final_norm_regret / float(n_runs)
    return avg_reward_t, avg_norm_regret_t, avg_final_norm_regret


def _tune_policy_family(
    mu: np.ndarray,
    candidates: list[_PolicyCandidate],
    tuning_runs: int,
    seed: int,
) -> _PolicyCandidate:
    best_candidate = candidates[0]
    best_regret = np.inf

    for idx, candidate in enumerate(candidates):
        candidate_seed = seed + 10_000 * idx
        final_norm_regret = _estimate_final_normalized_regret(
            mu=mu,
            runner=candidate.runner,
            params=candidate.params,
            n_runs=tuning_runs,
            seed=candidate_seed,
        )
        if final_norm_regret < best_regret:
            best_regret = final_norm_regret
            best_candidate = candidate
    return best_candidate


def _format_vgdts_value(value: PolicyParamValue) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _build_vgdts_label(params: VGdTSParams, tuned_keys: tuple[str, ...]) -> str:
    short_names = {
        "gamma_default": "g0",
        "gamma_min": "gmin",
        "gamma_max": "gmax",
        "lambda_vol": "lvol",
        "n0": "n0",
        "vol_low": "vlo",
        "vol_high": "vhi",
        "one_sided_negative_surprise": "one_side",
        "gamma_mapping": "map",
        "optimistic": "opt",
    }
    pieces = [
        f"{short_names.get(key, key)}={_format_vgdts_value(getattr(params, key))}"
        for key in tuned_keys
    ]
    return "VG-dTS" if not pieces else f"VG-dTS ({', '.join(pieces)})"


def _build_vgdts_candidates(
    base_params: VGdTSParams,
    vgdts_grid: dict[str, list[VGDTSScalar]] | None,
) -> list[_PolicyCandidate]:
    if not vgdts_grid:
        return [
            _PolicyCandidate(
                policy="VG-dTS",
                label="VG-dTS",
                runner=run_VG_dTS,
                params=base_params,
                best_params=asdict(base_params),
            )
        ]

    allowed_fields = {field.name for field in fields(VGdTSParams)}
    unknown = sorted(set(vgdts_grid) - allowed_fields)
    if unknown:
        raise ValueError(f"Unknown VG-dTS grid field(s): {', '.join(unknown)}")

    grid_keys = tuple(vgdts_grid.keys())
    grid_values: list[list[VGDTSScalar]] = []
    for key in grid_keys:
        values = list(vgdts_grid[key])
        if not values:
            raise ValueError(f"VG-dTS grid '{key}' must contain at least one value.")
        grid_values.append(values)

    candidates: list[_PolicyCandidate] = []
    for combo in product(*grid_values):
        overrides = dict(zip(grid_keys, combo))
        params = replace(base_params, **overrides)
        candidates.append(
            _PolicyCandidate(
                policy="VG-dTS",
                label=_build_vgdts_label(params, grid_keys),
                runner=run_VG_dTS,
                params=params,
                best_params=asdict(params),
            )
        )
    return candidates


def run_single_env_lambda_tuning_experiment(
    environment: str = "slow",
    n_runs: int = 100,
    tuning_runs: int | None = None,
    seed: int = 0,
    lambda_grid: list[float] | tuple[float, ...] | None = None,
    rexp3_gamma_grid: list[float] | tuple[float, ...] | None = None,
    rexp3_delta_grid: list[int] | tuple[int, ...] | None = None,
    environment_kwargs: dict[str, int | float] | None = None,
    vgdts_params: VGdTSParams | None = None,
    vgdts_grid: dict[str, list[VGDTSScalar]] | None = None,
    output_path: str | Path = "report/figures/single_env_lambda_tuning.png",
    show: bool = False,
) -> tuple[Path, dict[str, TunedPolicyResult]]:
    """
    Tune algorithm hyperparameters on one selected environment using final normalized regret,
    then plot average reward and normalized regret in Figure-1 style.

    Tuned families:
      - dTS (lambda/gamma grid)
      - dOTS (lambda/gamma grid)
      - Thompson Sampling (no tuning)
      - REXP3 (gamma and Delta grid)
      - Dynamic TS (lambda grid mapped to C = lambda/(1-lambda))
      - VG-dTS (full parameter grid when `vgdts_grid` is provided)
    """

    plt = _import_matplotlib_pyplot()

    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")
    if tuning_runs is None:
        tuning_runs = n_runs
    if tuning_runs <= 0:
        raise ValueError("tuning_runs must be > 0.")

    env_title, means_tk = _build_selected_environment_means(
        environment=environment,
        seed=seed,
        environment_kwargs=environment_kwargs,
    )
    mu = means_tk.T
    _, horizon = mu.shape
    t = np.arange(horizon)

    lambda_values = _coerce_float_grid(
        lambda_grid,
        default=[0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.97],
        name="lambda_grid",
        min_value=0.0,
        max_value=1.0,
    )
    rexp3_gamma_values = _coerce_float_grid(
        rexp3_gamma_grid,
        default=[0.05, 0.10, 0.20, 0.30, 0.40],
        name="rexp3_gamma_grid",
        min_value=0.0,
        max_value=1.0,
        include_max=True,
    )
    rexp3_delta_defaults = sorted(
        {
            max(10, horizon // 20),
            max(20, horizon // 10),
            max(40, horizon // 5),
        }
    )
    rexp3_delta_values = _coerce_int_grid(
        rexp3_delta_grid,
        default=rexp3_delta_defaults,
        name="rexp3_delta_grid",
        min_value=1,
    )

    vgdts_base_params = make_benchmark_vgdts_params() if vgdts_params is None else vgdts_params
    vgdts_candidates = _build_vgdts_candidates(vgdts_base_params, vgdts_grid)

    policy_families: list[tuple[str, list[_PolicyCandidate]]] = [
        (
            "dTS",
            [
                _PolicyCandidate(
                    policy="dTS",
                    label=f"dTS (λ={lam:.3f})",
                    runner=run_dTS,
                    params=dTSParams(alpha0=1.0, beta0=1.0, gamma=lam),
                    best_params={"lambda": lam, "gamma": lam},
                )
                for lam in lambda_values
            ],
        ),
        (
            "dOTS",
            [
                _PolicyCandidate(
                    policy="dOTS",
                    label=f"dOTS (λ={lam:.3f})",
                    runner=run_dOTS,
                    params=dOTSParams(alpha0=1.0, beta0=1.0, gamma=lam),
                    best_params={"lambda": lam, "gamma": lam},
                )
                for lam in lambda_values
            ],
        ),
        (
            "TS",
            [
                _PolicyCandidate(
                    policy="TS",
                    label="Thompson Sampling",
                    runner=run_TS,
                    params=TSParams(alpha0=1.0, beta0=1.0),
                    best_params={},
                )
            ],
        ),
        (
            "REXP3",
            [
                _PolicyCandidate(
                    policy="REXP3",
                    label=f"REXP3 (λ={gamma:.3f}, Δ={delta})",
                    runner=run_REXP3,
                    params=REXP3Params(gamma=gamma, Delta=delta),
                    best_params={"lambda": gamma, "gamma": gamma, "Delta": delta},
                )
                for gamma in rexp3_gamma_values
                for delta in rexp3_delta_values
            ],
        ),
        (
            "Dynamic TS",
            [
                _PolicyCandidate(
                    policy="Dynamic TS",
                    label=f"Dynamic TS (λ={lam:.3f}, C={lam / (1.0 - lam):.2f})",
                    runner=run_DTS,
                    params=DTSParams(alpha0=1.0, beta0=1.0, C=lam / (1.0 - lam)),
                    best_params={"lambda": lam, "C": lam / (1.0 - lam)},
                )
                for lam in lambda_values
            ],
        ),
        (
            "VG-dTS",
            vgdts_candidates,
        ),
    ]

    tuned_results: dict[str, TunedPolicyResult] = {}
    curve_payload: list[tuple[str, np.ndarray, np.ndarray]] = []

    for family_idx, (family_name, candidates) in enumerate(policy_families):
        best_candidate = _tune_policy_family(
            mu=mu,
            candidates=candidates,
            tuning_runs=tuning_runs,
            seed=seed + 100_000 * (family_idx + 1),
        )
        avg_reward_t, avg_norm_regret_t, final_norm_regret = _compute_avg_curves_and_final_normalized_regret(
            mu=mu,
            runner=best_candidate.runner,
            params=best_candidate.params,
            n_runs=n_runs,
            seed=seed + 300_000 * (family_idx + 1),
        )
        curve_payload.append(
            (
                f"{best_candidate.label}, R_T/T={final_norm_regret:.3f}",
                avg_reward_t,
                avg_norm_regret_t,
            )
        )
        tuned_results[family_name] = TunedPolicyResult(
            policy=family_name,
            best_params=best_candidate.best_params,
            final_normalized_regret=final_norm_regret,
        )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)
    ax_reward, ax_regret = axes
    oracle_mean = mu.max(axis=0)
    ax_reward.plot(t, oracle_mean, linestyle="--", color="black", linewidth=2.0, label="Oracle")

    for label, avg_reward_t, avg_norm_regret_t in curve_payload:
        ax_reward.plot(t, avg_reward_t, linewidth=1.1, label=label)
        ax_regret.plot(t, avg_norm_regret_t, linewidth=1.1, label=label)

    ax_reward.set_title(f"{env_title} - Tuned Policies (Reward)")
    ax_reward.set_ylabel("Average Reward")
    ax_reward.grid(alpha=0.25)
    ax_reward.legend(fontsize=8, ncol=2)

    ax_regret.set_title(f"{env_title} - Tuned Policies (Normalized Regret)")
    ax_regret.set_xlabel("Timesteps")
    ax_regret.set_ylabel("Normalized Regret")
    ax_regret.grid(alpha=0.25)
    ax_regret.legend(fontsize=8, ncol=2)

    for ax in (ax_reward, ax_regret):
        ax.set_ylim(-0.2, 1.2)
        ax.autoscale(enable=False, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)

    if show:
        plt.show()
    plt.close(fig)
    return output_path, tuned_results


@dataclass(frozen=True)
class PolicySpec:
    name: str
    runner: Callable[[np.ndarray, Any, np.random.Generator], np.ndarray]
    params: Any


def abrupt_means(
    horizon: int = 1000,
    n_arms: int = 4,
    cycle_len: int = 250,
    change_times: list[int] | None = None,
    levels: list[float] | None = None,
) -> np.ndarray:
    """
    Abruptly varying env used in Raj2017 Figure 1:
      - repeats every cycle_len
      - within each cycle, all arms start at 0
      - arm k jumps to levels[k] at change_times[k] and stays there until cycle ends
    Returns means with shape (T, K).
    """
    if change_times is None:
        change_times = [50, 100, 150, 200]
    if levels is None:
        levels = [0.10, 0.37, 0.63, 0.90]
    if len(change_times) != n_arms or len(levels) != n_arms:
        raise ValueError("change_times and levels must have length n_arms.")

    means = np.zeros((horizon, n_arms), dtype=float)
    for t in range(horizon):
        tau = t % cycle_len
        for k in range(n_arms):
            if tau >= change_times[k]:
                means[t, k] = levels[k]
    return means


def _compute_curves_streaming(
    mu: np.ndarray,
    policy: PolicySpec,
    n_runs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    mu: (K, T)
    Returns:
      avg_reward_t: (T,)
      avg_norm_regret_t: (T,) where norm regret is cum_regret(t)/(t+1)
    Uses streaming accumulation (doesn't store all runs).
    """
    K, T = mu.shape
    mu_star = mu.max(axis=0)              # (T,)
    oracle_cum = np.cumsum(mu_star)       # (T,)
    denom = (np.arange(T, dtype=float) + 1.0)

    sum_reward_t = np.zeros(T, dtype=float)
    sum_norm_regret_t = np.zeros(T, dtype=float)

    # independent RNG per run
    ss = np.random.SeedSequence(seed)
    run_seeds = ss.spawn(n_runs)

    for i in range(n_runs):
        rng = np.random.default_rng(run_seeds[i])
        rewards = policy.runner(mu, policy.params, rng).astype(float)  # (T,)

        sum_reward_t += rewards

        alg_cum = np.cumsum(rewards)           # (T,)
        cum_regret = oracle_cum - alg_cum      # (T,)
        norm_regret = cum_regret / denom       # (T,)

        sum_norm_regret_t += norm_regret

    avg_reward_t = sum_reward_t / float(n_runs)
    avg_norm_regret_t = sum_norm_regret_t / float(n_runs)
    return avg_reward_t, avg_norm_regret_t


def reproduce_figure1(
    n_runs: int = 200,
    seed: int = 0,
    output_path: str | Path = "report/figures/figure1_reproduction.png",
    show: bool = True,
) -> Path:
    """
    Reproduce Raj2017 Figure 1:
      Top row: Average Reward over time
      Bottom row: Normalized Regret over time (cum_regret(t)/(t+1))
    """
    plt = _import_matplotlib_pyplot()

    # (name, means(T,K), gamma_for_dTS/dOTS, (gamma_REXP3, Delta_REXP3), C_for_DTS)
    env_specs = [
        ("Slow",  slow_varying_sinusoid_means(horizon=5000, n_arms=4, period=1000), 0.80, (0.114, 250), 250),
        ("Fast",  slow_varying_sinusoid_means(horizon=1000, n_arms=4, period=100),  0.40, (0.359, 25),  25),
        ("Abrupt", abrupt_means(horizon=1000, n_arms=4, cycle_len=250),             0.60, (0.254, 50),  50),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

    for col, (env_name, means_TK, gamma_dt, (gamma_exp3, delta_exp3), C_dts) in enumerate(env_specs):
        mu = means_TK.T  # convert (T,K) -> (K,T) for policies
        K, T = mu.shape
        t = np.arange(T)

        # Oracle mean curve for the top plot
        oracle_mean = mu.max(axis=0)

        policies = [
            PolicySpec(f"dTS (γ={gamma_dt:.2f})", run_dTS, dTSParams(alpha0=1.0, beta0=1.0, gamma=gamma_dt)),
            PolicySpec(f"dOTS (γ={gamma_dt:.2f})", run_dOTS, dOTSParams(alpha0=1.0, beta0=1.0, gamma=gamma_dt)),
            PolicySpec(
                f"vgDTS+O (γ0={gamma_dt:.2f})",
                run_VG_dTS,
                VGdTSParams(
                    alpha0=1.0,
                    beta0=1.0,
                    gamma_default=gamma_dt,
                    gamma_min=0.05,
                    gamma_max=0.995,
                    lambda_vol=0.94,
                    n0=5.0,
                    gamma_mapping="inverse_linear",
                    vol_low=0.0,
                    vol_high=2.0,
                ),
            ),
            PolicySpec("Thompson Sampling", run_TS, TSParams(alpha0=1.0, beta0=1.0)),
            PolicySpec(f"REXP3 (γ={gamma_exp3:.3f}, Δ={delta_exp3})", run_REXP3, REXP3Params(gamma=gamma_exp3, Delta=delta_exp3)),
            PolicySpec(f"Dynamic TS (C={C_dts})", run_DTS, DTSParams(alpha0=1.0, beta0=1.0, C=float(C_dts))),
        ]

        # --- Top row: Average Reward ---
        ax_top = axes[0, col]
        ax_top.plot(t, oracle_mean, linestyle="--", linewidth=2.0, color="black", label="Oracle")

        # --- Bottom row: Normalized Regret ---
        ax_bot = axes[1, col]

        for pi, p in enumerate(policies):
            # separate seeds per env + policy so results are stable but independent-ish
            pol_seed = seed + 10_000 * col + 1_000 * pi
            avg_reward_t, avg_norm_regret_t = _compute_curves_streaming(mu, p, n_runs=n_runs, seed=pol_seed)

            ax_top.plot(t, avg_reward_t, linewidth=1, label=p.name)
            ax_bot.plot(t, avg_norm_regret_t, linewidth=1, label=p.name)

        ax_top.set_title(f"{env_name} Varying Environment")
        ax_top.set_xlabel("Timesteps")
        ax_top.set_ylabel("Average Reward")
        ax_top.grid(alpha=0.25)
        ax_top.legend(fontsize=8)

        ax_bot.set_title(f"{env_name} Varying Environment")
        ax_bot.set_xlabel("Timesteps")
        ax_bot.set_ylabel("Normalized Regret")
        ax_bot.grid(alpha=0.25)
        ax_bot.legend(fontsize=8)

        for ax in (ax_top, ax_bot):
            ax.set_ylim(-0.2, 1.2)
            ax.autoscale(enable=False, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def reproduce_rigorous_env_suite(
    n_runs: int = 200,
    seed: int = 0,
    horizon: int = 2000,
    n_arms: int = 4,
    output_path: str | Path = "report/figures/rigorous_env_suite_reward_regret.png",
    show: bool = True,
) -> Path:
    """
    Evaluate policies on stronger non-stationary environments:
      - mixed regime (stable/drift/burst/switch)
      - random breakpoints
      - random drift amplitude
    Top row: average reward. Bottom row: normalized regret.
    """
    plt = _import_matplotlib_pyplot()

    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_arms < 4:
        raise ValueError("n_arms must be >= 4.")

    breakpoint_min_segment = max(1, min(max(5, horizon // 100), horizon))
    breakpoint_max = min(12, max(0, horizon // breakpoint_min_segment - 1))
    breakpoint_min = min(3, breakpoint_max)

    env_specs = [
        (
            "Mixed Regime",
            mixed_regime_challenge_means(
                horizon=horizon,
                n_arms=n_arms,
                burst_prob=0.05,
                burst_duration=20,
                switch_interval=max(50, horizon // 8),
                seed=seed + 11,
            ),
        ),
        (
            "Random Breakpoints",
            random_breakpoint_means(
                horizon=horizon,
                n_arms=n_arms,
                min_breakpoints=breakpoint_min,
                max_breakpoints=breakpoint_max,
                min_segment=breakpoint_min_segment,
                min_jump=0.10,
                seed=seed + 29,
            ),
        ),
        (
            "Random Drift Amplitude",
            random_drift_amplitude_means(
                horizon=horizon,
                n_arms=n_arms,
                amplitude_min=0.05,
                amplitude_max=0.30,
                amplitude_step_std=0.015,
                period_min=80,
                period_max=max(120, horizon // 4),
                drift_step_std=0.006,
                seed=seed + 47,
            ),
        ),
    ]

    # Tuned on the rigorous env family using final normalized regret.
    dts_gamma = 0.85
    dots_gamma = 0.85
    vgdts_params = VGdTSParams(
        alpha0=1.0,
        beta0=1.0,
        gamma_default=0.45,
        gamma_min=0.12,
        gamma_max=0.995,
        lambda_vol=0.97,
        n0=0.0,
        gamma_mapping="inverse_linear",
        vol_low=0.0,
        vol_high=3.7,
        optimistic=True,
        one_sided_negative_surprise=False
    )
    rexp3_gamma = 0.32
    rexp3_delta = max(20, int(0.11 * horizon))
    dts_c = float(max(20, int(0.04 * horizon)))

    fig, axes = plt.subplots(2, len(env_specs), figsize=(16, 8), constrained_layout=True)

    for col, (env_name, means_TK) in enumerate(env_specs):
        mu = means_TK.T
        _, t_horizon = mu.shape
        t = np.arange(t_horizon)
        oracle_mean = mu.max(axis=0)

        policies = [
            PolicySpec(f"dTS (γ={dts_gamma:.2f})", run_dTS, dTSParams(alpha0=1.0, beta0=1.0, gamma=dts_gamma)),
            PolicySpec(f"dOTS (γ={dots_gamma:.2f})", run_dOTS, dOTSParams(alpha0=1.0, beta0=1.0, gamma=dots_gamma)),
            PolicySpec(
                f"vgDTS+O (γ0={vgdts_params.gamma_default:.2f}, γmin={vgdts_params.gamma_min:.2f})",
                run_VG_dTS,
                vgdts_params,
            ),
            PolicySpec("Thompson Sampling", run_TS, TSParams(alpha0=1.0, beta0=1.0)),
            PolicySpec(
                f"REXP3 (γ={rexp3_gamma:.2f}, Δ={rexp3_delta})",
                run_REXP3,
                REXP3Params(gamma=rexp3_gamma, Delta=rexp3_delta),
            ),
            PolicySpec(f"Dynamic TS (C={int(dts_c)})", run_DTS, DTSParams(alpha0=1.0, beta0=1.0, C=dts_c)),
        ]

        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        ax_top.plot(t, oracle_mean, linestyle="--", linewidth=2.0, color="black", label="Oracle")

        for pi, p in enumerate(policies):
            pol_seed = seed + 10_000 * col + 1_000 * pi
            avg_reward_t, avg_norm_regret_t = _compute_curves_streaming(mu, p, n_runs=n_runs, seed=pol_seed)
            ax_top.plot(t, avg_reward_t, linewidth=1, label=p.name)
            ax_bot.plot(t, avg_norm_regret_t, linewidth=1, label=p.name)

        ax_top.set_title(env_name)
        ax_top.set_xlabel("Timesteps")
        ax_top.set_ylabel("Average Reward")
        ax_top.grid(alpha=0.25)
        ax_top.legend(fontsize=8)

        ax_bot.set_title(env_name)
        ax_bot.set_xlabel("Timesteps")
        ax_bot.set_ylabel("Normalized Regret")
        ax_bot.grid(alpha=0.25)
        ax_bot.legend(fontsize=8)

        for ax in (ax_top, ax_bot):
            ax.set_ylim(-0.2, 1.2)
            ax.autoscale(enable=False, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)
    return output_path


@dataclass(frozen=True)
class BenchmarkPolicySpec:
    name: str
    runner: Callable[[np.ndarray, Any, np.random.Generator], np.ndarray]
    params: Any


def _compute_avg_reward_and_final_norm_regret(
    mu: np.ndarray,
    runner: Callable[[np.ndarray, Any, np.random.Generator], np.ndarray],
    params: Any,
    n_runs: int,
    seed: int,
) -> tuple[float, float]:
    k, horizon = mu.shape
    if k <= 0 or horizon <= 0:
        raise ValueError("mu must have shape (K, T) with K,T > 0.")
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")

    oracle_total = float(np.sum(mu.max(axis=0)))
    sum_reward_total = 0.0
    sum_norm_regret = 0.0

    run_seeds = np.random.SeedSequence(seed).spawn(n_runs)
    for run_seed in run_seeds:
        rng = np.random.default_rng(run_seed)
        rewards = runner(mu, params, rng).astype(float)
        if rewards.shape != (horizon,):
            raise ValueError("Policy runner must return rewards with shape (T,).")
        reward_total = float(np.sum(rewards))
        sum_reward_total += reward_total
        sum_norm_regret += (oracle_total - reward_total) / float(horizon)

    avg_reward = sum_reward_total / float(n_runs * horizon)
    avg_final_norm_regret = sum_norm_regret / float(n_runs)
    return avg_reward, avg_final_norm_regret


def _greedy_recovering_oracle_expected_total(reward_table: np.ndarray, horizon: int) -> float:
    n_arms, z_size = reward_table.shape
    z_max = z_size - 1
    delays = np.zeros(n_arms, dtype=int)
    total = 0.0
    for _ in range(horizon):
        current = reward_table[np.arange(n_arms), delays]
        arm = int(np.argmax(current))
        total += float(current[arm])
        delays = np.minimum(delays + 1, z_max)
        delays[arm] = 0
    return total


def _run_recovering_benchmark(
    reward_table: np.ndarray,
    horizon: int,
    n_runs: int,
    seed: int,
    vgdts_params: VGdTSParams,
) -> dict[str, dict[str, float]]:
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")

    greedy_oracle_total = _greedy_recovering_oracle_expected_total(reward_table, horizon)

    policies = [
        (
            "VG-dTS",
            lambda table, h, rng, env_seed: run_vgdts_recovering(
                table,
                horizon=h,
                params=vgdts_params,
                rng=rng,
                env_seed=env_seed,
            ),
        ),
        (
            "1RGP-UCB (1910.14354)",
            lambda table, h, rng, env_seed: run_drpg_ucb(
                table,
                horizon=h,
                params=RGPParams(
                    lookahead=1,
                    lengthscale=5.0,
                    kernel_amplitude=1.0,
                    noise_std=0.10,
                    max_sequences=2000,
                ),
                rng=rng,
                env_seed=env_seed,
            ),
        ),
        (
            "1RGP-TS (1910.14354)",
            lambda table, h, rng, env_seed: run_drpg_ts(
                table,
                horizon=h,
                params=RGPParams(
                    lookahead=1,
                    lengthscale=5.0,
                    kernel_amplitude=1.0,
                    noise_std=0.10,
                    max_sequences=2000,
                ),
                rng=rng,
                env_seed=env_seed,
            ),
        ),
    ]

    output: dict[str, dict[str, float]] = {}
    run_seeds = np.random.SeedSequence(seed).spawn(n_runs)
    for policy_name, runner in policies:
        reward_totals = []
        norm_regrets = []
        for run_idx, run_seed in enumerate(run_seeds):
            rng = np.random.default_rng(run_seed)
            env_seed = seed + 10_000 * (run_idx + 1)
            rewards = runner(reward_table, horizon, rng, env_seed).astype(float)
            reward_total = float(np.sum(rewards))
            reward_totals.append(reward_total)
            norm_regrets.append((greedy_oracle_total - reward_total) / float(horizon))

        output[policy_name] = {
            "avg_reward": float(np.mean(reward_totals) / float(horizon)),
            "avg_final_normalized_regret_vs_greedy_oracle": float(np.mean(norm_regrets)),
        }
    return output


def _plot_nonstationary_heatmap(
    matrix: np.ndarray,
    env_names: list[str],
    policy_names: list[str],
    output_path: Path,
) -> Path:
    plt = _import_matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(policy_names)), max(5, 0.75 * len(env_names))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(len(policy_names)))
    ax.set_xticklabels(policy_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(env_names)))
    ax.set_yticklabels(env_names)
    ax.set_title("Final Normalized Regret (Lower is Better)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("R_T / T")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_recovering_bars(
    recovering_summary: dict[str, dict[str, float]],
    output_path: Path,
) -> Path:
    plt = _import_matplotlib_pyplot()

    names = list(recovering_summary.keys())
    rewards = np.array([recovering_summary[name]["avg_reward"] for name in names], dtype=float)
    regrets = np.array(
        [
            recovering_summary[name]["avg_final_normalized_regret_vs_greedy_oracle"]
            for name in names
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax_reward, ax_regret = axes

    ax_reward.bar(names, rewards)
    ax_reward.set_title("Recovering Bandit: Avg Reward")
    ax_reward.set_ylabel("Average reward per step")
    ax_reward.tick_params(axis="x", rotation=20)

    ax_regret.bar(names, regrets)
    ax_regret.set_title("Recovering Bandit: Regret vs Greedy Oracle")
    ax_regret.set_ylabel("Final normalized regret")
    ax_regret.tick_params(axis="x", rotation=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def run_paper_algorithm_benchmarks(
    n_runs_nonstationary: int = 20,
    n_runs_recovering: int = 20,
    seed: int = 0,
    output_dir: str | Path = "report",
    nonstationary_env_suite: list[tuple[str, np.ndarray]] | None = None,
    recovering_table: np.ndarray | None = None,
    recovering_horizon: int = 1200,
    vgdts_params: VGdTSParams | None = None,
    make_plots: bool = True,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """
    Benchmark paper algorithms vs VG-dTS on:
      - Existing non-stationary environments
      - Paper-specific switching and recovering environments
    """
    if n_runs_nonstationary <= 0 or n_runs_recovering <= 0:
        raise ValueError("n_runs_nonstationary and n_runs_recovering must be > 0.")

    output_root = Path(output_dir)
    fig_dir = output_root / "figures"
    results_dir = output_root / "results"

    if nonstationary_env_suite is None:
        env_suite: list[tuple[str, np.ndarray]] = [
            (
                "Slow Sine (existing)",
                slow_varying_sinusoid_means(horizon=1200, n_arms=4, period=300).T,
            ),
            (
                "Fast Sine (existing)",
                slow_varying_sinusoid_means(horizon=800, n_arms=4, period=80).T,
            ),
            (
                "Abrupt Cycle (existing)",
                _abrupt_means(horizon=800, n_arms=4, cycle_len=200).T,
            ),
            (
                "Mixed Regime (existing)",
                mixed_regime_challenge_means(
                    horizon=1200,
                    n_arms=4,
                    burst_prob=0.05,
                    burst_duration=20,
                    switch_interval=150,
                    seed=seed + 11,
                ).T,
            ),
            (
                "Random Breakpoints (existing)",
                random_breakpoint_means(
                    horizon=1200,
                    n_arms=4,
                    min_breakpoints=3,
                    max_breakpoints=8,
                    min_segment=20,
                    min_jump=0.10,
                    seed=seed + 19,
                ).T,
            ),
            (
                "Random Drift (existing)",
                random_drift_amplitude_means(
                    horizon=1200,
                    n_arms=4,
                    amplitude_min=0.05,
                    amplitude_max=0.30,
                    amplitude_step_std=0.015,
                    period_min=80,
                    period_max=250,
                    drift_step_std=0.006,
                    seed=seed + 23,
                ).T,
            ),
            (
                "Global Switching (1302.3721)",
                global_switching_means(
                    horizon=1200,
                    n_arms=4,
                    switch_prob=0.02,
                    seed=seed + 31,
                ).T,
            ),
            (
                "Per-Arm Switching (1302.3721)",
                per_arm_switching_means(
                    horizon=1200,
                    n_arms=4,
                    switch_prob=0.02,
                    seed=seed + 37,
                ).T,
            ),
        ]
    else:
        env_suite = [(name, np.asarray(mu, dtype=float)) for name, mu in nonstationary_env_suite]

    vgdts_params_eff = make_benchmark_vgdts_params() if vgdts_params is None else vgdts_params

    policies = [
        BenchmarkPolicySpec(
            name="VG-dTS",
            runner=run_VG_dTS,
            params=vgdts_params_eff,
        ),
        BenchmarkPolicySpec(
            name="dTS",
            runner=run_dTS,
            params=dTSParams(alpha0=1.0, beta0=1.0, gamma=0.75),
        ),
        BenchmarkPolicySpec(
            name="dOTS",
            runner=run_dOTS,
            params=dOTSParams(alpha0=1.0, beta0=1.0, gamma=0.75),
        ),
        BenchmarkPolicySpec(
            name="TS",
            runner=run_TS,
            params=TSParams(alpha0=1.0, beta0=1.0),
        ),
        BenchmarkPolicySpec(
            name="REXP3",
            runner=run_REXP3,
            params=REXP3Params(gamma=0.1136, Delta=250),
        ),
        BenchmarkPolicySpec(
            name="Dynamic TS",
            runner=run_DTS,
            params=DTSParams(alpha0=1.0, beta0=1.0, C=250.0),
        ),
        BenchmarkPolicySpec(
            name="SW-UCB (0805.3415)",
            runner=run_sw_ucb,
            params=SWUCBParams(tau=200, xi=0.5),
        ),
        BenchmarkPolicySpec(
            name="D-UCB (0805.3415)",
            runner=run_d_ucb,
            params=DUCBParams(gamma=0.98, xi=0.5),
        ),
        BenchmarkPolicySpec(
            name="CUSUM-UCB (1711.03539)",
            runner=run_cusum_ucb,
            params=CUSUMUCBParams(
                xi=0.5,
                epsilon=0.05,
                threshold=8.0,
                warmup=40,
                random_explore=0.05,
            ),
        ),
        BenchmarkPolicySpec(
            name="GLR-klUCB (1902.01575)",
            runner=run_glr_klucb,
            params=GLRklUCBParams(
                alpha=1.0,
                threshold_scale=1.5,
                min_segment_len=20,
                max_history=400,
            ),
        ),
        BenchmarkPolicySpec(
            name="AdaSwitch (1902.07010)",
            runner=run_adaswitch,
            params=AdaSwitchParams(
                xi=0.5,
                min_window=16,
                max_windows=8,
                reset_threshold=0.18,
                min_pulls_for_reset=40,
            ),
        ),
        BenchmarkPolicySpec(
            name="SW-TS (Trovo 2020)",
            runner=run_sw_ts,
            params=SWTSParams(alpha0=1.0, beta0=1.0, tau=200),
        ),
        BenchmarkPolicySpec(
            name="gamma-SWGTS (2409.05181)",
            runner=run_gamma_swgts,
            params=GammaSWGTSParams(alpha0=1.0, beta0=1.0, tau=200, gamma=0.7),
        ),
        BenchmarkPolicySpec(
            name="Global-CTS (1302.3721)",
            runner=run_global_cts,
            params=GlobalCTSParams(alpha0=1.0, beta0=1.0, hazard=0.02, max_runlengths=160),
        ),
        BenchmarkPolicySpec(
            name="AFF-TS (1712.03134)",
            runner=run_aff_ts,
            params=AFFTSParams(alpha0=2.0, beta0=2.0, eta=0.001, lambda_init=0.95),
        ),
        BenchmarkPolicySpec(
            name="D-LinUCB one-hot (1909.09146)",
            runner=run_dlinucb_onehot,
            params=DLinUCBParams(
                gamma=0.98,
                lambda_reg=1.0,
                delta=0.05,
                sigma=0.5,
                action_norm_bound=1.0,
                theta_norm_bound=1.0,
            ),
        ),
        BenchmarkPolicySpec(
            name="Beta-SWTS (2409.05181)",
            runner=run_beta_swts,
            params=BetaSWTSParams(alpha0=1.0, beta0=1.0, tau=200),
        ),
    ]

    nonstationary_summary: dict[str, dict[str, dict[str, float]]] = {}
    env_names: list[str] = []
    policy_names = [spec.name for spec in policies]
    heatmap_values = np.zeros((len(env_suite), len(policies)), dtype=float)

    for env_idx, (env_name, mu) in enumerate(env_suite):
        env_names.append(env_name)
        nonstationary_summary[env_name] = {}
        for policy_idx, policy in enumerate(policies):
            avg_reward, final_norm_regret = _compute_avg_reward_and_final_norm_regret(
                mu=mu,
                runner=policy.runner,
                params=policy.params,
                n_runs=n_runs_nonstationary,
                seed=seed + 100_000 * (env_idx + 1) + 10_000 * (policy_idx + 1),
            )
            nonstationary_summary[env_name][policy.name] = {
                "avg_reward": avg_reward,
                "final_normalized_regret": final_norm_regret,
            }
            heatmap_values[env_idx, policy_idx] = final_norm_regret

    if recovering_table is None:
        recovering_table_eff = make_recovering_reward_table(n_arms=8, z_max=30, seed=seed + 211)
    else:
        recovering_table_eff = np.asarray(recovering_table, dtype=float)
    recovering_summary = _run_recovering_benchmark(
        reward_table=recovering_table_eff,
        horizon=recovering_horizon,
        n_runs=n_runs_recovering,
        seed=seed + 307,
        vgdts_params=vgdts_params_eff,
    )

    nonstationary_plot_path = fig_dir / "paper_algorithms_nonstationary_heatmap.png"
    recovering_plot_path = fig_dir / "paper_algorithms_recovering_benchmark.png"
    if make_plots:
        try:
            nonstationary_plot_path = _plot_nonstationary_heatmap(
                matrix=heatmap_values,
                env_names=env_names,
                policy_names=policy_names,
                output_path=nonstationary_plot_path,
            )
            recovering_plot_path = _plot_recovering_bars(
                recovering_summary=recovering_summary,
                output_path=recovering_plot_path,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "matplotlib":
                raise ModuleNotFoundError(
                    "matplotlib is required for plotting. Install it or call with make_plots=False."
                ) from exc
            raise

    summary: dict[str, Any] = {
        "config": {
            "n_runs_nonstationary": n_runs_nonstationary,
            "n_runs_recovering": n_runs_recovering,
            "seed": seed,
            "vgdts_params": asdict(vgdts_params_eff),
        },
        "nonstationary": nonstationary_summary,
        "recovering": recovering_summary,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "paper_algorithms_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return nonstationary_plot_path, recovering_plot_path, summary_path, summary
