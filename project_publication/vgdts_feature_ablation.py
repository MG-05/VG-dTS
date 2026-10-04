from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np

from project_publication.pipeline import (
    ALGORITHM_ORDER,
    build_publication_environment_suite,
    evaluate_environment_suite,
    evaluate_environment_suite_fixed,
)
from src.adts.policies import VGdTSParams, run_VG_dTS
from src.adts.vgdts_config import make_benchmark_vgdts_params


FEATURE_ORDER: tuple[str, ...] = (
    "shock_stale",
    "dual_memory",
    "likelihood_surprise",
    "two_timescale",
    "adaptive_optimism",
    "online_calibration",
)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "shock_stale": "Global shock gate + stale-arm revisit pressure",
    "dual_memory": "Dual-memory posterior (short+long) with volatility gating",
    "likelihood_surprise": "Likelihood-based surprise (-log predictive probability)",
    "two_timescale": "Two-timescale volatility (fast-slow divergence)",
    "adaptive_optimism": "Adaptive optimism temperature",
    "online_calibration": "Online quantile calibration of volatility map",
}

FEATURE_OVERRIDES: dict[str, dict[str, Any]] = {
    "shock_stale": {
        "global_shock_enabled": True,
        "global_shock_lambda": 0.90,
        "global_shock_threshold": 2.0,
        "global_shock_gamma_floor": 0.20,
        "stale_arm_revisit_enabled": True,
        "stale_arm_revisit_scale": 0.30,
        "stale_arm_revisit_half_life": 50.0,
        "stale_arm_revisit_clip": 0.50,
    },
    "dual_memory": {
        "dual_memory_enabled": True,
        "gamma_long": 0.995,
        "dual_memory_mix_power": 1.0,
    },
    "likelihood_surprise": {
        "surprise_mode": "neg_log_likelihood",
        "surprise_clip": 8.0,
    },
    "two_timescale": {
        "two_timescale_volatility_enabled": True,
        "lambda_vol_fast": 0.80,
        "lambda_vol_slow": 0.97,
    },
    "adaptive_optimism": {
        "adaptive_optimism_enabled": True,
        "optimism_base": 0.50,
        "optimism_var_weight": 0.60,
        "optimism_vol_weight": 0.80,
    },
    "online_calibration": {
        "online_vol_calibration_enabled": True,
        "calibration_window": 200,
        "calibration_quantile_low": 0.20,
        "calibration_quantile_high": 0.80,
        "calibration_min_count": 20,
        "calibration_min_span": 1e-3,
    },
}

FEATURE_TUNING_KNOBS: dict[str, tuple[str, list[Any]]] = {
    "shock_stale": ("stale_arm_revisit_scale", [0.20, 0.35]),
    "dual_memory": ("dual_memory_mix_power", [0.80, 1.40]),
    "likelihood_surprise": ("vol_high", [1.8, 2.8]),
    "two_timescale": ("lambda_vol_fast", [0.75, 0.85]),
    "adaptive_optimism": ("optimism_vol_weight", [0.60, 1.00]),
    "online_calibration": ("calibration_quantile_high", [0.75, 0.90]),
}


def _features_to_id(features: tuple[str, ...]) -> str:
    return "none" if not features else "+".join(features)


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


def _make_variant_params(base: VGdTSParams, features: tuple[str, ...]) -> VGdTSParams:
    overrides: dict[str, Any] = {}
    for feature in features:
        overrides.update(FEATURE_OVERRIDES[feature])
    return replace(base, **overrides)


def _make_variant_tuning_grid(features: tuple[str, ...]) -> dict[str, list[Any]]:
    # Keep per-variant sweeps bounded so all variants can be evaluated under one budget.
    grid: dict[str, list[Any]] = {
        "gamma_default": [0.35, 0.50],
    }

    feature_knobs = [FEATURE_TUNING_KNOBS[name] for name in FEATURE_ORDER if name in features]
    for key, values in feature_knobs[:2]:
        grid[key] = values
    return grid


def _build_vgdts_candidates(base: VGdTSParams, grid: dict[str, list[Any]]) -> list[VGdTSParams]:
    keys = tuple(grid.keys())
    values = [list(grid[key]) for key in keys]
    if not values:
        return [base]

    candidates: list[VGdTSParams] = []
    for combo in product(*values):
        overrides = dict(zip(keys, combo))
        candidates.append(replace(base, **overrides))
    return candidates


def _estimate_final_normalized_regret(
    mu: np.ndarray,
    params: VGdTSParams,
    n_runs: int,
    seed: int,
) -> float:
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")

    horizon = int(mu.shape[1])
    oracle_total = float(np.max(mu, axis=0).sum())
    total = 0.0

    run_seeds = np.random.SeedSequence(seed).spawn(n_runs)
    for run_seed in run_seeds:
        rng = np.random.default_rng(run_seed)
        rewards = run_VG_dTS(mu=mu, params=params, rng=rng).astype(float)
        alg_total = float(rewards.sum())
        total += (oracle_total - alg_total) / float(horizon)

    return total / float(n_runs)


def _evaluate_variant_fixed(
    env_suite,
    params: VGdTSParams,
    eval_runs: int,
    seed: int,
) -> dict[str, Any]:
    per_env: dict[str, dict[str, Any]] = {}
    regrets: list[float] = []

    for env_idx, env in enumerate(env_suite):
        mu = env.means_tk.T
        regret = _estimate_final_normalized_regret(
            mu=mu,
            params=params,
            n_runs=eval_runs,
            seed=seed + 1_000_000 * (env_idx + 1),
        )
        regrets.append(regret)
        per_env[env.key] = {
            "environment_name": env.name,
            "final_normalized_regret": float(regret),
        }

    return {
        "avg_final_normalized_regret": float(np.mean(regrets)),
        "per_environment": per_env,
    }


def _evaluate_variant_tuned(
    env_suite,
    base_params: VGdTSParams,
    tuning_grid: dict[str, list[Any]],
    tuning_runs: int,
    eval_runs: int,
    seed: int,
) -> dict[str, Any]:
    per_env: dict[str, dict[str, Any]] = {}
    regrets: list[float] = []

    for env_idx, env in enumerate(env_suite):
        mu = env.means_tk.T
        candidates = _build_vgdts_candidates(base=base_params, grid=tuning_grid)

        best_params = candidates[0]
        best_tuning_regret = float("inf")
        for cand_idx, candidate in enumerate(candidates):
            regret = _estimate_final_normalized_regret(
                mu=mu,
                params=candidate,
                n_runs=tuning_runs,
                seed=seed + 100_000 * (env_idx + 1) + 1_000 * (cand_idx + 1),
            )
            if regret < best_tuning_regret:
                best_tuning_regret = regret
                best_params = candidate

        eval_regret = _estimate_final_normalized_regret(
            mu=mu,
            params=best_params,
            n_runs=eval_runs,
            seed=seed + 2_000_000 * (env_idx + 1),
        )
        regrets.append(eval_regret)
        per_env[env.key] = {
            "environment_name": env.name,
            "tuning_final_normalized_regret": float(best_tuning_regret),
            "final_normalized_regret": float(eval_regret),
            "best_params": asdict(best_params),
        }

    return {
        "avg_final_normalized_regret": float(np.mean(regrets)),
        "per_environment": per_env,
    }


def _evaluate_all_fixed_combos(
    env_suite,
    base_params: VGdTSParams,
    eval_runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    combos: list[tuple[str, ...]] = []
    for r in range(len(FEATURE_ORDER) + 1):
        combos.extend(combinations(FEATURE_ORDER, r))

    for idx, features in enumerate(combos):
        variant_id = _features_to_id(features)
        params = _make_variant_params(base=base_params, features=features)
        eval_result = _evaluate_variant_fixed(
            env_suite=env_suite,
            params=params,
            eval_runs=eval_runs,
            seed=seed + 10_000_000 * (idx + 1),
        )
        results.append(
            {
                "variant_id": variant_id,
                "features": list(features),
                "n_features": len(features),
                "fixed": eval_result,
            }
        )
        print(
            f"[fixed-screen] {idx + 1:02d}/{len(combos)} {variant_id} "
            f"avg R_T/T={eval_result['avg_final_normalized_regret']:.4f}",
            flush=True,
        )

    results.sort(key=lambda item: item["fixed"]["avg_final_normalized_regret"])
    return results


def _select_top_multifeature_combos(
    fixed_screen_results: list[dict[str, Any]],
    top_k: int,
) -> list[tuple[str, ...]]:
    selected: list[tuple[str, ...]] = []
    for item in fixed_screen_results:
        features = tuple(item["features"])
        if len(features) < 2:
            continue
        selected.append(features)
        if len(selected) >= top_k:
            break
    return selected


def _summarize_against_baselines(
    variant_eval: dict[str, Any],
    baseline_evals,
) -> dict[str, Any]:
    env_keys = list(variant_eval["per_environment"].keys())

    wins_vs_best_non_vg = 0
    first_place_envs = 0
    deltas_vs_best_non_vg: list[float] = []
    deltas_vs_baseline_vgdts: list[float] = []
    ranks: list[int] = []
    per_env: dict[str, dict[str, Any]] = {}

    for env_key in env_keys:
        variant_regret = float(variant_eval["per_environment"][env_key]["final_normalized_regret"])
        baseline_eval = baseline_evals[env_key]

        all_regrets = {
            algo: float(baseline_eval.policies[algo].eval_final_normalized_regret)
            for algo in ALGORITHM_ORDER
        }
        baseline_vgdts = all_regrets["VG-dTS"]
        all_regrets["VG-dTS"] = variant_regret

        non_vg = {algo: regret for algo, regret in all_regrets.items() if algo != "VG-dTS"}
        best_non_vg_algo = min(non_vg, key=non_vg.get)
        best_non_vg = non_vg[best_non_vg_algo]

        delta_best_non_vg = variant_regret - best_non_vg
        delta_baseline_vgdts = variant_regret - baseline_vgdts
        rank = 1 + sum(
            1 for _, regret in all_regrets.items() if regret < variant_regret - 1e-12
        )

        if delta_best_non_vg < 0.0:
            wins_vs_best_non_vg += 1
        if rank == 1:
            first_place_envs += 1

        deltas_vs_best_non_vg.append(delta_best_non_vg)
        deltas_vs_baseline_vgdts.append(delta_baseline_vgdts)
        ranks.append(rank)

        per_env[env_key] = {
            "variant_final_normalized_regret": variant_regret,
            "best_non_vg_algorithm": best_non_vg_algo,
            "best_non_vg_final_normalized_regret": best_non_vg,
            "baseline_vgdts_final_normalized_regret": baseline_vgdts,
            "delta_vs_best_non_vg": delta_best_non_vg,
            "delta_vs_baseline_vgdts": delta_baseline_vgdts,
            "rank_among_all_policies": rank,
        }

    return {
        "avg_delta_vs_best_non_vg": float(np.mean(deltas_vs_best_non_vg)),
        "avg_delta_vs_baseline_vgdts": float(np.mean(deltas_vs_baseline_vgdts)),
        "avg_rank_among_all_policies": float(np.mean(ranks)),
        "wins_vs_best_non_vg_env_count": int(wins_vs_best_non_vg),
        "first_place_env_count": int(first_place_envs),
        "per_environment": per_env,
    }


def run_vgdts_feature_ablation(
    output_dir: str | Path = "report/project_publication/vgdts_feature_ablation",
    horizon: int = 1200,
    n_arms: int = 4,
    seed: int = 7,
    fixed_screen_eval_runs: int = 8,
    tuning_runs: int = 4,
    eval_runs: int = 8,
    top_combo_count: int = 3,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[setup] Building environment suite...", flush=True)
    env_suite = build_publication_environment_suite(horizon=horizon, n_arms=n_arms, seed=seed)
    base_params = make_benchmark_vgdts_params()

    print("[stage 1/4] Fixed-parameter exhaustive screening over 64 feature combinations...", flush=True)
    fixed_screen = _evaluate_all_fixed_combos(
        env_suite=env_suite,
        base_params=base_params,
        eval_runs=fixed_screen_eval_runs,
        seed=seed + 101,
    )

    selected_combo_features = _select_top_multifeature_combos(
        fixed_screen_results=fixed_screen,
        top_k=top_combo_count,
    )
    selected_combo_ids = [_features_to_id(features) for features in selected_combo_features]
    print(f"[stage 1/4] Selected top multifeature combos: {', '.join(selected_combo_ids)}", flush=True)

    tuned_variant_features: list[tuple[str, ...]] = [tuple()]
    tuned_variant_features.extend((feature,) for feature in FEATURE_ORDER)
    tuned_variant_features.append(tuple(FEATURE_ORDER))
    tuned_variant_features.extend(selected_combo_features)

    deduped: list[tuple[str, ...]] = []
    seen_ids: set[str] = set()
    for features in tuned_variant_features:
        variant_id = _features_to_id(features)
        if variant_id in seen_ids:
            continue
        seen_ids.add(variant_id)
        deduped.append(features)
    tuned_variant_features = deduped

    print("[stage 2/4] Evaluating tuned VG-dTS variants...", flush=True)
    tuned_variants: list[dict[str, Any]] = []
    for idx, features in enumerate(tuned_variant_features):
        variant_id = _features_to_id(features)
        params = _make_variant_params(base=base_params, features=features)
        tuning_grid = _make_variant_tuning_grid(features=features)
        tuned_eval = _evaluate_variant_tuned(
            env_suite=env_suite,
            base_params=params,
            tuning_grid=tuning_grid,
            tuning_runs=tuning_runs,
            eval_runs=eval_runs,
            seed=seed + 501 + 10_000 * (idx + 1),
        )
        fixed_eval = _evaluate_variant_fixed(
            env_suite=env_suite,
            params=params,
            eval_runs=eval_runs,
            seed=seed + 1501 + 10_000 * (idx + 1),
        )
        tuned_variants.append(
            {
                "variant_id": variant_id,
                "features": list(features),
                "description": [FEATURE_DESCRIPTIONS[name] for name in features],
                "tuning_grid": tuning_grid,
                "tuned": tuned_eval,
                "fixed": fixed_eval,
            }
        )
        print(
            f"[tuned] {idx + 1:02d}/{len(tuned_variant_features)} {variant_id} "
            f"avg tuned R_T/T={tuned_eval['avg_final_normalized_regret']:.4f} "
            f"avg fixed R_T/T={fixed_eval['avg_final_normalized_regret']:.4f}",
            flush=True,
        )

    print("[stage 3/4] Running full baseline suite (tuned + fixed) for comparisons...", flush=True)
    baseline_vgdts_grid = {
        "gamma_default": [0.35, 0.50],
        "gamma_min": [0.10, 0.20],
        "vol_high": [2.2, 3.0],
    }
    baseline_tuned = evaluate_environment_suite(
        env_suite=env_suite,
        tuning_runs=tuning_runs,
        eval_runs=eval_runs,
        seed=seed + 2_501,
        lambda_grid=[0.45, 0.75],
        rexp3_gamma_grid=[0.10, 0.30],
        rexp3_delta_grid=[max(50, horizon // 20), max(120, horizon // 8)],
        beta_swts_tau_grid=[max(60, horizon // 20), max(160, horizon // 8)],
        sw_ucb_tau_grid=[max(60, horizon // 20), max(160, horizon // 8)],
        d_ucb_gamma_grid=[0.95, 0.99],
        cusum_epsilon_grid=[0.025, 0.05],
        cusum_threshold_grid=[6.0, 10.0],
        glr_alpha_grid=[0.5, 1.0],
        glr_threshold_scale_grid=[1.0, 1.5],
        adaswitch_reset_threshold_grid=[0.12, 0.20],
        sw_ts_tau_grid=[max(60, horizon // 20), max(160, horizon // 8)],
        gamma_swgts_tau_grid=[max(60, horizon // 20), max(160, horizon // 8)],
        gamma_swgts_gamma_grid=[0.5, 0.7],
        vgdts_params=base_params,
        vgdts_grid=baseline_vgdts_grid,
    )
    baseline_fixed = evaluate_environment_suite_fixed(
        env_suite=env_suite,
        eval_runs=eval_runs,
        seed=seed + 3_501,
        vgdts_params=base_params,
        dts_gamma=0.75,
        dots_gamma=0.75,
        rexp3_gamma=0.1136,
        rexp3_delta=max(120, horizon // 8),
        dynamic_c=250.0,
        sw_ucb_tau=max(120, horizon // 8),
        d_ucb_gamma=0.98,
        cusum_epsilon=0.05,
        cusum_threshold=8.0,
        glr_alpha=1.0,
        glr_threshold_scale=1.5,
        adaswitch_reset_threshold=0.18,
        sw_ts_tau=max(120, horizon // 8),
        gamma_swgts_tau=max(120, horizon // 8),
        gamma_swgts_gamma=0.7,
        beta_swts_tau=max(120, horizon // 8),
    )

    print("[stage 4/4] Summarizing and ranking variants...", flush=True)
    for variant in tuned_variants:
        variant["tuned_vs_baselines"] = _summarize_against_baselines(
            variant_eval=variant["tuned"],
            baseline_evals=baseline_tuned,
        )
        variant["fixed_vs_baselines"] = _summarize_against_baselines(
            variant_eval=variant["fixed"],
            baseline_evals=baseline_fixed,
        )

    tuned_ranked = sorted(
        tuned_variants,
        key=lambda item: item["tuned"]["avg_final_normalized_regret"],
    )
    fixed_ranked = sorted(
        tuned_variants,
        key=lambda item: item["fixed"]["avg_final_normalized_regret"],
    )

    summary: dict[str, Any] = {
        "config": {
            "horizon": horizon,
            "n_arms": n_arms,
            "seed": seed,
            "fixed_screen_eval_runs": fixed_screen_eval_runs,
            "tuning_runs": tuning_runs,
            "eval_runs": eval_runs,
            "top_combo_count": top_combo_count,
        },
        "feature_order": list(FEATURE_ORDER),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "fixed_screen_ranked": fixed_screen,
        "selected_multifeature_combos": selected_combo_ids,
        "tuned_variants_ranked": tuned_ranked,
        "fixed_variants_ranked": fixed_ranked,
        "best_tuned_variant_id": tuned_ranked[0]["variant_id"],
        "best_fixed_variant_id": fixed_ranked[0]["variant_id"],
        "baseline_tuned": {
            env_key: {
                algo: float(env_eval.policies[algo].eval_final_normalized_regret)
                for algo in ALGORITHM_ORDER
            }
            for env_key, env_eval in baseline_tuned.items()
        },
        "baseline_fixed": {
            env_key: {
                algo: float(env_eval.policies[algo].eval_final_normalized_regret)
                for algo in ALGORITHM_ORDER
            }
            for env_key, env_eval in baseline_fixed.items()
        },
    }

    json_path = _save_json(results_dir / "vgdts_feature_ablation_summary.json", summary)
    print(f"[done] Wrote summary to {json_path}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VG-dTS feature ablation against non-stationary baselines.",
    )
    parser.add_argument("--output-dir", default="report/project_publication/vgdts_feature_ablation")
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--n-arms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fixed-screen-eval-runs", type=int, default=8)
    parser.add_argument("--tuning-runs", type=int, default=4)
    parser.add_argument("--eval-runs", type=int, default=8)
    parser.add_argument("--top-combo-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_vgdts_feature_ablation(
        output_dir=args.output_dir,
        horizon=args.horizon,
        n_arms=args.n_arms,
        seed=args.seed,
        fixed_screen_eval_runs=args.fixed_screen_eval_runs,
        tuning_runs=args.tuning_runs,
        eval_runs=args.eval_runs,
        top_combo_count=args.top_combo_count,
    )


if __name__ == "__main__":
    main()
