from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from project_publication.pipeline import (
    ALGORITHM_ORDER,
    build_publication_environment_suite,
    evaluate_environment_suite,
    evaluate_environment_suite_fixed,
)
from project_publication.vgdts_feature_ablation import (
    _evaluate_variant_fixed,
    _evaluate_variant_tuned,
    _summarize_against_baselines,
)
from src.adts.policies import VGdTSParams
from src.adts.vgdts_config import make_benchmark_vgdts_params


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


def _variant_specs(base: VGdTSParams) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "baseline_vgdts",
            "label": "Benchmark VG-dTS",
            "params": base,
            "tuning_grid": {
                "gamma_default": [0.35, 0.50],
                "gamma_min": [0.10, 0.20],
                "vol_high": [2.2, 3.0],
            },
        },
        {
            "variant_id": "dual_memory+likelihood_surprise",
            "label": "Dual-memory + likelihood surprise",
            "params": replace(
                base,
                dual_memory_enabled=True,
                gamma_long=0.995,
                dual_memory_mix_power=1.0,
                surprise_mode="neg_log_likelihood",
            ),
            "tuning_grid": {
                "gamma_default": [0.35, 0.50],
                "dual_memory_mix_power": [0.80, 1.40],
                "vol_high": [1.8, 2.8],
            },
        },
        {
            "variant_id": "dual_memory+likelihood_surprise+confidence_bonus",
            "label": "Dual-memory + likelihood surprise + confidence bonus",
            "params": replace(
                base,
                dual_memory_enabled=True,
                gamma_long=0.995,
                dual_memory_mix_power=1.0,
                surprise_mode="neg_log_likelihood",
                confidence_bonus_enabled=True,
                confidence_bonus_scale=0.60,
                confidence_bonus_vol_weight=0.80,
                confidence_bonus_coldstart_weight=0.40,
                confidence_bonus_clip=0.75,
            ),
            "tuning_grid": {
                "gamma_default": [0.35, 0.50],
                "dual_memory_mix_power": [0.80, 1.40],
                "vol_high": [1.8, 2.8],
                "confidence_bonus_scale": [0.40, 0.70],
                "confidence_bonus_vol_weight": [0.50, 1.00],
            },
        },
    ]


def run_vgdts_confidence_bonus_study(
    output_dir: str | Path = "report/project_publication/vgdts_confidence_bonus_study",
    horizon: int = 800,
    n_arms: int = 4,
    seed: int = 11,
    tuning_runs: int = 4,
    eval_runs: int = 10,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[setup] Building environment suite...", flush=True)
    env_suite = build_publication_environment_suite(horizon=horizon, n_arms=n_arms, seed=seed)
    base_params = make_benchmark_vgdts_params()
    variants = _variant_specs(base=base_params)

    print("[stage 1/3] Running baseline suite (tuned + fixed) with all 16 policies...", flush=True)
    baseline_tuned = evaluate_environment_suite(
        env_suite=env_suite,
        tuning_runs=tuning_runs,
        eval_runs=eval_runs,
        seed=seed + 1_001,
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
        global_cts_hazard_grid=[0.01, 0.02, 0.05],
        global_cts_max_runlengths_grid=[max(80, horizon // 20), max(160, horizon // 10)],
        dlinucb_gamma_grid=[0.95, 0.98],
        dlinucb_delta_grid=[0.05, 0.10],
        vgdts_params=base_params,
        vgdts_grid={
            "gamma_default": [0.35, 0.50],
            "gamma_min": [0.10, 0.20],
            "vol_high": [2.2, 3.0],
        },
    )
    baseline_fixed = evaluate_environment_suite_fixed(
        env_suite=env_suite,
        eval_runs=eval_runs,
        seed=seed + 2_001,
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
        global_cts_hazard=0.02,
        global_cts_max_runlengths=max(160, horizon // 10),
        dlinucb_gamma=0.98,
        dlinucb_delta=0.05,
        beta_swts_tau=max(120, horizon // 8),
    )

    print("[stage 2/3] Evaluating VG-dTS variants before/after confidence bonus...", flush=True)
    variant_results: list[dict[str, Any]] = []
    for idx, spec in enumerate(variants):
        tuned_eval = _evaluate_variant_tuned(
            env_suite=env_suite,
            base_params=spec["params"],
            tuning_grid=spec["tuning_grid"],
            tuning_runs=tuning_runs,
            eval_runs=eval_runs,
            seed=seed + 3_001 + 10_000 * (idx + 1),
        )
        fixed_eval = _evaluate_variant_fixed(
            env_suite=env_suite,
            params=spec["params"],
            eval_runs=eval_runs,
            seed=seed + 4_001 + 10_000 * (idx + 1),
        )
        variant_results.append(
            {
                "variant_id": spec["variant_id"],
                "label": spec["label"],
                "base_params": asdict(spec["params"]),
                "tuning_grid": spec["tuning_grid"],
                "tuned": tuned_eval,
                "fixed": fixed_eval,
                "tuned_vs_baselines": _summarize_against_baselines(
                    variant_eval=tuned_eval,
                    baseline_evals=baseline_tuned,
                ),
                "fixed_vs_baselines": _summarize_against_baselines(
                    variant_eval=fixed_eval,
                    baseline_evals=baseline_fixed,
                ),
            }
        )
        print(
            f"[variant] {idx + 1:02d}/{len(variants)} {spec['variant_id']} "
            f"tuned={tuned_eval['avg_final_normalized_regret']:.4f} "
            f"fixed={fixed_eval['avg_final_normalized_regret']:.4f}",
            flush=True,
        )

    print("[stage 3/3] Building summary...", flush=True)
    variant_by_id = {item["variant_id"]: item for item in variant_results}
    baseline_variant = variant_by_id["baseline_vgdts"]
    before_bonus = variant_by_id["dual_memory+likelihood_surprise"]
    after_bonus = variant_by_id["dual_memory+likelihood_surprise+confidence_bonus"]

    summary: dict[str, Any] = {
        "config": {
            "horizon": horizon,
            "n_arms": n_arms,
            "seed": seed,
            "tuning_runs": tuning_runs,
            "eval_runs": eval_runs,
        },
        "algorithm_order": list(ALGORITHM_ORDER),
        "variants": variant_results,
        "headline": {
            "best_tuned_variant_id": min(
                variant_results,
                key=lambda item: item["tuned"]["avg_final_normalized_regret"],
            )["variant_id"],
            "best_fixed_variant_id": min(
                variant_results,
                key=lambda item: item["fixed"]["avg_final_normalized_regret"],
            )["variant_id"],
        },
        "before_after": {
            "tuned_delta_after_minus_before": (
                after_bonus["tuned"]["avg_final_normalized_regret"]
                - before_bonus["tuned"]["avg_final_normalized_regret"]
            ),
            "fixed_delta_after_minus_before": (
                after_bonus["fixed"]["avg_final_normalized_regret"]
                - before_bonus["fixed"]["avg_final_normalized_regret"]
            ),
            "tuned_delta_after_minus_baseline": (
                after_bonus["tuned"]["avg_final_normalized_regret"]
                - baseline_variant["tuned"]["avg_final_normalized_regret"]
            ),
            "fixed_delta_after_minus_baseline": (
                after_bonus["fixed"]["avg_final_normalized_regret"]
                - baseline_variant["fixed"]["avg_final_normalized_regret"]
            ),
        },
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

    json_path = _save_json(results_dir / "vgdts_confidence_bonus_study.json", summary)
    print(f"[done] Wrote summary to {json_path}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare VG-dTS dual-memory+likelihood surprise before/after confidence bonus.",
    )
    parser.add_argument("--output-dir", default="report/project_publication/vgdts_confidence_bonus_study")
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--n-arms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tuning-runs", type=int, default=4)
    parser.add_argument("--eval-runs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_vgdts_confidence_bonus_study(
        output_dir=args.output_dir,
        horizon=args.horizon,
        n_arms=args.n_arms,
        seed=args.seed,
        tuning_runs=args.tuning_runs,
        eval_runs=args.eval_runs,
    )


if __name__ == "__main__":
    main()
