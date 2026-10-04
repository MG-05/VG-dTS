from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from project_publication.pipeline import build_publication_environment_suite
from src.adts.policies import VGdTSParams, run_VG_dTS_v2, run_VG_dTS_v21
from src.adts.vgdts_config import make_benchmark_vgdts_v21_params


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


def _evaluate_variant_fixed(
    env_suite,
    params: VGdTSParams,
    runner,
    eval_runs: int,
    seed: int,
) -> dict[str, Any]:
    per_env: dict[str, float] = {}
    regrets: list[float] = []

    for env_idx, env in enumerate(env_suite):
        mu = env.means_tk.T
        horizon = int(mu.shape[1])
        oracle_total = float(np.max(mu, axis=0).sum())

        seed_env = seed + 2_000_000 * (env_idx + 1)
        seed_policy = seed_env + 500_000  # VG-dTS is policy slot 0 in the publication benchmark.
        run_seeds = np.random.SeedSequence(seed_policy).spawn(eval_runs)

        total_regret = 0.0
        for run_seed in run_seeds:
            rng = np.random.default_rng(run_seed)
            rewards = runner(mu=mu, params=params, rng=rng).astype(float)
            alg_total = float(rewards.sum())
            total_regret += (oracle_total - alg_total) / float(horizon)

        regret = total_regret / float(eval_runs)
        per_env[env.key] = float(regret)
        regrets.append(float(regret))

    return {
        "avg_final_normalized_regret": float(np.mean(regrets)),
        "per_env": per_env,
    }


def _load_fixed_baselines(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["algorithm"]: float(row["avg_final_normalized_regret"])
        for row in payload["ranked_avg"]
        if row["algorithm"] != "VG-dTS"
    }


def _rank_vs_baselines(variant_regret: float, baseline_regrets: dict[str, float]) -> int:
    merged = dict(baseline_regrets)
    merged["VG-dTS"] = float(variant_regret)
    ranked = sorted(merged.items(), key=lambda item: item[1])
    return 1 + [name for name, _ in ranked].index("VG-dTS")


def _variant_specs(base: VGdTSParams) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "v2_baseline",
            "label": "VG-dTS v2 baseline",
            "runner": run_VG_dTS_v2,
            "params": base,
        },
        {
            "variant_id": "v21_shock_score_only",
            "label": "v2 + shock score",
            "runner": run_VG_dTS_v21,
            "params": replace(base, shock_score_enabled=True),
        },
        {
            "variant_id": "v21_shock_n_eff_only",
            "label": "v2 + shock-to-effective-sample-size",
            "runner": run_VG_dTS_v21,
            "params": replace(base, shock_n_eff_enabled=True),
        },
        {
            "variant_id": "v21_selective_revisit_only",
            "label": "v2 + selective stale-arm revisit",
            "runner": run_VG_dTS_v21,
            "params": replace(base, selective_revisit_enabled=True),
        },
        {
            "variant_id": "v21_shock_bundle",
            "label": "v2 + shock score + shock-to-effective-sample-size",
            "runner": run_VG_dTS_v21,
            "params": replace(base, shock_score_enabled=True, shock_n_eff_enabled=True),
        },
        {
            "variant_id": "v21_full",
            "label": "v2.1 full",
            "runner": run_VG_dTS_v21,
            "params": replace(
                base,
                shock_score_enabled=True,
                shock_n_eff_enabled=True,
                selective_revisit_enabled=True,
            ),
        },
    ]


def run_vgdts_v21_component_study(
    output_dir: str | Path = "report/project_publication/vgdts_v21_component_study",
    horizon: int = 5000,
    n_arms: int = 4,
    seed: int = 999,
    eval_runs: int = 1,
    baseline_json: str | Path = "report/project_publication/baselines16_h5000_fixed_e1_quick_vgdts_v2.json",
) -> dict[str, Any]:
    output_root = Path(output_dir)
    baseline_path = Path(baseline_json)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {baseline_path}")

    env_suite = build_publication_environment_suite(horizon=horizon, n_arms=n_arms, seed=seed)
    base_params = replace(
        make_benchmark_vgdts_v21_params(),
        shock_score_enabled=False,
        shock_n_eff_enabled=False,
        selective_revisit_enabled=False,
    )
    baseline_regrets = _load_fixed_baselines(baseline_path)

    variants: list[dict[str, Any]] = []
    for spec in _variant_specs(base_params):
        eval_result = _evaluate_variant_fixed(
            env_suite=env_suite,
            params=spec["params"],
            runner=spec["runner"],
            eval_runs=eval_runs,
            seed=seed,
        )
        variants.append(
            {
                "variant_id": spec["variant_id"],
                "label": spec["label"],
                "params": asdict(spec["params"]),
                "avg_final_normalized_regret": eval_result["avg_final_normalized_regret"],
                "per_env": eval_result["per_env"],
                "rank_vs_baselines": _rank_vs_baselines(
                    variant_regret=eval_result["avg_final_normalized_regret"],
                    baseline_regrets=baseline_regrets,
                ),
            }
        )

    variants_by_id = {variant["variant_id"]: variant for variant in variants}
    baseline_avg = variants_by_id["v2_baseline"]["avg_final_normalized_regret"]
    baseline_per_env = variants_by_id["v2_baseline"]["per_env"]
    for variant in variants:
        variant["delta_vs_v2"] = float(variant["avg_final_normalized_regret"] - baseline_avg)
        variant["per_env_delta_vs_v2"] = {
            env_key: float(regret - baseline_per_env[env_key])
            for env_key, regret in variant["per_env"].items()
        }

    variants_sorted = sorted(variants, key=lambda item: item["avg_final_normalized_regret"])
    summary = {
        "config": {
            "horizon": horizon,
            "n_arms": n_arms,
            "seed": seed,
            "eval_runs": eval_runs,
            "baseline_json": str(baseline_path),
        },
        "variants": variants_sorted,
    }
    _save_json(
        output_root / "results" / "vgdts_v21_component_ablation_h5000_e1.json",
        summary,
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated v2.1 component ablations on top of VG-dTS v2."
    )
    parser.add_argument("--output-dir", default="report/project_publication/vgdts_v21_component_study")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--n-arms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--eval-runs", type=int, default=1)
    parser.add_argument(
        "--baseline-json",
        default="report/project_publication/baselines16_h5000_fixed_e1_quick_vgdts_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_vgdts_v21_component_study(
        output_dir=args.output_dir,
        horizon=args.horizon,
        n_arms=args.n_arms,
        seed=args.seed,
        eval_runs=args.eval_runs,
        baseline_json=args.baseline_json,
    )
    print("Variant summary (lower is better):")
    for variant in summary["variants"]:
        print(
            f"{variant['variant_id']:28s} "
            f"avg={variant['avg_final_normalized_regret']:.15f} "
            f"delta_vs_v2={variant['delta_vs_v2']:+.6f} "
            f"rank={variant['rank_vs_baselines']}/16"
        )


if __name__ == "__main__":
    main()
