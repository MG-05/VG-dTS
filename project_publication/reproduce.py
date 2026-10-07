"""Paper-only experiment orchestration, replay, plotting, and numerical checks."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from src.adts.policies import (
    TSParams, dTSParams, dOTSParams, DTSParams, REXP3Params, VGdTSParams,
    BetaSWTSParams, run_TS, run_dTS, run_dOTS, run_DTS, run_REXP3,
    run_VG_dTS, run_beta_swts,
)
from .pipeline import (
    ALGORITHM_ORDER, ALGORITHM_COLORS, EnvironmentSpec, EnvironmentEvaluation,
    TunedPolicyEvaluation, PolicyCandidate, build_publication_environment_suite,
    _build_single_environment, _build_policy_candidates_tuned,
    _build_policy_candidates_fixed, _evaluate_environment_with_candidates,
    _evaluation_to_json_dict, _save_json, plot_original_paper_environments,
    plot_random_signal_failure, plot_environment_oracle_gallery, plot_optimized_heatmap,
)

TUNED = "optimized_heatmap_8env_7policies"
FIXED = "fixed_params_heatmap_8env_7policies"
RANDOM = "rigorous_pure_random_signal_tuned"
POLICIES = {
    "VG-dTS": (VGdTSParams, run_VG_dTS), "dTS": (dTSParams, run_dTS),
    "dOTS": (dOTSParams, run_dOTS), "TS": (TSParams, run_TS),
    "REXP3": (REXP3Params, run_REXP3), "Dynamic TS": (DTSParams, run_DTS),
    "Beta-SWTS": (BetaSWTSParams, run_beta_swts),
}


def _read_reference(reference_dir: Path) -> dict:
    return {name: json.loads((reference_dir / f"{name}.json").read_text())
            for name in (TUNED, FIXED, RANDOM)}


def _saved_candidates(record: dict) -> dict[str, list[PolicyCandidate]]:
    result = {}
    for name in ALGORITHM_ORDER:
        original = record["policies"][name]["best_params"]
        values = {key: value for key, value in original.items() if key != "lambda"}
        cls, runner = POLICIES[name]
        result[name] = [PolicyCandidate(runner, cls(**values), dict(original))]
    return result


def _run_job(job: tuple) -> tuple[str, EnvironmentEvaluation]:
    group, env, candidates, eval_runs, seed, tuning_runs, reference = job
    print(f"Running {group}/{env.key}", flush=True)
    result = _evaluate_environment_with_candidates(env, candidates, eval_runs, seed, tuning_runs)
    # In replay mode, keep the archived tuning score explicitly as provenance;
    # only the evaluation scores/curves have been recomputed.
    if reference is not None and group != "fixed":
        result = EnvironmentEvaluation(env, {
            name: TunedPolicyEvaluation(
                p.best_params, reference["policies"][name]["tuning_final_normalized_regret"],
                p.eval_final_normalized_regret, p.avg_reward_t, p.avg_norm_regret_t,
            ) for name, p in result.policies.items()
        })
    return group, result


def _save_evaluation(output_dir: Path, group: str, result: EnvironmentEvaluation) -> None:
    stem = output_dir / "runs" / group / result.environment.key
    record = _evaluation_to_json_dict(result)
    record["runtime"] = {"numpy": np.__version__, "python": platform.python_version()}
    record["environment_sha256"] = hashlib.sha256(result.environment.means_tk.tobytes()).hexdigest()
    _save_json(stem.with_suffix(".json"), record)
    arrays = {"means_tk": result.environment.means_tk}
    for i, name in enumerate(ALGORITHM_ORDER):
        arrays[f"reward_{i}"] = result.policies[name].avg_reward_t
        arrays[f"regret_{i}"] = result.policies[name].avg_norm_regret_t
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)


def _load_evaluations(output_dir: Path, group: str, keys: list[str]) -> dict[str, EnvironmentEvaluation]:
    result = {}
    for key in keys:
        stem = output_dir / "runs" / group / key
        record = json.loads(stem.with_suffix(".json").read_text())
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
            env = EnvironmentSpec(key, record["environment"]["name"], arrays["means_tk"])
            policies = {}
            for i, name in enumerate(ALGORITHM_ORDER):
                p = record["policies"][name]
                policies[name] = TunedPolicyEvaluation(
                    p["best_params"], p["tuning_final_normalized_regret"],
                    p["eval_final_normalized_regret"], arrays[f"reward_{i}"], arrays[f"regret_{i}"],
                )
        result[key] = EnvironmentEvaluation(env, policies)
    return result


def render(output_dir: Path) -> None:
    """Redraw all twelve manuscript PNGs from saved simulation curves."""
    import matplotlib
    matplotlib.use("Agg")  # Reproduction saves files and never needs a GUI session.
    metadata = json.loads((output_dir / "metadata.json").read_text())
    if metadata["status"] not in {"simulated", "complete"}:
        raise ValueError("The simulation is incomplete; plots cannot be regenerated yet.")
    keys = metadata["environment_order"]
    tuned = _load_evaluations(output_dir, "tuned", keys)
    fixed = _load_evaluations(output_dir, "fixed", keys)
    random = _load_evaluations(output_dir, "random", ["pure_random_signal"])["pure_random_signal"]
    fig_dir = output_dir / "figures"
    plot_environment_oracle_gallery([e.environment for e in tuned.values()], fig_dir / "environment_oracle")
    plot_original_paper_environments(tuned, fig_dir / "original_paper_envs_tuned.png")
    plot_random_signal_failure(random, fig_dir / f"{RANDOM}.png")
    for name, mode, evaluations in [(TUNED, "optimized", tuned), (FIXED, "fixed", fixed)]:
        plot_optimized_heatmap(evaluations, fig_dir / f"{name}.png",
                               title="Optimized Final Normalized Regret" if mode == "optimized" else "Fixed-Parameter Final Normalized Regret")
        payload = {"algorithms": list(ALGORITHM_ORDER), "algorithm_colors": ALGORITHM_COLORS,
                   "mode": mode, "evaluations": {k: _evaluation_to_json_dict(v) for k, v in evaluations.items()}}
        if mode == "fixed":
            payload["fixed_parameters"] = {k: v.best_params for k, v in next(iter(fixed.values())).policies.items()}
        _save_json(output_dir / "results" / f"{name}.json", payload)
    _save_json(output_dir / "results" / f"{RANDOM}.json", _evaluation_to_json_dict(random))


def reproduce(*, output_dir: Path, reference_dir: Path, mode: str = "paper", seed: int = 0,
              tuning_runs: int = 70, eval_runs: int = 1000, workers: int = 1) -> None:
    if mode in {"paper", "retune"} and np.__version__ != "1.26.4":
        raise RuntimeError("Paper reproduction requires numpy==1.26.4. Install requirements.txt; NumPy 2.x changes the breakpoint realization.")
    if mode not in {"paper", "retune", "smoke"}:
        raise ValueError("Unknown reproduction mode.")
    if min(tuning_runs, eval_runs, workers) <= 0 or seed < 0:
        raise ValueError("Run counts/workers must be positive and seed nonnegative.")
    output_dir, reference_dir = Path(output_dir), Path(reference_dir)
    # Results are evidence. Refuse to overwrite either an old run or the paper.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new --output-dir.")
    references = _read_reference(reference_dir)
    horizon = 120 if mode == "smoke" else 5000
    if mode == "smoke":
        tuning_runs, eval_runs = 1, 2
    suite = build_publication_environment_suite(horizon, 4, seed)
    random_env = _build_single_environment("random_signal", horizon, 4, seed)
    grids = {} if mode != "smoke" else dict(
        lambda_grid=[0.75], rexp3_gamma_grid=[0.2], rexp3_delta_grid=[20],
        beta_swts_tau_grid=[20], vgdts_grid={},
    )
    tuned_candidates = _build_policy_candidates_tuned(horizon, **grids)
    fixed_candidates = _build_policy_candidates_fixed(horizon)
    jobs = []
    seed_records = []
    for group, environments, offset in [("tuned", suite, 0), ("fixed", suite, 4_000_000), ("random", [random_env], 0)]:
        for i, env in enumerate(environments):
            base_seed = seed + (9_000_000 if group == "random" else offset + (2_000_000 if group == "fixed" else 1_000_000) * (i + 1))
            record = references[RANDOM] if group == "random" else references[FIXED if group == "fixed" else TUNED]["evaluations"][env.key]
            candidates = fixed_candidates if group == "fixed" else tuned_candidates
            if mode == "paper":
                candidates = _saved_candidates(record)
            count = tuning_runs if mode != "paper" and group != "fixed" else None
            jobs.append((group, env, candidates, eval_runs, base_seed, count, record if mode == "paper" else None))
            seed_records.append({"group": group, "environment": env.key, "base_seed": base_seed})
    import matplotlib
    source_root = Path(__file__).resolve().parents[1]
    sources = sorted((source_root / "src").rglob("*.py")) + sorted((source_root / "project_publication").glob("*.py"))
    metadata = {
        "status": "running", "mode": mode, "horizon": horizon, "n_arms": 4, "seed": seed,
        "tuning_runs": tuning_runs if mode != "paper" else None, "eval_runs": eval_runs,
        "tuning_scores_source": "archived reference" if mode == "paper" else "recomputed",
        "workers": workers, "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "environment_order": [e.key for e in suite],
        "algorithm_order": list(ALGORITHM_ORDER), "jobs": seed_records,
        "candidate_counts": {k: len(v) for k, v in tuned_candidates.items()} if mode != "paper" else None,
        "tuning_grids": {k: [c.best_params for c in v] for k, v in tuned_candidates.items()} if mode != "paper" else None,
        "seed_schedule": {"tuning": "base_seed + 10000*(policy_index+1) + 100*(candidate_index+1)",
                          "evaluation": "base_seed + 500000*(policy_index+1)", "rollouts": "numpy.random.SeedSequence(seed).spawn(n_runs); default_rng(child)"},
        "source_sha256": {p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "reference_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(reference_dir.glob("*.json"))},
    }
    _save_json(output_dir / "metadata.json", metadata)
    started = time.monotonic()
    def save(result):
        group, evaluation = result
        _save_evaluation(output_dir, group, evaluation)
        print(f"Saved {group}/{evaluation.environment.key}", flush=True)
    if workers == 1:
        for job in jobs:
            save(_run_job(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                save(future.result())
    metadata.update(status="simulated", simulation_seconds=time.monotonic() - started)
    _save_json(output_dir / "metadata.json", metadata)
    render(output_dir)
    metadata["status"] = "complete"
    _save_json(output_dir / "metadata.json", metadata)
    print(f"Saved figures, numerical summaries, curves, and run metadata to {output_dir}", flush=True)


def verify(output_dir: Path, reference_dir: Path, atol: float = 1e-12) -> bool:
    """Check all 119 final regrets; retune runs also check parameters/tuning scores."""
    metadata = json.loads((output_dir / "metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["horizon"] != 5000 or metadata["eval_runs"] != 1000 or metadata["seed"] != 0:
        raise ValueError("Exact paper verification requires a completed T=5000, seed=0, 1000-evaluation-run reproduction.")
    expected, actual = _read_reference(reference_dir), _read_reference(output_dir / "results")
    mismatches = []
    max_error = 0.0
    count = 0
    for artifact in (TUNED, FIXED, RANDOM):
        ref_envs = {"pure_random_signal": expected[artifact]} if artifact == RANDOM else expected[artifact]["evaluations"]
        got_envs = {"pure_random_signal": actual[artifact]} if artifact == RANDOM else actual[artifact]["evaluations"]
        for env, record in ref_envs.items():
            for name in ALGORITHM_ORDER:
                ref, got = record["policies"][name], got_envs[env]["policies"][name]
                error = abs(ref["eval_final_normalized_regret"] - got["eval_final_normalized_regret"])
                max_error = max(max_error, error)
                count += 1
                if not np.isfinite(error) or error > atol:
                    mismatches.append(f"{artifact}/{env}/{name}: evaluation error {error:.3g}")
                if metadata["mode"] == "retune":
                    tuning_error = abs(ref["tuning_final_normalized_regret"] - got["tuning_final_normalized_regret"])
                    if got["best_params"] != ref["best_params"] or not np.isfinite(tuning_error) or tuning_error > atol:
                        mismatches.append(f"{artifact}/{env}/{name}: tuning/parameters differ")
    report = {"passed": not mismatches, "cells": count, "max_absolute_error": max_error,
              "atol": atol, "mismatches": mismatches, "mode": metadata["mode"]}
    _save_json(output_dir / "verification.json", report)
    print(json.dumps(report, indent=2))
    return not mismatches
