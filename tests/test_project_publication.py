from __future__ import annotations

from pathlib import Path
import sys

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_publication.pipeline import build_publication_environment_suite, run_publication_bundle


def test_build_publication_environment_suite_has_expected_keys() -> None:
    suite = build_publication_environment_suite(horizon=120, n_arms=4, seed=3)
    keys = [env.key for env in suite]
    assert keys == [
        "slow",
        "fast",
        "abrupt",
        "mixed",
        "random_breakpoints",
        "random_drift",
        "global_switching",
        "per_arm_switching",
    ]
    assert all(env.means_tk.shape == (120, 4) for env in suite)


def test_run_publication_bundle_tiny_smoke(tmp_path) -> None:
    artifacts = run_publication_bundle(
        output_dir=tmp_path,
        horizon=80,
        n_arms=4,
        seed=5,
        tuning_runs=1,
        eval_runs=1,
        single_environment_key="fast",
        lambda_grid=[0.75],
        rexp3_gamma_grid=[0.2],
        rexp3_delta_grid=[20],
        beta_swts_tau_grid=[20],
        vgdts_grid={
            "gamma_default": [0.45],
            "gamma_min": [0.12],
            "lambda_vol": [0.88],
            "n0": [0.0],
            "vol_high": [2.6],
            "one_sided_negative_surprise": [False],
        },
        show=False,
    )

    assert artifacts.single_env_plot.exists()
    assert artifacts.original_envs_plot.exists()
    assert artifacts.random_signal_plot.exists()
    assert artifacts.optimized_heatmap_plot.exists()
    assert artifacts.fixed_heatmap_plot.exists()
    assert len(artifacts.oracle_environment_plots) == 9
    assert len(artifacts.fixed_environment_plots) == 8
    assert all(path.exists() for path in artifacts.oracle_environment_plots.values())
    assert all(path.exists() for path in artifacts.fixed_environment_plots.values())

    assert artifacts.single_env_summary_json.exists()
    assert artifacts.original_envs_summary_json.exists()
    assert artifacts.random_signal_summary_json.exists()
    assert artifacts.optimized_heatmap_summary_json.exists()
    assert artifacts.fixed_heatmap_summary_json.exists()
