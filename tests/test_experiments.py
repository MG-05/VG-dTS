from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from src.adts.experiments import TunedPolicyResult, run_single_env_lambda_tuning_experiment


def test_single_env_lambda_tuning_experiment_runs_and_saves_plot(tmp_path) -> None:
    output_path = tmp_path / "single_env_tuned.png"

    plot_path, tuned = run_single_env_lambda_tuning_experiment(
        environment="fast",
        n_runs=3,
        tuning_runs=2,
        seed=7,
        lambda_grid=[0.5, 0.8],
        rexp3_gamma_grid=[0.1, 0.3],
        rexp3_delta_grid=[8, 16],
        vgdts_grid={"gamma_default": [0.45, 0.85], "n0": [0.0, 25.0]},
        environment_kwargs={"horizon": 80, "n_arms": 4, "period": 20},
        output_path=output_path,
        show=False,
    )

    assert plot_path == output_path
    assert output_path.exists()

    assert set(tuned) == {"dTS", "dOTS", "TS", "REXP3", "Dynamic TS", "VG-dTS"}
    assert all(isinstance(result, TunedPolicyResult) for result in tuned.values())
    assert tuned["TS"].best_params == {}
    assert tuned["dTS"].best_params["lambda"] in {0.5, 0.8}
    assert tuned["Dynamic TS"].best_params["lambda"] in {0.5, 0.8}
    assert tuned["VG-dTS"].best_params["gamma_default"] in {0.45, 0.85}
    assert tuned["VG-dTS"].best_params["n0"] in {0.0, 25.0}
    for result in tuned.values():
        assert np.isfinite(result.final_normalized_regret)


def test_single_env_lambda_tuning_rejects_unknown_environment() -> None:
    with pytest.raises(ValueError, match="Unknown environment"):
        run_single_env_lambda_tuning_experiment(
            environment="invalid_environment",
            n_runs=1,
            tuning_runs=1,
            show=False,
        )
