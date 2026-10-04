from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adts import tuning_cli


def test_parse_env_kwargs_parses_numeric_values() -> None:
    kwargs = tuning_cli._parse_env_kwargs(["horizon=80", "period=20", "min_jump=0.10"])
    assert kwargs == {"horizon": 80, "period": 20, "min_jump": 0.10}


def test_parse_env_kwargs_rejects_invalid_item() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        tuning_cli._parse_env_kwargs(["horizon:80"])


def test_parse_vgdts_grid_parses_scalar_values() -> None:
    grid = tuning_cli._parse_vgdts_grid(
        [
            "gamma_default=0.45,0.85",
            "n0=0,25",
            "optimistic=true,false",
            "gamma_mapping=inverse_linear,logistic",
        ]
    )
    assert grid == {
        "gamma_default": [0.45, 0.85],
        "n0": [0, 25],
        "optimistic": [True, False],
        "gamma_mapping": ["inverse_linear", "logistic"],
    }


def test_main_invokes_tuning_entrypoint(monkeypatch, capsys, tmp_path) -> None:
    class FakeTunedResult:
        def __init__(self, best_params: dict[str, float], final_normalized_regret: float) -> None:
            self.best_params = best_params
            self.final_normalized_regret = final_normalized_regret

    captured_call: dict[str, object] = {}

    def fake_run_single_env_lambda_tuning_experiment(**kwargs):
        captured_call.update(kwargs)
        return tmp_path / "cli_plot.png", {
            "TS": FakeTunedResult(best_params={}, final_normalized_regret=0.125),
            "dTS": FakeTunedResult(best_params={"lambda": 0.5}, final_normalized_regret=0.075),
        }

    fake_experiments_module = types.ModuleType("src.adts.experiments")
    fake_experiments_module.run_single_env_lambda_tuning_experiment = (
        fake_run_single_env_lambda_tuning_experiment
    )
    monkeypatch.setitem(sys.modules, "src.adts.experiments", fake_experiments_module)

    exit_code = tuning_cli.main(
        [
            "--environment",
            "fast",
            "--n-runs",
            "3",
            "--tuning-runs",
            "2",
            "--seed",
            "11",
            "--lambda-grid",
            "0.5",
            "0.8",
            "--rexp3-gamma-grid",
            "0.1",
            "0.3",
            "--rexp3-delta-grid",
            "8",
            "16",
            "--vgdts-grid",
            "gamma_default=0.5,0.8",
            "--vgdts-grid",
            "n0=0,25",
            "--env-kw",
            "horizon=80",
            "--env-kw",
            "period=20",
            "--output-path",
            str(tmp_path / "result.png"),
        ]
    )

    assert exit_code == 0
    assert captured_call["environment"] == "fast"
    assert captured_call["n_runs"] == 3
    assert captured_call["tuning_runs"] == 2
    assert captured_call["seed"] == 11
    assert captured_call["lambda_grid"] == [0.5, 0.8]
    assert captured_call["rexp3_gamma_grid"] == [0.1, 0.3]
    assert captured_call["rexp3_delta_grid"] == [8, 16]
    assert captured_call["vgdts_grid"] == {"gamma_default": [0.5, 0.8], "n0": [0, 25]}
    assert captured_call["environment_kwargs"] == {"horizon": 80, "period": 20}

    output = capsys.readouterr().out
    assert "Saved plot to:" in output
    assert "Best parameters by policy:" in output
