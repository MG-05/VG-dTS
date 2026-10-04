from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adts.envs import make_recovering_reward_table
from src.adts.experiments import run_paper_algorithm_benchmarks
from src.adts.vgdts_config import make_benchmark_vgdts_params


def test_paper_algorithm_benchmarks_tiny_smoke(tmp_path) -> None:
    rng = np.random.default_rng(0)
    tiny_mu = rng.uniform(0.05, 0.95, size=(4, 40))
    recovering_table = make_recovering_reward_table(n_arms=4, z_max=8, seed=9)

    nonstationary_plot, recovering_plot, summary_path, summary = run_paper_algorithm_benchmarks(
        n_runs_nonstationary=1,
        n_runs_recovering=1,
        seed=3,
        output_dir=tmp_path,
        nonstationary_env_suite=[("tiny", tiny_mu)],
        recovering_table=recovering_table,
        recovering_horizon=40,
        vgdts_params=make_benchmark_vgdts_params(gamma_default=0.9, n0=25.0),
        make_plots=False,
    )

    assert nonstationary_plot.name == "paper_algorithms_nonstationary_heatmap.png"
    assert recovering_plot.name == "paper_algorithms_recovering_benchmark.png"
    assert summary_path.exists()
    assert "nonstationary" in summary
    assert "recovering" in summary
    assert "tiny" in summary["nonstationary"]
    expected_policy_subset = {
        "SW-UCB (0805.3415)",
        "D-UCB (0805.3415)",
        "CUSUM-UCB (1711.03539)",
        "GLR-klUCB (1902.01575)",
        "AdaSwitch (1902.07010)",
        "SW-TS (Trovo 2020)",
        "gamma-SWGTS (2409.05181)",
    }
    assert expected_policy_subset.issubset(summary["nonstationary"]["tiny"].keys())
    assert summary["config"]["vgdts_params"]["gamma_default"] == 0.9
    assert summary["config"]["vgdts_params"]["n0"] == 25.0
