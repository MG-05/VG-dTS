from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_publication.pipeline import (  # noqa: E402
    ALGORITHM_ORDER,
    build_publication_environment_suite,
    evaluate_environment_suite,
    evaluate_environment_suite_fixed,
)


def test_publication_pipeline_includes_extended_nonstationary_baselines() -> None:
    env_suite = build_publication_environment_suite(horizon=80, n_arms=4, seed=4)[:1]

    tuned = evaluate_environment_suite(
        env_suite=env_suite,
        tuning_runs=1,
        eval_runs=1,
        seed=4,
        lambda_grid=[0.75],
        rexp3_gamma_grid=[0.2],
        rexp3_delta_grid=[20],
        beta_swts_tau_grid=[20],
        sw_ucb_tau_grid=[20],
        d_ucb_gamma_grid=[0.97],
        cusum_epsilon_grid=[0.05],
        cusum_threshold_grid=[4.0],
        glr_alpha_grid=[1.0],
        glr_threshold_scale_grid=[1.2],
        adaswitch_reset_threshold_grid=[0.2],
        sw_ts_tau_grid=[20],
        gamma_swgts_tau_grid=[20],
        gamma_swgts_gamma_grid=[0.7],
        global_cts_hazard_grid=[0.02],
        global_cts_max_runlengths_grid=[32],
        dlinucb_gamma_grid=[0.97],
        dlinucb_delta_grid=[0.05],
        vgdts_grid={
            "gamma_default": [0.45],
            "gamma_min": [0.12],
            "lambda_vol": [0.88],
            "n0": [0.0],
            "vol_high": [2.6],
            "one_sided_negative_surprise": [False],
        },
    )
    fixed = evaluate_environment_suite_fixed(env_suite=env_suite, eval_runs=1, seed=17)

    env_key = env_suite[0].key
    assert set(tuned[env_key].policies.keys()) == set(ALGORITHM_ORDER)
    assert set(fixed[env_key].policies.keys()) == set(ALGORITHM_ORDER)

    assert "SW-UCB (0805.3415)" in tuned[env_key].policies
    assert "D-UCB (0805.3415)" in tuned[env_key].policies
    assert "CUSUM-UCB (1711.03539)" in tuned[env_key].policies
    assert "GLR-klUCB (1902.01575)" in tuned[env_key].policies
    assert "AdaSwitch (1902.07010)" in tuned[env_key].policies
    assert "SW-TS (Trovo 2020)" in tuned[env_key].policies
    assert "gamma-SWGTS (2409.05181)" in tuned[env_key].policies
    assert "Global-CTS (1302.3721)" in tuned[env_key].policies
    assert "D-LinUCB one-hot (1909.09146)" in tuned[env_key].policies
    assert "Global-CTS (1302.3721)" in fixed[env_key].policies
    assert "D-LinUCB one-hot (1909.09146)" in fixed[env_key].policies
