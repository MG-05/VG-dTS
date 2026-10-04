from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adts.envs import make_recovering_reward_table
from src.adts.policies import (
    AdaSwitchParams,
    AFFTSParams,
    BetaSWTSParams,
    CUSUMUCBParams,
    DLinUCBParams,
    DUCBParams,
    GLRklUCBParams,
    GammaSWGTSParams,
    GlobalCTSParams,
    RGPParams,
    SWTSParams,
    SWUCBParams,
    VGdTSParams,
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
)


def _make_mu(k: int = 4, t: int = 60) -> np.ndarray:
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.05, 0.95, size=(k, t))
    return mu


def _assert_rewards(rewards: np.ndarray, t: int) -> None:
    assert rewards.shape == (t,)
    assert set(np.unique(rewards)).issubset({0, 1})


def test_nonstationary_policy_runners_return_binary_rewards() -> None:
    mu = _make_mu(k=4, t=50)
    rng = np.random.default_rng(42)

    r1 = run_global_cts(mu, GlobalCTSParams(hazard=0.05, max_runlengths=64), rng)
    r2 = run_aff_ts(mu, AFFTSParams(eta=0.001), np.random.default_rng(1))
    r3 = run_dlinucb_onehot(mu, DLinUCBParams(gamma=0.97), np.random.default_rng(2))
    r4 = run_beta_swts(mu, BetaSWTSParams(tau=20), np.random.default_rng(3))
    r5 = run_sw_ucb(mu, SWUCBParams(tau=20, xi=0.5), np.random.default_rng(4))
    r6 = run_d_ucb(mu, DUCBParams(gamma=0.97, xi=0.5), np.random.default_rng(5))
    r7 = run_cusum_ucb(
        mu,
        CUSUMUCBParams(xi=0.5, epsilon=0.05, threshold=4.0, warmup=10, random_explore=0.05),
        np.random.default_rng(6),
    )
    r8 = run_glr_klucb(
        mu,
        GLRklUCBParams(alpha=1.0, threshold_scale=1.2, min_segment_len=5, max_history=60),
        np.random.default_rng(7),
    )
    r9 = run_adaswitch(
        mu,
        AdaSwitchParams(xi=0.5, min_window=4, max_windows=5, reset_threshold=0.2, min_pulls_for_reset=10),
        np.random.default_rng(8),
    )
    r10 = run_sw_ts(mu, SWTSParams(tau=20), np.random.default_rng(9))
    r11 = run_gamma_swgts(mu, GammaSWGTSParams(tau=20, gamma=0.7), np.random.default_rng(10))

    _assert_rewards(r1, 50)
    _assert_rewards(r2, 50)
    _assert_rewards(r3, 50)
    _assert_rewards(r4, 50)
    _assert_rewards(r5, 50)
    _assert_rewards(r6, 50)
    _assert_rewards(r7, 50)
    _assert_rewards(r8, 50)
    _assert_rewards(r9, 50)
    _assert_rewards(r10, 50)
    _assert_rewards(r11, 50)


def test_recovering_policy_runners_return_binary_rewards() -> None:
    table = make_recovering_reward_table(n_arms=4, z_max=8, seed=5)
    horizon = 40

    r_ucb = run_drpg_ucb(
        table,
        horizon=horizon,
        params=RGPParams(lookahead=1, max_sequences=1000),
        rng=np.random.default_rng(10),
        env_seed=123,
    )
    r_ts = run_drpg_ts(
        table,
        horizon=horizon,
        params=RGPParams(lookahead=1, max_sequences=1000),
        rng=np.random.default_rng(11),
        env_seed=123,
    )
    r_vg = run_vgdts_recovering(
        table,
        horizon=horizon,
        params=VGdTSParams(),
        rng=np.random.default_rng(12),
        env_seed=123,
    )

    _assert_rewards(r_ucb, horizon)
    _assert_rewards(r_ts, horizon)
    _assert_rewards(r_vg, horizon)
