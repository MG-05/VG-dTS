from __future__ import annotations

import numpy as np
import pytest

from src.adts.envs import (
    BernoulliNonStationaryEnv,
    make_random_breakpoint_env,
    mixed_regime_challenge_means,
    random_breakpoint_means,
    random_drift_amplitude_means,
)


def _assert_mean_matrix(means: np.ndarray, horizon: int, n_arms: int) -> None:
    assert means.shape == (horizon, n_arms)
    assert np.all(means >= 0.0)
    assert np.all(means <= 1.0)


def test_mixed_regime_challenge_has_expected_arm_behaviors() -> None:
    horizon = 800
    means = mixed_regime_challenge_means(
        horizon=horizon,
        n_arms=4,
        burst_prob=0.08,
        burst_duration=12,
        switch_interval=80,
        seed=7,
    )
    _assert_mean_matrix(means, horizon=horizon, n_arms=4)

    stable_arm = means[:, 0]
    drifting_arm = means[:, 1]
    bursty_arm = means[:, 2]
    switching_arm = means[:, 3]

    assert np.allclose(stable_arm, stable_arm[0])
    assert drifting_arm[-1] > drifting_arm[0]
    assert np.count_nonzero(bursty_arm > 0.5) > 0
    assert (bursty_arm.max() - bursty_arm.min()) > 0.4
    assert np.count_nonzero(np.diff(switching_arm)) >= 3
    assert len(np.unique(np.round(switching_arm, decimals=8))) == 2


def test_random_breakpoint_env_is_reproducible_with_seed() -> None:
    kwargs = dict(
        horizon=600,
        n_arms=3,
        min_breakpoints=2,
        max_breakpoints=5,
        min_segment=30,
        min_mean=0.1,
        max_mean=0.9,
        min_jump=0.1,
    )
    means_a = random_breakpoint_means(seed=19, **kwargs)
    means_b = random_breakpoint_means(seed=19, **kwargs)
    means_c = random_breakpoint_means(seed=20, **kwargs)

    _assert_mean_matrix(means_a, horizon=kwargs["horizon"], n_arms=kwargs["n_arms"])
    assert np.array_equal(means_a, means_b)
    assert not np.array_equal(means_a, means_c)

    for arm in range(kwargs["n_arms"]):
        transitions = int(np.count_nonzero(np.diff(means_a[:, arm])))
        assert kwargs["min_breakpoints"] <= transitions <= kwargs["max_breakpoints"]


def test_random_drift_amplitude_env_is_reproducible_and_nonstationary() -> None:
    kwargs = dict(
        horizon=700,
        n_arms=4,
        amplitude_min=0.05,
        amplitude_max=0.30,
        amplitude_step_std=0.02,
        period_min=40,
        period_max=160,
        drift_step_std=0.006,
    )
    means_a = random_drift_amplitude_means(seed=31, **kwargs)
    means_b = random_drift_amplitude_means(seed=31, **kwargs)
    means_c = random_drift_amplitude_means(seed=32, **kwargs)

    _assert_mean_matrix(means_a, horizon=kwargs["horizon"], n_arms=kwargs["n_arms"])
    assert np.array_equal(means_a, means_b)
    assert not np.array_equal(means_a, means_c)
    assert np.all(np.std(means_a, axis=0) > 0.03)


def test_make_random_breakpoint_env_returns_valid_bernoulli_env() -> None:
    env = make_random_breakpoint_env(
        horizon=120,
        n_arms=3,
        min_breakpoints=1,
        max_breakpoints=3,
        min_segment=20,
        means_seed=13,
        seed=5,
    )
    assert isinstance(env, BernoulliNonStationaryEnv)
    assert env.horizon == 120
    assert env.n_arms == 3
    assert env.sample(arm=0, t=0) in {0, 1}


def test_mixed_regime_requires_four_arms() -> None:
    with pytest.raises(ValueError, match="n_arms must be >= 4"):
        mixed_regime_challenge_means(horizon=100, n_arms=3)


def test_random_breakpoint_rejects_infeasible_breakpoint_budget() -> None:
    with pytest.raises(ValueError, match="horizon too short"):
        random_breakpoint_means(
            horizon=50,
            n_arms=2,
            min_breakpoints=1,
            max_breakpoints=4,
            min_segment=12,
        )
