from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .envs import BernoulliNonStationaryEnv

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DynamicOracleResult:
    chosen_arms: IntArray
    expected_rewards: FloatArray
    sampled_rewards: IntArray
    cumulative_expected_reward: FloatArray
    cumulative_sampled_reward: FloatArray


def dynamic_oracle_actions(mean_rewards: FloatArray) -> tuple[IntArray, FloatArray]:
    """
    Dynamic oracle from the paper:
    at each t, choose the arm with the largest mean reward mu_{k,t}.
    """

    # 2D array with shape (T, K) -> rewards for all K arms at each time step t in T
    means = np.asarray(mean_rewards, dtype=float)
    # print(f"mean rewards: {means}")
    if means.ndim != 2:
        raise ValueError("mean_rewards must have shape (horizon, n_arms).")

    # list of arms with the largest mean at each time step t
    chosen_arms = np.argmax(means, axis=1).astype(np.int64)
    # vector of time t from 0 to T-1
    t = np.arange(means.shape[0])
    # list of rewards
    expected_rewards = means[t, chosen_arms]
    # return chosen arms and expected rewards array for each timestep t
    return chosen_arms, expected_rewards


def run_dynamic_oracle(env: BernoulliNonStationaryEnv) -> DynamicOracleResult:
    chosen_arms, expected_rewards = dynamic_oracle_actions(env.mean_rewards)

    sampled_rewards = np.array(
        [env.sample(arm=int(arm), t=t) for t, arm in enumerate(chosen_arms)],
        dtype=np.int64,
    )

    cumulative_expected_reward = np.cumsum(expected_rewards)
    cumulative_sampled_reward = np.cumsum(sampled_rewards)

    return DynamicOracleResult(
        chosen_arms=chosen_arms,
        expected_rewards=expected_rewards,
        sampled_rewards=sampled_rewards,
        cumulative_expected_reward=cumulative_expected_reward,
        cumulative_sampled_reward=cumulative_sampled_reward,
    )
