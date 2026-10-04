from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adts.envs import (
    RecoveringBanditEnv,
    global_switching_means,
    make_recovering_reward_table,
    per_arm_switching_means,
)


def test_switching_mean_generators_shape_and_bounds() -> None:
    g = global_switching_means(horizon=60, n_arms=4, switch_prob=0.2, seed=1)
    p = per_arm_switching_means(horizon=60, n_arms=4, switch_prob=0.2, seed=2)
    assert g.shape == (60, 4)
    assert p.shape == (60, 4)
    assert np.all((0.0 <= g) & (g <= 1.0))
    assert np.all((0.0 <= p) & (p <= 1.0))


def test_recovering_env_step_updates_delays() -> None:
    table = make_recovering_reward_table(n_arms=3, z_max=5, seed=7)
    env = RecoveringBanditEnv(reward_table=table, seed=0)
    assert env.delays.tolist() == [0, 0, 0]

    _, z0, _ = env.step(1)
    assert z0 == 0
    assert env.delays.tolist() == [1, 0, 1]

    _, z1, _ = env.step(1)
    assert z1 == 0
    assert env.delays.tolist() == [2, 0, 2]
