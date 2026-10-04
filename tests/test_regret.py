from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional
def regret_from_rewards(mu: np.ndarray, rewards: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    mu: shape (K, T)
    rewards: shape (T,) realized rewards in {0,1}
    Returns:
      cum_regret[t] = sum_{n<=t} mu_star[n] - sum_{n<=t} rewards[n]
      final_regret, final_norm_regret
    """
    mu_star = mu.max(axis=0)  # dynamic oracle expected reward each t
    oracle_cum = np.cumsum(mu_star)
    alg_cum = np.cumsum(rewards.astype(float))
    cum_regret = oracle_cum - alg_cum
    final_regret = float(cum_regret[-1])
    final_norm = final_regret / mu.shape[1]
    return cum_regret, final_regret, final_norm