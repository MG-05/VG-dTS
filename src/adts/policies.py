from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TSParams:
    alpha0: float = 1.0
    beta0: float = 1.0


@dataclass
class DTSParams:
    """
    Dynamic Thompson Sampling as described in Appendix B:
    - Maintain Beta(alpha_k, beta_k) per arm (initialized to alpha0, beta0).
    - Select arm via TS sampling.
    - Update only played arm with observed reward.
    - Apply discounting ONLY to played arm AFTER threshold (alpha+beta > C):
         (alpha, beta) <- (C/(C+1)) * (alpha, beta)
      Then do the usual Bernoulli conjugate update.
    """
    alpha0: float = 1.0
    beta0: float = 1.0
    C: float = 250.0  # threshold


@dataclass
class dTSParams:
    """
    Discounted Thompson Sampling (Algorithm 1):
    Maintain discounted success/failure counts S_k, F_k (floats).
    Each step:
      - sample theta_k ~ Beta(S_k + alpha0, F_k + beta0)
      - play argmax theta_k, observe Bernoulli reward r in {0,1}
      - for played arm: S <- gamma*S + r, F <- gamma*F + (1-r)
      - for unplayed arms: S <- gamma*S, F <- gamma*F
    """
    alpha0: float = 1.0
    beta0: float = 1.0
    gamma: float = 0.75


@dataclass
class dOTSParams(dTSParams):
    """
    Discounted Optimistic Thompson Sampling (Eq. (6)):
      tilde ~ Beta(S+alpha0, F+beta0)
      theta = max(E[tilde], tilde)
    """
    pass


@dataclass
class VGdTSParams:
    """
    Volatility-Gated discounted TS (VG-dTS):
      - Maintains Beta(alpha_k, beta_k) per arm.
      - Uses played-arm posterior-surprise EWMA volatility to build
        per-arm discount gamma_{k,t}.
      - Uses optimistic action scores when enabled:
          theta_k = max(Beta sample_k, posterior mean_k)
      - Applies uncertainty gating via effective sample size n_eff.
      - Uses prior-preserving discounting for all arms each round.
    """

    alpha0: float = 1.0
    beta0: float = 1.0

    # EWMA state for surprise-based volatility proxy
    lambda_vol: float = 0.94
    mean0: float = 0.0
    var0: float = 0.0

    # Action/value estimation options
    optimistic: bool = True
    surprise_clip: float = 8.0
    one_sided_negative_surprise: bool = False

    # Discount range and low-evidence fallback
    gamma_min: float = 0.30
    gamma_max: float = 0.995
    gamma_default: float = 0.90
    n0: float = 25.0

    # Volatility->discount mapping: {"inverse_linear", "logistic"}
    gamma_mapping: str = "inverse_linear"

    # Inverse-linear mapping controls
    vol_low: float = 0.0
    vol_high: float = 2.0

    # Logistic mapping controls
    logistic_mid: float = 1.0
    logistic_slope: float = 4.0

    eps: float = 1e-12


@dataclass
class REXP3Params:
    """
    REXP3 as a restarting EXP3:
      - Reset weights every Delta steps.
      - EXP3 probabilities:
           p_i = (1-gamma)*w_i/sum(w) + gamma/K
      - Reward estimate:
           xhat = r / p_i
      - Update:
           w_i <- w_i * exp((gamma/K) * xhat)
    gamma: exploration/egalitarianism factor
    Delta: restart period (batch length)
    """
    gamma: float = 0.1136
    Delta: int = 250


def run_TS(mu: np.ndarray, params: TSParams, rng: np.random.Generator) -> np.ndarray:
    K, T = mu.shape
    alpha = np.full(K, params.alpha0, dtype=float)
    beta = np.full(K, params.beta0, dtype=float)
    rewards = np.zeros(T, dtype=int)

    for t in range(T):
        theta = rng.beta(alpha, beta)
        arm = int(np.argmax(theta))
        r = int(rng.random() < mu[arm, t])
        rewards[t] = r
        alpha[arm] += r
        beta[arm] += (1 - r)

    return rewards


def run_DTS(mu: np.ndarray, params: DTSParams, rng: np.random.Generator) -> np.ndarray:
    K, T = mu.shape
    alpha = np.full(K, params.alpha0, dtype=float)
    beta = np.full(K, params.beta0, dtype=float)
    rewards = np.zeros(T, dtype=int)

    disc = params.C / (params.C + 1.0)

    for t in range(T):
        theta = rng.beta(alpha, beta)
        arm = int(np.argmax(theta))
        r = int(rng.random() < mu[arm, t])
        rewards[t] = r

        # apply discount to played arm only after threshold
        if (alpha[arm] + beta[arm]) > params.C:
            alpha[arm] *= disc
            beta[arm] *= disc

        # conjugate update on played arm
        alpha[arm] += r
        beta[arm] += (1 - r)

    return rewards


def run_dTS(mu: np.ndarray, params: dTSParams, rng: np.random.Generator) -> np.ndarray:
    K, T = mu.shape
    S = np.zeros(K, dtype=float)
    F = np.zeros(K, dtype=float)
    rewards = np.zeros(T, dtype=int)

    for t in range(T):
        theta = rng.beta(S + params.alpha0, F + params.beta0)
        arm = int(np.argmax(theta))
        r = int(rng.random() < mu[arm, t])
        rewards[t] = r

        # discount all arms
        S *= params.gamma
        F *= params.gamma

        # then update played arm
        S[arm] += r
        F[arm] += (1 - r)

    return rewards


def run_dOTS(mu: np.ndarray, params: dOTSParams, rng: np.random.Generator) -> np.ndarray:
    K, T = mu.shape
    S = np.zeros(K, dtype=float)
    F = np.zeros(K, dtype=float)
    rewards = np.zeros(T, dtype=int)

    for t in range(T):
        a = S + params.alpha0
        b = F + params.beta0
        tilde = rng.beta(a, b)
        mean = a / (a + b)
        theta = np.maximum(mean, tilde)  # Eq. (6)
        arm = int(np.argmax(theta))
        r = int(rng.random() < mu[arm, t])
        rewards[t] = r

        S *= params.gamma
        F *= params.gamma
        S[arm] += r
        F[arm] += (1 - r)

    return rewards


def _validate_vgdts_params(params: VGdTSParams) -> None:
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if not (0.0 <= params.lambda_vol <= 1.0):
        raise ValueError("lambda_vol must be in [0, 1].")
    if params.gamma_min > params.gamma_max:
        raise ValueError("gamma_min must be <= gamma_max.")
    if params.n0 < 0.0:
        raise ValueError("n0 must be >= 0.")
    if params.eps <= 0.0:
        raise ValueError("eps must be > 0.")
    if params.surprise_clip <= 0.0:
        raise ValueError("surprise_clip must be > 0.")
    if params.gamma_mapping not in {"inverse_linear", "logistic"}:
        raise ValueError("gamma_mapping must be 'inverse_linear' or 'logistic'.")
    if params.logistic_slope <= 0.0:
        raise ValueError("logistic_slope must be > 0 for monotone decay.")
    if params.gamma_mapping == "inverse_linear" and params.vol_high <= params.vol_low:
        raise ValueError("vol_high must be > vol_low for inverse-linear mapping.")


def _inverse_linear_gamma_with_bounds(
    vol: np.ndarray,
    vol_low: float,
    vol_high: float,
    params: VGdTSParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear normalization of volatility in [vol_low, vol_high], then inverted:
      low vol -> gamma_max, high vol -> gamma_min.
    """
    span = max(vol_high - vol_low, params.eps)
    norm = np.clip((vol - vol_low) / span, 0.0, 1.0)
    gamma = params.gamma_max - norm * (params.gamma_max - params.gamma_min)
    gamma = np.clip(gamma, params.gamma_min, params.gamma_max)
    return gamma, norm


def _gamma_vol_inverse_linear(vol: np.ndarray, params: VGdTSParams) -> np.ndarray:
    gamma, _ = _inverse_linear_gamma_with_bounds(
        vol=vol,
        vol_low=params.vol_low,
        vol_high=params.vol_high,
        params=params,
    )
    return gamma


def _gamma_vol_logistic(vol: np.ndarray, params: VGdTSParams) -> np.ndarray:
    """
    Logistic decay from gamma_max to gamma_min as volatility rises.
    """
    x = params.logistic_slope * (vol - params.logistic_mid)
    x = np.clip(x, -60.0, 60.0)  # numerically stable exp
    decay = 1.0 / (1.0 + np.exp(x))
    gamma = params.gamma_min + decay * (params.gamma_max - params.gamma_min)
    return np.clip(gamma, params.gamma_min, params.gamma_max)


def _gamma_from_volatility(vol: np.ndarray, params: VGdTSParams) -> np.ndarray:
    if params.gamma_mapping == "inverse_linear":
        return _gamma_vol_inverse_linear(vol, params)
    return _gamma_vol_logistic(vol, params)


def _standardized_surprise(
    x_t: float,
    p_hat: float,
    params: VGdTSParams,
) -> float:
    """
    Standardized prediction error used for volatility tracking.
    If one_sided_negative_surprise=True, only downside errors contribute.
    """
    denom = math.sqrt(max(p_hat * (1.0 - p_hat), params.eps))
    if params.one_sided_negative_surprise:
        surprise_num = max(p_hat - x_t, 0.0)
    else:
        surprise_num = abs(x_t - p_hat)
    surprise = surprise_num / denom
    return float(np.clip(surprise, 0.0, params.surprise_clip))


def run_VG_dTS(
    mu: np.ndarray,
    params: VGdTSParams,
    rng: np.random.Generator,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-arm Volatility-Gated discounted Thompson Sampling (VG-dTS).

    Core steps each round:
      1) Draw Beta samples and build action scores:
            score_k = sample_k or max(sample_k, mean_k) if optimistic=True
         then choose argmax(score_k).
      2) Observe Bernoulli reward for selected arm.
      3) Compute selected-arm standardized posterior surprise:
            s_t = error(x_t, p_hat_t) / sqrt(p_hat_t(1-p_hat_t) + eps)
         where error(x, p) = |x-p| by default and max(p-x, 0) when
         one_sided_negative_surprise=True.
         where x_t is selected-arm return proxy (reward if returns is None).
      4) Update played-arm surprise EWMA statistics.
      5) Map per-arm local volatility proxy -> gamma.
      6) Apply uncertainty-gated adaptive discount:
            n_eff = alpha + beta - 2
            w = n_eff / (n_eff + n0)
            gamma = (1-w)*gamma_default + w*gamma_vol(v)
      7) Restless prior-preserving discount for all arms:
            alpha <- 1 + gamma*(alpha - 1)
            beta  <- 1 + gamma*(beta - 1)
      8) Conjugate update on selected arm.

    If `returns` is None, observed Bernoulli rewards are used as the
    selected-arm surprise proxy.
    """
    _validate_vgdts_params(params)

    K, T = mu.shape
    alpha = np.full(K, params.alpha0, dtype=float)
    beta = np.full(K, params.beta0, dtype=float)
    surprise_mean = np.full(K, params.mean0, dtype=float)
    surprise_var = np.full(K, params.var0, dtype=float)

    rewards = np.zeros(T, dtype=int)

    gamma_default = float(np.clip(params.gamma_default, params.gamma_min, params.gamma_max))

    returns_arr = None
    if returns is not None:
        returns_arr = np.asarray(returns, dtype=float)
        if returns_arr.shape != mu.shape:
            raise ValueError("returns must have shape (K, T) matching mu.")

    one_minus_lambda = 1.0 - params.lambda_vol

    for t in range(T):
        sample = rng.beta(alpha, beta)
        mean = alpha / (alpha + beta)
        n_eff = np.maximum(alpha + beta - 2.0, 0.0)

        if params.optimistic:
            theta = np.maximum(sample, mean)
        else:
            theta = sample
        arm = int(np.argmax(theta))

        r = int(rng.random() < mu[arm, t])

        rewards[t] = r

        # Selected-arm posterior surprise (standardized prediction error).
        x_t = float(r if returns_arr is None else returns_arr[arm, t])
        p_hat = float(mean[arm])
        surprise = _standardized_surprise(x_t=x_t, p_hat=p_hat, params=params)

        # Update local surprise EWMA only on the played arm (bandit feedback).
        surprise_mean[arm] = params.lambda_vol * surprise_mean[arm] + one_minus_lambda * surprise
        centered = surprise - surprise_mean[arm]
        surprise_var[arm] = params.lambda_vol * surprise_var[arm] + one_minus_lambda * (centered * centered)
        surprise_var[arm] = max(surprise_var[arm], 0.0)

        vol_proxy = np.sqrt(np.maximum(surprise_var, 0.0))
        gamma_vol = _gamma_from_volatility(vol_proxy, params)

        if params.n0 == 0.0:
            w = (n_eff > 0.0).astype(float)
        else:
            w = n_eff / (n_eff + params.n0)
        w = np.clip(w, 0.0, 1.0)
        gamma = (1.0 - w) * gamma_default + w * gamma_vol
        gamma = np.clip(gamma, params.gamma_min, params.gamma_max)

        # Prior-preserving restless discount of all arms.
        alpha = 1.0 + gamma * (alpha - 1.0)
        beta = 1.0 + gamma * (beta - 1.0)

        # Selected-arm Bernoulli update.
        alpha[arm] += r
        beta[arm] += (1 - r)

    return rewards


def run_REXP3(mu: np.ndarray, params: REXP3Params, rng: np.random.Generator) -> np.ndarray:
    K, T = mu.shape
    rewards = np.zeros(T, dtype=int)

    w = np.ones(K, dtype=float)

    def reset():
        nonlocal w
        w = np.ones(K, dtype=float)

    reset()

    for t in range(T):
        # restart each batch
        if t % params.Delta == 0 and t != 0:
            reset()

        W = w.sum()
        p = (1.0 - params.gamma) * (w / W) + params.gamma / K

        arm = int(rng.choice(K, p=p))
        r = float(rng.random() < mu[arm, t])
        rewards[t] = int(r)

        # importance-weighted reward estimate
        xhat = r / max(p[arm], 1e-12)

        # standard EXP3 update
        w[arm] *= math.exp((params.gamma / K) * xhat)

    return rewards


@dataclass
class BetaSWTSParams:
    alpha0: float = 1.0
    beta0: float = 1.0
    tau: int = 200


def _validate_mu(mu: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(mu, dtype=float)
    if arr.ndim != 2:
        raise ValueError("mu must have shape (K, T).")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("mu entries must be in [0, 1].")
    return int(arr.shape[0]), int(arr.shape[1])


def _pull_from_mu(mu: np.ndarray, arm: int, t: int, rng: np.random.Generator) -> int:
    return int(rng.random() < float(mu[arm, t]))


def run_beta_swts(
    mu: np.ndarray,
    params: BetaSWTSParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Beta-SWTS from 2409.05181 for Bernoulli rewards.
    """
    k, horizon = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if params.tau <= 0:
        raise ValueError("tau must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    pulls = np.zeros(k, dtype=int)
    successes = np.zeros(k, dtype=float)
    history: list[tuple[int, int]] = []

    for t in range(horizon):
        alpha = params.alpha0 + successes
        beta = params.beta0 + np.maximum(pulls.astype(float) - successes, 0.0)
        theta = rng.beta(alpha, beta)
        arm = int(np.argmax(theta))

        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward

        pulls[arm] += 1
        successes[arm] += reward
        history.append((arm, reward))

        if len(history) > params.tau:
            old_arm, old_reward = history.pop(0)
            pulls[old_arm] -= 1
            successes[old_arm] -= old_reward

    return rewards
