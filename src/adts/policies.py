# This is where the thompson sampling algorithms go
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from .envs import RecoveringBanditEnv


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

    # --- Enhancement toggles (all default-off to preserve legacy behavior) ---
    global_shock_enabled: bool = False
    global_shock_lambda: float = 0.90
    global_shock_threshold: float = 2.0
    global_shock_gamma_floor: float = 0.20

    stale_arm_revisit_enabled: bool = False
    stale_arm_revisit_scale: float = 0.30
    stale_arm_revisit_half_life: float = 50.0
    stale_arm_revisit_clip: float = 0.50

    dual_memory_enabled: bool = False
    gamma_long: float = 0.995
    dual_memory_mix_power: float = 1.0

    surprise_mode: str = "standardized"  # {"standardized", "neg_log_likelihood"}

    two_timescale_volatility_enabled: bool = False
    lambda_vol_fast: float = 0.80
    lambda_vol_slow: float = 0.97

    adaptive_optimism_enabled: bool = False
    optimism_base: float = 0.50
    optimism_var_weight: float = 0.60
    optimism_vol_weight: float = 0.80

    confidence_bonus_enabled: bool = False
    confidence_bonus_scale: float = 0.60
    confidence_bonus_vol_weight: float = 0.80
    confidence_bonus_coldstart_weight: float = 0.40
    confidence_bonus_clip: float = 0.75

    # v2.1 add-ons (all default-off; only used by run_VG_dTS_v21)
    shock_score_enabled: bool = False
    shock_score_scale: float = 0.12
    shock_score_decay: float = 0.92
    shock_n_eff_enabled: bool = False
    shock_n_eff_scale: float = 0.75
    selective_revisit_enabled: bool = False
    selective_revisit_margin: float = 0.03

    online_vol_calibration_enabled: bool = False
    calibration_window: int = 200
    calibration_quantile_low: float = 0.20
    calibration_quantile_high: float = 0.80
    calibration_min_count: int = 20
    calibration_min_span: float = 1e-3

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
    if params.global_shock_lambda < 0.0 or params.global_shock_lambda > 1.0:
        raise ValueError("global_shock_lambda must be in [0, 1].")
    if params.global_shock_threshold <= 0.0:
        raise ValueError("global_shock_threshold must be > 0.")
    if params.global_shock_gamma_floor <= 0.0:
        raise ValueError("global_shock_gamma_floor must be > 0.")
    if params.stale_arm_revisit_scale < 0.0:
        raise ValueError("stale_arm_revisit_scale must be >= 0.")
    if params.stale_arm_revisit_half_life <= 0.0:
        raise ValueError("stale_arm_revisit_half_life must be > 0.")
    if params.stale_arm_revisit_clip < 0.0:
        raise ValueError("stale_arm_revisit_clip must be >= 0.")
    if params.gamma_long <= 0.0 or params.gamma_long > 1.0:
        raise ValueError("gamma_long must be in (0, 1].")
    if params.dual_memory_mix_power <= 0.0:
        raise ValueError("dual_memory_mix_power must be > 0.")
    if params.surprise_mode not in {"standardized", "neg_log_likelihood"}:
        raise ValueError("surprise_mode must be 'standardized' or 'neg_log_likelihood'.")
    if params.lambda_vol_fast < 0.0 or params.lambda_vol_fast > 1.0:
        raise ValueError("lambda_vol_fast must be in [0, 1].")
    if params.lambda_vol_slow < 0.0 or params.lambda_vol_slow > 1.0:
        raise ValueError("lambda_vol_slow must be in [0, 1].")
    if params.two_timescale_volatility_enabled and params.lambda_vol_fast >= params.lambda_vol_slow:
        raise ValueError("Require lambda_vol_fast < lambda_vol_slow for two-timescale volatility.")
    if params.confidence_bonus_scale < 0.0:
        raise ValueError("confidence_bonus_scale must be >= 0.")
    if params.confidence_bonus_vol_weight < 0.0:
        raise ValueError("confidence_bonus_vol_weight must be >= 0.")
    if params.confidence_bonus_coldstart_weight < 0.0:
        raise ValueError("confidence_bonus_coldstart_weight must be >= 0.")
    if params.confidence_bonus_clip < 0.0:
        raise ValueError("confidence_bonus_clip must be >= 0.")
    if params.shock_score_scale < 0.0:
        raise ValueError("shock_score_scale must be >= 0.")
    if not (0.0 <= params.shock_score_decay <= 1.0):
        raise ValueError("shock_score_decay must be in [0, 1].")
    if params.shock_n_eff_scale < 0.0:
        raise ValueError("shock_n_eff_scale must be >= 0.")
    if params.selective_revisit_margin < 0.0:
        raise ValueError("selective_revisit_margin must be >= 0.")
    if params.calibration_window <= 1:
        raise ValueError("calibration_window must be > 1.")
    if params.calibration_min_count <= 1:
        raise ValueError("calibration_min_count must be > 1.")
    if not (0.0 <= params.calibration_quantile_low < params.calibration_quantile_high <= 1.0):
        raise ValueError("Require 0 <= calibration_quantile_low < calibration_quantile_high <= 1.")
    if params.calibration_min_span <= 0.0:
        raise ValueError("calibration_min_span must be > 0.")


def _vgdts_has_enhancements(params: VGdTSParams) -> bool:
    return any(
        (
            params.global_shock_enabled,
            params.stale_arm_revisit_enabled,
            params.dual_memory_enabled,
            params.surprise_mode != "standardized",
            params.two_timescale_volatility_enabled,
            params.adaptive_optimism_enabled,
            params.confidence_bonus_enabled,
            params.online_vol_calibration_enabled,
        )
    )


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


def _negative_log_predictive_surprise(
    x_t: float,
    p_hat: float,
    params: VGdTSParams,
) -> float:
    """
    Bernoulli negative log-predictive surprise:
      -log P(X=x_t | p_hat)
    with optional downside-only filtering.
    """
    p_clip = float(np.clip(p_hat, params.eps, 1.0 - params.eps))
    if params.one_sided_negative_surprise and x_t >= p_hat:
        return 0.0

    if x_t >= 0.5:
        prob = p_clip
    else:
        prob = 1.0 - p_clip

    return float(np.clip(-math.log(max(prob, params.eps)), 0.0, params.surprise_clip))


def _compute_surprise(x_t: float, p_hat: float, params: VGdTSParams) -> float:
    if params.surprise_mode == "neg_log_likelihood":
        return _negative_log_predictive_surprise(x_t=x_t, p_hat=p_hat, params=params)
    return _standardized_surprise(x_t=x_t, p_hat=p_hat, params=params)


def _variance_gated_confidence_bonus(
    posterior_std: np.ndarray,
    vol_norm: np.ndarray,
    n_eff: np.ndarray,
    params: VGdTSParams,
) -> np.ndarray:
    coldstart = 1.0 / np.sqrt(np.maximum(n_eff, 0.0) + 1.0)
    multiplier = 1.0 + params.confidence_bonus_vol_weight * vol_norm
    multiplier += params.confidence_bonus_coldstart_weight * coldstart
    bonus = params.confidence_bonus_scale * multiplier * posterior_std
    return np.clip(bonus, 0.0, params.confidence_bonus_clip)


def _run_vgdts_legacy(
    mu: np.ndarray,
    params: VGdTSParams,
    rng: np.random.Generator,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
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


def _run_vgdts_enhanced(
    mu: np.ndarray,
    params: VGdTSParams,
    rng: np.random.Generator,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    K, T = mu.shape
    alpha = np.full(K, params.alpha0, dtype=float)
    beta = np.full(K, params.beta0, dtype=float)
    alpha_long = np.full(K, params.alpha0, dtype=float) if params.dual_memory_enabled else None
    beta_long = np.full(K, params.beta0, dtype=float) if params.dual_memory_enabled else None

    surprise_mean = np.full(K, params.mean0, dtype=float)
    surprise_var = np.full(K, params.var0, dtype=float)
    if params.two_timescale_volatility_enabled:
        surprise_mean_fast = np.full(K, params.mean0, dtype=float)
        surprise_var_fast = np.full(K, params.var0, dtype=float)
        surprise_mean_slow = np.full(K, params.mean0, dtype=float)
        surprise_var_slow = np.full(K, params.var0, dtype=float)
    else:
        surprise_mean_fast = None
        surprise_var_fast = None
        surprise_mean_slow = None
        surprise_var_slow = None

    rewards = np.zeros(T, dtype=int)
    last_played = np.full(K, -1, dtype=int)
    vol_history: list[list[float]] = [[] for _ in range(K)]

    gamma_default = float(np.clip(params.gamma_default, params.gamma_min, params.gamma_max))
    gamma_long = float(np.clip(params.gamma_long, params.gamma_min, params.gamma_max))

    returns_arr = None
    if returns is not None:
        returns_arr = np.asarray(returns, dtype=float)
        if returns_arr.shape != mu.shape:
            raise ValueError("returns must have shape (K, T) matching mu.")

    one_minus_lambda = 1.0 - params.lambda_vol
    one_minus_fast = 1.0 - params.lambda_vol_fast
    one_minus_slow = 1.0 - params.lambda_vol_slow
    global_surprise = 0.0

    for t in range(T):
        if params.two_timescale_volatility_enabled:
            vol_fast = np.sqrt(np.maximum(surprise_var_fast, 0.0))
            vol_slow = np.sqrt(np.maximum(surprise_var_slow, 0.0))
            vol_proxy = np.maximum(vol_fast - vol_slow, 0.0)
        else:
            vol_proxy = np.sqrt(np.maximum(surprise_var, 0.0))

        vol_low_eff = np.full(K, params.vol_low, dtype=float)
        vol_high_eff = np.full(K, params.vol_high, dtype=float)
        if params.online_vol_calibration_enabled:
            for k in range(K):
                hist = vol_history[k]
                if len(hist) >= params.calibration_min_count:
                    arr = np.asarray(hist, dtype=float)
                    q_low = float(np.quantile(arr, params.calibration_quantile_low))
                    q_high = float(np.quantile(arr, params.calibration_quantile_high))
                    if q_high - q_low < params.calibration_min_span:
                        q_high = q_low + params.calibration_min_span
                    vol_low_eff[k] = q_low
                    vol_high_eff[k] = q_high

        vol_span = np.maximum(vol_high_eff - vol_low_eff, params.eps)
        vol_norm = np.clip((vol_proxy - vol_low_eff) / vol_span, 0.0, 1.0)

        n_eff = np.maximum(alpha + beta - 2.0, 0.0)
        if params.dual_memory_enabled:
            assert alpha_long is not None
            assert beta_long is not None
            n_eff_long = np.maximum(alpha_long + beta_long - 2.0, 0.0)

            short_weight = np.power(vol_norm, params.dual_memory_mix_power)
            short_weight = np.clip(short_weight, 0.0, 1.0)

            sample_short = rng.beta(alpha, beta)
            mean_short = alpha / (alpha + beta)
            sample_long = rng.beta(alpha_long, beta_long)
            mean_long = alpha_long / (alpha_long + beta_long)

            sample = short_weight * sample_short + (1.0 - short_weight) * sample_long
            mean = short_weight * mean_short + (1.0 - short_weight) * mean_long
            action_n_eff = short_weight * n_eff + (1.0 - short_weight) * n_eff_long
        else:
            sample = rng.beta(alpha, beta)
            mean = alpha / (alpha + beta)
            action_n_eff = n_eff

        posterior_std = np.sqrt(np.maximum(mean * (1.0 - mean), 0.0) / (np.maximum(action_n_eff, 0.0) + 1.0))
        if params.adaptive_optimism_enabled and params.optimistic:
            std_norm = np.clip(posterior_std / 0.5, 0.0, 1.0)
            optimism_temp = params.optimism_base + params.optimism_var_weight * std_norm
            optimism_temp -= params.optimism_vol_weight * vol_norm
            optimism_temp = np.clip(optimism_temp, 0.0, 1.0)
            theta = sample + optimism_temp * np.maximum(mean - sample, 0.0)
        elif params.optimistic:
            theta = np.maximum(sample, mean)
        else:
            theta = sample

        if params.confidence_bonus_enabled and params.confidence_bonus_scale > 0.0:
            theta = theta + _variance_gated_confidence_bonus(
                posterior_std=posterior_std,
                vol_norm=vol_norm,
                n_eff=action_n_eff,
                params=params,
            )

        if params.stale_arm_revisit_enabled and params.stale_arm_revisit_scale > 0.0:
            age = np.maximum(t - last_played, 1).astype(float)
            age_ratio = age / (age + params.stale_arm_revisit_half_life)
            predictive_std = np.sqrt(np.maximum(mean * (1.0 - mean), 0.0))
            revisit_bonus = params.stale_arm_revisit_scale * np.sqrt(age_ratio) * predictive_std
            revisit_bonus = np.clip(revisit_bonus, 0.0, params.stale_arm_revisit_clip)
            theta = theta + revisit_bonus

        arm = int(np.argmax(theta))
        r = int(rng.random() < mu[arm, t])
        rewards[t] = r
        last_played[arm] = t

        x_t = float(r if returns_arr is None else returns_arr[arm, t])
        p_hat = float(mean[arm])
        surprise = _compute_surprise(x_t=x_t, p_hat=p_hat, params=params)

        if params.two_timescale_volatility_enabled:
            assert surprise_mean_fast is not None
            assert surprise_var_fast is not None
            assert surprise_mean_slow is not None
            assert surprise_var_slow is not None

            surprise_mean_fast[arm] = params.lambda_vol_fast * surprise_mean_fast[arm] + one_minus_fast * surprise
            centered_fast = surprise - surprise_mean_fast[arm]
            surprise_var_fast[arm] = params.lambda_vol_fast * surprise_var_fast[arm] + one_minus_fast * (
                centered_fast * centered_fast
            )
            surprise_var_fast[arm] = max(surprise_var_fast[arm], 0.0)

            surprise_mean_slow[arm] = params.lambda_vol_slow * surprise_mean_slow[arm] + one_minus_slow * surprise
            centered_slow = surprise - surprise_mean_slow[arm]
            surprise_var_slow[arm] = params.lambda_vol_slow * surprise_var_slow[arm] + one_minus_slow * (
                centered_slow * centered_slow
            )
            surprise_var_slow[arm] = max(surprise_var_slow[arm], 0.0)

            vol_fast = np.sqrt(np.maximum(surprise_var_fast, 0.0))
            vol_slow = np.sqrt(np.maximum(surprise_var_slow, 0.0))
            vol_proxy = np.maximum(vol_fast - vol_slow, 0.0)
        else:
            surprise_mean[arm] = params.lambda_vol * surprise_mean[arm] + one_minus_lambda * surprise
            centered = surprise - surprise_mean[arm]
            surprise_var[arm] = params.lambda_vol * surprise_var[arm] + one_minus_lambda * (centered * centered)
            surprise_var[arm] = max(surprise_var[arm], 0.0)
            vol_proxy = np.sqrt(np.maximum(surprise_var, 0.0))

        if params.online_vol_calibration_enabled:
            hist = vol_history[arm]
            hist.append(float(vol_proxy[arm]))
            if len(hist) > params.calibration_window:
                hist.pop(0)

            vol_low_eff = np.full(K, params.vol_low, dtype=float)
            vol_high_eff = np.full(K, params.vol_high, dtype=float)
            for k in range(K):
                arm_hist = vol_history[k]
                if len(arm_hist) >= params.calibration_min_count:
                    arr = np.asarray(arm_hist, dtype=float)
                    q_low = float(np.quantile(arr, params.calibration_quantile_low))
                    q_high = float(np.quantile(arr, params.calibration_quantile_high))
                    if q_high - q_low < params.calibration_min_span:
                        q_high = q_low + params.calibration_min_span
                    vol_low_eff[k] = q_low
                    vol_high_eff[k] = q_high
        else:
            vol_low_eff = np.full(K, params.vol_low, dtype=float)
            vol_high_eff = np.full(K, params.vol_high, dtype=float)

        if params.gamma_mapping == "inverse_linear":
            span = np.maximum(vol_high_eff - vol_low_eff, params.eps)
            norm = np.clip((vol_proxy - vol_low_eff) / span, 0.0, 1.0)
            gamma_vol = params.gamma_max - norm * (params.gamma_max - params.gamma_min)
            gamma_vol = np.clip(gamma_vol, params.gamma_min, params.gamma_max)
        else:
            gamma_vol = _gamma_vol_logistic(vol_proxy, params)

        n_eff = np.maximum(alpha + beta - 2.0, 0.0)
        if params.n0 == 0.0:
            w = (n_eff > 0.0).astype(float)
        else:
            w = n_eff / (n_eff + params.n0)
        w = np.clip(w, 0.0, 1.0)
        gamma = (1.0 - w) * gamma_default + w * gamma_vol

        if params.global_shock_enabled:
            global_surprise = params.global_shock_lambda * global_surprise
            global_surprise += (1.0 - params.global_shock_lambda) * surprise
            shock_level = (global_surprise - params.global_shock_threshold) / max(params.global_shock_threshold, params.eps)
            shock_level = float(np.clip(shock_level, 0.0, 1.0))
            shock_gamma = float(np.clip(params.global_shock_gamma_floor, params.gamma_min, params.gamma_max))
            gamma = (1.0 - shock_level) * gamma + shock_level * shock_gamma

        gamma = np.clip(gamma, params.gamma_min, params.gamma_max)

        alpha = 1.0 + gamma * (alpha - 1.0)
        beta = 1.0 + gamma * (beta - 1.0)

        if params.dual_memory_enabled:
            assert alpha_long is not None
            assert beta_long is not None
            alpha_long = 1.0 + gamma_long * (alpha_long - 1.0)
            beta_long = 1.0 + gamma_long * (beta_long - 1.0)

        alpha[arm] += r
        beta[arm] += (1 - r)

        if params.dual_memory_enabled:
            assert alpha_long is not None
            assert beta_long is not None
            alpha_long[arm] += r
            beta_long[arm] += (1 - r)

    return rewards


def run_VG_dTS(
    mu: np.ndarray,
    params: VGdTSParams,
    rng: np.random.Generator,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-arm Volatility-Gated discounted Thompson Sampling (VG-dTS).
    """
    _validate_vgdts_params(params)
    if _vgdts_has_enhancements(params):
        return _run_vgdts_enhanced(mu=mu, params=params, rng=rng, returns=returns)
    return _run_vgdts_legacy(mu=mu, params=params, rng=rng, returns=returns)


def run_VG_dTS_v2(
    mu: np.ndarray,
    params: VGdTSParams,
    rng: np.random.Generator,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    VG-dTS v2: frozen structure built from three mechanisms:
      - dual-memory posterior mixing
      - likelihood surprise (negative log predictive probability)
      - volatility-aware confidence bonus

    We keep scalar knobs configurable through `params`, but enforce the
    mechanism set so v2 remains a single coherent algorithm.
    """
    v2_params = replace(
        params,
        dual_memory_enabled=True,
        surprise_mode="neg_log_likelihood",
        confidence_bonus_enabled=True,
    )
    return run_VG_dTS(mu=mu, params=v2_params, rng=rng, returns=returns)


def _vgdts_v21_has_addons(params: VGdTSParams) -> bool:
    return any(
        (
            params.shock_score_enabled,
            params.shock_n_eff_enabled,
            params.selective_revisit_enabled,
        )
    )


def run_VG_dTS_v21(
    mu: np.ndarray,
    params: VGdTSParams,
    rng: np.random.Generator,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    VG-dTS v2.1: preserve v2 core and add three soft exploration add-ons:
      - shock score
      - shock-driven effective sample size modulation
      - selective stale-arm revisit

    When all add-ons are disabled, this matches v2 exactly.
    """
    core_params = replace(
        params,
        dual_memory_enabled=True,
        surprise_mode="neg_log_likelihood",
        confidence_bonus_enabled=True,
    )
    _validate_vgdts_params(core_params)
    if not _vgdts_v21_has_addons(core_params):
        return run_VG_dTS_v2(mu=mu, params=core_params, rng=rng, returns=returns)

    k_arms, horizon = mu.shape
    alpha = np.full(k_arms, core_params.alpha0, dtype=float)
    beta = np.full(k_arms, core_params.beta0, dtype=float)
    alpha_long = np.full(k_arms, core_params.alpha0, dtype=float)
    beta_long = np.full(k_arms, core_params.beta0, dtype=float)
    surprise_mean = np.full(k_arms, core_params.mean0, dtype=float)
    surprise_var = np.full(k_arms, core_params.var0, dtype=float)
    shock_state = np.zeros(k_arms, dtype=float)
    last_played = np.full(k_arms, -1, dtype=int)
    rewards = np.zeros(horizon, dtype=int)

    gamma_default = float(np.clip(core_params.gamma_default, core_params.gamma_min, core_params.gamma_max))
    gamma_long = float(np.clip(core_params.gamma_long, core_params.gamma_min, core_params.gamma_max))
    one_minus_lambda = 1.0 - core_params.lambda_vol
    one_minus_shock = 1.0 - core_params.shock_score_decay

    returns_arr = None
    if returns is not None:
        returns_arr = np.asarray(returns, dtype=float)
        if returns_arr.shape != mu.shape:
            raise ValueError("returns must have shape (K, T) matching mu.")

    for t in range(horizon):
        vol_proxy = np.sqrt(np.maximum(surprise_var, 0.0))
        vol_span = max(core_params.vol_high - core_params.vol_low, core_params.eps)
        vol_norm = np.clip((vol_proxy - core_params.vol_low) / vol_span, 0.0, 1.0)

        n_eff_short = np.maximum(alpha + beta - 2.0, 0.0)
        n_eff_long = np.maximum(alpha_long + beta_long - 2.0, 0.0)
        short_weight = np.power(vol_norm, core_params.dual_memory_mix_power)
        short_weight = np.clip(short_weight, 0.0, 1.0)

        sample_short = rng.beta(alpha, beta)
        mean_short = alpha / (alpha + beta)
        sample_long = rng.beta(alpha_long, beta_long)
        mean_long = alpha_long / (alpha_long + beta_long)

        sample = short_weight * sample_short + (1.0 - short_weight) * sample_long
        mean = short_weight * mean_short + (1.0 - short_weight) * mean_long
        action_n_eff = short_weight * n_eff_short + (1.0 - short_weight) * n_eff_long

        shock_norm = np.clip(shock_state / max(core_params.surprise_clip, core_params.eps), 0.0, 1.0)
        bonus_n_eff = action_n_eff
        if core_params.shock_n_eff_enabled and core_params.shock_n_eff_scale > 0.0:
            bonus_n_eff = action_n_eff / (1.0 + core_params.shock_n_eff_scale * shock_norm)

        posterior_std = np.sqrt(np.maximum(mean * (1.0 - mean), 0.0) / (np.maximum(bonus_n_eff, 0.0) + 1.0))
        theta = np.maximum(sample, mean) if core_params.optimistic else sample
        theta = theta + _variance_gated_confidence_bonus(
            posterior_std=posterior_std,
            vol_norm=vol_norm,
            n_eff=bonus_n_eff,
            params=core_params,
        )

        if core_params.shock_score_enabled and core_params.shock_score_scale > 0.0:
            theta = theta + core_params.shock_score_scale * shock_norm * posterior_std

        if core_params.selective_revisit_enabled and core_params.stale_arm_revisit_scale > 0.0:
            theta_before_revisit = theta.copy()
            age = np.maximum(t - last_played, 1).astype(float)
            age_ratio = age / (age + core_params.stale_arm_revisit_half_life)
            revisit_bonus = core_params.stale_arm_revisit_scale * np.sqrt(age_ratio) * posterior_std
            revisit_bonus = np.clip(revisit_bonus, 0.0, core_params.stale_arm_revisit_clip)
            candidate_mask = (np.max(theta_before_revisit) - theta_before_revisit) <= core_params.selective_revisit_margin
            theta = theta + candidate_mask.astype(float) * revisit_bonus

        arm = int(np.argmax(theta))
        reward = int(rng.random() < mu[arm, t])
        rewards[t] = reward
        last_played[arm] = t

        x_t = float(reward if returns_arr is None else returns_arr[arm, t])
        p_hat = float(mean[arm])
        surprise = _negative_log_predictive_surprise(x_t=x_t, p_hat=p_hat, params=core_params)

        shock_state *= core_params.shock_score_decay
        shock_state[arm] += one_minus_shock * surprise

        surprise_mean[arm] = core_params.lambda_vol * surprise_mean[arm] + one_minus_lambda * surprise
        centered = surprise - surprise_mean[arm]
        surprise_var[arm] = core_params.lambda_vol * surprise_var[arm] + one_minus_lambda * (centered * centered)
        surprise_var[arm] = max(surprise_var[arm], 0.0)

        vol_proxy = np.sqrt(np.maximum(surprise_var, 0.0))
        gamma_vol = _gamma_from_volatility(vol_proxy, core_params)
        n_eff_discount = np.maximum(alpha + beta - 2.0, 0.0)
        if core_params.n0 == 0.0:
            w = (n_eff_discount > 0.0).astype(float)
        else:
            w = n_eff_discount / (n_eff_discount + core_params.n0)
        w = np.clip(w, 0.0, 1.0)
        gamma = (1.0 - w) * gamma_default + w * gamma_vol
        gamma = np.clip(gamma, core_params.gamma_min, core_params.gamma_max)

        alpha = 1.0 + gamma * (alpha - 1.0)
        beta = 1.0 + gamma * (beta - 1.0)
        alpha_long = 1.0 + gamma_long * (alpha_long - 1.0)
        beta_long = 1.0 + gamma_long * (beta_long - 1.0)

        alpha[arm] += reward
        beta[arm] += (1 - reward)
        alpha_long[arm] += reward
        beta_long[arm] += (1 - reward)

    return rewards


def run_VG_dTS_v3(
    mu: np.ndarray,
    params: VGdTSParams,
    rng: np.random.Generator,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    VG-dTS v3: hazard-gated reset TS with dual surprise channels.

    Core mechanisms (frozen structure):
      1) Per-arm hazard-gated posterior reset mix between short-memory posterior
         and reset prior for action sampling.
      2) Mandatory uncertainty bonus with age-aware inflation for stale arms.
      3) Hazard derived from two channels:
         - likelihood surprise: -log P(X_t | p_hat)
         - signed residual drift: (x_t - p_hat)

    Scalar behavior remains configurable through `VGdTSParams`.
    Component toggles used for ablations:
      - `global_shock_enabled`: enable/disable hazard-gated reset mixing
      - `stale_arm_revisit_enabled`: enable/disable age-aware uncertainty inflation
      - `two_timescale_volatility_enabled`: enable/disable signed-drift channel
    """
    _validate_vgdts_params(params)

    k_arms, horizon = mu.shape
    alpha = np.full(k_arms, params.alpha0, dtype=float)
    beta = np.full(k_arms, params.beta0, dtype=float)
    rewards = np.zeros(horizon, dtype=int)

    like_state = np.zeros(k_arms, dtype=float)
    drift_state = np.zeros(k_arms, dtype=float)
    hazard_state = np.zeros(k_arms, dtype=float)
    last_played = np.full(k_arms, -1, dtype=int)

    prior_mass = params.alpha0 + params.beta0
    gamma_default = float(np.clip(params.gamma_default, params.gamma_min, params.gamma_max))
    gamma_reset = float(np.clip(params.global_shock_gamma_floor, params.gamma_min, params.gamma_max))

    like_decay = float(np.clip(params.lambda_vol, 0.0, 1.0))
    drift_decay = float(np.clip(params.lambda_vol_slow, 0.0, 1.0))
    hazard_decay = float(np.clip(params.global_shock_lambda, 0.0, 1.0))

    one_minus_like = 1.0 - like_decay
    one_minus_drift = 1.0 - drift_decay
    one_minus_hazard = 1.0 - hazard_decay

    hazard_slope = float(max(params.logistic_slope, params.eps))
    hazard_mid = float(params.logistic_mid)
    drift_weight = float(max(params.confidence_bonus_coldstart_weight, 0.0))
    vol_span = float(max(params.vol_high - params.vol_low, params.eps))
    hazard_enabled = bool(params.global_shock_enabled)
    age_bonus_enabled = bool(params.stale_arm_revisit_enabled)
    drift_channel_enabled = bool(params.two_timescale_volatility_enabled)

    returns_arr = None
    if returns is not None:
        returns_arr = np.asarray(returns, dtype=float)
        if returns_arr.shape != mu.shape:
            raise ValueError("returns must have shape (K, T) matching mu.")

    for t in range(horizon):
        # Hazard-gated reset mixture for action-time posterior.
        if hazard_enabled:
            alpha_mix = (1.0 - hazard_state) * alpha + hazard_state * params.alpha0
            beta_mix = (1.0 - hazard_state) * beta + hazard_state * params.beta0
        else:
            alpha_mix = alpha
            beta_mix = beta

        sample = rng.beta(alpha_mix, beta_mix)
        mean = alpha_mix / (alpha_mix + beta_mix)
        n_eff = np.maximum(alpha_mix + beta_mix - prior_mass, 0.0)
        posterior_std = np.sqrt(np.maximum(mean * (1.0 - mean), 0.0) / (n_eff + 1.0))

        theta = np.maximum(sample, mean) if params.optimistic else sample

        # Mandatory uncertainty bonus (v3 keeps this always on).
        like_norm = np.clip((like_state - params.vol_low) / vol_span, 0.0, 1.0)
        coldstart = 1.0 / np.sqrt(n_eff + 1.0)
        bonus_multiplier = 1.0 + params.confidence_bonus_vol_weight * like_norm
        bonus_multiplier += params.confidence_bonus_coldstart_weight * coldstart
        bonus = params.confidence_bonus_scale * bonus_multiplier * posterior_std
        bonus = np.clip(bonus, 0.0, params.confidence_bonus_clip)
        theta = theta + bonus

        # Mandatory age-aware uncertainty inflation for stale arms.
        if age_bonus_enabled and params.stale_arm_revisit_scale > 0.0:
            age = np.maximum(t - last_played, 1).astype(float)
            age_ratio = age / (age + params.stale_arm_revisit_half_life)
            age_bonus = params.stale_arm_revisit_scale * np.sqrt(age_ratio) * posterior_std
            age_bonus = np.clip(age_bonus, 0.0, params.stale_arm_revisit_clip)
            theta = theta + age_bonus

        arm = int(np.argmax(theta))
        reward = int(rng.random() < mu[arm, t])
        rewards[t] = reward
        last_played[arm] = t

        x_t = float(reward if returns_arr is None else returns_arr[arm, t])
        p_hat = float(mean[arm])
        like_surprise = _negative_log_predictive_surprise(x_t=x_t, p_hat=p_hat, params=params)
        signed_residual = x_t - p_hat

        # Decay all channel states, then update played arm.
        like_state *= like_decay
        drift_state *= drift_decay
        like_state[arm] += one_minus_like * like_surprise
        if drift_channel_enabled:
            drift_state[arm] += one_minus_drift * signed_residual

        # Dual-channel hazard map.
        like_norm = np.clip((like_state - params.vol_low) / vol_span, 0.0, 1.0)
        if hazard_enabled:
            hazard_input = like_norm
            if drift_channel_enabled:
                drift_signal = np.clip(-drift_state, -1.0, 1.0)  # negative residuals raise hazard.
                hazard_input = hazard_input + drift_weight * drift_signal
            logit = np.clip(hazard_slope * (hazard_input - hazard_mid), -60.0, 60.0)
            hazard_raw = 1.0 / (1.0 + np.exp(-logit))
            hazard_state = hazard_decay * hazard_state + one_minus_hazard * hazard_raw

        # Base volatility-adapted discount before hazard reset pressure.
        gamma_vol = params.gamma_max - like_norm * (params.gamma_max - params.gamma_min)
        gamma_vol = np.clip(gamma_vol, params.gamma_min, params.gamma_max)

        n_eff_short = np.maximum(alpha + beta - prior_mass, 0.0)
        if params.n0 == 0.0:
            w = (n_eff_short > 0.0).astype(float)
        else:
            w = n_eff_short / (n_eff_short + params.n0)
        w = np.clip(w, 0.0, 1.0)
        gamma_base = (1.0 - w) * gamma_default + w * gamma_vol

        if hazard_enabled:
            gamma = (1.0 - hazard_state) * gamma_base + hazard_state * gamma_reset
        else:
            gamma = gamma_base
        gamma = np.clip(gamma, params.gamma_min, params.gamma_max)

        # Prior-preserving discounting with hazard-gated reset pressure.
        alpha = params.alpha0 + gamma * (alpha - params.alpha0)
        beta = params.beta0 + gamma * (beta - params.beta0)
        alpha[arm] += reward
        beta[arm] += (1 - reward)

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
class GlobalCTSParams:
    alpha0: float = 1.0
    beta0: float = 1.0
    hazard: float = 0.02
    max_runlengths: int = 200


@dataclass
class AFFTSParams:
    alpha0: float = 2.0
    beta0: float = 2.0
    eta: float = 0.001
    lambda_init: float = 0.95
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    eps: float = 1e-12


@dataclass
class DLinUCBParams:
    gamma: float = 0.98
    lambda_reg: float = 1.0
    delta: float = 0.05
    sigma: float = 0.5
    action_norm_bound: float = 1.0
    theta_norm_bound: float = 1.0
    jitter: float = 1e-9


@dataclass
class BetaSWTSParams:
    alpha0: float = 1.0
    beta0: float = 1.0
    tau: int = 200


@dataclass
class SWUCBParams:
    tau: int = 200
    xi: float = 0.5


@dataclass
class DUCBParams:
    gamma: float = 0.99
    xi: float = 0.5


@dataclass
class CUSUMUCBParams:
    xi: float = 0.5
    epsilon: float = 0.05
    threshold: float = 8.0
    warmup: int = 50
    random_explore: float = 0.05


@dataclass
class GLRklUCBParams:
    alpha: float = 1.0
    threshold_scale: float = 1.5
    min_segment_len: int = 20
    max_history: int = 0


@dataclass
class AdaSwitchParams:
    xi: float = 0.5
    min_window: int = 16
    max_windows: int = 8
    reset_threshold: float = 0.18
    min_pulls_for_reset: int = 40


@dataclass
class SWTSParams:
    alpha0: float = 1.0
    beta0: float = 1.0
    tau: int = 200


@dataclass
class GammaSWGTSParams:
    alpha0: float = 1.0
    beta0: float = 1.0
    tau: int = 200
    gamma: float = 0.7


@dataclass
class RGPParams:
    lookahead: int = 1
    lengthscale: float = 5.0
    kernel_amplitude: float = 1.0
    noise_std: float = 0.1
    max_sequences: int = 5000
    min_var: float = 1e-9


def _validate_mu(mu: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(mu, dtype=float)
    if arr.ndim != 2:
        raise ValueError("mu must have shape (K, T).")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("mu entries must be in [0, 1].")
    return int(arr.shape[0]), int(arr.shape[1])


def _pull_from_mu(mu: np.ndarray, arm: int, t: int, rng: np.random.Generator) -> int:
    return int(rng.random() < float(mu[arm, t]))


def _kl_bernoulli(p: float, q: float, eps: float = 1e-12) -> float:
    p_clip = float(np.clip(p, eps, 1.0 - eps))
    q_clip = float(np.clip(q, eps, 1.0 - eps))
    return p_clip * math.log(p_clip / q_clip) + (1.0 - p_clip) * math.log((1.0 - p_clip) / (1.0 - q_clip))


def _kl_ucb_bernoulli(mean: float, n: int, bonus: float) -> float:
    if n <= 0:
        return 1.0
    if bonus <= 0.0:
        return float(np.clip(mean, 0.0, 1.0))

    mean_clip = float(np.clip(mean, 0.0, 1.0))
    lo = mean_clip
    hi = 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if n * _kl_bernoulli(mean_clip, mid) <= bonus:
            lo = mid
        else:
            hi = mid
    return float(np.clip(lo, 0.0, 1.0))


def run_global_cts(
    mu: np.ndarray,
    params: GlobalCTSParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Global-CTS from 1302.3721 (particle-truncated runlength mixture).
    """
    k, horizon = _validate_mu(mu)
    if not (0.0 <= params.hazard <= 1.0):
        raise ValueError("hazard must be in [0, 1].")
    if params.max_runlengths <= 0:
        raise ValueError("max_runlengths must be > 0.")
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")

    rewards = np.zeros(horizon, dtype=int)

    weights = np.array([1.0], dtype=float)
    alpha = np.full((1, k), params.alpha0, dtype=float)
    beta = np.full((1, k), params.beta0, dtype=float)

    for t in range(horizon):
        state_idx = int(rng.choice(len(weights), p=weights))
        theta = rng.beta(alpha[state_idx], beta[state_idx])
        arm = int(np.argmax(theta))
        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward

        denom = alpha[:, arm] + beta[:, arm]
        if reward == 1:
            likelihood = alpha[:, arm] / np.maximum(denom, 1e-12)
        else:
            likelihood = beta[:, arm] / np.maximum(denom, 1e-12)

        growth_weights = (1.0 - params.hazard) * weights * likelihood
        cp_weight = params.hazard * float(np.sum(weights * likelihood))
        next_weights = np.concatenate(([cp_weight], growth_weights), axis=0)

        n_states_next = len(next_weights)
        next_alpha = np.empty((n_states_next, k), dtype=float)
        next_beta = np.empty((n_states_next, k), dtype=float)

        next_alpha[0] = params.alpha0
        next_beta[0] = params.beta0
        next_alpha[0, arm] += reward
        next_beta[0, arm] += (1 - reward)

        next_alpha[1:] = alpha
        next_beta[1:] = beta
        next_alpha[1:, arm] += reward
        next_beta[1:, arm] += (1 - reward)

        mass = float(np.sum(next_weights))
        if mass <= 0.0 or not np.isfinite(mass):
            next_weights = np.full(n_states_next, 1.0 / float(n_states_next), dtype=float)
        else:
            next_weights = next_weights / mass

        if n_states_next > params.max_runlengths:
            keep = np.argpartition(next_weights, -params.max_runlengths)[-params.max_runlengths:]
            keep = keep[np.argsort(next_weights[keep])[::-1]]
            next_weights = next_weights[keep]
            next_alpha = next_alpha[keep]
            next_beta = next_beta[keep]
            next_weights = next_weights / np.sum(next_weights)

        weights = next_weights
        alpha = next_alpha
        beta = next_beta

    return rewards


def run_aff_ts(
    mu: np.ndarray,
    params: AFFTSParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    AFF-TS from 1712.03134 using adaptive forgetting-factor moments.
    """
    k, horizon = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if params.eta <= 0.0:
        raise ValueError("eta must be > 0.")
    if not (0.0 <= params.lambda_min <= params.lambda_max <= 1.0):
        raise ValueError("Require 0 <= lambda_min <= lambda_max <= 1.")
    if params.eps <= 0.0:
        raise ValueError("eps must be > 0.")

    rewards = np.zeros(horizon, dtype=int)

    lam = np.full(k, np.clip(params.lambda_init, params.lambda_min, params.lambda_max), dtype=float)
    m = np.zeros(k, dtype=float)
    w = np.zeros(k, dtype=float)
    m_dot = np.zeros(k, dtype=float)
    w_dot = np.zeros(k, dtype=float)
    last_selected = np.full(k, -1, dtype=int)

    init_rounds = min(k, horizon)
    for t in range(init_rounds):
        arm = t
        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward
        m[arm] = float(reward)
        w[arm] = 1.0
        m_dot[arm] = 0.0
        w_dot[arm] = 0.0
        last_selected[arm] = t

    for t in range(init_rounds, horizon):
        age = t - last_selected
        age = np.maximum(age, 0)
        decay = np.power(np.clip(lam, params.lambda_min, params.lambda_max), age * k, dtype=float)
        m_tilde = decay * m
        w_tilde = decay * w

        alpha = params.alpha0 + np.maximum(m_tilde, 0.0)
        beta = params.beta0 + np.maximum(w_tilde - m_tilde, 0.0)

        samples = rng.beta(alpha, beta)
        arm = int(np.argmax(samples))
        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward

        prev_m = m[arm]
        prev_w = max(w[arm], params.eps)
        prev_mean = prev_m / prev_w
        lam_old = lam[arm]

        m_new = lam_old * prev_m + reward
        w_new = lam_old * prev_w + 1.0
        mean_new = m_new / max(w_new, params.eps)

        grad_term = (m_dot[arm] - w_dot[arm] * prev_mean) / prev_w
        gradient = 2.0 * (prev_mean - mean_new) * grad_term

        m_dot_new = lam_old * m_dot[arm] + prev_m
        w_dot_new = lam_old * w_dot[arm] + prev_w
        lam_new = np.clip(lam_old - params.eta * gradient, params.lambda_min, params.lambda_max)

        m[arm] = m_new
        w[arm] = w_new
        m_dot[arm] = m_dot_new
        w_dot[arm] = w_dot_new
        lam[arm] = lam_new
        last_selected[arm] = t

    return rewards


def run_dlinucb_onehot(
    mu: np.ndarray,
    params: DLinUCBParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    D-LinUCB (1909.09146) with one-hot arm features for K-armed Bernoulli setup.
    """
    k, horizon = _validate_mu(mu)
    if not (0.0 < params.gamma < 1.0):
        raise ValueError("gamma must be in (0, 1).")
    if params.lambda_reg <= 0.0:
        raise ValueError("lambda_reg must be > 0.")
    if not (0.0 < params.delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    if params.sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if params.action_norm_bound <= 0.0 or params.theta_norm_bound <= 0.0:
        raise ValueError("action_norm_bound and theta_norm_bound must be > 0.")
    if params.jitter <= 0.0:
        raise ValueError("jitter must be > 0.")

    rewards = np.zeros(horizon, dtype=int)

    eye = np.eye(k, dtype=float)
    v = params.lambda_reg * eye.copy()
    v_tilde = params.lambda_reg * eye.copy()
    b = np.zeros(k, dtype=float)
    theta_hat = np.zeros(k, dtype=float)

    denom_gamma = max(1.0 - params.gamma**2, 1e-12)
    for t in range(horizon):
        inv_v = np.linalg.inv(v + params.jitter * eye)
        radius_term = 1.0 + (
            (params.action_norm_bound**2) * (1.0 - params.gamma ** (2 * max(t, 1)))
        ) / (params.lambda_reg * k * denom_gamma)
        beta_t = np.sqrt(params.lambda_reg) * params.theta_norm_bound
        beta_t += params.sigma * np.sqrt(
            2.0 * np.log(1.0 / params.delta) + k * np.log(max(radius_term, 1.0))
        )

        conf_mat = inv_v @ v_tilde @ inv_v
        conf = np.sqrt(np.clip(np.diag(conf_mat), 0.0, None))
        scores = theta_hat + beta_t * conf
        arm = int(np.argmax(scores))

        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward

        x = eye[arm]
        outer = np.outer(x, x)
        v = params.gamma * v + outer + (1.0 - params.gamma) * params.lambda_reg * eye
        v_tilde = (
            (params.gamma**2) * v_tilde
            + outer
            + (1.0 - params.gamma**2) * params.lambda_reg * eye
        )
        b = params.gamma * b + reward * x
        theta_hat = np.linalg.solve(v + params.jitter * eye, b)

    return rewards


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


def run_sw_ts(
    mu: np.ndarray,
    params: SWTSParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Classical Sliding-Window Thompson Sampling for Bernoulli rewards.
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


def run_gamma_swgts(
    mu: np.ndarray,
    params: GammaSWGTSParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Gamma-SWGTS: generalized sliding-window TS with power-likelihood scaling.
    """
    k, horizon = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if params.tau <= 0:
        raise ValueError("tau must be > 0.")
    if params.gamma <= 0.0:
        raise ValueError("gamma must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    pulls = np.zeros(k, dtype=int)
    successes = np.zeros(k, dtype=float)
    history: list[tuple[int, int]] = []

    for t in range(horizon):
        failures = np.maximum(pulls.astype(float) - successes, 0.0)
        alpha = params.alpha0 + params.gamma * successes
        beta = params.beta0 + params.gamma * failures
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


def run_sw_ucb(
    mu: np.ndarray,
    params: SWUCBParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sliding-Window UCB baseline for bounded rewards in [0,1].
    """
    k, horizon = _validate_mu(mu)
    if params.tau <= 0:
        raise ValueError("tau must be > 0.")
    if params.xi <= 0.0:
        raise ValueError("xi must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    pulls = np.zeros(k, dtype=float)
    successes = np.zeros(k, dtype=float)
    history: list[tuple[int, int]] = []

    for t in range(horizon):
        if np.any(pulls <= 0.0):
            arm = int(np.where(pulls <= 0.0)[0][0])
        else:
            means = successes / np.maximum(pulls, 1e-12)
            log_term = math.log(max(2.0, float(min(t + 1, params.tau))))
            bonus = np.sqrt(params.xi * log_term / np.maximum(pulls, 1e-12))
            arm = int(np.argmax(means + bonus))

        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward

        pulls[arm] += 1.0
        successes[arm] += float(reward)
        history.append((arm, reward))

        if len(history) > params.tau:
            old_arm, old_reward = history.pop(0)
            pulls[old_arm] -= 1.0
            successes[old_arm] -= float(old_reward)

    return rewards


def run_d_ucb(
    mu: np.ndarray,
    params: DUCBParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Discounted UCB baseline with exponentially discounted sufficient statistics.
    """
    k, horizon = _validate_mu(mu)
    if not (0.0 < params.gamma < 1.0):
        raise ValueError("gamma must be in (0, 1).")
    if params.xi <= 0.0:
        raise ValueError("xi must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    discounted_pulls = np.zeros(k, dtype=float)
    discounted_successes = np.zeros(k, dtype=float)
    n_gamma = 0.0

    for t in range(horizon):
        if np.any(discounted_pulls <= 1e-12):
            arm = int(np.where(discounted_pulls <= 1e-12)[0][0])
        else:
            means = discounted_successes / np.maximum(discounted_pulls, 1e-12)
            log_term = math.log(max(2.0, n_gamma))
            bonus = np.sqrt(params.xi * log_term / np.maximum(discounted_pulls, 1e-12))
            arm = int(np.argmax(means + bonus))

        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward

        discounted_pulls *= params.gamma
        discounted_successes *= params.gamma
        discounted_pulls[arm] += 1.0
        discounted_successes[arm] += float(reward)
        n_gamma = params.gamma * n_gamma + 1.0

    return rewards


def run_cusum_ucb(
    mu: np.ndarray,
    params: CUSUMUCBParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    CUSUM-UCB style policy with arm-wise CUSUM change detectors and local restarts.
    """
    k, horizon = _validate_mu(mu)
    if params.xi <= 0.0:
        raise ValueError("xi must be > 0.")
    if params.epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")
    if params.threshold <= 0.0:
        raise ValueError("threshold must be > 0.")
    if params.warmup <= 1:
        raise ValueError("warmup must be > 1.")
    if not (0.0 <= params.random_explore <= 1.0):
        raise ValueError("random_explore must be in [0, 1].")

    rewards = np.zeros(horizon, dtype=int)
    pulls = np.zeros(k, dtype=float)
    successes = np.zeros(k, dtype=float)
    warmup_count = np.zeros(k, dtype=int)
    warmup_sum = np.zeros(k, dtype=float)
    ref_mean = np.zeros(k, dtype=float)
    g_plus = np.zeros(k, dtype=float)
    g_minus = np.zeros(k, dtype=float)

    for t in range(horizon):
        if np.any(pulls <= 0.0):
            arm = int(np.where(pulls <= 0.0)[0][0])
        elif rng.random() < params.random_explore:
            arm = int(rng.integers(0, k))
        else:
            means = successes / np.maximum(pulls, 1e-12)
            log_term = math.log(max(2.0, float(t + 1)))
            bonus = np.sqrt(params.xi * log_term / np.maximum(pulls, 1e-12))
            arm = int(np.argmax(means + bonus))

        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward

        pulls[arm] += 1.0
        successes[arm] += float(reward)

        if warmup_count[arm] < params.warmup:
            warmup_count[arm] += 1
            warmup_sum[arm] += float(reward)
            ref_mean[arm] = warmup_sum[arm] / float(warmup_count[arm])
            continue

        drift_pos = float(reward) - ref_mean[arm] - params.epsilon
        drift_neg = ref_mean[arm] - float(reward) - params.epsilon
        g_plus[arm] = max(0.0, g_plus[arm] + drift_pos)
        g_minus[arm] = max(0.0, g_minus[arm] + drift_neg)

        if max(g_plus[arm], g_minus[arm]) > params.threshold:
            pulls[arm] = 1.0
            successes[arm] = float(reward)
            warmup_count[arm] = 1
            warmup_sum[arm] = float(reward)
            ref_mean[arm] = float(reward)
            g_plus[arm] = 0.0
            g_minus[arm] = 0.0

    return rewards


def run_glr_klucb(
    mu: np.ndarray,
    params: GLRklUCBParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    GLR-klUCB style policy: KL-UCB indices with arm-local GLR restart detection.
    """
    k, horizon = _validate_mu(mu)
    if params.alpha <= 0.0:
        raise ValueError("alpha must be > 0.")
    if params.threshold_scale <= 0.0:
        raise ValueError("threshold_scale must be > 0.")
    if params.min_segment_len <= 1:
        raise ValueError("min_segment_len must be > 1.")
    if params.max_history < 0:
        raise ValueError("max_history must be >= 0.")

    rewards = np.zeros(horizon, dtype=int)
    histories: list[list[int]] = [[] for _ in range(k)]

    for t in range(horizon):
        if any(len(hist) == 0 for hist in histories):
            arm = next(i for i, hist in enumerate(histories) if len(hist) == 0)
        else:
            bonus = math.log(max(2.0, float(t + 1))) + params.alpha * math.log(
                max(math.log(max(3.0, float(t + 1))), 1.0)
            )
            scores = np.zeros(k, dtype=float)
            for arm_idx in range(k):
                hist = histories[arm_idx]
                n = len(hist)
                mean = float(np.mean(hist))
                scores[arm_idx] = _kl_ucb_bernoulli(mean=mean, n=n, bonus=bonus)
            arm = int(np.argmax(scores))

        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward
        histories[arm].append(int(reward))

        if params.max_history > 0 and len(histories[arm]) > params.max_history:
            histories[arm] = histories[arm][-params.max_history:]

        hist = histories[arm]
        n = len(hist)
        m = params.min_segment_len
        if n < 2 * m:
            continue

        arr = np.asarray(hist, dtype=float)
        prefix = np.cumsum(arr)
        total = float(prefix[-1])
        pooled = total / float(n)
        threshold = params.threshold_scale * math.log(max(2.0, float(t + 1)))
        best_stat = -np.inf
        best_split = -1

        for split in range(m, n - m + 1):
            left_sum = float(prefix[split - 1])
            right_sum = total - left_sum
            p_left = left_sum / float(split)
            p_right = right_sum / float(n - split)
            stat = split * _kl_bernoulli(p_left, pooled) + (n - split) * _kl_bernoulli(p_right, pooled)
            if stat > best_stat:
                best_stat = stat
                best_split = split

        if best_stat > threshold and best_split > 0:
            histories[arm] = hist[best_split:]

    return rewards


def run_adaswitch(
    mu: np.ndarray,
    params: AdaSwitchParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    AdaSwitch-style adaptive-window UCB using dyadic windows and local memory resets.
    """
    k, horizon = _validate_mu(mu)
    if params.xi <= 0.0:
        raise ValueError("xi must be > 0.")
    if params.min_window <= 1:
        raise ValueError("min_window must be > 1.")
    if params.max_windows <= 0:
        raise ValueError("max_windows must be > 0.")
    if params.reset_threshold <= 0.0:
        raise ValueError("reset_threshold must be > 0.")
    if params.min_pulls_for_reset <= params.min_window:
        raise ValueError("min_pulls_for_reset must exceed min_window.")

    rewards = np.zeros(horizon, dtype=int)
    histories: list[list[int]] = [[] for _ in range(k)]

    for t in range(horizon):
        if any(len(hist) == 0 for hist in histories):
            arm = next(i for i, hist in enumerate(histories) if len(hist) == 0)
        else:
            log_term = math.log(max(2.0, float(t + 1)))
            scores = np.zeros(k, dtype=float)
            for arm_idx in range(k):
                hist = histories[arm_idx]
                arr = np.asarray(hist, dtype=float)
                n = len(hist)
                windows: list[int] = []
                w = params.min_window
                while w < n and len(windows) < params.max_windows - 1:
                    windows.append(w)
                    w *= 2
                windows.append(n)

                best_index = -np.inf
                for window in windows:
                    seg = arr[-window:]
                    mean = float(np.mean(seg))
                    bonus = math.sqrt(params.xi * log_term / float(window))
                    best_index = max(best_index, mean + bonus)
                scores[arm_idx] = best_index
            arm = int(np.argmax(scores))

        reward = _pull_from_mu(mu, arm, t, rng)
        rewards[t] = reward
        histories[arm].append(int(reward))

        hist = histories[arm]
        if len(hist) >= params.min_pulls_for_reset:
            arr = np.asarray(hist, dtype=float)
            short_w = min(params.min_window, len(hist))
            short_mean = float(np.mean(arr[-short_w:]))
            long_mean = float(np.mean(arr))
            if abs(short_mean - long_mean) > params.reset_threshold:
                histories[arm] = hist[-short_w:]

    return rewards


def _sqexp_kernel(
    x: np.ndarray,
    y: np.ndarray,
    lengthscale: float,
    amplitude: float,
) -> np.ndarray:
    xx = np.atleast_1d(x).astype(float)[:, None]
    yy = np.atleast_1d(y).astype(float)[None, :]
    dist2 = (xx - yy) ** 2
    scale = max(lengthscale, 1e-9)
    return (amplitude**2) * np.exp(-0.5 * dist2 / (scale**2))


def _gp_grid_posterior(
    z_grid: np.ndarray,
    z_obs: list[int],
    y_obs: list[float],
    params: RGPParams,
) -> tuple[np.ndarray, np.ndarray]:
    if not z_obs:
        prior_var = np.full(len(z_grid), params.kernel_amplitude**2, dtype=float)
        return np.zeros(len(z_grid), dtype=float), prior_var

    x = np.asarray(z_obs, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    k_xx = _sqexp_kernel(x, x, params.lengthscale, params.kernel_amplitude)
    k_xx = k_xx + (params.noise_std**2) * np.eye(len(x), dtype=float)
    k_xs = _sqexp_kernel(x, z_grid, params.lengthscale, params.kernel_amplitude)

    # Solve K^{-1} y and K^{-1} K_xs in one factorization.
    alpha = np.linalg.solve(k_xx, y)
    mean = k_xs.T @ alpha

    v = np.linalg.solve(k_xx, k_xs)
    k_ss_diag = np.diag(_sqexp_kernel(z_grid, z_grid, params.lengthscale, params.kernel_amplitude))
    var = k_ss_diag - np.sum(k_xs * v, axis=0)
    var = np.maximum(var, params.min_var)
    return mean, var


def _enumerate_sequences(
    n_arms: int,
    depth: int,
    max_sequences: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    total = n_arms**depth
    if total <= max_sequences:
        grid = np.indices((n_arms,) * depth).reshape(depth, -1).T
        for row in grid:
            yield row.astype(int)
        return

    for _ in range(max_sequences):
        yield rng.integers(0, n_arms, size=depth, dtype=int)


def _choose_sequence_ucb(
    delays: np.ndarray,
    gp_means: np.ndarray,
    gp_vars: np.ndarray,
    depth: int,
    alpha_t: float,
    max_sequences: int,
    rng: np.random.Generator,
) -> np.ndarray:
    best_seq: np.ndarray | None = None
    best_score = -np.inf
    z_max = gp_means.shape[1] - 1

    for seq in _enumerate_sequences(delays.shape[0], depth, max_sequences, rng):
        d = delays.copy()
        mean_sum = 0.0
        var_sum = 0.0
        for arm in seq:
            z = int(d[arm])
            mean_sum += float(gp_means[arm, z])
            var_sum += float(gp_vars[arm, z])
            d = np.minimum(d + 1, z_max)
            d[arm] = 0
        score = mean_sum + alpha_t * np.sqrt(max(var_sum, 0.0))
        if score > best_score:
            best_score = score
            best_seq = seq.copy()

    if best_seq is None:
        raise RuntimeError("Failed to choose a sequence.")
    return best_seq


def _choose_sequence_ts(
    delays: np.ndarray,
    gp_means: np.ndarray,
    gp_vars: np.ndarray,
    depth: int,
    max_sequences: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = rng.normal(gp_means, np.sqrt(np.maximum(gp_vars, 1e-12)))
    best_seq: np.ndarray | None = None
    best_score = -np.inf
    z_max = gp_means.shape[1] - 1

    for seq in _enumerate_sequences(delays.shape[0], depth, max_sequences, rng):
        d = delays.copy()
        score = 0.0
        for arm in seq:
            z = int(d[arm])
            score += float(sampled[arm, z])
            d = np.minimum(d + 1, z_max)
            d[arm] = 0
        if score > best_score:
            best_score = score
            best_seq = seq.copy()

    if best_seq is None:
        raise RuntimeError("Failed to choose a sequence.")
    return best_seq


def run_drpg_ucb(
    reward_table: np.ndarray,
    horizon: int,
    params: RGPParams,
    rng: np.random.Generator,
    env_seed: int | None = None,
) -> np.ndarray:
    """
    dRGP-UCB from 1910.14354 on a finite-z recovering environment.
    """
    if params.lookahead <= 0:
        raise ValueError("lookahead must be > 0.")
    if params.lengthscale <= 0.0 or params.kernel_amplitude <= 0.0:
        raise ValueError("lengthscale and kernel_amplitude must be > 0.")
    if params.noise_std <= 0.0:
        raise ValueError("noise_std must be > 0.")
    if params.max_sequences <= 0:
        raise ValueError("max_sequences must be > 0.")
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")

    env = RecoveringBanditEnv(reward_table=reward_table, seed=env_seed)
    n_arms = env.n_arms
    z_grid = np.arange(env.z_max + 1, dtype=float)

    obs_z: list[list[int]] = [[] for _ in range(n_arms)]
    obs_y: list[list[float]] = [[] for _ in range(n_arms)]
    rewards = np.zeros(horizon, dtype=int)

    t = 0
    while t < horizon:
        depth = min(params.lookahead, horizon - t)
        gp_means = np.zeros((n_arms, env.z_max + 1), dtype=float)
        gp_vars = np.zeros((n_arms, env.z_max + 1), dtype=float)
        for arm in range(n_arms):
            mean, var = _gp_grid_posterior(z_grid, obs_z[arm], obs_y[arm], params)
            gp_means[arm] = mean
            gp_vars[arm] = var

        alpha_t = np.sqrt(2.0 * np.log((max(2, n_arms * (env.z_max + 1))) ** depth * (t + depth + 1) ** 2))
        sequence = _choose_sequence_ucb(
            delays=env.delays,
            gp_means=gp_means,
            gp_vars=gp_vars,
            depth=depth,
            alpha_t=alpha_t,
            max_sequences=params.max_sequences,
            rng=rng,
        )

        for arm in sequence:
            reward, delay_before, _ = env.step(int(arm))
            rewards[t] = reward
            obs_z[int(arm)].append(delay_before)
            obs_y[int(arm)].append(float(reward))
            t += 1

    return rewards


def run_drpg_ts(
    reward_table: np.ndarray,
    horizon: int,
    params: RGPParams,
    rng: np.random.Generator,
    env_seed: int | None = None,
) -> np.ndarray:
    """
    dRGP-TS from 1910.14354 on a finite-z recovering environment.
    """
    if params.lookahead <= 0:
        raise ValueError("lookahead must be > 0.")
    if params.lengthscale <= 0.0 or params.kernel_amplitude <= 0.0:
        raise ValueError("lengthscale and kernel_amplitude must be > 0.")
    if params.noise_std <= 0.0:
        raise ValueError("noise_std must be > 0.")
    if params.max_sequences <= 0:
        raise ValueError("max_sequences must be > 0.")
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")

    env = RecoveringBanditEnv(reward_table=reward_table, seed=env_seed)
    n_arms = env.n_arms
    z_grid = np.arange(env.z_max + 1, dtype=float)

    obs_z: list[list[int]] = [[] for _ in range(n_arms)]
    obs_y: list[list[float]] = [[] for _ in range(n_arms)]
    rewards = np.zeros(horizon, dtype=int)

    t = 0
    while t < horizon:
        depth = min(params.lookahead, horizon - t)
        gp_means = np.zeros((n_arms, env.z_max + 1), dtype=float)
        gp_vars = np.zeros((n_arms, env.z_max + 1), dtype=float)
        for arm in range(n_arms):
            mean, var = _gp_grid_posterior(z_grid, obs_z[arm], obs_y[arm], params)
            gp_means[arm] = mean
            gp_vars[arm] = var

        sequence = _choose_sequence_ts(
            delays=env.delays,
            gp_means=gp_means,
            gp_vars=gp_vars,
            depth=depth,
            max_sequences=params.max_sequences,
            rng=rng,
        )

        for arm in sequence:
            reward, delay_before, _ = env.step(int(arm))
            rewards[t] = reward
            obs_z[int(arm)].append(delay_before)
            obs_y[int(arm)].append(float(reward))
            t += 1

    return rewards


def run_vgdts_recovering(
    reward_table: np.ndarray,
    horizon: int,
    params: VGdTSParams,
    rng: np.random.Generator,
    env_seed: int | None = None,
) -> np.ndarray:
    """
    Interactive VG-dTS runner for recovering bandits (action-dependent means).
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")

    env = RecoveringBanditEnv(reward_table=reward_table, seed=env_seed)
    k = env.n_arms

    alpha = np.full(k, params.alpha0, dtype=float)
    beta = np.full(k, params.beta0, dtype=float)
    surprise_mean = np.full(k, params.mean0, dtype=float)
    surprise_var = np.full(k, params.var0, dtype=float)
    rewards = np.zeros(horizon, dtype=int)

    gamma_default = float(np.clip(params.gamma_default, params.gamma_min, params.gamma_max))
    one_minus_lambda = 1.0 - params.lambda_vol

    for t in range(horizon):
        sample = rng.beta(alpha, beta)
        mean = alpha / (alpha + beta)
        score = np.maximum(sample, mean) if params.optimistic else sample
        arm = int(np.argmax(score))

        reward, _, _ = env.step(arm)
        rewards[t] = reward

        p_hat = float(mean[arm])
        surprise = _standardized_surprise(x_t=float(reward), p_hat=p_hat, params=params)

        surprise_mean[arm] = params.lambda_vol * surprise_mean[arm] + one_minus_lambda * surprise
        centered = surprise - surprise_mean[arm]
        surprise_var[arm] = params.lambda_vol * surprise_var[arm] + one_minus_lambda * (centered * centered)
        surprise_var[arm] = max(surprise_var[arm], 0.0)

        vol = np.sqrt(np.maximum(surprise_var, 0.0))
        if params.gamma_mapping == "logistic":
            x = params.logistic_slope * (vol - params.logistic_mid)
            x = np.clip(x, -60.0, 60.0)
            decay = 1.0 / (1.0 + np.exp(x))
            gamma_vol = params.gamma_min + decay * (params.gamma_max - params.gamma_min)
        else:
            span = max(params.vol_high - params.vol_low, params.eps)
            norm = np.clip((vol - params.vol_low) / span, 0.0, 1.0)
            gamma_vol = params.gamma_max - norm * (params.gamma_max - params.gamma_min)
        gamma_vol = np.clip(gamma_vol, params.gamma_min, params.gamma_max)

        n_eff = np.maximum(alpha + beta - 2.0, 0.0)
        if params.n0 == 0.0:
            w = (n_eff > 0.0).astype(float)
        else:
            w = n_eff / (n_eff + params.n0)
        w = np.clip(w, 0.0, 1.0)
        gamma = (1.0 - w) * gamma_default + w * gamma_vol
        gamma = np.clip(gamma, params.gamma_min, params.gamma_max)

        alpha = 1.0 + gamma * (alpha - 1.0)
        beta = 1.0 + gamma * (beta - 1.0)
        alpha[arm] += reward
        beta[arm] += (1 - reward)

    return rewards
