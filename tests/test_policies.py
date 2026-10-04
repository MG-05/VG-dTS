from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.adts.policies import (
    VGdTSParams,
    _standardized_surprise,
    _variance_gated_confidence_bonus,
    run_VG_dTS,
    run_VG_dTS_v2,
    run_VG_dTS_v21,
    run_VG_dTS_v3,
)
from src.adts.vgdts_config import make_benchmark_vgdts_v21_params


def _reference_vgdts_original(
    mu: np.ndarray,
    params: VGdTSParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Original VG-dTS update path before adding extra gating features."""
    k_arms, horizon = mu.shape
    alpha = np.full(k_arms, params.alpha0, dtype=float)
    beta = np.full(k_arms, params.beta0, dtype=float)
    surprise_mean = np.full(k_arms, params.mean0, dtype=float)
    surprise_var = np.full(k_arms, params.var0, dtype=float)
    rewards = np.zeros(horizon, dtype=int)

    gamma_default = float(np.clip(params.gamma_default, params.gamma_min, params.gamma_max))
    one_minus_lambda = 1.0 - params.lambda_vol

    for t in range(horizon):
        sample = rng.beta(alpha, beta)
        mean = alpha / (alpha + beta)
        theta = np.maximum(sample, mean) if params.optimistic else sample
        arm = int(np.argmax(theta))

        reward = int(rng.random() < mu[arm, t])
        rewards[t] = reward

        x_t = float(reward)
        p_hat = float(mean[arm])
        denom = np.sqrt(max(p_hat * (1.0 - p_hat), params.eps))
        surprise = abs(x_t - p_hat) / denom
        surprise = float(np.clip(surprise, 0.0, params.surprise_clip))

        surprise_mean[arm] = params.lambda_vol * surprise_mean[arm] + one_minus_lambda * surprise
        centered = surprise - surprise_mean[arm]
        surprise_var[arm] = params.lambda_vol * surprise_var[arm] + one_minus_lambda * (centered * centered)
        surprise_var[arm] = max(surprise_var[arm], 0.0)

        vol_proxy = np.sqrt(np.maximum(surprise_var, 0.0))

        if params.gamma_mapping == "inverse_linear":
            span = max(params.vol_high - params.vol_low, params.eps)
            norm = np.clip((vol_proxy - params.vol_low) / span, 0.0, 1.0)
            gamma_vol = params.gamma_max - norm * (params.gamma_max - params.gamma_min)
            gamma_vol = np.clip(gamma_vol, params.gamma_min, params.gamma_max)
        else:
            x = params.logistic_slope * (vol_proxy - params.logistic_mid)
            x = np.clip(x, -60.0, 60.0)
            decay = 1.0 / (1.0 + np.exp(x))
            gamma_vol = params.gamma_min + decay * (params.gamma_max - params.gamma_min)
            gamma_vol = np.clip(gamma_vol, params.gamma_min, params.gamma_max)

        n_eff = np.maximum(alpha + beta - 2.0, 0.0)
        w = n_eff / (n_eff + params.n0) if params.n0 > 0.0 else (n_eff > 0.0).astype(float)
        w = np.clip(w, 0.0, 1.0)
        gamma = (1.0 - w) * gamma_default + w * gamma_vol
        gamma = np.clip(gamma, params.gamma_min, params.gamma_max)

        alpha = 1.0 + gamma * (alpha - 1.0)
        beta = 1.0 + gamma * (beta - 1.0)

        alpha[arm] += reward
        beta[arm] += (1 - reward)

    return rewards


def test_vgdts_matches_original_update_path() -> None:
    rng = np.random.default_rng(123)
    mu = rng.uniform(0.1, 0.9, size=(4, 300))

    params = VGdTSParams(
        optimistic=True,
        gamma_mapping="inverse_linear",
    )

    rewards_ref = _reference_vgdts_original(mu, params, np.random.default_rng(7))
    rewards_new = run_VG_dTS(mu, params, np.random.default_rng(7))
    assert np.array_equal(rewards_ref, rewards_new)


def test_vgdts_runs_and_emits_binary_rewards() -> None:
    rng = np.random.default_rng(321)
    mu = rng.uniform(0.05, 0.95, size=(4, 250))

    params = VGdTSParams(
        optimistic=True,
        gamma_mapping="inverse_linear",
    )

    rewards = run_VG_dTS(mu, params, np.random.default_rng(11))
    assert rewards.shape == (mu.shape[1],)
    assert set(np.unique(rewards)).issubset({0, 1})


def test_vgdts_rejects_invalid_inverse_linear_bounds() -> None:
    mu = np.full((3, 40), 0.5, dtype=float)
    params = VGdTSParams(gamma_mapping="inverse_linear", vol_low=1.0, vol_high=0.5)

    with pytest.raises(ValueError, match="vol_high must be > vol_low"):
        run_VG_dTS(mu, params, np.random.default_rng(0))


def test_vgdts_rejects_invalid_logistic_slope() -> None:
    mu = np.full((3, 40), 0.5, dtype=float)
    params = VGdTSParams(gamma_mapping="logistic", logistic_slope=0.0)

    with pytest.raises(ValueError, match="logistic_slope must be > 0"):
        run_VG_dTS(mu, params, np.random.default_rng(0))


def test_vgdts_rejects_negative_confidence_bonus_scale() -> None:
    mu = np.full((3, 40), 0.5, dtype=float)
    params = VGdTSParams(confidence_bonus_enabled=True, confidence_bonus_scale=-0.1)

    with pytest.raises(ValueError, match="confidence_bonus_scale must be >= 0"):
        run_VG_dTS(mu, params, np.random.default_rng(0))


def test_standardized_surprise_one_sided_ignores_positive_errors() -> None:
    params_two_sided = VGdTSParams(one_sided_negative_surprise=False, surprise_clip=50.0)
    params_one_sided = VGdTSParams(one_sided_negative_surprise=True, surprise_clip=50.0)

    x_t = 1.0
    p_hat = 0.2

    s_two_sided = _standardized_surprise(x_t=x_t, p_hat=p_hat, params=params_two_sided)
    s_one_sided = _standardized_surprise(x_t=x_t, p_hat=p_hat, params=params_one_sided)

    assert s_two_sided > 0.0
    assert s_one_sided == 0.0


def test_standardized_surprise_one_sided_keeps_negative_errors() -> None:
    params_two_sided = VGdTSParams(one_sided_negative_surprise=False, surprise_clip=50.0)
    params_one_sided = VGdTSParams(one_sided_negative_surprise=True, surprise_clip=50.0)

    x_t = 0.0
    p_hat = 0.8

    s_two_sided = _standardized_surprise(x_t=x_t, p_hat=p_hat, params=params_two_sided)
    s_one_sided = _standardized_surprise(x_t=x_t, p_hat=p_hat, params=params_one_sided)

    assert s_two_sided > 0.0
    assert np.isclose(s_two_sided, s_one_sided)


def test_vgdts_one_sided_surprise_changes_trajectory() -> None:
    rng = np.random.default_rng(222)
    mu = rng.uniform(0.05, 0.95, size=(4, 400))

    params_two_sided = VGdTSParams(
        optimistic=True,
        gamma_mapping="inverse_linear",
        one_sided_negative_surprise=False,
    )
    params_one_sided = VGdTSParams(
        optimistic=True,
        gamma_mapping="inverse_linear",
        one_sided_negative_surprise=True,
    )

    rewards_two_sided = run_VG_dTS(mu, params_two_sided, np.random.default_rng(13))
    rewards_one_sided = run_VG_dTS(mu, params_one_sided, np.random.default_rng(13))
    assert not np.array_equal(rewards_two_sided, rewards_one_sided)


def test_variance_gated_confidence_bonus_increases_with_volatility_and_low_counts() -> None:
    params = VGdTSParams(
        confidence_bonus_enabled=True,
        confidence_bonus_scale=0.6,
        confidence_bonus_vol_weight=0.8,
        confidence_bonus_coldstart_weight=0.4,
        confidence_bonus_clip=1.0,
    )
    posterior_std = np.array([0.2, 0.2, 0.2], dtype=float)
    vol_norm = np.array([0.0, 1.0, 0.0], dtype=float)
    n_eff = np.array([25.0, 25.0, 0.0], dtype=float)

    bonus = _variance_gated_confidence_bonus(
        posterior_std=posterior_std,
        vol_norm=vol_norm,
        n_eff=n_eff,
        params=params,
    )

    assert bonus[1] > bonus[0]
    assert bonus[2] > bonus[0]


def test_vgdts_confidence_bonus_changes_trajectory() -> None:
    rng = np.random.default_rng(19)
    mu = rng.uniform(0.05, 0.95, size=(4, 400))

    params_base = VGdTSParams(
        optimistic=True,
        gamma_mapping="inverse_linear",
        dual_memory_enabled=True,
        surprise_mode="neg_log_likelihood",
    )
    params_bonus = VGdTSParams(
        optimistic=True,
        gamma_mapping="inverse_linear",
        dual_memory_enabled=True,
        surprise_mode="neg_log_likelihood",
        confidence_bonus_enabled=True,
        confidence_bonus_scale=0.6,
        confidence_bonus_vol_weight=0.8,
        confidence_bonus_coldstart_weight=0.4,
        confidence_bonus_clip=0.75,
    )

    rewards_base = run_VG_dTS(mu, params_base, np.random.default_rng(23))
    rewards_bonus = run_VG_dTS(mu, params_bonus, np.random.default_rng(23))
    assert not np.array_equal(rewards_base, rewards_bonus)


def test_vgdts_v2_enforces_frozen_mechanism_set() -> None:
    rng = np.random.default_rng(41)
    mu = rng.uniform(0.05, 0.95, size=(4, 300))

    params = VGdTSParams(
        dual_memory_enabled=False,
        surprise_mode="standardized",
        confidence_bonus_enabled=False,
    )
    rewards_v2 = run_VG_dTS_v2(mu, params, np.random.default_rng(29))

    expected_v2_params = replace(
        params,
        dual_memory_enabled=True,
        surprise_mode="neg_log_likelihood",
        confidence_bonus_enabled=True,
    )
    rewards_expected = run_VG_dTS(mu, expected_v2_params, np.random.default_rng(29))
    assert np.array_equal(rewards_v2, rewards_expected)


def test_vgdts_v21_matches_v2_when_addons_disabled() -> None:
    rng = np.random.default_rng(61)
    mu = rng.uniform(0.05, 0.95, size=(4, 300))
    params = VGdTSParams()

    rewards_v2 = run_VG_dTS_v2(mu, params, np.random.default_rng(37))
    rewards_v21 = run_VG_dTS_v21(mu, params, np.random.default_rng(37))
    assert np.array_equal(rewards_v2, rewards_v21)


def test_vgdts_v21_benchmark_preset_is_shock_score_only() -> None:
    params = make_benchmark_vgdts_v21_params()

    assert params.shock_score_enabled is True
    assert params.shock_n_eff_enabled is False
    assert params.selective_revisit_enabled is False


def test_vgdts_v21_addon_changes_trajectory() -> None:
    rng = np.random.default_rng(62)
    mu = rng.uniform(0.05, 0.95, size=(4, 300))

    params_base = VGdTSParams()
    params_full = VGdTSParams(
        shock_score_enabled=True,
        shock_n_eff_enabled=True,
        selective_revisit_enabled=True,
    )

    rewards_base = run_VG_dTS_v21(mu, params_base, np.random.default_rng(41))
    rewards_full = run_VG_dTS_v21(mu, params_full, np.random.default_rng(41))
    assert not np.array_equal(rewards_base, rewards_full)


def test_vgdts_v3_component_toggles_change_trajectory() -> None:
    rng = np.random.default_rng(52)
    mu = rng.uniform(0.05, 0.95, size=(4, 300))

    params_ablation = VGdTSParams(
        stale_arm_revisit_enabled=False,
        global_shock_enabled=False,
        two_timescale_volatility_enabled=False,
    )
    params_full = VGdTSParams(
        stale_arm_revisit_enabled=True,
        global_shock_enabled=True,
        two_timescale_volatility_enabled=True,
    )

    rewards_ablation = run_VG_dTS_v3(mu, params_ablation, np.random.default_rng(31))
    rewards_full = run_VG_dTS_v3(mu, params_full, np.random.default_rng(31))
    assert not np.array_equal(rewards_ablation, rewards_full)
