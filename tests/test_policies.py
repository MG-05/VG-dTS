import numpy as np
import pytest
from src.adts.policies import VGdTSParams, _standardized_surprise, run_VG_dTS


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
