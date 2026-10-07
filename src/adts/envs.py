from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class BernoulliNonStationaryEnv:
    """Bernoulli bandit with time-varying mean rewards."""

    mean_rewards: FloatArray
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        means = np.asarray(self.mean_rewards, dtype=float)
        if means.ndim != 2:
            raise ValueError("mean_rewards must have shape (horizon, n_arms).")
        if np.any(means < 0.0) or np.any(means > 1.0):
            raise ValueError("All Bernoulli means must be in [0, 1].")
        self.mean_rewards = means
        self.rng = np.random.default_rng(self.seed)

    @property
    def horizon(self) -> int:
        return int(self.mean_rewards.shape[0])

    @property
    def n_arms(self) -> int:
        return int(self.mean_rewards.shape[1])

    def means_at(self, t: int) -> FloatArray:
        return self.mean_rewards[t]

    def sample(self, arm: int, t: int) -> int:
        return int(self.rng.random() < self.mean_rewards[t, arm])


def _validate_horizon_n_arms(
    horizon: int,
    n_arms: int,
    *,
    min_arms: int,
) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_arms < min_arms:
        raise ValueError(f"n_arms must be >= {min_arms}.")


def _validate_prob(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")


def slow_varying_sinusoid_means(
    horizon: int = 5000,
    n_arms: int = 4,
    period: int = 1000,
    offsets: FloatArray | None = None,
) -> FloatArray:
    """
    Slow-varying sinusoidal means used in the paper's non-stationary setup.
    Mean rewards are constrained to [0, 1] via (sin(.) + 1) / 2.
    """

    _validate_horizon_n_arms(horizon, n_arms, min_arms=2)
    if period <= 0:
        raise ValueError("period must be > 0.")

    if offsets is None:
        offsets = np.linspace(0.0, 2.0 * np.pi, num=n_arms, endpoint=False)
    else:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (n_arms,):
            raise ValueError("offsets must have shape (n_arms,).")

    t = np.arange(horizon, dtype=float)
    phase = (2.0 * np.pi * t[:, None] / float(period)) + offsets[None, :]
    means = 0.5 + 0.5 * np.sin(phase)
    return np.clip(means, 0.0, 1.0)


def mixed_regime_challenge_means(
    horizon: int = 5000,
    n_arms: int = 4,
    stable_mean: float = 0.65,
    drift_start: float = 0.20,
    drift_end: float = 0.85,
    burst_base: float = 0.10,
    burst_peak: float = 0.95,
    burst_prob: float = 0.04,
    burst_duration: int = 20,
    switch_low: float = 0.15,
    switch_high: float = 0.85,
    switch_interval: int = 200,
    seed: int | None = None,
) -> FloatArray:
    """
    Strong heterogeneous benchmark:
      arm 0: stable mean
      arm 1: drifting mean
      arm 2: bursty mean (rare spikes)
      arm 3: abruptly switching mean (piecewise constant)

    If n_arms > 4, extra arms are randomized drifting sinusoids.
    """

    _validate_horizon_n_arms(horizon, n_arms, min_arms=4)
    for name, value in (
        ("stable_mean", stable_mean),
        ("drift_start", drift_start),
        ("drift_end", drift_end),
        ("burst_base", burst_base),
        ("burst_peak", burst_peak),
        ("switch_low", switch_low),
        ("switch_high", switch_high),
    ):
        _validate_prob(name, float(value))

    if not (0.0 <= burst_prob <= 1.0):
        raise ValueError("burst_prob must be in [0, 1].")
    if burst_duration <= 0:
        raise ValueError("burst_duration must be > 0.")
    if switch_interval <= 0:
        raise ValueError("switch_interval must be > 0.")

    rng = np.random.default_rng(seed)
    means = np.zeros((horizon, n_arms), dtype=float)

    # Arm 0: stationary baseline
    means[:, 0] = stable_mean

    # Arm 1: smooth drift over horizon
    means[:, 1] = np.linspace(drift_start, drift_end, num=horizon, dtype=float)

    # Arm 2: mostly low baseline with occasional bursts
    burst_arm = np.full(horizon, burst_base, dtype=float)
    burst_low = min(burst_base, burst_peak)
    burst_high = max(burst_base, burst_peak)
    t = 0
    while t < horizon:
        if rng.random() < burst_prob:
            burst_len = int(rng.integers(1, burst_duration + 1))
            burst_level = float(rng.uniform(0.5 * (burst_low + burst_high), burst_high))
            end = min(horizon, t + burst_len)
            burst_arm[t:end] = burst_level
        t += 1
    means[:, 2] = burst_arm

    # Arm 3: abrupt low/high switching with random segment lengths
    switch_low_eff = min(switch_low, switch_high)
    switch_high_eff = max(switch_low, switch_high)
    switch_arm = np.empty(horizon, dtype=float)
    t = 0
    level = switch_low_eff
    min_seg = max(2, switch_interval // 2)
    max_seg = max(min_seg + 1, int(1.5 * switch_interval) + 1)
    while t < horizon:
        seg_len = int(rng.integers(min_seg, max_seg))
        end = min(horizon, t + seg_len)
        switch_arm[t:end] = level
        level = switch_high_eff if level == switch_low_eff else switch_low_eff
        t = end
    means[:, 3] = switch_arm

    if n_arms > 4:
        t_grid = np.arange(horizon, dtype=float)
        for arm in range(4, n_arms):
            base = rng.uniform(0.2, 0.8)
            amp = rng.uniform(0.05, 0.25)
            period = float(rng.integers(80, 450))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            means[:, arm] = base + amp * np.sin((2.0 * np.pi * t_grid / period) + phase)

    return np.clip(means, 0.0, 1.0)


def _segment_lengths_with_minimum(
    horizon: int,
    n_segments: int,
    min_segment: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    required = n_segments * min_segment
    if required > horizon:
        raise ValueError("horizon is too small for requested min_segment and breakpoints.")
    lengths = np.full(n_segments, min_segment, dtype=np.int64)
    slack = horizon - required
    if slack > 0:
        lengths += rng.multinomial(slack, np.full(n_segments, 1.0 / float(n_segments)))
    return lengths


def random_breakpoint_means(
    horizon: int = 5000,
    n_arms: int = 4,
    min_breakpoints: int = 3,
    max_breakpoints: int = 10,
    min_segment: int = 20,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
    min_jump: float = 0.08,
    seed: int | None = None,
) -> FloatArray:
    """
    Piecewise-constant benchmark with random breakpoint locations and levels.
    Useful for testing adaptation to abrupt regime changes.
    """

    _validate_horizon_n_arms(horizon, n_arms, min_arms=1)
    if min_breakpoints < 0:
        raise ValueError("min_breakpoints must be >= 0.")
    if max_breakpoints < min_breakpoints:
        raise ValueError("max_breakpoints must be >= min_breakpoints.")
    if min_segment <= 0:
        raise ValueError("min_segment must be > 0.")
    _validate_prob("min_mean", min_mean)
    _validate_prob("max_mean", max_mean)
    if max_mean <= min_mean:
        raise ValueError("max_mean must be > min_mean.")
    if min_jump < 0.0:
        raise ValueError("min_jump must be >= 0.")
    if min_jump > (max_mean - min_mean):
        raise ValueError("min_jump cannot exceed max_mean - min_mean.")
    if (max_breakpoints + 1) * min_segment > horizon:
        raise ValueError("horizon too short for max_breakpoints and min_segment.")

    rng = np.random.default_rng(seed)
    means = np.zeros((horizon, n_arms), dtype=float)

    for arm in range(n_arms):
        n_breaks = int(rng.integers(min_breakpoints, max_breakpoints + 1))
        n_segments = n_breaks + 1
        lengths = _segment_lengths_with_minimum(
            horizon=horizon,
            n_segments=n_segments,
            min_segment=min_segment,
            rng=rng,
        )

        levels = np.empty(n_segments, dtype=float)
        levels[0] = rng.uniform(min_mean, max_mean)
        for idx in range(1, n_segments):
            candidate = float(rng.uniform(min_mean, max_mean))
            attempts = 0
            while abs(candidate - levels[idx - 1]) < min_jump and attempts < 32:
                candidate = float(rng.uniform(min_mean, max_mean))
                attempts += 1
            if abs(candidate - levels[idx - 1]) < min_jump:
                prev = levels[idx - 1]
                if prev < 0.5 * (min_mean + max_mean):
                    candidate = min(max_mean, prev + min_jump)
                else:
                    candidate = max(min_mean, prev - min_jump)
            levels[idx] = candidate

        start = 0
        for seg_idx, seg_len in enumerate(lengths):
            end = start + int(seg_len)
            means[start:end, arm] = levels[seg_idx]
            start = end

    return np.clip(means, 0.0, 1.0)


def random_drift_amplitude_means(
    horizon: int = 5000,
    n_arms: int = 4,
    amplitude_min: float = 0.05,
    amplitude_max: float = 0.30,
    amplitude_step_std: float = 0.01,
    period_min: int = 80,
    period_max: int = 400,
    baseline_low: float = 0.20,
    baseline_high: float = 0.80,
    drift_step_std: float = 0.004,
    seed: int | None = None,
) -> FloatArray:
    """
    Smoothly non-stationary benchmark where sinusoid amplitude drifts over time.
    This avoids tuning to a single fixed-amplitude cartoon trajectory.
    """

    _validate_horizon_n_arms(horizon, n_arms, min_arms=1)
    if amplitude_min < 0.0:
        raise ValueError("amplitude_min must be >= 0.")
    if amplitude_max <= amplitude_min:
        raise ValueError("amplitude_max must be > amplitude_min.")
    if amplitude_step_std < 0.0:
        raise ValueError("amplitude_step_std must be >= 0.")
    if period_min <= 1:
        raise ValueError("period_min must be > 1.")
    if period_max < period_min:
        raise ValueError("period_max must be >= period_min.")
    _validate_prob("baseline_low", baseline_low)
    _validate_prob("baseline_high", baseline_high)
    if baseline_high <= baseline_low:
        raise ValueError("baseline_high must be > baseline_low.")
    if drift_step_std < 0.0:
        raise ValueError("drift_step_std must be >= 0.")

    rng = np.random.default_rng(seed)
    t_grid = np.arange(horizon, dtype=float)
    means = np.zeros((horizon, n_arms), dtype=float)

    for arm in range(n_arms):
        base = rng.uniform(baseline_low, baseline_high)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        period = float(rng.integers(period_min, period_max + 1))
        trend_scale = rng.uniform(0.02, 0.12)

        amp = np.empty(horizon, dtype=float)
        amp[0] = rng.uniform(amplitude_min, amplitude_max)
        if horizon > 1:
            amp_noise = rng.normal(loc=0.0, scale=amplitude_step_std, size=horizon - 1)
            for t in range(1, horizon):
                amp[t] = np.clip(amp[t - 1] + amp_noise[t - 1], amplitude_min, amplitude_max)

        trend = np.cumsum(rng.normal(loc=0.0, scale=drift_step_std, size=horizon))
        trend_norm = np.max(np.abs(trend))
        if trend_norm > 0.0:
            trend = trend / trend_norm

        means[:, arm] = base + amp * np.sin((2.0 * np.pi * t_grid / period) + phase) + trend_scale * trend

    return np.clip(means, 0.0, 1.0)


def make_slow_varying_sinusoid_env(
    horizon: int = 5000,
    n_arms: int = 4,
    period: int = 1000,
    offsets: FloatArray | None = None,
    seed: int | None = None,
) -> BernoulliNonStationaryEnv:
    means = slow_varying_sinusoid_means(
        horizon=horizon,
        n_arms=n_arms,
        period=period,
        offsets=offsets,
    )
    return BernoulliNonStationaryEnv(mean_rewards=means, seed=seed)


def make_mixed_regime_challenge_env(
    horizon: int = 5000,
    n_arms: int = 4,
    stable_mean: float = 0.65,
    drift_start: float = 0.20,
    drift_end: float = 0.85,
    burst_base: float = 0.10,
    burst_peak: float = 0.95,
    burst_prob: float = 0.04,
    burst_duration: int = 20,
    switch_low: float = 0.15,
    switch_high: float = 0.85,
    switch_interval: int = 200,
    means_seed: int | None = None,
    seed: int | None = None,
) -> BernoulliNonStationaryEnv:
    means = mixed_regime_challenge_means(
        horizon=horizon,
        n_arms=n_arms,
        stable_mean=stable_mean,
        drift_start=drift_start,
        drift_end=drift_end,
        burst_base=burst_base,
        burst_peak=burst_peak,
        burst_prob=burst_prob,
        burst_duration=burst_duration,
        switch_low=switch_low,
        switch_high=switch_high,
        switch_interval=switch_interval,
        seed=means_seed,
    )
    return BernoulliNonStationaryEnv(mean_rewards=means, seed=seed)


def make_random_breakpoint_env(
    horizon: int = 5000,
    n_arms: int = 4,
    min_breakpoints: int = 3,
    max_breakpoints: int = 10,
    min_segment: int = 20,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
    min_jump: float = 0.08,
    means_seed: int | None = None,
    seed: int | None = None,
) -> BernoulliNonStationaryEnv:
    means = random_breakpoint_means(
        horizon=horizon,
        n_arms=n_arms,
        min_breakpoints=min_breakpoints,
        max_breakpoints=max_breakpoints,
        min_segment=min_segment,
        min_mean=min_mean,
        max_mean=max_mean,
        min_jump=min_jump,
        seed=means_seed,
    )
    return BernoulliNonStationaryEnv(mean_rewards=means, seed=seed)


def make_random_drift_amplitude_env(
    horizon: int = 5000,
    n_arms: int = 4,
    amplitude_min: float = 0.05,
    amplitude_max: float = 0.30,
    amplitude_step_std: float = 0.01,
    period_min: int = 80,
    period_max: int = 400,
    baseline_low: float = 0.20,
    baseline_high: float = 0.80,
    drift_step_std: float = 0.004,
    means_seed: int | None = None,
    seed: int | None = None,
) -> BernoulliNonStationaryEnv:
    means = random_drift_amplitude_means(
        horizon=horizon,
        n_arms=n_arms,
        amplitude_min=amplitude_min,
        amplitude_max=amplitude_max,
        amplitude_step_std=amplitude_step_std,
        period_min=period_min,
        period_max=period_max,
        baseline_low=baseline_low,
        baseline_high=baseline_high,
        drift_step_std=drift_step_std,
        seed=means_seed,
    )
    return BernoulliNonStationaryEnv(mean_rewards=means, seed=seed)


def global_switching_means(
    horizon: int = 2000,
    n_arms: int = 4,
    switch_prob: float = 0.02,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
    seed: int | None = None,
) -> FloatArray:
    """
    Global switching environment inspired by 1302.3721:
    all arms redraw their Bernoulli means at each changepoint.
    """
    _validate_horizon_n_arms(horizon, n_arms, min_arms=1)
    if not (0.0 <= switch_prob <= 1.0):
        raise ValueError("switch_prob must be in [0, 1].")
    if not (0.0 <= min_mean < max_mean <= 1.0):
        raise ValueError("Require 0 <= min_mean < max_mean <= 1.")

    rng = np.random.default_rng(seed)
    means = np.zeros((horizon, n_arms), dtype=float)

    current = rng.uniform(min_mean, max_mean, size=n_arms)
    means[0] = current
    for t in range(1, horizon):
        if rng.random() < switch_prob:
            current = rng.uniform(min_mean, max_mean, size=n_arms)
        means[t] = current
    return means


def per_arm_switching_means(
    horizon: int = 2000,
    n_arms: int = 4,
    switch_prob: float = 0.02,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
    seed: int | None = None,
) -> FloatArray:
    """
    Per-arm switching environment inspired by 1302.3721:
    each arm independently redraws its Bernoulli mean at changepoints.
    """
    _validate_horizon_n_arms(horizon, n_arms, min_arms=1)
    if not (0.0 <= switch_prob <= 1.0):
        raise ValueError("switch_prob must be in [0, 1].")
    if not (0.0 <= min_mean < max_mean <= 1.0):
        raise ValueError("Require 0 <= min_mean < max_mean <= 1.")

    rng = np.random.default_rng(seed)
    means = np.zeros((horizon, n_arms), dtype=float)

    current = rng.uniform(min_mean, max_mean, size=n_arms)
    means[0] = current
    for t in range(1, horizon):
        switches = rng.random(n_arms) < switch_prob
        if np.any(switches):
            current = current.copy()
            current[switches] = rng.uniform(min_mean, max_mean, size=int(np.sum(switches)))
        means[t] = current
    return means


def abrupt_means(
    horizon: int = 1000,
    n_arms: int = 4,
    cycle_len: int = 250,
    change_times: list[int] | None = None,
    levels: list[float] | None = None,
) -> np.ndarray:
    """
    Abruptly varying env used in Raj2017 Figure 1:
      - repeats every cycle_len
      - within each cycle, all arms start at 0
      - arm k jumps to levels[k] at change_times[k] and stays there until cycle ends
    Returns means with shape (T, K).
    """
    if change_times is None:
        change_times = [50, 100, 150, 200]
    if levels is None:
        levels = [0.10, 0.37, 0.63, 0.90]
    if len(change_times) != n_arms or len(levels) != n_arms:
        raise ValueError("change_times and levels must have length n_arms.")

    means = np.zeros((horizon, n_arms), dtype=float)
    for t in range(horizon):
        tau = t % cycle_len
        for k in range(n_arms):
            if tau >= change_times[k]:
                means[t, k] = levels[k]
    return means
