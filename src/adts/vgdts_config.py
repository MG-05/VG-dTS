from __future__ import annotations

from dataclasses import fields, replace

from .policies import VGdTSParams

VGDTSScalar = bool | int | float | str


def make_benchmark_vgdts_params(**overrides: VGDTSScalar) -> VGdTSParams:
    """
    Shared VG-dTS benchmark configuration used by experiment entrypoints.
    """
    allowed = {field.name for field in fields(VGdTSParams)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown VG-dTS parameter override(s): {', '.join(unknown)}")

    base = VGdTSParams(
        alpha0=1.0,
        beta0=1.0,
        gamma_default=0.45,
        gamma_min=0.12,
        gamma_max=0.995,
        lambda_vol=0.88,
        n0=0.0,
        gamma_mapping="inverse_linear",
        vol_low=0.0,
        vol_high=2.6,
        optimistic=True,
    )
    return replace(base, **overrides)


def make_default_vgdts_tuning_grid() -> dict[str, list[VGDTSScalar]]:
    """
    Modest default sweep over the main VG-dTS memory/adaptation controls.
    """
    return {
        "gamma_default": [0.45, 0.85],
        "gamma_min": [0.12, 0.70],
        "lambda_vol": [0.88, 0.97],
        "n0": [0.0, 25.0],
        "vol_high": [2.6, 3.5],
        "one_sided_negative_surprise": [False, True],
    }


def make_benchmark_vgdts_v2_params(**overrides: VGDTSScalar) -> VGdTSParams:
    """
    Frozen VG-dTS v2 benchmark configuration.

    v2 keeps the same VGdTSParams container for compatibility, but fixes the
    structural mechanisms to:
      - dual-memory posterior
      - likelihood surprise
      - confidence bonus
    """
    allowed = {field.name for field in fields(VGdTSParams)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown VG-dTS parameter override(s): {', '.join(unknown)}")

    base = VGdTSParams(
        alpha0=1.0,
        beta0=1.0,
        gamma_default=0.45,
        gamma_min=0.12,
        gamma_max=0.995,
        lambda_vol=0.88,
        n0=0.0,
        gamma_mapping="inverse_linear",
        vol_low=0.0,
        vol_high=2.6,
        optimistic=True,
        dual_memory_enabled=True,
        gamma_long=0.995,
        dual_memory_mix_power=1.0,
        surprise_mode="neg_log_likelihood",
        confidence_bonus_enabled=True,
        confidence_bonus_scale=0.6,
        confidence_bonus_vol_weight=0.8,
        confidence_bonus_coldstart_weight=0.4,
        confidence_bonus_clip=0.75,
    )
    return replace(base, **overrides)


def make_default_vgdts_v2_tuning_grid() -> dict[str, list[VGDTSScalar]]:
    """
    Small default v2 sweep over memory and bonus strength controls.
    """
    return {
        "gamma_default": [0.35, 0.50],
        "gamma_min": [0.10, 0.20],
        "vol_high": [2.2, 3.0],
        "dual_memory_mix_power": [0.8, 1.4],
        "confidence_bonus_scale": [0.4, 0.7],
    }


def make_benchmark_vgdts_v21_params(**overrides: VGDTSScalar) -> VGdTSParams:
    """
    Frozen VG-dTS v2.1 benchmark configuration.

    Starts from v2 and freezes the leading soft-addition candidate:
      - shock score enabled
      - shock-driven effective sample size modulation disabled
      - selective stale-arm revisit disabled
    """
    allowed = {field.name for field in fields(VGdTSParams)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown VG-dTS parameter override(s): {', '.join(unknown)}")

    base = VGdTSParams(
        alpha0=1.0,
        beta0=1.0,
        gamma_default=0.45,
        gamma_min=0.12,
        gamma_max=0.995,
        lambda_vol=0.88,
        n0=0.0,
        gamma_mapping="inverse_linear",
        vol_low=0.0,
        vol_high=2.6,
        optimistic=True,
        dual_memory_enabled=True,
        gamma_long=0.995,
        dual_memory_mix_power=1.0,
        surprise_mode="neg_log_likelihood",
        confidence_bonus_enabled=True,
        confidence_bonus_scale=0.6,
        confidence_bonus_vol_weight=0.8,
        confidence_bonus_coldstart_weight=0.4,
        confidence_bonus_clip=0.75,
        shock_score_enabled=True,
        shock_score_scale=0.12,
        shock_score_decay=0.92,
        shock_n_eff_enabled=False,
        shock_n_eff_scale=0.75,
        selective_revisit_enabled=False,
        selective_revisit_margin=0.03,
        stale_arm_revisit_scale=0.18,
        stale_arm_revisit_half_life=60.0,
        stale_arm_revisit_clip=0.25,
    )
    return replace(base, **overrides)


def make_default_vgdts_v21_tuning_grid() -> dict[str, list[VGDTSScalar]]:
    """
    Small v2.1 sweep centered on the v2 core plus shock-score strength.
    """
    return {
        "gamma_default": [0.35, 0.50],
        "gamma_min": [0.10, 0.20],
        "vol_high": [2.2, 3.0],
        "dual_memory_mix_power": [0.8, 1.4],
        "confidence_bonus_scale": [0.4, 0.7],
        "shock_score_scale": [0.08, 0.16],
        "shock_score_decay": [0.88, 0.96],
    }


def make_benchmark_vgdts_v3_params(**overrides: VGDTSScalar) -> VGdTSParams:
    """
    Frozen VG-dTS v3 benchmark configuration.

    v3 focuses on:
      - hazard-gated posterior reset mixing
      - mandatory uncertainty + age-aware inflation
      - dual surprise channels (likelihood + signed residual drift)
    """
    allowed = {field.name for field in fields(VGdTSParams)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown VG-dTS parameter override(s): {', '.join(unknown)}")

    base = VGdTSParams(
        alpha0=1.0,
        beta0=1.0,
        gamma_default=0.42,
        gamma_min=0.10,
        gamma_max=0.995,
        n0=0.0,
        optimistic=True,
        # likelihood-surprise channel
        lambda_vol=0.90,
        vol_low=0.0,
        vol_high=2.8,
        # signed-residual channel and hazard map
        lambda_vol_slow=0.96,
        logistic_mid=0.55,
        logistic_slope=5.0,
        global_shock_enabled=True,
        global_shock_lambda=0.85,
        global_shock_gamma_floor=0.15,
        two_timescale_volatility_enabled=True,
        # uncertainty bonus (always used by v3 runner)
        confidence_bonus_scale=0.65,
        confidence_bonus_vol_weight=0.80,
        confidence_bonus_coldstart_weight=0.40,
        confidence_bonus_clip=0.80,
        # age-aware stale-arm inflation (always used by v3 runner)
        stale_arm_revisit_enabled=True,
        stale_arm_revisit_scale=0.30,
        stale_arm_revisit_half_life=50.0,
        stale_arm_revisit_clip=0.50,
    )
    return replace(base, **overrides)


def make_default_vgdts_v3_tuning_grid() -> dict[str, list[VGDTSScalar]]:
    """
    Modest v3 sweep over a small set of adaptation controls.
    """
    return {
        "gamma_default": [0.36, 0.48],
        "gamma_min": [0.08, 0.14],
        "vol_high": [2.4, 3.2],
        "logistic_mid": [0.45, 0.65],
        "confidence_bonus_scale": [0.5, 0.8],
        "stale_arm_revisit_scale": [0.2, 0.4],
    }
