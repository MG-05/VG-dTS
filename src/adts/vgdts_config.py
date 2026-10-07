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
