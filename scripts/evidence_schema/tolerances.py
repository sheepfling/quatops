"""Acceptance tolerances shared by converter tests and crystallized records."""

from __future__ import annotations

from scripts.evidence_schema.crystallized import PrecisionTolerance

_POLICY_VALUES: dict[str, tuple[tuple[str, float], ...]] = {
    "Eigen": (("float32", 2e-6), ("float64", 4e-12)),
    "MuJoCo": (("float64", 3e-12),),
    "OpenCV": (("float32", 2.5e-6), ("float64", 4e-12)),
    "ROS 2 tf2": (("float64", 4e-12),),
    "SpiceyPy / CSPICE": (("float64", 3e-12),),
    "CesiumJS": (("orientation", 2e-11), ("ecef-position", 2e-9)),
    "Three.js GLTFLoader": (("float64", 2e-12),),
    "OpenUSD": (("float64", 3e-12),),
}


def tolerances_for_target(target: str) -> list[PrecisionTolerance]:
    """Return the repository converter tests' exact absolute error limits."""
    try:
        values = _POLICY_VALUES[target]
    except KeyError as exc:
        raise ValueError(f"no converter comparison policy for {target!r}") from exc
    return [
        PrecisionTolerance(profile=profile, absolute=absolute, relative=0.0)
        for profile, absolute in values
    ]
