"""Interpretable persistence/hysteresis Recovery-router candidates."""
from __future__ import annotations

from typing import Any


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"unsupported router comparison: {operator}")


def candidate_fires(spec: dict[str, Any], features: dict[str, Any]) -> bool:
    """Evaluate one JSON-serializable rule against current causal features."""
    if int(features["approach_elapsed_frames"]) < int(
        spec.get("minimum_approach_frames", 0)
    ):
        return False
    active = set(features["trigger_active_types"])
    for branch in spec["branches"]:
        required_any = set(branch.get("trigger_any", []))
        if required_any and active.isdisjoint(required_any):
            continue
        required_all = set(branch.get("trigger_all", []))
        if not required_all.issubset(active):
            continue
        conditions = branch.get("conditions", [])
        if all(_compare(
            float(features[condition["feature"]]),
            condition["operator"],
            float(condition["value"]),
        ) for condition in conditions):
            return True
    return False


def default_candidate_catalog() -> list[dict[str, Any]]:
    """Small rule grid; thresholds are diagnostics, never task gates."""
    any_trigger = [
        "moving_away", "terminal_plateau", "gate_stall",
        "near_timeout_terminal", "cube_displacement",
    ]
    catalog: list[dict[str, Any]] = [{
        "name": "old_first_trigger",
        "description": "Reference: switch on the first legacy detector hit.",
        "minimum_approach_frames": 0,
        "branches": [{"trigger_any": any_trigger}],
    }]
    for patience in (10, 15, 20, 25, 30, 40):
        catalog.append({
            "name": f"patience_{patience}",
            "description": (
                f"Any legacy signal, but only after {patience} Approach frames."
            ),
            "minimum_approach_frames": patience,
            "branches": [{"trigger_any": any_trigger}],
        })
    catalog.extend([
        {
            "name": "cube_displacement_only",
            "description": "Recover only after the 3 mm cube-motion detector.",
            "minimum_approach_frames": 8,
            "branches": [{"trigger_any": ["cube_displacement"]}],
        },
        {
            "name": "cube_outward",
            "description": "Cube-motion detector plus positive radial velocity.",
            "minimum_approach_frames": 8,
            "branches": [{
                "trigger_any": ["cube_displacement"],
                "conditions": [{
                    "feature": "cube_radial_velocity_m_s",
                    "operator": ">", "value": 0.0,
                }],
            }],
        },
        {
            "name": "cube_outward_5mm_s",
            "description": "Cube-motion detector moving outward faster than 5 mm/s.",
            "minimum_approach_frames": 8,
            "branches": [{
                "trigger_any": ["cube_displacement"],
                "conditions": [{
                    "feature": "cube_radial_velocity_m_s",
                    "operator": ">", "value": 0.005,
                }],
            }],
        },
    ])
    for duration in (3, 4, 5):
        catalog.append({
            "name": f"moving_away_persistent_{duration}",
            "description": (
                "Moving-away detector with "
                f"{duration} consecutive increasing-error frames."
            ),
            "minimum_approach_frames": 10,
            "branches": [{
                "trigger_any": ["moving_away"],
                "conditions": [{
                    "feature": "moving_away_consecutive_frames",
                    "operator": ">=", "value": duration,
                }],
            }],
        })
        catalog.append({
            "name": f"moving_away_persistent_{duration}_error15",
            "description": (
                f"Persistent-{duration} moving away while error exceeds 15 mm."
            ),
            "minimum_approach_frames": 10,
            "branches": [{
                "trigger_any": ["moving_away"],
                "conditions": [
                    {
                        "feature": "moving_away_consecutive_frames",
                        "operator": ">=", "value": duration,
                    },
                    {
                        "feature": "terminal_error_m",
                        "operator": ">", "value": 0.015,
                    },
                ],
            }],
        })
    for duration in (5, 8, 10):
        catalog.append({
            "name": f"plateau_persistent_{duration}",
            "description": f"Plateau detector with {duration} non-improving frames.",
            "minimum_approach_frames": 12,
            "branches": [{
                "trigger_any": ["terminal_plateau"],
                "conditions": [{
                    "feature": "plateau_duration_frames",
                    "operator": ">=", "value": duration,
                }],
            }],
        })
        catalog.append({
            "name": f"gate_stall_persistent_{duration}",
            "description": f"Gate-stall detector persisting for {duration} frames.",
            "minimum_approach_frames": 12,
            "branches": [{
                "trigger_any": ["gate_stall"],
                "conditions": [{
                    "feature": "gate_stall_duration_frames",
                    "operator": ">=", "value": duration,
                }],
            }],
        })
    catalog.extend([
        {
            "name": "late_terminal_only",
            "description": "Only the final-ten-frame near-terminal timeout signal.",
            "minimum_approach_frames": 40,
            "branches": [{"trigger_any": ["near_timeout_terminal"]}],
        },
        {
            "name": "selective_hysteresis",
            "description": (
                "Cube moving outward, persistent moving-away, persistent stall, "
                "or the near-timeout terminal signal after a patience window."
            ),
            "minimum_approach_frames": 12,
            "branches": [
                {
                    "trigger_any": ["cube_displacement"],
                    "conditions": [{
                        "feature": "cube_radial_velocity_m_s",
                        "operator": ">", "value": 0.0,
                    }],
                },
                {
                    "trigger_any": ["moving_away"],
                    "conditions": [
                        {
                            "feature": "moving_away_consecutive_frames",
                            "operator": ">=", "value": 5,
                        },
                        {
                            "feature": "terminal_error_m",
                            "operator": ">", "value": 0.015,
                        },
                    ],
                },
                {
                    "trigger_any": ["gate_stall", "terminal_plateau"],
                    "conditions": [{
                        "feature": "non_improving_consecutive_frames",
                        "operator": ">=", "value": 8,
                    }],
                },
                {"trigger_any": ["near_timeout_terminal"]},
            ],
        },
    ])
    return catalog
