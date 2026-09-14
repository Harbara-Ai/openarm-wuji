"""Low-dimensional, interpretable temporal features for Recovery routing."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .recovery_router import primary_trigger


def _slope(values: Sequence[float], window: int, fps: float) -> float:
    count = min(len(values), window)
    if count < 2:
        return 0.0
    sample = np.asarray(values[-count:], dtype=float)
    time = np.arange(count, dtype=float) / fps
    return float(np.polyfit(time, sample, 1)[0])


def _recent_min(values: Sequence[float], window: int) -> float:
    return float(min(values[-min(len(values), window):]))


def _trailing_true(values: Sequence[bool]) -> int:
    count = 0
    for value in reversed(values):
        if not value:
            break
        count += 1
    return count


def extract_router_features(
    *,
    errors_m: Sequence[float],
    cube_displacements_m: Sequence[float],
    cube_radial_velocities_m_s: Sequence[float],
    arm_velocity_norms_rad_s: Sequence[float],
    arm_velocity_max_abs_rad_s: Sequence[float],
    grasp_velocity_norms_m_s: Sequence[float],
    grasp_velocity_toward_target_m_s: Sequence[float],
    gate_stall_flags: Sequence[bool],
    gate_consecutive: int,
    gate_frames_total: int,
    orientation_error_deg: float,
    active_triggers: dict[str, float],
    frame: int,
    timeout_frames: int,
    fps: float = 30.0,
) -> dict[str, object]:
    """Build features from observations available at the routing instant."""
    if not errors_m:
        raise ValueError("errors_m must contain the current frame")
    lengths = {
        len(errors_m), len(cube_displacements_m),
        len(cube_radial_velocities_m_s), len(arm_velocity_norms_rad_s),
        len(arm_velocity_max_abs_rad_s), len(grasp_velocity_norms_m_s),
        len(grasp_velocity_toward_target_m_s), len(gate_stall_flags),
    }
    if len(lengths) != 1:
        raise ValueError(f"router history length mismatch: {sorted(lengths)}")

    errors = np.asarray(errors_m, dtype=float)
    error_diffs = np.diff(errors)
    moving_away_flags = (error_diffs > 0.0).tolist()
    # A <=0.1 mm decrease in one control frame is treated as no meaningful
    # progress.  This is a diagnostic hysteresis scale, not a task gate.
    non_improving_flags = (error_diffs >= -0.0001).tolist()
    entered_basin = bool(np.any(errors <= 0.012))
    basin_min = float(errors[errors <= 0.012].min()) if entered_basin else None
    current = float(errors[-1])

    result: dict[str, object] = {
        "approach_frame": int(frame),
        "approach_elapsed_frames": int(frame + 1),
        "remaining_timeout_frames": int(max(0, timeout_frames - frame - 1)),
        "terminal_error_m": current,
        "terminal_error_min_so_far_m": float(errors.min()),
        "orientation_error_deg": float(orientation_error_deg),
        "moving_away_consecutive_frames": _trailing_true(moving_away_flags),
        "non_improving_consecutive_frames": _trailing_true(non_improving_flags),
        "plateau_duration_frames": _trailing_true(non_improving_flags),
        "gate_stall_duration_frames": _trailing_true(gate_stall_flags),
        "gate_consecutive_frames": int(gate_consecutive),
        "gate_frames_total": int(gate_frames_total),
        "entered_12mm_basin": entered_basin,
        "basin_exit_distance_m": (
            float(max(0.0, current - basin_min)) if basin_min is not None else 0.0
        ),
        "cube_displacement_m": float(cube_displacements_m[-1]),
        "cube_radial_velocity_m_s": float(cube_radial_velocities_m_s[-1]),
        "arm_joint_velocity_l2_rad_s": float(arm_velocity_norms_rad_s[-1]),
        "arm_joint_velocity_max_abs_rad_s": float(
            arm_velocity_max_abs_rad_s[-1]
        ),
        "grasp_center_velocity_m_s": float(grasp_velocity_norms_m_s[-1]),
        "grasp_center_velocity_toward_target_m_s": float(
            grasp_velocity_toward_target_m_s[-1]
        ),
        "trigger_primary_type": primary_trigger(active_triggers),
        "trigger_active_types": sorted(active_triggers),
    }
    for window in (3, 5, 10):
        recent_min = _recent_min(errors_m, window)
        result[f"error_slope_{window}_m_s"] = _slope(errors_m, window, fps)
        result[f"recent_error_min_{window}_m"] = recent_min
        result[f"distance_from_recent_min_{window}_m"] = current - recent_min
        result[f"cube_displacement_slope_{window}_m_s"] = _slope(
            cube_displacements_m, window, fps
        )
        result[f"cube_displacement_change_{window}_m"] = float(
            cube_displacements_m[-1]
            - cube_displacements_m[-min(len(cube_displacements_m), window)]
        )
    return result
