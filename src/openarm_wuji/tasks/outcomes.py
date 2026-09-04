from __future__ import annotations

import numpy as np

from .se3 import pose_drift


OUTCOMES = (
    "rigid_success",
    "settled_after_slip",
    "persistent_slip",
    "drop",
    "never_lift",
    "approach_push",
)


def _drift_metrics(samples: list[dict], anchor_index: int) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    anchor_index = min(max(anchor_index, 0), len(samples) - 1)
    anchor = samples[anchor_index]
    translation = []
    rotation = []
    for sample in samples[anchor_index:]:
        tr, rot = pose_drift(
            anchor["object_relative_position_m"],
            anchor["object_relative_quaternion_wxyz"],
            sample["object_relative_position_m"],
            sample["object_relative_quaternion_wxyz"],
        )
        translation.append(tr)
        rotation.append(rot)
    return max(translation, default=0.0), max(rotation, default=0.0)


def evaluate_lift_outcome(*, task_success: bool, approach_push: bool,
                          grasp_close_start: dict, lift_samples: list[dict],
                          control_hz: float, success_hold_frames: int,
                          baseline: dict) -> dict:
    """Evaluate task, relative-pose stability, and an exclusive outcome label.

    `lift_samples` includes the pre-lift pose at index 0 followed by post-action poses.
    The CD-WM thresholds are consumed only from the explicitly named baseline config.
    """
    if not lift_samples:
        lift_samples = [grasp_close_start]
    margin = int(baseline["stability_anchor_margin_frames"])
    max_translation, max_rotation = _drift_metrics(lift_samples, margin)
    translation_limit = float(baseline["max_relative_translation_drift_m"])
    rotation_limit = float(baseline["max_relative_rotation_drift_deg"])
    pose_stable = (
        len(lift_samples) > margin
        and max_translation < translation_limit
        and max_rotation < rotation_limit
    )

    final_window = lift_samples[-min(success_hold_frames, len(lift_samples)):]
    final_translation, final_rotation = _drift_metrics(final_window, 0)
    final_window_pose_stable = (
        final_translation < translation_limit and final_rotation < rotation_limit
    )
    establishment_translation, establishment_rotation = pose_drift(
        grasp_close_start["object_relative_position_m"],
        grasp_close_start["object_relative_quaternion_wxyz"],
        lift_samples[0]["object_relative_position_m"],
        lift_samples[0]["object_relative_quaternion_wxyz"],
    )

    heights = np.asarray([sample["cube_height_m"] for sample in lift_samples])
    peak_height = float(np.max(heights))
    final_height = float(heights[-1])
    minimum_lift = float(baseline["min_object_lift_m"])
    duration_steps = max(1, int(round(
        float(baseline["lift_duration_s"]) * control_hz
    )))
    protocol_window = lift_samples[:duration_steps + 1]
    start_grasp_z = float(lift_samples[0]["grasp_center_position_m"][2])
    gripper_lift_in_window = max(
        float(sample["grasp_center_position_m"][2]) - start_grasp_z
        for sample in protocol_window
    )
    protocol_gripper_lift = float(baseline["gripper_lift_m"])
    external_protocol_reached = gripper_lift_in_window >= protocol_gripper_lift
    external_object_lifted = final_height >= minimum_lift
    grasp_stable = bool(pose_stable and external_object_lifted)
    final_window_stable = bool(final_window_pose_stable and external_object_lifted)

    if approach_push:
        outcome = "approach_push"
    elif peak_height < minimum_lift:
        outcome = "never_lift"
    elif final_height < minimum_lift:
        outcome = "drop"
    elif grasp_stable:
        outcome = "rigid_success"
    elif final_window_stable:
        outcome = "settled_after_slip"
    else:
        outcome = "persistent_slip"
    if outcome not in OUTCOMES:
        raise RuntimeError(f"unrecognized outcome: {outcome}")

    return {
        "task_success": bool(task_success),
        "grasp_stable": bool(grasp_stable),
        "outcome": outcome,
        "max_relative_translation_drift_m": max_translation,
        "max_relative_rotation_drift_deg": max_rotation,
        "final_window_translation_drift_m": final_translation,
        "final_window_rotation_drift_deg": final_rotation,
        "final_window_stable": bool(final_window_stable),
        "establishment_translation_m": establishment_translation,
        "establishment_rotation_deg": establishment_rotation,
        "gripper_lift_within_baseline_window_m": gripper_lift_in_window,
        "external_lift_protocol_reached": bool(external_protocol_reached),
        "external_object_lifted": bool(external_object_lifted),
        "external_baseline": {
            "name": str(baseline["name"]),
            "source_url": str(baseline["source_url"]),
            "lift_duration_s": float(baseline["lift_duration_s"]),
            "gripper_lift_m": protocol_gripper_lift,
            "min_object_lift_m": minimum_lift,
            "stability_anchor_margin_frames": margin,
            "max_relative_translation_drift_m": translation_limit,
            "max_relative_rotation_drift_deg": rotation_limit,
            "threshold_status": "external_baseline_not_wuji_calibrated",
        },
    }
