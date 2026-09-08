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


def _slope_per_second(values: list[float], control_hz: float) -> float:
    if len(values) < 2:
        return 0.0
    time = np.arange(len(values), dtype=float) / control_hz
    centered_time = time - np.mean(time)
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0.0:
        return 0.0
    centered_values = np.asarray(values, dtype=float) - np.mean(values)
    return float(np.dot(centered_time, centered_values) / denominator)


def _post_settle_metrics(samples: list[dict], *, control_hz: float,
                         post_settle: dict, translation_limit: float,
                         rotation_limit: float) -> dict:
    """Measure convergence over a fixed trailing window after lift transients."""
    window_intervals = max(1, int(round(
        float(post_settle["window_s"]) * control_hz
    )))
    required_samples = window_intervals + 1
    window = samples[-required_samples:]
    anchor = window[0]
    translation_drift = []
    rotation_drift = []
    translation_speed = []
    angular_speed = []
    for index, sample in enumerate(window):
        translation, rotation = pose_drift(
            anchor["object_relative_position_m"],
            anchor["object_relative_quaternion_wxyz"],
            sample["object_relative_position_m"],
            sample["object_relative_quaternion_wxyz"],
        )
        translation_drift.append(translation)
        rotation_drift.append(rotation)
        if index:
            step_translation, step_rotation = pose_drift(
                window[index - 1]["object_relative_position_m"],
                window[index - 1]["object_relative_quaternion_wxyz"],
                sample["object_relative_position_m"],
                sample["object_relative_quaternion_wxyz"],
            )
            translation_speed.append(step_translation * control_hz)
            angular_speed.append(step_rotation * control_hz)
    max_translation = max(translation_drift, default=0.0)
    max_rotation = max(rotation_drift, default=0.0)
    max_translation_speed = max(translation_speed, default=0.0)
    max_angular_speed = max(angular_speed, default=0.0)
    rms_translation_speed = float(np.sqrt(np.mean(np.square(
        translation_speed
    )))) if translation_speed else 0.0
    rms_angular_speed = float(np.sqrt(np.mean(np.square(
        angular_speed
    )))) if angular_speed else 0.0
    translation_slope = _slope_per_second(translation_drift, control_hz)
    rotation_slope = _slope_per_second(rotation_drift, control_hz)
    translation_speed_limit = float(
        post_settle["rms_relative_translation_speed_m_s"]
    )
    angular_speed_limit = float(
        post_settle["rms_relative_angular_speed_deg_s"]
    )
    enough_samples = len(window) >= required_samples
    stable = bool(
        enough_samples
        and max_translation < translation_limit
        and max_rotation < rotation_limit
        and rms_translation_speed < translation_speed_limit
        and rms_angular_speed < angular_speed_limit
        and translation_slope < translation_speed_limit
        and rotation_slope < angular_speed_limit
    )
    return {
        "stable": stable,
        "window_s": float(post_settle["window_s"]),
        "window_samples": len(window),
        "required_samples": required_samples,
        "max_translation_drift_m": max_translation,
        "max_rotation_drift_deg": max_rotation,
        "max_relative_translation_speed_m_s": max_translation_speed,
        "max_relative_angular_speed_deg_s": max_angular_speed,
        "rms_relative_translation_speed_m_s": rms_translation_speed,
        "rms_relative_angular_speed_deg_s": rms_angular_speed,
        "translation_drift_slope_m_s": translation_slope,
        "rotation_drift_slope_deg_s": rotation_slope,
        "threshold_source": str(post_settle["threshold_source"]),
        "rms_relative_translation_speed_threshold_m_s": float(
            post_settle["rms_relative_translation_speed_m_s"]
        ),
        "rms_relative_angular_speed_threshold_deg_s": float(
            post_settle["rms_relative_angular_speed_deg_s"]
        ),
    }


def evaluate_lift_outcome(*, task_success: bool, approach_push: bool,
                          grasp_close_start: dict, lift_samples: list[dict],
                          control_hz: float, success_hold_frames: int,
                          baseline: dict, post_settle: dict | None = None) -> dict:
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
    if post_settle is None:
        duration = float(baseline["lift_duration_s"])
        post_settle = {
            "window_s": duration,
            "threshold_source": "derived_from_external_baseline",
            "rms_relative_translation_speed_m_s": translation_limit / duration,
            "rms_relative_angular_speed_deg_s": rotation_limit / duration,
        }
    post_settle_metrics = _post_settle_metrics(
        lift_samples,
        control_hz=control_hz,
        post_settle=post_settle,
        translation_limit=translation_limit,
        rotation_limit=rotation_limit,
    )
    post_settle_stable = bool(
        post_settle_metrics["stable"] and external_object_lifted
    )
    continuous_slip = bool(external_object_lifted and not post_settle_stable)

    if approach_push:
        outcome = "approach_push"
    elif peak_height < minimum_lift:
        outcome = "never_lift"
    elif final_height < minimum_lift:
        outcome = "drop"
    elif grasp_stable:
        outcome = "rigid_success"
    elif post_settle_stable:
        outcome = "settled_after_slip"
    else:
        outcome = "persistent_slip"
    if outcome not in OUTCOMES:
        raise RuntimeError(f"unrecognized outcome: {outcome}")

    return {
        "task_success": bool(task_success),
        "grasp_stable": bool(grasp_stable),
        "post_settle_stable": post_settle_stable,
        "continuous_slip": continuous_slip,
        "outcome": outcome,
        "max_relative_translation_drift_m": max_translation,
        "max_relative_rotation_drift_deg": max_rotation,
        "final_window_translation_drift_m": final_translation,
        "final_window_rotation_drift_deg": final_rotation,
        "final_window_stable": bool(final_window_stable),
        "post_settle_metrics": post_settle_metrics,
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
