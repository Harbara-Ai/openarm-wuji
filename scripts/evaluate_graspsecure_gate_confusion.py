"""Measure GraspSecure-gate confusion by lifting both PASS and FAIL states.

This is a diagnostic evaluator, not a deployment controller.  The frozen
Reach/Approach/Recovery/GraspSecure policies and the existing scripted Lift
are reused without changing their gates, targets, gains, or action semantics.
The GraspSecure gate and the 25 mm pre-Lift safety reference are recorded as
labels; both are bypassed for the forced-Lift diagnostic unless the simulator
state is non-finite or the cube is grossly outside a generous workspace bound.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from openarm_wuji.policy import ApproachPolicy, ReachPolicy, RecoveryPolicy
from openarm_wuji.policy.grasp_secure_controller import GraspSecurePolicy
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import capture_simulator_snapshot
from openarm_wuji.tasks import (
    ReachGraspLiftTask,
    initialize_task_from_handoff,
    run_scripted_lift_from_handoff,
)
from openarm_wuji.tasks.grasp_preload_stage import FINGER_NAMES, finger_force_vector
from scripts.evaluate_grasp_secure_act import _run_grasp, _set_grasp_deterministic
from scripts.evaluate_staged_with_recovery import (
    _run_episode as run_frozen_upstream,
    _set_deterministic,
)


FEATURE_NAMES = (
    "preload_l2_rad",
    *(f"preload_finger{finger}_joint{joint}_rad"
      for finger in range(1, 6) for joint in range(1, 5)),
    "active_finger_count",
    "thumb_present",
    *(f"force_finger{finger}_n" for finger in range(1, 6)),
    "total_force_n",
    "force_hhi",
    "effective_force_fingers",
    "contact_count",
    "contact_persistence_s",
    "designated_tip_fraction",
    "distal_non_tip_fraction",
    "interior_contact_fraction",
    "edge_contact_fraction",
    "corner_contact_fraction",
    "contact_slip_mean_m_s",
    "contact_slip_max_m_s",
    "cube_palm_x_m",
    "cube_palm_y_m",
    "cube_palm_z_m",
    "cube_palm_rotation_deg",
    "cube_linear_speed_m_s",
    "cube_angular_speed_rad_s",
    "relative_linear_speed_m_s",
    "relative_angular_speed_rad_s",
    "resultant_force_n",
    "resultant_moment_nm",
    "prelift_cube_motion_m",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2), encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_plain(value), separators=(",", ":")) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]


def _safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _as_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def confusion_metrics(gate_pass: np.ndarray, lift_success: np.ndarray
                      ) -> dict[str, Any]:
    """Return a load-bearing confusion matrix for a static grasp gate."""
    predicted = np.asarray(gate_pass, dtype=bool)
    actual = np.asarray(lift_success, dtype=bool)
    if predicted.shape != actual.shape or predicted.ndim != 1:
        raise ValueError("gate_pass and lift_success must be same-size 1-D arrays")
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision_P_lift_success_given_gate_pass": _safe_div(tp, tp + fp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_positive_fraction_among_gate_pass": _safe_div(fp, tp + fp),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "P_lift_success_given_gate_fail": _safe_div(fn, fn + tn),
        "recall_sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "negative_predictive_value": _safe_div(tn, tn + fn),
        "accuracy": _safe_div(tp + tn, len(actual)),
    }


def _quat_rotation_deg(quaternion_wxyz: np.ndarray) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return float("nan")
    w = float(np.clip(abs(quaternion[0] / norm), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def _terminal_contact_slip(task, contacts: list[dict[str, Any]]) -> list[float]:
    """Compute finger-versus-cube tangential point speed for each contact."""
    import mujoco

    data = task.data
    model = task.model
    cube_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY,
        task.cube_body_id, cube_velocity, 0,
    )
    cube_angular = cube_velocity[:3]
    cube_linear = cube_velocity[3:]
    cube_com = np.asarray(data.xpos[task.cube_body_id], dtype=float)
    speeds = []
    for contact in contacts:
        body_id = int(contact["other_body_id"])
        finger_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY,
            body_id, finger_velocity, 0,
        )
        point = np.asarray(contact["position_world_m"], dtype=float)
        finger_point = finger_velocity[3:] + np.cross(
            finger_velocity[:3], point - np.asarray(data.xpos[body_id])
        )
        cube_point = cube_linear + np.cross(cube_angular, point - cube_com)
        relative = finger_point - cube_point
        normal = np.asarray(contact["normal_on_cube_world"], dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        tangential = relative - float(np.dot(relative, normal)) * normal
        speeds.append(float(np.linalg.norm(tangential)))
    return speeds


def terminal_features(*, robot, task, grasp: dict[str, Any],
                      prelift_cube_motion_m: float,
                      minimum_force_n: float) -> dict[str, Any]:
    """Capture pre-Lift features only; no future Lift information is used."""
    telemetry = task.task_telemetry()
    forces = finger_force_vector(telemetry)
    active = forces >= minimum_force_n
    contacts = telemetry["contacts"]
    slips = _terminal_contact_slip(task, contacts)
    preload = (
        robot.data.ctrl[robot.hand_actuator_ids]
        - robot.data.qpos[robot.hand_qpos_ids]
    ).astype(float)
    positive_force = np.maximum(forces, 0.0)
    total_force = float(np.sum(positive_force))
    shares = positive_force / total_force if total_force > 1e-12 else np.zeros(5)
    hhi = float(np.sum(shares ** 2)) if total_force > 1e-12 else 1.0
    roles = Counter(str(contact["contact_role"]) for contact in contacts)
    regions = Counter(
        "corner" if contact["corner_contact"] else
        "edge" if contact["edge_contact"] else "interior"
        for contact in contacts
    )
    count = len(contacts)
    cube_linear = np.asarray(
        telemetry["cube_linear_velocity_world_m_s"], dtype=float
    )
    cube_angular = np.asarray(
        telemetry["cube_angular_velocity_world_rad_s"], dtype=float
    )
    palm_linear = np.asarray(
        telemetry["grasp_center_linear_velocity_world_m_s"], dtype=float
    )
    palm_angular = np.asarray(
        telemetry["grasp_center_angular_velocity_world_rad_s"], dtype=float
    )
    relative_position = np.asarray(telemetry["object_relative_position_m"])
    relative_quaternion = np.asarray(
        telemetry["object_relative_quaternion_wxyz"]
    )
    resultant_force = np.asarray(
        telemetry["contact_resultant_force_world_n"], dtype=float
    )
    resultant_moment = np.asarray(
        telemetry["contact_resultant_moment_about_cube_world_nm"], dtype=float
    )
    result: dict[str, Any] = {
        "preload_l2_rad": float(np.linalg.norm(preload)),
        "per_joint_preload_rad": preload,
        "active_finger_mask": active,
        "active_fingers": [
            name for name, flag in zip(FINGER_NAMES, active, strict=True) if flag
        ],
        "active_finger_count": int(np.count_nonzero(active)),
        "thumb_present": bool(active[0]),
        "per_finger_normal_force_n": forces,
        "total_force_n": total_force,
        "force_hhi": hhi,
        "effective_force_fingers": float(1.0 / hhi) if total_force > 1e-12 else 0.0,
        "contact_count": count,
        "contact_topology": "+".join(
            name for name, flag in zip(FINGER_NAMES, active, strict=True) if flag
        ),
        "contact_persistence_s": float(
            grasp["longest_persistent_multifinger_frames"] / robot.control_hz
        ),
        "designated_tip_fraction": _safe_div(
            roles.get("designated_tip_geom", 0), count
        ) or 0.0,
        "distal_non_tip_fraction": _safe_div(
            roles.get("distal_non_tip_geom", 0), count
        ) or 0.0,
        "interior_contact_fraction": _safe_div(regions.get("interior", 0), count) or 0.0,
        "edge_contact_fraction": _safe_div(regions.get("edge", 0), count) or 0.0,
        "corner_contact_fraction": _safe_div(regions.get("corner", 0), count) or 0.0,
        "contact_slip_mean_m_s": float(np.mean(slips)) if slips else None,
        "contact_slip_max_m_s": float(np.max(slips)) if slips else None,
        "cube_pose_world": np.r_[
            telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"]
        ],
        "cube_pose_relative_to_palm": np.r_[relative_position, relative_quaternion],
        "cube_palm_rotation_deg": _quat_rotation_deg(relative_quaternion),
        "cube_linear_velocity_world_m_s": cube_linear,
        "cube_angular_velocity_world_rad_s": cube_angular,
        "grasp_center_linear_velocity_world_m_s": palm_linear,
        "grasp_center_angular_velocity_world_rad_s": palm_angular,
        "cube_linear_speed_m_s": float(np.linalg.norm(cube_linear)),
        "cube_angular_speed_rad_s": float(np.linalg.norm(cube_angular)),
        "relative_linear_speed_m_s": float(np.linalg.norm(cube_linear - palm_linear)),
        "relative_angular_speed_rad_s": float(np.linalg.norm(cube_angular - palm_angular)),
        "resultant_force_world_n": resultant_force,
        "resultant_moment_world_nm": resultant_moment,
        "resultant_force_n": float(np.linalg.norm(resultant_force)),
        "resultant_moment_nm": float(np.linalg.norm(resultant_moment)),
        "prelift_cube_motion_m": float(prelift_cube_motion_m),
        "grasp_gate_components": {
            "grasp_success": bool(grasp["grasp_success"]),
            "preload_success": bool(grasp["preload_success"]),
            "terminal_hold_success": bool(grasp["terminal_hold_success"]),
            "stable_contact_window_seen": bool(grasp["stable_contact_window_seen"]),
            "failure_stage": grasp["failure_stage"],
        },
    }
    return result


def _feature_row(sample: dict[str, Any]) -> dict[str, float]:
    feature = sample["prelift_features"]
    row = {"preload_l2_rad": float(feature["preload_l2_rad"])}
    preload = feature["per_joint_preload_rad"]
    for finger in range(5):
        for joint in range(4):
            row[f"preload_finger{finger + 1}_joint{joint + 1}_rad"] = float(
                preload[finger * 4 + joint]
            )
    row.update({
        "active_finger_count": float(feature["active_finger_count"]),
        "thumb_present": float(feature["thumb_present"]),
        **{
            f"force_finger{index + 1}_n": float(value)
            for index, value in enumerate(feature["per_finger_normal_force_n"])
        },
        "total_force_n": float(feature["total_force_n"]),
        "force_hhi": float(feature["force_hhi"]),
        "effective_force_fingers": float(feature["effective_force_fingers"]),
        "contact_count": float(feature["contact_count"]),
        "contact_persistence_s": float(feature["contact_persistence_s"]),
        "designated_tip_fraction": float(feature["designated_tip_fraction"]),
        "distal_non_tip_fraction": float(feature["distal_non_tip_fraction"]),
        "interior_contact_fraction": float(feature["interior_contact_fraction"]),
        "edge_contact_fraction": float(feature["edge_contact_fraction"]),
        "corner_contact_fraction": float(feature["corner_contact_fraction"]),
        "contact_slip_mean_m_s": _as_float(feature["contact_slip_mean_m_s"]),
        "contact_slip_max_m_s": _as_float(feature["contact_slip_max_m_s"]),
        "cube_palm_x_m": float(feature["cube_pose_relative_to_palm"][0]),
        "cube_palm_y_m": float(feature["cube_pose_relative_to_palm"][1]),
        "cube_palm_z_m": float(feature["cube_pose_relative_to_palm"][2]),
        "cube_palm_rotation_deg": float(feature["cube_palm_rotation_deg"]),
        "cube_linear_speed_m_s": float(feature["cube_linear_speed_m_s"]),
        "cube_angular_speed_rad_s": float(feature["cube_angular_speed_rad_s"]),
        "relative_linear_speed_m_s": float(feature["relative_linear_speed_m_s"]),
        "relative_angular_speed_rad_s": float(feature["relative_angular_speed_rad_s"]),
        "resultant_force_n": float(feature["resultant_force_n"]),
        "resultant_moment_nm": float(feature["resultant_moment_nm"]),
        "prelift_cube_motion_m": float(feature["prelift_cube_motion_m"]),
    })
    if tuple(row) != FEATURE_NAMES:
        raise RuntimeError("feature construction order does not match FEATURE_NAMES")
    return row


def _state_is_physically_valid(*, robot, task, nominal_cube: np.ndarray,
                               workspace_radius_m: float) -> tuple[bool, dict[str, Any]]:
    telemetry = task.task_telemetry()
    finite = bool(np.isfinite(np.r_[
        robot.data.qpos, robot.data.qvel, robot.data.ctrl,
        telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"],
        telemetry["cube_linear_velocity_world_m_s"],
        telemetry["cube_angular_velocity_world_rad_s"],
    ]).all())
    cube = np.asarray(telemetry["cube_position_m"], dtype=float)
    distance = float(np.linalg.norm(cube - nominal_cube))
    workspace_ok = bool(distance <= workspace_radius_m)
    return finite and workspace_ok, {
        "finite": finite,
        "cube_distance_from_nominal_m": distance,
        "diagnostic_workspace_radius_m": workspace_radius_m,
        "workspace_ok": workspace_ok,
    }


def _save_snapshot(path: Path, snapshot: dict[str, np.ndarray],
                   metadata: dict[str, Any]) -> None:
    np.savez_compressed(
        path, **snapshot,
        metadata_json=np.asarray(json.dumps(_plain(metadata))),
    )


def _lift_outcome(lift: dict[str, Any]) -> str:
    if lift["success"]:
        return "success"
    reason = str(lift.get("failure_reason"))
    return {
        "drop": "drop",
        "environment_support": "environment_support",
        "height_not_held": "height_hold_failure",
    }.get(reason, "other")


def run_one(*, seed: int, rollout_index: int, robot, config: dict[str, Any],
            reach, approach, recovery, grasp, router: dict[str, Any],
            preload_gate_rad: float, grasp_timeout_frames: int,
            workspace_radius_m: float, output: Path) -> dict[str, Any]:
    """Run a frozen staged episode and force Lift after either gate label."""
    upstream_dir = output / "upstream"
    grasp_dir = output / "grasp"
    terminal_dir = output / "terminal_states"
    lift_dir = output / "lift"
    for directory in (upstream_dir, grasp_dir, terminal_dir, lift_dir):
        directory.mkdir(parents=True, exist_ok=True)
    upstream = run_frozen_upstream(
        seed=seed, rollout_index=rollout_index, robot=robot, config=config,
        reach_policy=reach, approach_policy=approach,
        recovery_policy=recovery, output=upstream_dir, router_spec=router,
    )
    row: dict[str, Any] = {
        "rollout_index": rollout_index,
        "seed": seed,
        "reach_success": bool(upstream["reach_success"]),
        "approach_stage_success": bool(upstream["new_success"]),
        "recovery_attempted": bool(upstream.get("recovery_attempted", False)),
        "graspsecure_attempted": False,
        "graspsecure_pass": None,
        "formal_safety_pass": None,
        "formal_safety_rejected": None,
        "graspsecure_gate_overridden": False,
        "prelift_safety_overridden": False,
        "diagnostic_forced_lift": False,
        "physical_validity_pass": None,
        "lift_attempted": False,
        "lift_success": None,
        "lift_outcome": None,
        "expert_action_supplied_to_policy": False,
        "state_reset_between_grasp_and_lift": False,
    }
    if not upstream["new_success"]:
        row["upstream_failure"] = upstream["new_outcome"]
        return row

    task = ReachGraspLiftTask(robot, config)
    initialize_task_from_handoff(task)
    row["graspsecure_attempted"] = True
    label = f"rollout_{rollout_index:06d}_seed_{seed:06d}"
    grasp_path = grasp_dir / f"{label}.npz"
    grasp_metrics = _run_grasp(
        robot=robot, task=task, policy=grasp, seed=seed,
        source_kind="live_staged_handoff_gate_diagnostic",
        timeout_frames=grasp_timeout_frames,
        preload_gate_rad=preload_gate_rad, output=grasp_dir,
        episode_label=label, save_images=False,
    )
    row["graspsecure_pass"] = bool(grasp_metrics["grasp_preload_success"])
    row["graspsecure_failure_stage"] = grasp_metrics["failure_stage"]
    row["graspsecure_trajectory"] = grasp_path

    upstream_motion = float(upstream["new"]["maximum_cube_displacement_m"])
    grasp_motion = float(grasp_metrics["maximum_cube_displacement_m"])
    prelift_motion = max(upstream_motion, grasp_motion)
    safety_reference = float(config["grasp"]["max_approach_cube_displacement_m"])
    row["prelift_cube_motion_m"] = prelift_motion
    row["prelift_safety_reference_m"] = safety_reference
    row["formal_safety_pass"] = bool(prelift_motion <= safety_reference)
    row["formal_safety_rejected"] = not row["formal_safety_pass"]
    row["graspsecure_gate_overridden"] = not row["graspsecure_pass"]
    row["prelift_safety_overridden"] = not row["formal_safety_pass"]
    row["diagnostic_forced_lift"] = bool(
        row["graspsecure_gate_overridden"] or row["prelift_safety_overridden"]
    )
    row["prelift_features"] = terminal_features(
        robot=robot, task=task, grasp=grasp_metrics,
        prelift_cube_motion_m=prelift_motion,
        minimum_force_n=float(config["grasp"]["min_normal_force_n"]),
    )

    nominal_cube = np.asarray(config["reset"]["cube_nominal_position_m"], dtype=float)
    valid, validity = _state_is_physically_valid(
        robot=robot, task=task, nominal_cube=nominal_cube,
        workspace_radius_m=workspace_radius_m,
    )
    row["physical_validity_pass"] = valid
    row["physical_validity"] = validity
    snapshot_path = terminal_dir / f"terminal_{rollout_index:06d}_seed_{seed:06d}.npz"
    _save_snapshot(
        snapshot_path,
        capture_simulator_snapshot(robot, task, observation=robot.get_observation()),
        {
            "seed": seed,
            "rollout_index": rollout_index,
            "graspsecure_pass": row["graspsecure_pass"],
            "formal_safety_pass": row["formal_safety_pass"],
            "diagnostic_forced_lift": row["diagnostic_forced_lift"],
            "prelift_features": row["prelift_features"],
        },
    )
    row["terminal_snapshot"] = snapshot_path
    if not valid:
        row["diagnostic_exclusion_reason"] = "nonfinite_or_gross_workspace_escape"
        return row

    row["lift_attempted"] = True
    lift_metrics, lift_arrays = run_scripted_lift_from_handoff(
        task, seed=seed, terminal_hold_s=1.0
    )
    lift_path = lift_dir / f"lift_{rollout_index:06d}_seed_{seed:06d}.npz"
    np.savez_compressed(
        lift_path, **lift_arrays,
        metrics_json=np.asarray(json.dumps(_plain(lift_metrics))),
        prelift_features_json=np.asarray(json.dumps(_plain(row["prelift_features"]))),
        graspsecure_pass=np.asarray(row["graspsecure_pass"]),
        formal_safety_pass=np.asarray(row["formal_safety_pass"]),
        diagnostic_forced_lift=np.asarray(row["diagnostic_forced_lift"]),
    )
    row["lift_trajectory"] = lift_path
    row["lift"] = lift_metrics
    row["lift_success"] = bool(lift_metrics["success"])
    row["lift_outcome"] = _lift_outcome(lift_metrics)
    return row


def threshold_scan(values: np.ndarray, labels: np.ndarray,
                   current_threshold: float) -> list[dict[str, Any]]:
    """Evaluate preload-only thresholds without changing the formal gate."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    if len(values) == 0:
        return []
    thresholds = np.unique(np.r_[
        np.linspace(float(values.min()), float(values.max()), 101),
        float(current_threshold),
    ])
    rows = []
    for threshold in thresholds:
        metrics = confusion_metrics(values >= threshold, labels)
        precision = metrics["precision_P_lift_success_given_gate_pass"]
        recall = metrics["recall_sensitivity"]
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None
            and precision + recall > 0 else 0.0
        )
        rows.append({
            "threshold_rad": float(threshold),
            "is_current_threshold": bool(abs(threshold - current_threshold) < 1e-12),
            "precision": precision,
            "recall": recall,
            "false_negative_rate": metrics["false_negative_rate"],
            "specificity": metrics["specificity"],
            "f1": float(f1),
            "confusion": {key: metrics[key] for key in ("TP", "FP", "FN", "TN")},
        })
    return rows


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    comparisons = pos[:, None] - neg[None, :]
    return float((np.count_nonzero(comparisons > 0)
                  + 0.5 * np.count_nonzero(comparisons == 0))
                 / comparisons.size)


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray
                            ) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    confusion = confusion_metrics(probabilities >= 0.5, labels)
    recall = confusion["recall_sensitivity"]
    specificity = confusion["specificity"]
    return {
        "roc_auc": _auc(labels, probabilities),
        "accuracy": confusion["accuracy"],
        "balanced_accuracy": (
            0.5 * (recall + specificity)
            if recall is not None and specificity is not None else None
        ),
        "precision": confusion["precision_P_lift_success_given_gate_pass"],
        "recall": recall,
        "specificity": specificity,
        "confusion": {key: confusion[key] for key in ("TP", "FP", "FN", "TN")},
    }


def _impute_and_standardize(train: np.ndarray, test: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float).copy()
    test = np.asarray(test, dtype=float).copy()
    medians = np.nanmedian(train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-9] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0,
                  iterations: int = 100) -> np.ndarray:
    """Small deterministic class-balanced L2 logistic regression via IRLS."""
    x = np.c_[np.ones(len(x)), np.asarray(x, dtype=float)]
    y = np.asarray(y, dtype=float)
    positives = max(float(np.sum(y)), 1.0)
    negatives = max(float(len(y) - np.sum(y)), 1.0)
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * positives),
                             len(y) / (2.0 * negatives))
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.r_[0.0, np.ones(x.shape[1] - 1)]
    for _ in range(iterations):
        logits = np.clip(x @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ (sample_weight * (probability - y)) + l2 * penalty * beta
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = x.T @ (x * curvature[:, None])
        hessian += np.diag(l2 * penalty + 1e-8)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return beta


def _predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    logits = np.clip(np.c_[np.ones(len(x)), x] @ beta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _stratified_folds(labels: np.ndarray, folds: int, seed: int
                      ) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    for value in (False, True):
        indices = np.flatnonzero(labels == value)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            buckets[offset % folds].append(int(index))
    return [np.asarray(sorted(bucket), dtype=int) for bucket in buckets]


def cross_validated_logistic(features: np.ndarray, labels: np.ndarray,
                             names: tuple[str, ...], seed: int = 7
                             ) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    minimum_class = int(min(np.count_nonzero(labels), np.count_nonzero(~labels)))
    folds = min(5, minimum_class)
    if folds < 2:
        return {"available": False, "reason": "fewer than two samples in one class"}
    probabilities = np.zeros(len(labels), dtype=float)
    fold_rows = []
    all_indices = np.arange(len(labels))
    for fold, test_index in enumerate(_stratified_folds(labels, folds, seed)):
        train_index = np.setdiff1d(all_indices, test_index, assume_unique=True)
        x_train, x_test, _, _ = _impute_and_standardize(
            features[train_index], features[test_index]
        )
        beta = _fit_logistic(x_train, labels[train_index])
        probabilities[test_index] = _predict_logistic(beta, x_test)
        fold_rows.append({
            "fold": fold,
            "train_samples": len(train_index),
            "test_samples": len(test_index),
            **_classification_metrics(labels[test_index], probabilities[test_index]),
        })
    x_full, _, mean, std = _impute_and_standardize(features, features)
    beta = _fit_logistic(x_full, labels)
    coefficients = sorted(
        ({
            "feature": name,
            "standardized_coefficient": float(beta[index + 1]),
            "absolute_coefficient": float(abs(beta[index + 1])),
        } for index, name in enumerate(names)),
        key=lambda item: item["absolute_coefficient"], reverse=True,
    )
    return {
        "available": True,
        "implementation": "NumPy class-balanced L2 logistic regression (IRLS)",
        "features_are_pre_lift_only": True,
        "folds": folds,
        "out_of_fold": _classification_metrics(labels, probabilities),
        "fold_metrics": fold_rows,
        "full_fit_intercept": float(beta[0]),
        "full_fit_coefficients": coefficients,
        "imputation_and_scaling": {
            "median": np.nan_to_num(np.nanmedian(
                np.where(np.isfinite(features), features, np.nan), axis=0
            )),
            "mean_after_imputation": mean,
            "std_after_imputation": std,
        },
        "out_of_fold_probabilities": probabilities,
    }


def _distribution(values: list[float]) -> dict[str, Any] | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _group_analysis(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "preload_l2_rad", "active_finger_count", "thumb_present",
        "force_finger1_n", "force_finger2_n", "force_finger3_n",
        "force_finger4_n", "force_finger5_n",
        "total_force_n", "force_hhi", "effective_force_fingers",
        "contact_count", "contact_persistence_s", "designated_tip_fraction",
        "distal_non_tip_fraction", "interior_contact_fraction",
        "edge_contact_fraction", "corner_contact_fraction",
        "contact_slip_mean_m_s", "contact_slip_max_m_s",
        "cube_palm_x_m", "cube_palm_y_m", "cube_palm_z_m",
        "cube_palm_rotation_deg",
        "cube_linear_speed_m_s", "cube_angular_speed_rad_s",
        "relative_linear_speed_m_s", "relative_angular_speed_rad_s",
        "resultant_force_n", "resultant_moment_nm", "prelift_cube_motion_m",
    )
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key in ("TP", "FP", "FN", "TN")}
    for sample in samples:
        key = (
            "TP" if sample["graspsecure_pass"] and sample["lift_success"] else
            "FP" if sample["graspsecure_pass"] else
            "FN" if sample["lift_success"] else "TN"
        )
        groups[key].append(sample)
    result = {}
    for key, rows in groups.items():
        feature_rows = [_feature_row(row) for row in rows]
        result[key] = {
            "count": len(rows),
            "lift_outcome_distribution": dict(Counter(
                str(row["lift_outcome"]) for row in rows
            )),
            "graspsecure_failure_stage_distribution": dict(Counter(
                str(row["graspsecure_failure_stage"]) for row in rows
            )),
            "contact_topology_distribution": dict(Counter(
                str(row["prelift_features"]["contact_topology"]) for row in rows
            )),
            "features": {
                field: _distribution([
                    _as_float(row[field]) for row in feature_rows
                ]) for field in fields
            },
        }
    return result


def _svg_polyline_plot(path: Path, scan: list[dict[str, Any]]) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 45, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = np.asarray([row["threshold_rad"] for row in scan])
    xmin, xmax = float(x_values.min()), float(x_values.max())
    if xmax <= xmin:
        xmax = xmin + 1.0
    series = (
        ("precision", "#2563eb"),
        ("recall", "#16a34a"),
        ("false_negative_rate", "#dc2626"),
    )
    def x_pos(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w
    def y_pos(value: float) -> float:
        return top + (1.0 - value) * plot_h
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">Preload-only threshold diagnostic</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = y_pos(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    for tick in range(6):
        value = xmin + tick * (xmax - xmin) / 5
        x = x_pos(value)
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.2f}</text>')
    lines.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">preload threshold (rad)</text>',
    ])
    for offset, (name, color) in enumerate(series):
        points = []
        for row in scan:
            value = row[name]
            if value is not None:
                points.append(f'{x_pos(row["threshold_rad"]):.1f},{y_pos(value):.1f}')
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>')
        legend_x = left + 15 + offset * 220
        lines.append(f'<line x1="{legend_x}" y1="{top + 15}" x2="{legend_x + 28}" y2="{top + 15}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 36}" y="{top + 20}" font-family="sans-serif" font-size="13">{name}</text>')
    current = next((row for row in scan if row["is_current_threshold"]), None)
    if current is not None:
        x = x_pos(current["threshold_rad"])
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="6 5"/>')
        lines.append(f'<text x="{x + 5:.1f}" y="{top + plot_h - 10}" font-family="sans-serif" font-size="12" fill="#7c3aed">current 1.1775</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def _svg_bar_plot(path: Path, coefficients: list[dict[str, Any]], limit: int = 15) -> None:
    rows = coefficients[:limit]
    width, row_height = 1050, 30
    height = 80 + row_height * len(rows)
    left, right, top = 330, 50, 45
    plot_w = width - left - right
    maximum = max((row["absolute_coefficient"] for row in rows), default=1.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="525" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">Pre-Lift logistic feature coefficients (full fit)</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        bar = row["absolute_coefficient"] / max(maximum, 1e-12) * plot_w
        color = "#16a34a" if row["standardized_coefficient"] >= 0 else "#dc2626"
        lines.append(f'<text x="{left - 10}" y="{y + 18}" text-anchor="end" font-family="sans-serif" font-size="12">{row["feature"]}</text>')
        lines.append(f'<rect x="{left}" y="{y + 4}" width="{bar:.1f}" height="18" fill="{color}" opacity="0.82"/>')
        lines.append(f'<text x="{left + bar + 6:.1f}" y="{y + 18}" font-family="sans-serif" font-size="12">{row["standardized_coefficient"]:+.3f}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_parquet(path: Path, samples: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for sample in samples:
        row = {
            "seed": int(sample["seed"]),
            "rollout_index": int(sample["rollout_index"]),
            "graspsecure_pass": bool(sample["graspsecure_pass"]),
            "formal_safety_pass": bool(sample["formal_safety_pass"]),
            "diagnostic_forced_lift": bool(sample["diagnostic_forced_lift"]),
            "lift_success": bool(sample["lift_success"]),
            "lift_outcome": str(sample["lift_outcome"]),
            "graspsecure_failure_stage": str(sample["graspsecure_failure_stage"]),
            "contact_topology": str(sample["prelift_features"]["contact_topology"]),
            **_feature_row(sample),
        }
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def _median_iqr(group: dict[str, Any], field: str, scale: float = 1.0) -> str:
    distribution = group["features"].get(field)
    if distribution is None:
        return "N/A"
    return (
        f'{distribution["median"] * scale:.3f} '
        f'[{distribution["q25"] * scale:.3f}, '
        f'{distribution["q75"] * scale:.3f}]'
    )


def _write_report(path: Path, summary: dict[str, Any], output: Path) -> None:
    counts = summary["collection"]
    matrix = summary["confusion_matrix"]
    groups = summary["error_group_analysis"]
    scan = summary["preload_threshold_scan"]
    current = scan["current_threshold_preload_only"]
    best = scan["best_f1"]
    logistic = summary["pre_lift_feature_analysis"]["logistic_regression"]
    top = logistic.get("full_fit_coefficients", [])[:10]
    relative_output = output.resolve().as_posix()
    rows = []
    for group in ("TP", "FP", "FN", "TN"):
        rows.append(
            f'| {group} | {groups[group]["count"]} | '
            f'{_median_iqr(groups[group], "preload_l2_rad")} | '
            f'{_median_iqr(groups[group], "active_finger_count")} | '
            f'{_median_iqr(groups[group], "total_force_n")} | '
            f'{_median_iqr(groups[group], "contact_persistence_s")} | '
            f'{_median_iqr(groups[group], "contact_slip_mean_m_s", 1000.0)} | '
            f'{_median_iqr(groups[group], "prelift_cube_motion_m", 1000.0)} |'
        )
    coefficient_rows = "\n".join(
        f'| {item["feature"]} | {item["standardized_coefficient"]:+.4f} |'
        for item in top
    ) or "| N/A | N/A |"
    force_rows = []
    topology_rows = []
    outcome_rows = []
    for group in ("TP", "FP", "FN", "TN"):
        forces = " / ".join(
            _median_iqr(groups[group], f"force_finger{finger}_n")
            for finger in range(1, 6)
        )
        force_rows.append(f"| {group} | {forces} |")
        topologies = groups[group]["contact_topology_distribution"]
        topology_text = "; ".join(
            f"{name or 'none'}: {count}"
            for name, count in sorted(
                topologies.items(), key=lambda item: (-item[1], item[0])
            )
        )
        topology_rows.append(f"| {group} | {topology_text} |")
        outcomes = ", ".join(
            f"{name}: {count}"
            for name, count in groups[group]["lift_outcome_distribution"].items()
        )
        outcome_rows.append(f"| {group} | {outcomes} |")
    formal_safety = summary["formal_safety_analysis"]
    report = f"""# GraspSecure gate confusion-matrix diagnostic

## 结论

**{summary['decision']['case']}: {summary['decision']['statement']}**

本实验没有训练或修改 Reach、Approach、Recovery、GraspSecure、router 或 scripted Lift。GraspSecure gate 与原 25 mm pre-Lift safety reference 只作为标签；只要状态 finite 且 cube 未超出宽松的 diagnostic workspace guard，PASS 和 FAIL 都执行同一个 Lift。

最关键结果：

- `P(Lift success | GraspSecure PASS) = {_fmt_rate(matrix['precision_P_lift_success_given_gate_pass'])}`；
- `P(Lift success | GraspSecure FAIL) = {_fmt_rate(matrix['P_lift_success_given_gate_fail'])}`；
- recall = `{_fmt_rate(matrix['recall_sensitivity'])}`，specificity = `{_fmt_rate(matrix['specificity'])}`；
- false positives = `{matrix['FP']}`，false negatives = `{matrix['FN']}`。

## Protocol 与样本

- fresh seeds：`{counts['seed_start']}–{counts['seed_end_inclusive']}`；
- full staged rollouts：`{counts['completed_rollouts']}`；
- 到达 GraspSecure terminal evaluation：`{counts['terminal_states']}`；
- GraspSecure PASS / FAIL：`{counts['gate_pass']} / {counts['gate_fail']}`；
- formal 25 mm safety rejects：`{counts['formal_safety_rejects']}`；
- diagnostic forced Lift：`{counts['diagnostic_forced_lifts']}`；
- physical invalid/workspace exclusions：`{counts['physical_exclusions']}`；
- actual Lift attempts：`{counts['lift_attempts']}`。

GraspSecure FAIL 统一在 ACT 跑满 80-frame terminal evaluation 后取状态；PASS 在原 gate 首次通过时取状态。没有 reset、snapshot restore 或 expert action 插入 GraspSecure→Lift handoff。Lift 保持 terminal 20D hand target，轨迹、IK、gain、rate limit 和 1 s unsupported hold definition 均未修改。

## 2×2 confusion matrix

| | Lift SUCCESS | Lift FAIL |
|---|---:|---:|
| GraspSecure PASS | {matrix['TP']} | {matrix['FP']} |
| GraspSecure FAIL | {matrix['FN']} | {matrix['TN']} |

| Metric | Result |
|---|---:|
| precision / `P(success | PASS)` | {_fmt_rate(matrix['precision_P_lift_success_given_gate_pass'])} |
| false-positive rate | {_fmt_rate(matrix['false_positive_rate'])} |
| false-positive fraction among PASS | {_fmt_rate(matrix['false_positive_fraction_among_gate_pass'])} |
| false-negative count | {matrix['FN']} |
| `P(success | FAIL)` | {_fmt_rate(matrix['P_lift_success_given_gate_fail'])} |
| recall | {_fmt_rate(matrix['recall_sensitivity'])} |
| specificity | {_fmt_rate(matrix['specificity'])} |
| accuracy | {_fmt_rate(matrix['accuracy'])} |

25 mm formal pre-Lift safety reference 也只按标签评估：

- `P(Lift success | safety PASS) = {_fmt_rate(formal_safety['precision_P_lift_success_given_gate_pass'])}`；
- `P(Lift success | safety REJECT) = {_fmt_rate(formal_safety['P_lift_success_given_gate_fail'])}`。

## False-positive / false-negative terminal features

数值格式为 median `[Q25, Q75]`；slip 与 cube motion 单位为 mm/s、mm。

| Group | n | preload L2 rad | active fingers | total force N | persistence s | contact slip mm/s | cube motion mm |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

完整 per-joint preload、per-finger force、topology、relative pose/velocity 与 outcome 分布在 `summary.json` 和 `samples.parquet`。FP 表示静态 gate 通过但卸载失败；FN 表示静态 gate 拒绝但实际可承载，是本实验用于消除选择偏差的核心样本。

### Error-mode interpretation

- **FP 不是 preload 不足造成。** FP preload median 为 `{groups['FP']['features']['preload_l2_rad']['median']:.3f} rad`，与 TP 的 `{groups['TP']['features']['preload_l2_rad']['median']:.3f} rad` 重叠；两组也几乎都是五指接触。FP 的 terminal slip median 甚至低于 TP，因此单帧低 slip、五指 topology 和高 preload 都不足以证明可承重。
- **FP 的真实 Lift failure** 为 `{groups['FP']['lift_outcome_distribution']}`。这说明 gate 主要漏掉的是卸载后支撑、掉落和高度保持，而非形式上的 GraspSecure terminal 条件。
- **FN 是结构化漏检。** FN 中 `{groups['FN']['contact_topology_distribution'].get('finger1+finger2+finger3+finger4+finger5', 0)}/{groups['FN']['count']}` 仍为五指 topology；其 contact persistence median `{groups['FN']['features']['contact_persistence_s']['median']:.3f} s`，明显长于 TP 的 `{groups['TP']['features']['contact_persistence_s']['median']:.3f} s`，但 preload median 仅 `{groups['FN']['features']['preload_l2_rad']['median']:.3f} rad`。这些是真正能 Lift、却因 preload/terminal-hold conjunction 被拒绝的 grasp。
- **TN 与 FN 的主要差别不只是 preload。** TN 的 active-finger median 为 `{groups['TN']['features']['active_finger_count']['median']:.1f}`、total force median `{groups['TN']['features']['total_force_n']['median']:.3f} N`，而 FN 分别为 `{groups['FN']['features']['active_finger_count']['median']:.1f}` 和 `{groups['FN']['features']['total_force_n']['median']:.3f} N`；需要把接触覆盖、force distribution、relative pose/motion 与 persistence 联合考虑。

每指 terminal normal-force median `[Q25, Q75]`，顺序固定为 finger1/thumb → finger5/little：

| Group | f1 / f2 / f3 / f4 / f5 (N) |
|---|---|
{chr(10).join(force_rows)}

| Group | terminal contact topology distribution |
|---|---|
{chr(10).join(topology_rows)}

| Group | scripted Lift outcome distribution |
|---|---|
{chr(10).join(outcome_rows)}

## Preload threshold scan

这是 **preload-only diagnostic scan**，没有修改正式 conjunction gate。

- 当前 threshold：`{summary['preload_threshold_scan']['current_threshold_rad']:.6f} rad`；preload-only precision/recall/FNR = `{_fmt_rate(current['precision'])}` / `{_fmt_rate(current['recall'])}` / `{_fmt_rate(current['false_negative_rate'])}`。
- best-F1 threshold：`{best['threshold_rad']:.4f} rad`；precision/recall/FNR = `{_fmt_rate(best['precision'])}` / `{_fmt_rate(best['recall'])}` / `{_fmt_rate(best['false_negative_rate'])}`。
- preload-only ROC AUC：`{summary['preload_threshold_scan']['roc_auc']:.4f}`。

结论：当前 threshold 并非简单“太宽松”或“太严格”。降低阈值会减少 FN，但同时保留大量 FP；提高阈值会迅速损失 recall。即使本批样本内选择 best-F1 点，precision 也只有 `{_fmt_rate(best['precision'])}`，所以 preload 本身不是充分的 load-bearing 判据。

![Preload threshold scan](../outputs/graspsecure_gate_confusion/preload_threshold_scan.svg)

## Pre-Lift feature analysis

模型只使用 Lift 前信息；不包含 drop time、Lift drift、terminal Lift contact 或任何 future label feature。NumPy class-balanced L2 logistic regression 使用 stratified {logistic.get('folds', 'N/A')}-fold out-of-fold evaluation。

| CV metric | Result |
|---|---:|
| ROC AUC | {logistic.get('out_of_fold', {}).get('roc_auc', float('nan')):.4f} |
| balanced accuracy | {logistic.get('out_of_fold', {}).get('balanced_accuracy', float('nan')):.4f} |
| precision | {_fmt_rate(logistic.get('out_of_fold', {}).get('precision'))} |
| recall | {_fmt_rate(logistic.get('out_of_fold', {}).get('recall'))} |
| specificity | {_fmt_rate(logistic.get('out_of_fold', {}).get('specificity'))} |

标准化 full-fit coefficient 只用于解释关联，不是部署 classifier：

| Feature | coefficient toward Lift success |
|---|---:|
{coefficient_rows}

![Pre-Lift feature coefficients](../outputs/graspsecure_gate_confusion/pre_lift_feature_coefficients.svg)

## 解释与限制

{summary['decision']['rationale']}

- 本实验是 deterministic fresh-seed observational diagnostic；它量化 gate 判别能力，但不把 logistic coefficient 写成干预因果。
- threshold scan 在同一批样本上选择 best F1，只用于说明 preload 的分离上限；任何正式 gate 修改都必须在 held-out seeds 上预注册验证。
- `8 mm / 6°` 仍只是 GraspSecure static gate 中的 external diagnostic，不被宣称为 Wuji 最终标准。
- forced Lift 是仿真诊断路径，不得复制到真实机器人部署安全逻辑。

## Artifacts

- `{relative_output}/summary.json`
- `{relative_output}/samples.parquet`
- `{relative_output}/progress.jsonl`
- `{relative_output}/terminal_states/`
- `{relative_output}/grasp/`
- `{relative_output}/lift/`
- `{relative_output}/preload_threshold_scan.svg`
- `{relative_output}/pre_lift_feature_coefficients.svg`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def analyze(rows: list[dict[str, Any]], *, preload_gate_rad: float,
            output: Path, report: Path, collection: dict[str, Any]) -> dict[str, Any]:
    samples = [
        row for row in rows
        if row["graspsecure_attempted"] and row["lift_attempted"]
    ]
    gate = np.asarray([row["graspsecure_pass"] for row in samples], dtype=bool)
    success = np.asarray([row["lift_success"] for row in samples], dtype=bool)
    matrix = confusion_metrics(gate, success)
    feature_rows = [_feature_row(row) for row in samples]
    features = np.asarray([
        [row[name] for name in FEATURE_NAMES] for row in feature_rows
    ], dtype=float)
    scan_rows = threshold_scan(
        features[:, FEATURE_NAMES.index("preload_l2_rad")],
        success, preload_gate_rad,
    )
    current = min(
        scan_rows, key=lambda row: abs(row["threshold_rad"] - preload_gate_rad)
    )
    best = max(
        scan_rows,
        key=lambda row: (row["f1"], row["specificity"] or 0.0),
    )
    preload_auc = _auc(success, features[:, FEATURE_NAMES.index("preload_l2_rad")])
    logistic = cross_validated_logistic(features, success, FEATURE_NAMES)
    fail_success = matrix["P_lift_success_given_gate_fail"] or 0.0
    pass_success = matrix["precision_P_lift_success_given_gate_pass"] or 0.0
    best_precision = best["precision"] or 0.0
    best_recall = best["recall"] or 0.0
    if fail_success <= 0.05:
        case = "Case A"
        statement = "GraspSecure FAIL almost never lifts; improve PASS precision."
        rationale = (
            "GraspSecure FAIL 中的 Lift success 接近零，说明 gate recall 较好；"
            "主要错误是 PASS 内仍包含不能承重的 false positives。"
        )
    elif best_precision >= 0.80 and best_recall >= 0.80:
        case = "Case C"
        statement = "A preload threshold can materially improve both precision and recall."
        rationale = (
            "本批数据的 preload-only scan 找到 precision/recall 同时至少 80% 的阈值。"
            "这只是同样本诊断，下一步必须在 held-out seeds 预注册验证后才能改 gate。"
        )
    elif matrix["FP"] > 0 and matrix["FN"] > 0 and fail_success >= 0.10:
        case = "Case B"
        statement = "The current gate has material false positives and false negatives."
        rationale = (
            f"PASS 的 load-bearing precision 只有 {pass_success:.1%}，同时 FAIL 中仍有 "
            f"{fail_success:.1%} 能 Lift；两种方向的错误都存在。"
            "如果 preload AUC/scan 同时有限，应重做 gate 定义而非只移动一个阈值，"
            "优先加入 geometry、persistence 或小幅 physical load probe。"
        )
    else:
        case = "Case D"
        statement = "Preload alone cannot separate load-bearing grasps."
        rationale = (
            "preload-only threshold 无法同时给出高 precision 与 recall。"
            "应加入 contact geometry/persistence 或 physical load probe。"
        )
    summary = {
        "schema_version": 1,
        "experiment": "GraspSecure gate confusion matrix with diagnostic forced Lift",
        "collection": collection,
        "frozen_protocol": {
            "policies_retrained": False,
            "formal_graspsecure_gate_modified": False,
            "formal_prelift_safety_guard_modified": False,
            "scripted_lift_modified": False,
            "gate_and_safety_used_as_labels_only_in_diagnostic": True,
            "expert_action_used": False,
            "state_reset_between_grasp_and_lift": False,
            "lift_success_definition_modified": False,
        },
        "confusion_matrix": matrix,
        "formal_safety_analysis": confusion_metrics(
            np.asarray([row["formal_safety_pass"] for row in samples], dtype=bool),
            success,
        ),
        "lift_outcome_distribution": dict(Counter(
            str(row["lift_outcome"]) for row in samples
        )),
        "graspsecure_failure_stage_distribution": dict(Counter(
            str(row["graspsecure_failure_stage"]) for row in samples
            if not row["graspsecure_pass"]
        )),
        "error_group_analysis": _group_analysis(samples),
        "preload_threshold_scan": {
            "current_threshold_rad": preload_gate_rad,
            "current_threshold_preload_only": current,
            "best_f1": best,
            "roc_auc": preload_auc,
            "rows": scan_rows,
            "formal_gate_was_not_changed": True,
        },
        "pre_lift_feature_analysis": {
            "feature_names": FEATURE_NAMES,
            "logistic_regression": logistic,
            "future_lift_information_used": False,
        },
        "decision": {
            "case": case,
            "statement": statement,
            "rationale": rationale,
        },
        "samples": [{
            "seed": row["seed"],
            "rollout_index": row["rollout_index"],
            "graspsecure_pass": row["graspsecure_pass"],
            "graspsecure_failure_stage": row["graspsecure_failure_stage"],
            "formal_safety_pass": row["formal_safety_pass"],
            "diagnostic_forced_lift": row["diagnostic_forced_lift"],
            "lift_success": row["lift_success"],
            "lift_outcome": row["lift_outcome"],
            "prelift_features": row["prelift_features"],
            "terminal_snapshot": row["terminal_snapshot"],
            "graspsecure_trajectory": row["graspsecure_trajectory"],
            "lift_trajectory": row["lift_trajectory"],
        } for row in samples],
    }
    _write_json(output / "summary.json", summary)
    _write_parquet(output / "samples.parquet", samples)
    _svg_polyline_plot(output / "preload_threshold_scan.svg", scan_rows)
    if logistic.get("available"):
        _svg_bar_plot(
            output / "pre_lift_feature_coefficients.svg",
            logistic["full_fit_coefficients"],
        )
    _write_report(report, summary, output)
    return summary


def _collection_counts(rows: list[dict[str, Any]], *, seed_start: int,
                       target: int, min_class: int, max_rollouts: int
                       ) -> dict[str, Any]:
    terminals = [row for row in rows if row["graspsecure_attempted"]]
    attempts = [row for row in terminals if row["lift_attempted"]]
    return {
        "seed_start": seed_start,
        "seed_end_inclusive": rows[-1]["seed"] if rows else None,
        "target_terminal_states": target,
        "minimum_per_gate_class": min_class,
        "maximum_rollouts": max_rollouts,
        "completed_rollouts": len(rows),
        "reach_success": sum(row["reach_success"] for row in rows),
        "approach_stage_success": len(terminals),
        "terminal_states": len(terminals),
        "gate_pass": sum(row["graspsecure_pass"] is True for row in terminals),
        "gate_fail": sum(row["graspsecure_pass"] is False for row in terminals),
        "formal_safety_rejects": sum(
            row["formal_safety_rejected"] is True for row in terminals
        ),
        "diagnostic_forced_lifts": sum(
            row["diagnostic_forced_lift"] and row["lift_attempted"]
            for row in terminals
        ),
        "physical_exclusions": sum(
            row["physical_validity_pass"] is False for row in terminals
        ),
        "lift_attempts": len(attempts),
        "lift_success": sum(row["lift_success"] is True for row in attempts),
        "target_met": bool(
            len(terminals) >= target
            and sum(row["graspsecure_pass"] is True for row in terminals) >= min_class
            and sum(row["graspsecure_pass"] is False for row in terminals) >= min_class
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/graspsecure_gate_confusion",
    )
    parser.add_argument(
        "--report", type=Path,
        default=root / "docs/graspsecure_gate_confusion_matrix.md",
    )
    parser.add_argument("--mode", choices=("both", "collect", "analyze"), default="both")
    parser.add_argument("--seed-start", type=int, default=6000)
    parser.add_argument("--target-terminal-states", type=int, default=150)
    parser.add_argument("--minimum-per-gate-class", type=int, default=40)
    parser.add_argument("--max-rollouts", type=int, default=350)
    parser.add_argument("--hard-workspace-radius-m", type=float, default=0.5)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--model", type=Path,
        default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb",
    )
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/reach_grasp_lift.json",
    )
    parser.add_argument(
        "--synergies", type=Path,
        default=root / "configs/wuji_hand_left_synergies.json",
    )
    parser.add_argument(
        "--grasp-config", type=Path,
        default=root / "configs/grasp_secure_stage.json",
    )
    parser.add_argument(
        "--dataset-summary", type=Path,
        default=root / "outputs/grasp_preload_act/dataset/summary.json",
    )
    args = parser.parse_args()
    if args.overwrite and args.output.exists() and args.mode != "analyze":
        shutil.rmtree(args.output)
    if (args.mode in ("both", "collect") and args.output.exists()
            and not args.resume and not args.overwrite):
        raise FileExistsError(f"output exists: {args.output}; use --resume or --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / "progress.jsonl"

    task_config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = json.loads(args.grasp_config.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_summary.read_text(encoding="utf-8"))
    preload_gate = float(
        dataset["preload_statistics"]["evaluation_preload_gate"]
        ["hand_preload_l2_min_rad"]
    )
    paths = {
        key: root / value for key, value in {
            "reach": frozen["frozen_upstream"]["reach_checkpoint"],
            "approach": frozen["frozen_upstream"]["approach_checkpoint"],
            "recovery": frozen["frozen_upstream"]["recovery_checkpoint"],
            "grasp": frozen["checkpoint"],
            "router": frozen["frozen_upstream"]["router"],
        }.items()
    }
    rows = _load_jsonl(progress) if args.resume or args.mode == "analyze" else []
    if args.mode in ("both", "collect"):
        router = json.loads(paths["router"].read_text(encoding="utf-8"))
        reach = ReachPolicy(paths["reach"])
        approach = ApproachPolicy(
            paths["approach"],
            max_cube_displacement_m=float(
                task_config["grasp"]["max_approach_cube_displacement_m"]
            ),
        )
        recovery = RecoveryPolicy(
            paths["recovery"],
            max_cube_displacement_m=float(
                task_config["grasp"]["max_approach_cube_displacement_m"]
            ),
        )
        grasp = GraspSecurePolicy(paths["grasp"])
        for policy in (reach, approach, recovery):
            _set_deterministic(policy, args.inference_seed)
        _set_grasp_deterministic(grasp, args.inference_seed)
        robot = MujocoOpenArmWuji(
            args.model, args.synergies,
            arm_side=task_config["arm_side"], control_hz=30,
            image_height=240, image_width=320,
            front_camera=task_config["scene"]["front_camera_name"],
        )
        robot.connect()
        try:
            for index in range(len(rows), args.max_rollouts):
                counts = _collection_counts(
                    rows, seed_start=args.seed_start,
                    target=args.target_terminal_states,
                    min_class=args.minimum_per_gate_class,
                    max_rollouts=args.max_rollouts,
                )
                if counts["target_met"]:
                    break
                seed = args.seed_start + index
                row = run_one(
                    seed=seed, rollout_index=index, robot=robot,
                    config=task_config, reach=reach, approach=approach,
                    recovery=recovery, grasp=grasp, router=router,
                    preload_gate_rad=preload_gate,
                    grasp_timeout_frames=int(frozen["gate"]["timeout_frames"]),
                    workspace_radius_m=args.hard_workspace_radius_m,
                    output=args.output,
                )
                rows.append(row)
                _append_jsonl(progress, row)
                counts = _collection_counts(
                    rows, seed_start=args.seed_start,
                    target=args.target_terminal_states,
                    min_class=args.minimum_per_gate_class,
                    max_rollouts=args.max_rollouts,
                )
                print(
                    f"rollout {index + 1}/{args.max_rollouts} seed={seed} "
                    f"upstream={int(row['approach_stage_success'])} "
                    f"gate={row['graspsecure_pass']} "
                    f"safety={row['formal_safety_pass']} "
                    f"forced={int(row['diagnostic_forced_lift'])} "
                    f"lift={row['lift_success']} "
                    f"terminals={counts['terminal_states']} "
                    f"pass/fail={counts['gate_pass']}/{counts['gate_fail']}",
                    flush=True,
                )
                _write_json(args.output / "collection.partial.json", counts)
        finally:
            robot.disconnect()

    counts = _collection_counts(
        rows, seed_start=args.seed_start,
        target=args.target_terminal_states,
        min_class=args.minimum_per_gate_class,
        max_rollouts=args.max_rollouts,
    )
    _write_json(args.output / "collection.json", counts)
    manifest = {
        "schema_version": 1,
        "experiment": "GraspSecure gate confusion matrix",
        "frozen_components": paths,
        "graspsecure_preload_gate_rad": preload_gate,
        "formal_prelift_safety_reference_m": float(
            task_config["grasp"]["max_approach_cube_displacement_m"]
        ),
        "diagnostic_hard_workspace_radius_m": args.hard_workspace_radius_m,
        "gate_and_formal_safety_are_labels_only": True,
        "lift_definition_modified": False,
        "policy_training_performed": False,
    }
    _write_json(args.output / "manifest.json", manifest)
    if args.mode in ("both", "analyze"):
        if counts["lift_attempts"] == 0:
            raise RuntimeError("no forced-Lift samples are available for analysis")
        summary = analyze(
            rows, preload_gate_rad=preload_gate,
            output=args.output, report=args.report, collection=counts,
        )
        print(json.dumps({
            "collection": counts,
            "confusion": summary["confusion_matrix"],
            "decision": summary["decision"],
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
