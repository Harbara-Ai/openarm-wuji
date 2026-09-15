"""Evaluate physical grasp verification plus at most one GraspSecure retry.

No policy is trained here.  Conditional evaluation forks exact simulator
snapshots only to measure the counterfactual full-Lift outcome after probe
FAIL.  The deployable new branch itself remains continuous: probe, optional
return/settle, one frozen GraspSecure retry, second probe, then full Lift.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from openarm_wuji.policy import ApproachPolicy, ReachPolicy, RecoveryPolicy
from openarm_wuji.policy.grasp_secure_controller import GraspSecurePolicy
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import (
    capture_simulator_snapshot,
    restore_simulator_snapshot,
)
from openarm_wuji.tasks import ReachGraspLiftTask, initialize_task_from_handoff
from openarm_wuji.tasks.grasp_preload_stage import FINGER_NAMES, finger_force_vector
from openarm_wuji.tasks.grasp_probe_retry import (
    MicroLiftProbeSpec,
    continue_scripted_lift_after_probe,
    grasp_state_delta,
    return_probe_to_grasp,
    run_micro_lift_probe,
)
from openarm_wuji.tasks.scripted_lift_handoff import (
    run_scripted_lift_from_handoff,
)
from scripts.evaluate_grasp_secure_act import _run_grasp, _set_grasp_deterministic
from scripts.evaluate_graspsecure_gate_confusion import confusion_metrics
from scripts.evaluate_staged_with_recovery import (
    _run_episode as run_frozen_upstream,
    _set_deterministic,
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def _save_snapshot(path: Path, snapshot: Mapping[str, Any],
                   metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **snapshot,
        metadata_json=np.asarray(json.dumps(_plain(metadata))),
    )


def _save_trajectory(path: Path, arrays: Mapping[str, np.ndarray],
                     metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metrics_json=np.asarray(json.dumps(_plain(metrics))),
    )


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


def _safe_div(numerator: int | float, denominator: int | float
              ) -> float | None:
    return float(numerator / denominator) if denominator else None


def _fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def _fmt_number(value: float | None, *, scale: float = 1.0,
                digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * scale:.{digits}f}"


def _maximum_cube_displacement(
        arrays: Mapping[str, np.ndarray], reference_world_pose: Any,
        ) -> float | None:
    poses = np.asarray(arrays.get("cube_pose_world", []), dtype=float)
    if poses.ndim != 2 or len(poses) == 0 or poses.shape[1] < 3:
        return None
    reference = np.asarray(reference_world_pose, dtype=float)[:3]
    return float(np.max(np.linalg.norm(poses[:, :3] - reference[None, :], axis=1)))


def capture_grasp_signature(task) -> dict[str, Any]:
    """Capture the requested attempt-to-attempt state without future data."""
    robot = task.robot
    telemetry = task.task_telemetry()
    actual = robot.data.qpos[robot.hand_qpos_ids].astype(float).copy()
    target = robot.data.ctrl[robot.hand_actuator_ids].astype(float).copy()
    forces = finger_force_vector(telemetry)
    minimum_force = float(task.config["lift"]["min_normal_force_n"])
    active = forces >= minimum_force
    return {
        "target_hand_qpos_rad": target,
        "actual_hand_qpos_rad": actual,
        "per_joint_preload_rad": target - actual,
        "preload_l2_rad": float(np.linalg.norm(target - actual)),
        "active_finger_mask": active,
        "active_fingers": [
            name for name, flag in zip(FINGER_NAMES, active, strict=True) if flag
        ],
        "contact_topology": "+".join(
            name for name, flag in zip(FINGER_NAMES, active, strict=True) if flag
        ),
        "per_finger_normal_force_n": forces,
        "total_normal_force_n": float(np.sum(forces)),
        "cube_pose_relative_to_palm": np.r_[
            telemetry["object_relative_position_m"],
            telemetry["object_relative_quaternion_wxyz"],
        ],
        "cube_pose_world": np.r_[
            telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"]
        ],
        "cube_linear_velocity_world_m_s": telemetry[
            "cube_linear_velocity_world_m_s"
        ],
        "cube_angular_velocity_world_rad_s": telemetry[
            "cube_angular_velocity_world_rad_s"
        ],
    }


def meaningfully_different_grasp(delta: Mapping[str, Any]) -> bool:
    """Diagnostic-only state-change label; continuous values remain primary."""
    return bool(
        delta["topology_changed"]
        or delta["target_hand_l2_change_rad"] >= 0.05
        or delta["actual_hand_l2_change_rad"] >= 0.05
        or delta["cube_palm_translation_change_m"] >= 0.003
        or delta["cube_palm_rotation_change_deg"] >= 3.0
    )


def _restore_terminal(robot, config: dict[str, Any], *, seed: int,
                      snapshot: Mapping[str, Any]
                      ) -> tuple[ReachGraspLiftTask, dict[str, Any]]:
    task = ReachGraspLiftTask(robot, config)
    task.reset(seed)
    check = restore_simulator_snapshot(robot, task, snapshot)
    if not check["passed"]:
        raise RuntimeError(f"terminal snapshot restore failed: {check}")
    return task, check


def _state_physically_valid(task, *, workspace_radius_m: float = 0.5
                            ) -> tuple[bool, dict[str, Any]]:
    robot = task.robot
    telemetry = task.task_telemetry()
    cube = np.asarray(telemetry["cube_position_m"], dtype=float)
    reference = np.asarray(task.initial_cube_position, dtype=float)
    finite = bool(np.isfinite(np.r_[
        robot.data.qpos, robot.data.qvel, robot.data.ctrl,
        cube, telemetry["cube_quaternion_wxyz"],
    ]).all())
    displacement = float(np.linalg.norm(cube - reference))
    valid = bool(finite and displacement <= workspace_radius_m)
    return valid, {
        "valid": valid,
        "finite": finite,
        "cube_displacement_from_task_reference_m": displacement,
        "hard_workspace_radius_m": workspace_radius_m,
    }


def _outcome(metrics: Mapping[str, Any] | None) -> str:
    if metrics is None:
        return "not_attempted"
    return "success" if metrics["success"] else str(metrics["failure_reason"])


def evaluate_terminal_snapshot(
        *, robot, config: dict[str, Any], grasp_policy: GraspSecurePolicy,
        probe_spec: MicroLiftProbeSpec, snapshot: Mapping[str, Any],
        metadata: Mapping[str, Any], seed: int, index: int, output: Path,
        grasp_timeout_frames: int, preload_gate_rad: float,
        regrasp_settle_frames: int,
        counterfactual_after_probe_fail: bool,
        ) -> dict[str, Any]:
    """Matched baseline and continuous verification+retry from one terminal."""
    label = f"{index:06d}_seed_{seed:06d}"
    row: dict[str, Any] = {
        "index": index,
        "seed": seed,
        "graspsecure_pass_attempt1": bool(metadata["graspsecure_pass"]),
        "formal_safety_pass_attempt1": bool(metadata["formal_safety_pass"]),
        "counterfactual_restore_used_only_for_diagnostic": False,
        "maximum_regrasp_retries": 1,
    }

    # Matched old baseline from the exact same terminal state.  Its formal
    # system success still requires the historical GraspSecure+safety gates.
    task, baseline_restore = _restore_terminal(
        robot, config, seed=seed, snapshot=snapshot
    )
    signature1 = capture_grasp_signature(task)
    baseline_metrics, baseline_arrays = run_scripted_lift_from_handoff(
        task, seed=seed, terminal_hold_s=1.0
    )
    baseline_path = output / "baseline_lift" / f"baseline_{label}.npz"
    _save_trajectory(baseline_path, baseline_arrays, baseline_metrics)
    baseline_formal_eligible = bool(
        metadata["graspsecure_pass"] and metadata["formal_safety_pass"]
    )
    row["baseline"] = {
        "restore": baseline_restore,
        "formal_eligible": baseline_formal_eligible,
        "direct_lift_success": bool(baseline_metrics["success"]),
        "formal_system_success": bool(
            baseline_formal_eligible and baseline_metrics["success"]
        ),
        "lift_outcome": _outcome(baseline_metrics),
        "trajectory": baseline_path,
        "frames": len(baseline_arrays["phase"]),
        "maximum_cube_displacement_from_terminal_m": (
            _maximum_cube_displacement(
                baseline_arrays, signature1["cube_pose_world"]
            )
        ),
    }

    # New mechanism starts from the same exact snapshot.  From here onward it
    # is continuous except for the explicitly labelled conditional
    # counterfactual used to measure P(full Lift success | probe FAIL).
    task, new_restore = _restore_terminal(
        robot, config, seed=seed, snapshot=snapshot
    )
    probe1, probe1_arrays, context1 = run_micro_lift_probe(
        task, spec=probe_spec
    )
    probe1_path = output / "probe1" / f"probe1_{label}.npz"
    _save_trajectory(probe1_path, probe1_arrays, probe1)
    post_probe1 = capture_simulator_snapshot(robot, task)
    row["attempt1_signature"] = signature1
    row["probe1"] = {**probe1, "trajectory": probe1_path}
    row["new_restore"] = new_restore

    first_full = None
    first_full_arrays = None
    regrasp_metrics = None
    regrasp_return = None
    signature2 = None
    delta = None
    probe2 = None
    second_full = None
    actual_extra_frames = len(probe1_arrays["phase"])

    if probe1["pass"]:
        first_full, first_full_arrays = continue_scripted_lift_after_probe(
            task,
            context=context1,
            probe_arrays=probe1_arrays,
            seed=seed,
            terminal_hold_s=1.0,
        )
        path = output / "attempt1_full" / f"attempt1_full_{label}.npz"
        _save_trajectory(path, first_full_arrays, first_full)
        first_full["trajectory"] = path
        actual_extra_frames = len(first_full_arrays["phase"])
        row["probe1_counterfactual_full"] = {
            "available": True,
            "not_counterfactual_because_probe_passed": True,
            "success": bool(first_full["success"]),
            "outcome": _outcome(first_full),
        }
    else:
        if counterfactual_after_probe_fail:
            forced, forced_arrays = continue_scripted_lift_after_probe(
                task,
                context=context1,
                probe_arrays=probe1_arrays,
                seed=seed,
                terminal_hold_s=1.0,
                diagnostic_forced_after_probe_fail=True,
            )
            path = output / "probe1_forced_full" / f"forced_full_{label}.npz"
            _save_trajectory(path, forced_arrays, forced)
            row["probe1_counterfactual_full"] = {
                "available": True,
                "not_counterfactual_because_probe_passed": False,
                "success": bool(forced["success"]),
                "outcome": _outcome(forced),
                "trajectory": path,
            }
            # Resume the deployable branch at the exact post-probe state, not
            # the original grasp start.  This restore is evaluation-only.
            task, restore_after_counterfactual = _restore_terminal(
                robot, config, seed=seed, snapshot=post_probe1
            )
            row["counterfactual_restore_used_only_for_diagnostic"] = True
            row["post_probe_restore"] = restore_after_counterfactual
        else:
            row["probe1_counterfactual_full"] = {"available": False}

        regrasp_return, return_arrays = return_probe_to_grasp(
            task, context=context1, settle_frames=regrasp_settle_frames
        )
        return_path = output / "probe_return" / f"return_{label}.npz"
        _save_trajectory(return_path, return_arrays, regrasp_return)
        regrasp_return["trajectory"] = return_path
        actual_extra_frames += len(return_arrays["phase"])
        valid, validity = _state_physically_valid(task)
        row["pre_regrasp_physical_validity"] = validity
        if regrasp_return["success"] and valid:
            regrasp_output = output / "regrasp_act"
            regrasp_output.mkdir(parents=True, exist_ok=True)
            regrasp_metrics = _run_grasp(
                robot=robot,
                task=task,
                policy=grasp_policy,
                seed=seed,
                source_kind="post_probe_failure_one_retry",
                timeout_frames=grasp_timeout_frames,
                preload_gate_rad=preload_gate_rad,
                output=regrasp_output,
                episode_label=f"regrasp_{label}",
                save_images=False,
            )
            actual_extra_frames += int(regrasp_metrics["frames"])
            signature2 = capture_grasp_signature(task)
            delta = grasp_state_delta(signature1, signature2)
            delta["meaningfully_different"] = meaningfully_different_grasp(delta)
            valid2, validity2 = _state_physically_valid(task)
            row["post_regrasp_physical_validity"] = validity2
            if valid2:
                probe2, probe2_arrays, context2 = run_micro_lift_probe(
                    task, spec=probe_spec
                )
                actual_extra_frames += len(probe2_arrays["phase"])
                probe2_path = output / "probe2" / f"probe2_{label}.npz"
                _save_trajectory(probe2_path, probe2_arrays, probe2)
                probe2["trajectory"] = probe2_path
                if probe2["pass"]:
                    second_full, second_full_arrays = (
                        continue_scripted_lift_after_probe(
                            task,
                            context=context2,
                            probe_arrays=probe2_arrays,
                            seed=seed,
                            terminal_hold_s=1.0,
                        )
                    )
                    path = output / "attempt2_full" / f"attempt2_full_{label}.npz"
                    _save_trajectory(path, second_full_arrays, second_full)
                    second_full["trajectory"] = path
                    actual_extra_frames += (
                        len(second_full_arrays["phase"])
                        - len(probe2_arrays["phase"])
                    )

    final_full = first_full if first_full is not None else second_full
    final_full_arrays = (
        first_full_arrays if first_full is not None
        else second_full_arrays if second_full is not None
        else None
    )
    row.update({
        "regrasp_attempted": bool(not probe1["pass"]),
        "probe1_full_label_success": (
            bool(first_full["success"])
            if first_full is not None
            else row["probe1_counterfactual_full"].get("success")
        ),
        "single_attempt_success": bool(
            probe1["pass"] and first_full is not None and first_full["success"]
        ),
        "probe_return": regrasp_return,
        "regrasp": regrasp_metrics,
        "attempt2_signature": signature2,
        "attempt1_vs_attempt2": delta,
        "probe2": probe2,
        "attempt1_full": first_full,
        "attempt2_full": second_full,
        "new_system_success": bool(
            final_full is not None and final_full["success"]
        ),
        "new_system_lift_outcome": _outcome(final_full),
        "new_maximum_cube_displacement_from_attempt1_terminal_m": (
            _maximum_cube_displacement(
                final_full_arrays, signature1["cube_pose_world"]
            ) if final_full_arrays is not None else None
        ),
        "actual_new_branch_frames_after_terminal": actual_extra_frames,
        "baseline_branch_frames_after_terminal": len(baseline_arrays["phase"]),
        "extra_frames_vs_baseline": (
            actual_extra_frames - len(baseline_arrays["phase"])
        ),
        "state_reset_to_original_grasp_start_in_new_branch": False,
        "expert_action_used": False,
    })
    return row


def summarize_terminal_rows(rows: list[dict[str, Any]], *,
                            group: str) -> dict[str, Any]:
    n = len(rows)
    probe_pass = np.asarray([row["probe1"]["pass"] for row in rows], dtype=bool)
    labelled = [
        row for row in rows if row["probe1_full_label_success"] is not None
    ]
    predictive = confusion_metrics(
        np.asarray([row["probe1"]["pass"] for row in labelled], dtype=bool),
        np.asarray([row["probe1_full_label_success"] for row in labelled], dtype=bool),
    ) if labelled else None
    first_pass_rows = [row for row in rows if row["probe1"]["pass"]]
    first_fail_rows = [row for row in rows if not row["probe1"]["pass"]]
    retry_rows = [row for row in rows if row["regrasp_attempted"]]
    second_probe_rows = [row for row in retry_rows if row["probe2"] is not None]
    rescued = [
        row for row in retry_rows
        if not row["single_attempt_success"] and row["new_system_success"]
    ]
    changed = [
        row for row in retry_rows
        if row["attempt1_vs_attempt2"] is not None
    ]
    baseline_success = sum(row["baseline"]["formal_system_success"] for row in rows)
    direct_baseline_success = sum(row["baseline"]["direct_lift_success"] for row in rows)
    single = sum(row["single_attempt_success"] for row in rows)
    with_retry = sum(row["new_system_success"] for row in rows)
    failure_reasons = Counter(
        reason
        for row in rows
        for reason in row["probe1"]["failure_reasons"]
    )
    criteria_names = list(rows[0]["probe1"]["criteria"]) if rows else []
    old_gate_vs_post_probe = confusion_metrics(
        np.asarray([row["graspsecure_pass_attempt1"] for row in labelled],
                   dtype=bool),
        np.asarray([row["probe1_full_label_success"] for row in labelled],
                   dtype=bool),
    ) if labelled else None
    old_gate_vs_direct = confusion_metrics(
        np.asarray([row["graspsecure_pass_attempt1"] for row in rows],
                   dtype=bool),
        np.asarray([row["baseline"]["direct_lift_success"] for row in rows],
                   dtype=bool),
    ) if rows else None

    def probe_distributions(selected: list[dict[str, Any]]) -> dict[str, Any]:
        fields = (
            "terminal_palm_lift_m",
            "terminal_cube_lift_m",
            "supported_hold_fraction",
            "maximum_zero_contact_hold_frames",
            "maximum_relative_translation_drift_m",
            "terminal_relative_linear_speed_m_s",
            "settle_frames_before_hold",
        )
        return {
            field: _distribution([row["probe1"][field] for row in selected])
            for field in fields
        }

    future_success = [
        row for row in labelled if row["probe1_full_label_success"]
    ]
    future_failure = [
        row for row in labelled if not row["probe1_full_label_success"]
    ]
    core_names_without_drift_speed = [
        name for name in criteria_names if name != "no_fast_escape"
    ]
    core_probe_pass = np.asarray([
        all(row["probe1"]["criteria"][name]
            for name in core_names_without_drift_speed)
        for row in labelled
    ], dtype=bool)
    core_sensitivity = confusion_metrics(
        core_probe_pass,
        np.asarray([row["probe1_full_label_success"] for row in labelled],
                   dtype=bool),
    ) if labelled else None
    return {
        "group": group,
        "terminal_states": n,
        "first_probe": {
            "pass": int(np.count_nonzero(probe_pass)),
            "fail": int(n - np.count_nonzero(probe_pass)),
            "pass_rate": float(np.mean(probe_pass)) if n else None,
            "failure_reason_distribution": dict(failure_reasons),
            "criterion_pass_counts": {
                name: sum(row["probe1"]["criteria"][name] for row in rows)
                for name in criteria_names
            },
            "predictive_confusion": predictive,
            "P_full_lift_success_given_probe_pass": _safe_div(
                sum(row["probe1_full_label_success"] for row in first_pass_rows),
                len(first_pass_rows),
            ),
            "P_full_lift_success_given_probe_fail": _safe_div(
                sum(bool(row["probe1_full_label_success"])
                    for row in first_fail_rows
                    if row["probe1_full_label_success"] is not None),
                sum(row["probe1_full_label_success"] is not None
                    for row in first_fail_rows),
            ),
            "telemetry_by_post_probe_full_lift_outcome": {
                "success": probe_distributions(future_success),
                "failure": probe_distributions(future_failure),
            },
            "posthoc_remove_drift_and_speed_gate": {
                "diagnostic_only_not_a_new_formal_gate": True,
                "pass_count": int(np.count_nonzero(core_probe_pass)),
                "confusion": core_sensitivity,
            },
        },
        "old_graspsecure_gate_comparison": {
            "same_60_vs_post_probe_continuation": old_gate_vs_post_probe,
            "same_60_vs_direct_lift_from_terminal": old_gate_vs_direct,
        },
        "retry": {
            "attempted": len(retry_rows),
            "probe_return_success": sum(
                bool(row["probe_return"] and row["probe_return"]["success"])
                for row in retry_rows
            ),
            "physically_valid_before_regrasp": sum(
                bool(row.get("pre_regrasp_physical_validity", {}).get("valid"))
                for row in retry_rows
            ),
            "regrasp_gate_pass": sum(
                bool(row["regrasp"] and row["regrasp"]["grasp_preload_success"])
                for row in retry_rows
            ),
            "second_probe_attempted": len(second_probe_rows),
            "second_probe_pass": sum(row["probe2"]["pass"] for row in second_probe_rows),
            "second_probe_pass_rate": _safe_div(
                sum(row["probe2"]["pass"] for row in second_probe_rows),
                len(second_probe_rows),
            ),
            "rescued_to_full_success": len(rescued),
            "rescue_rate_given_first_probe_fail": _safe_div(
                len(rescued), len(first_fail_rows)
            ),
            "state_comparisons": len(changed),
            "physically_valid_after_regrasp": sum(
                bool(row.get("post_regrasp_physical_validity", {}).get("valid"))
                for row in retry_rows
            ),
            "meaningfully_different": sum(
                row["attempt1_vs_attempt2"]["meaningfully_different"]
                for row in changed
            ),
            "meaningfully_different_fraction": _safe_div(
                sum(row["attempt1_vs_attempt2"]["meaningfully_different"]
                    for row in changed), len(changed)
            ),
            "target_hand_l2_change_rad": _distribution([
                row["attempt1_vs_attempt2"]["target_hand_l2_change_rad"]
                for row in changed
            ]),
            "actual_hand_l2_change_rad": _distribution([
                row["attempt1_vs_attempt2"]["actual_hand_l2_change_rad"]
                for row in changed
            ]),
            "cube_palm_translation_change_m": _distribution([
                row["attempt1_vs_attempt2"]["cube_palm_translation_change_m"]
                for row in changed
            ]),
            "cube_palm_rotation_change_deg": _distribution([
                row["attempt1_vs_attempt2"]["cube_palm_rotation_change_deg"]
                for row in changed
            ]),
            "topology_changed": sum(
                row["attempt1_vs_attempt2"]["topology_changed"] for row in changed
            ),
        },
        "success_comparison": {
            "matched_formal_baseline_success": baseline_success,
            "matched_direct_lift_success_ignoring_old_gates": direct_baseline_success,
            "single_attempt_verified_success": single,
            "one_retry_success": with_retry,
            "single_attempt_rate": _safe_div(single, n),
            "one_retry_rate": _safe_div(with_retry, n),
            "absolute_retry_gain": _safe_div(with_retry - single, n),
            "matched_new_minus_formal_baseline": _safe_div(
                with_retry - baseline_success, n
            ),
        },
        "lift_outcomes": {
            "baseline_direct": dict(Counter(
                row["baseline"]["lift_outcome"] for row in rows
            )),
            "new_system": dict(Counter(
                row["new_system_lift_outcome"] for row in rows
            )),
            "probe1_continuation_label": dict(Counter(
                row["probe1_counterfactual_full"].get("outcome", "not_available")
                for row in rows
            )),
        },
        "timing": {
            "extra_frames_vs_baseline": _distribution([
                row["extra_frames_vs_baseline"] for row in rows
            ]),
            "extra_time_s_vs_baseline": _distribution([
                row["extra_frames_vs_baseline"] / 30.0 for row in rows
            ]),
        },
        "cube_displacement_from_terminal_m": {
            "baseline": _distribution([
                row["baseline"]["maximum_cube_displacement_from_terminal_m"]
                for row in rows
                if row["baseline"]["maximum_cube_displacement_from_terminal_m"]
                is not None
            ]),
            "new_system": _distribution([
                row["new_maximum_cube_displacement_from_attempt1_terminal_m"]
                for row in rows
                if row["new_maximum_cube_displacement_from_attempt1_terminal_m"]
                is not None
            ]),
        },
        "episodes": rows,
    }


def _conditional_sources(source_summary: Path, limit: int
                         ) -> list[dict[str, Any]]:
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    samples = source["samples"]
    if len(samples) < limit:
        raise ValueError(f"source contains only {len(samples)} samples")
    return samples[:limit]


def run_conditional(*, robot, config: dict[str, Any], grasp_policy,
                    probe_spec: MicroLiftProbeSpec, source_summary: Path,
                    limit: int, output: Path, grasp_timeout_frames: int,
                    preload_gate_rad: float, regrasp_settle_frames: int,
                    resume: bool) -> list[dict[str, Any]]:
    progress = output / "conditional" / "progress.jsonl"
    rows = _load_jsonl(progress) if resume else []
    sources = _conditional_sources(source_summary, limit)
    for index in range(len(rows), limit):
        source = sources[index]
        snapshot_path = Path(source["terminal_snapshot"])
        snapshot = _load_npz(snapshot_path)
        metadata = json.loads(str(snapshot.pop("metadata_json")))
        seed = int(source["seed"])
        row = evaluate_terminal_snapshot(
            robot=robot,
            config=config,
            grasp_policy=grasp_policy,
            probe_spec=probe_spec,
            snapshot=snapshot,
            metadata=metadata,
            seed=seed,
            index=index,
            output=output / "conditional",
            grasp_timeout_frames=grasp_timeout_frames,
            preload_gate_rad=preload_gate_rad,
            regrasp_settle_frames=regrasp_settle_frames,
            counterfactual_after_probe_fail=True,
        )
        row["source_terminal_snapshot"] = snapshot_path
        rows.append(row)
        _append_jsonl(progress, row)
        partial = summarize_terminal_rows(rows, group="conditional")
        partial.pop("episodes")
        _write_json(output / "conditional" / "summary.partial.json", partial)
        print(
            f"conditional {index + 1}/{limit} seed={seed} "
            f"probe1={int(row['probe1']['pass'])} "
            f"label={int(bool(row['probe1_full_label_success']))} "
            f"retry={int(row['regrasp_attempted'])} "
            f"probe2={None if row['probe2'] is None else int(row['probe2']['pass'])} "
            f"new={int(row['new_system_success'])}",
            flush=True,
        )
    return rows


def _fresh_terminal(*, robot, config: dict[str, Any], policies: dict[str, Any],
                    router: dict[str, Any], seed: int, index: int,
                    output: Path, grasp_timeout_frames: int,
                    preload_gate_rad: float) -> dict[str, Any]:
    upstream = run_frozen_upstream(
        seed=seed,
        rollout_index=index,
        robot=robot,
        config=config,
        reach_policy=policies["reach"],
        approach_policy=policies["approach"],
        recovery_policy=policies["recovery"],
        output=output / "upstream",
        router_spec=router,
    )
    result: dict[str, Any] = {
        "rollout_index": index,
        "seed": seed,
        "reach_success": bool(upstream["reach_success"]),
        "approach_stage_success": bool(upstream["new_success"]),
        "recovery_attempted": bool(upstream.get("recovery_attempted", False)),
        "terminal_reached": False,
        "physical_validity_pass": None,
        "upstream": upstream,
    }
    if not upstream["new_success"]:
        return result
    task = ReachGraspLiftTask(robot, config)
    initialize_task_from_handoff(task)
    grasp_output = output / "grasp1_act"
    grasp_output.mkdir(parents=True, exist_ok=True)
    grasp = _run_grasp(
        robot=robot,
        task=task,
        policy=policies["grasp"],
        seed=seed,
        source_kind="fresh_staged_probe_retry_handoff",
        timeout_frames=grasp_timeout_frames,
        preload_gate_rad=preload_gate_rad,
        output=grasp_output,
        episode_label=f"grasp1_{index:06d}_seed_{seed:06d}",
        save_images=False,
    )
    upstream_motion = float(upstream["new"]["maximum_cube_displacement_m"])
    grasp_motion = float(grasp["maximum_cube_displacement_m"])
    safety_reference = float(config["grasp"]["max_approach_cube_displacement_m"])
    metadata = {
        "seed": seed,
        "rollout_index": index,
        "graspsecure_pass": bool(grasp["grasp_preload_success"]),
        "formal_safety_pass": max(upstream_motion, grasp_motion) <= safety_reference,
        "prelift_cube_motion_m": max(upstream_motion, grasp_motion),
        "graspsecure": grasp,
    }
    valid, validity = _state_physically_valid(task)
    result.update({
        "terminal_reached": True,
        "physical_validity_pass": valid,
        "physical_validity": validity,
        "graspsecure": grasp,
        "metadata": metadata,
    })
    if not valid:
        return result
    snapshot = capture_simulator_snapshot(robot, task)
    path = output / "terminal_states" / (
        f"terminal_{index:06d}_seed_{seed:06d}.npz"
    )
    _save_snapshot(path, snapshot, metadata)
    result["terminal_snapshot"] = path
    result["snapshot"] = snapshot
    return result


def run_full(*, robot, config: dict[str, Any], policies: dict[str, Any],
             router: dict[str, Any], probe_spec: MicroLiftProbeSpec,
             seed_start: int, rollouts: int, output: Path,
             grasp_timeout_frames: int, preload_gate_rad: float,
             regrasp_settle_frames: int, resume: bool
             ) -> list[dict[str, Any]]:
    progress = output / "full" / "progress.jsonl"
    rows = _load_jsonl(progress) if resume else []
    for index in range(len(rows), rollouts):
        seed = seed_start + index
        fresh = _fresh_terminal(
            robot=robot,
            config=config,
            policies=policies,
            router=router,
            seed=seed,
            index=index,
            output=output / "full",
            grasp_timeout_frames=grasp_timeout_frames,
            preload_gate_rad=preload_gate_rad,
        )
        row: dict[str, Any] = {
            key: value for key, value in fresh.items() if key != "snapshot"
        }
        if fresh["terminal_reached"] and fresh["physical_validity_pass"]:
            terminal = evaluate_terminal_snapshot(
                robot=robot,
                config=config,
                grasp_policy=policies["grasp"],
                probe_spec=probe_spec,
                snapshot=fresh["snapshot"],
                metadata=fresh["metadata"],
                seed=seed,
                index=index,
                output=output / "full",
                grasp_timeout_frames=grasp_timeout_frames,
                preload_gate_rad=preload_gate_rad,
                regrasp_settle_frames=regrasp_settle_frames,
                counterfactual_after_probe_fail=False,
            )
            row["terminal_evaluation"] = terminal
            row["baseline_full_task_success"] = terminal[
                "baseline"
            ]["formal_system_success"]
            row["new_full_task_success"] = terminal["new_system_success"]
        else:
            row["terminal_evaluation"] = None
            row["baseline_full_task_success"] = False
            row["new_full_task_success"] = False
        rows.append(row)
        _append_jsonl(progress, row)
        print(
            f"full {index + 1}/{rollouts} seed={seed} "
            f"reach={int(row['reach_success'])} "
            f"approach={int(row['approach_stage_success'])} "
            f"terminal={int(row['terminal_reached'])} "
            f"baseline={int(row['baseline_full_task_success'])} "
            f"new={int(row['new_full_task_success'])}",
            flush=True,
        )
    return rows


def summarize_full(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    reach = sum(row["reach_success"] for row in rows)
    approach = sum(row["approach_stage_success"] for row in rows)
    terminals = [
        row["terminal_evaluation"] for row in rows
        if row["terminal_evaluation"] is not None
    ]
    terminal_summary = summarize_terminal_rows(terminals, group="full_terminal")
    terminal_summary.pop("episodes")
    baseline = sum(row["baseline_full_task_success"] for row in rows)
    new = sum(row["new_full_task_success"] for row in rows)
    baseline_drop = sum(
        row["terminal_evaluation"]["baseline"]["lift_outcome"] == "drop"
        and row["terminal_evaluation"]["baseline"]["formal_eligible"]
        for row in rows if row["terminal_evaluation"] is not None
    )
    new_outcomes = Counter(
        row["terminal_evaluation"]["new_system_lift_outcome"]
        for row in rows if row["terminal_evaluation"] is not None
    )
    return {
        "rollouts": total,
        "seed_start": rows[0]["seed"] if rows else None,
        "seed_end_inclusive": rows[-1]["seed"] if rows else None,
        "stage_counts": {
            "reach_success": reach,
            "approach_stage_success": approach,
            "graspsecure_terminal_states": len(terminals),
            "probe1_pass": sum(row["probe1"]["pass"] for row in terminals),
            "regrasp_attempted": sum(row["regrasp_attempted"] for row in terminals),
            "probe2_pass": sum(
                bool(row["probe2"] and row["probe2"]["pass"])
                for row in terminals
            ),
            "baseline_full_task_success": baseline,
            "new_full_task_success": new,
        },
        "stage_rates": {
            "P_Reach": _safe_div(reach, total),
            "P_Approach_given_Reach": _safe_div(approach, reach),
            "P_terminal_given_Approach": _safe_div(len(terminals), approach),
            "P_probe1_pass_given_terminal": _safe_div(
                sum(row["probe1"]["pass"] for row in terminals), len(terminals)
            ),
            "P_new_success_given_terminal": _safe_div(
                sum(row["new_system_success"] for row in terminals), len(terminals)
            ),
            "P_baseline_full_success": _safe_div(baseline, total),
            "P_new_full_success": _safe_div(new, total),
        },
        "matched_comparison": {
            "baseline_success": baseline,
            "new_success": new,
            "absolute_success_gain": _safe_div(new - baseline, total),
            "baseline_drop_count": baseline_drop,
            "new_lift_outcome_distribution": dict(new_outcomes),
            "cube_displacement_from_terminal_m": terminal_summary[
                "cube_displacement_from_terminal_m"
            ],
            "extra_frames_vs_baseline": terminal_summary["timing"][
                "extra_frames_vs_baseline"
            ],
            "extra_time_s_vs_baseline": terminal_summary["timing"][
                "extra_time_s_vs_baseline"
            ],
        },
        "terminal_state_analysis": terminal_summary,
        "episodes": rows,
    }


def choose_case(conditional: dict[str, Any],
                full: dict[str, Any] | None) -> dict[str, str]:
    precision = conditional["first_probe"][
        "P_full_lift_success_given_probe_pass"
    ] or 0.0
    fail_success = conditional["first_probe"][
        "P_full_lift_success_given_probe_fail"
    ] or 0.0
    gain = conditional["success_comparison"]["absolute_retry_gain"] or 0.0
    rescued = conditional["retry"]["rescued_to_full_success"]
    if precision < 0.70 or fail_success > 0.20:
        return {
            "case": "Case C",
            "statement": "Probe itself cannot yet cleanly separate good and bad grasps.",
            "next_step": "Redesign or recalibrate the probe before relying on retry logic.",
        }
    if gain < 0.05 or rescued < 2:
        return {
            "case": "Case B",
            "statement": "Probe works, but one deterministic regrasp rarely rescues failure.",
            "next_step": "Improve regrasp strategy/data while keeping the physical probe.",
        }
    if full is not None:
        terminal_rate = full["stage_rates"]["P_new_success_given_terminal"] or 0.0
        overall = full["stage_rates"]["P_new_full_success"] or 0.0
        upstream = (
            (full["stage_rates"]["P_Reach"] or 0.0)
            * (full["stage_rates"]["P_Approach_given_Reach"] or 0.0)
        )
        if terminal_rate >= 0.70 and overall + 0.10 < terminal_rate and upstream < 0.85:
            return {
                "case": "Case D",
                "statement": "Probe+retry works conditionally; upstream is now dominant.",
                "next_step": "Freeze grasp verification and optimize upstream later.",
            }
    return {
        "case": "Case A",
        "statement": "Probe predicts load-bearing and one retry clearly improves success.",
        "next_step": "Freeze the verification+one-retry mechanism.",
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    conditional = summary["conditional"]
    first = conditional["first_probe"]
    retry = conditional["retry"]
    success = conditional["success_comparison"]
    matrix = first["predictive_confusion"]
    old_gate = conditional["old_graspsecure_gate_comparison"][
        "same_60_vs_post_probe_continuation"
    ]
    criteria = first["criterion_pass_counts"]
    sensitivity = first["posthoc_remove_drift_and_speed_gate"]
    probe_config = summary["probe_config"]
    full = summary.get("full")
    full_text = "完整 fresh-seed matched evaluation 尚未运行。"
    full_artifact = (
        "`outputs/grasp_probe_regrasp_retry/full/` was intentionally not "
        "produced because the conditional prerequisite failed."
    )
    full_gate_note = (
        "因为 conditional experiment 没有证明 probe 有效，按预先协议没有启动 "
        "100–200 fresh-seed full evaluation；这不是缺失结果，而是防止把已证伪的 "
        "verification gate 接入完整系统。"
    )
    if full is not None:
        matched = full["matched_comparison"]
        stages = full["stage_counts"]
        rates = full["stage_rates"]
        full_text = f"""
| Metric | Baseline | Probe + one retry |
|---|---:|---:|
| full task success | {matched['baseline_success']} / {full['rollouts']} | {matched['new_success']} / {full['rollouts']} |
| full task rate | {_fmt_rate(rates['P_baseline_full_success'])} | {_fmt_rate(rates['P_new_full_success'])} |

- Reach：`{stages['reach_success']}/{full['rollouts']}`；Approach-stage：`{stages['approach_stage_success']}/{stages['reach_success']}`；terminal states：`{stages['graspsecure_terminal_states']}`。
- probe1 PASS：`{stages['probe1_pass']}`；regrasp attempts：`{stages['regrasp_attempted']}`；probe2 PASS：`{stages['probe2_pass']}`。
- matched absolute full-task gain：`{_fmt_rate(matched['absolute_success_gain'])}`。
- mean extra time：`{matched['extra_time_s_vs_baseline']['mean']:.3f} s` per terminal state。
- mean max cube displacement from matched terminal：baseline `{_fmt_number(matched['cube_displacement_from_terminal_m']['baseline']['mean'] if matched['cube_displacement_from_terminal_m']['baseline'] else None, scale=1000.0, digits=2)} mm`；new `{_fmt_number(matched['cube_displacement_from_terminal_m']['new_system']['mean'] if matched['cube_displacement_from_terminal_m']['new_system'] else None, scale=1000.0, digits=2)} mm`。
"""
        full_gate_note = "Conditional preregistered gate passed before this run."
        full_artifact = "`outputs/grasp_probe_regrasp_retry/full/`"
    target_change = retry["target_hand_l2_change_rad"]
    actual_change = retry["actual_hand_l2_change_rad"]
    translation_change = retry["cube_palm_translation_change_m"]
    rotation_change = retry["cube_palm_rotation_change_deg"]
    report = f"""# Grasp verification + one regrasp retry

## 结论

**{summary['decision']['case']}: {summary['decision']['statement']}**

本实验没有训练或修改任何 ACT，也没有修改原 scripted Lift、IK、gain、rate limit、controller 或 27D action semantics。micro-lift 使用原 120 mm S-curve 的前缀，Wuji 始终保持 live 20D terminal controller target。

## Conditional probe experiment

- terminal states：`{conditional['terminal_states']}`；
- first probe PASS/FAIL：`{first['pass']}/{first['fail']}`；
- first-probe pass rate：`{_fmt_rate(first['pass_rate'])}`；
- `P(full Lift success | probe PASS) = {_fmt_rate(first['P_full_lift_success_given_probe_pass'])}`；
- `P(full Lift success | probe FAIL) = {_fmt_rate(first['P_full_lift_success_given_probe_fail'])}`。

| | Full Lift success | Full Lift fail |
|---|---:|---:|
| Probe PASS | {matrix['TP']} | {matrix['FP']} |
| Probe FAIL | {matrix['FN']} | {matrix['TN']} |

这里 probe FAIL 后的 full-Lift label 只在 conditional counterfactual branch 中测量；deployable retry branch 不执行这次失败后的 full Lift，而是从真实 post-probe state 回落并 regrasp。

同一批 60 个 terminal states 上，旧 GraspSecure label 对同一个 post-probe continuation outcome 的 precision 为 `{_fmt_rate(old_gate['precision_P_lift_success_given_gate_pass'])}`，`P(success | old gate FAIL)` 为 `{_fmt_rate(old_gate['P_lift_success_given_gate_fail'])}`；虽然旧 gate 本身也弱，但 v1 probe 的 0 个 PASS 明显更差。

Probe criterion pass counts（分母 60）：

| Criterion | Pass |
|---|---:|
| finite | {criteria['finite']} |
| physical trajectory + hold completed | {criteria['trajectory_complete']} |
| palm stayed above minimum lift | {criteria['palm_moved']} |
| cube followed palm | {criteria['cube_followed']} |
| environment support released | {criteria['environment_support_released']} |
| contact retained | {criteria['contact_retained']} |
| no fast escape | {criteria['no_fast_escape']} |

Post-hoc sensitivity（不作为新 formal gate）：完全移除 translation-drift 与 terminal-speed checks 后，核心 palm/cube/support/contact 条件仍仅 `{sensitivity['pass_count']}/60` PASS；对应 TP/FP/FN/TN=`{sensitivity['confusion']['TP']}/{sensitivity['confusion']['FP']}/{sensitivity['confusion']['FN']}/{sensitivity['confusion']['TN']}`。因此 0 PASS 不能归因于 15 mm/30 mm/s 阈值本身。

## One regrasp retry

- first-probe failures / regrasp attempts：`{retry['attempted']}`；
- return succeeded / physically valid before regrasp：`{retry['probe_return_success']}/{retry['physically_valid_before_regrasp']}`；
- frozen GraspSecure gate passes on retry：`{retry['regrasp_gate_pass']}`；
- physically valid after regrasp / second probes attempted / passed：`{retry['physically_valid_after_regrasp']}/{retry['second_probe_attempted']}/{retry['second_probe_pass']}`；
- rescued to full success：`{retry['rescued_to_full_success']}`；
- rescue rate given probe1 FAIL：`{_fmt_rate(retry['rescue_rate_given_first_probe_fail'])}`；
- single-attempt verified success：`{success['single_attempt_verified_success']}/{conditional['terminal_states']} = {_fmt_rate(success['single_attempt_rate'])}`；
- one-retry success：`{success['one_retry_success']}/{conditional['terminal_states']} = {_fmt_rate(success['one_retry_rate'])}`；
- absolute retry gain：`{_fmt_rate(success['absolute_retry_gain'])}`。

Regrasp attempt1→attempt2 continuous state changes：

- meaningfully different：`{retry['meaningfully_different']}/{retry['state_comparisons']}`；
- median target/actual 20D L2 change：`{_fmt_number(target_change['median'] if target_change else None, digits=4)}/{_fmt_number(actual_change['median'] if actual_change else None, digits=4)} rad`；
- median cube-palm translation/rotation change：`{_fmt_number(translation_change['median'] if translation_change else None, scale=1000.0, digits=2)} mm / {_fmt_number(rotation_change['median'] if rotation_change else None, digits=2)}°`；
- topology changed：`{retry['topology_changed']}/{retry['state_comparisons']}`。

“meaningfully different” 仅是 diagnostic label：topology 改变，或 target/actual 20D L2 ≥0.05 rad，或 cube-palm translation ≥3 mm，或 rotation ≥3°。连续指标才是主要结果。

## Fresh-seed full staged matched comparison

{full_text}

{full_gate_note}

## Probe v1 definition

- 原 S-curve prefix target：{probe_config['height_m'] * 1000.0:.0f} mm（实际为首个超过该值的原 reference frame）；先等待 rate-limited arm 进入 physical micro-lift，再计 hold {probe_config['hold_s']:.1f} s；
- physical settle 最多 {probe_config['maximum_settle_s']:.1f} s；terminal palm lift ≥{probe_config['minimum_palm_lift_m'] * 1000.0:.0f} mm；cube lift ≥{probe_config['minimum_cube_lift_m'] * 1000.0:.0f} mm 且至少为 palm lift 的 {probe_config['minimum_cube_to_palm_lift_ratio'] * 100.0:.0f}%；
- hold 最后一帧无环境支撑，supported frames ≤1/3；
- hold 内 zero-contact longest run ≤3 frames；
- relative translation drift ≤{probe_config['maximum_relative_translation_drift_m'] * 1000.0:.0f} mm，terminal relative speed ≤{probe_config['maximum_terminal_relative_speed_m_s'] * 1000.0:.0f} mm/s；
- rotation、finger count/topology、antipodal、edge margin 均不作为 hard gate。

15 mm drift 只是本轮预注册的 diagnostic bound，不宣称为 Wuji 最终稳定标准。正式 60 条结果出来后没有为提高数字而重调。

## Interpretation

v1 probe 的主要问题不是 contact collapse：`{criteria['contact_retained']}/60` 保持了 contact。真正卡住的是 cube-follow（`{criteria['cube_followed']}/60`）、环境支撑释放（`{criteria['environment_support_released']}/60`）以及静态 micro-hold 下的 relative escape（`{criteria['no_fast_escape']}/60`）。与此同时，probe FAIL 后继续原上抬仍有 `{matrix['FN']}/60` 次完成 full Lift。这表明当前连续 scripted Lift 可以靠继续增加 arm target 进入承载状态，但在 15 mm reference 处插入静态 hold 会改变动力学，并不能作为无损的早期 load-bearing test。

Regrasp 确实不是简单重复：57 个可比状态全部触发了“不同 grasp”diagnostic，median actual-hand L2 变化 0.7540 rad，median cube-palm 平移/旋转变化 24.51 mm / 42.28°。但这种变化往往过大且没有转化成任何 probe2 PASS，因此“不同”不等于“更好”。

## Final decision

{summary['decision']['next_step']}

## Artifacts

- `outputs/grasp_probe_regrasp_retry/summary.json`
- `outputs/grasp_probe_regrasp_retry/manifest.json`
- `outputs/grasp_probe_regrasp_retry/conditional/`
- {full_artifact}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def build_summary(*, conditional_rows: list[dict[str, Any]],
                  full_rows: list[dict[str, Any]] | None,
                  config_path: Path, output: Path,
                  report: Path) -> dict[str, Any]:
    conditional = summarize_terminal_rows(
        conditional_rows, group="conditional"
    )
    full = summarize_full(full_rows) if full_rows is not None else None
    summary = {
        "schema_version": 1,
        "experiment": "grasp verification plus one regrasp retry",
        "frozen_protocol": {
            "policies_trained": False,
            "scripted_lift_modified": False,
            "controller_modified": False,
            "action_semantics": "27D absolute controller target",
            "maximum_regrasp_retries": 1,
            "new_branch_restores_original_grasp_start": False,
            "conditional_counterfactual_restore_is_deployable_logic": False,
        },
        "probe_config": json.loads(config_path.read_text(encoding="utf-8"))[
            "probe"
        ],
        "conditional": conditional,
        "full": full,
    }
    summary["decision"] = choose_case(conditional, full)
    _write_json(output / "summary.json", summary)
    _write_report(report, summary)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("conditional", "full", "analyze"),
                        default="conditional")
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/grasp_probe_regrasp_retry",
    )
    parser.add_argument(
        "--report", type=Path,
        default=root / "docs/grasp_probe_regrasp_retry.md",
    )
    parser.add_argument(
        "--probe-config", type=Path,
        default=root / "configs/grasp_probe_retry.json",
    )
    parser.add_argument(
        "--source-summary", type=Path,
        default=root / "outputs/graspsecure_gate_confusion/summary.json",
    )
    parser.add_argument("--conditional-limit", type=int, default=60)
    parser.add_argument("--full-rollouts", type=int, default=100)
    parser.add_argument("--full-seed-start", type=int, default=7000)
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

    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)
    if (args.mode != "analyze" and args.output.exists()
            and not args.resume and not args.overwrite):
        raise FileExistsError(f"output exists: {args.output}; use --resume/--overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    experiment = json.loads(args.probe_config.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = json.loads(args.grasp_config.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_summary.read_text(encoding="utf-8"))
    probe_spec = MicroLiftProbeSpec.from_mapping(experiment["probe"])
    probe_spec.validate()
    preload_gate = float(
        dataset["preload_statistics"]["evaluation_preload_gate"]
        ["hand_preload_l2_min_rad"]
    )
    grasp_timeout = int(frozen["gate"]["timeout_frames"])
    settle_frames = int(experiment["regrasp"]["return_settle_frames"])
    paths = {
        key: root / value for key, value in {
            "reach": frozen["frozen_upstream"]["reach_checkpoint"],
            "approach": frozen["frozen_upstream"]["approach_checkpoint"],
            "recovery": frozen["frozen_upstream"]["recovery_checkpoint"],
            "grasp": frozen["checkpoint"],
            "router": frozen["frozen_upstream"]["router"],
        }.items()
    }

    conditional_progress = args.output / "conditional" / "progress.jsonl"
    full_progress = args.output / "full" / "progress.jsonl"
    conditional_rows = _load_jsonl(conditional_progress)
    full_rows = _load_jsonl(full_progress) if full_progress.exists() else None

    if args.mode != "analyze":
        grasp = GraspSecurePolicy(paths["grasp"])
        _set_grasp_deterministic(grasp, args.inference_seed)
        policies: dict[str, Any] = {"grasp": grasp}
        router = None
        if args.mode == "full":
            policies.update({
                "reach": ReachPolicy(paths["reach"]),
                "approach": ApproachPolicy(
                    paths["approach"],
                    max_cube_displacement_m=float(
                        config["grasp"]["max_approach_cube_displacement_m"]
                    ),
                ),
                "recovery": RecoveryPolicy(
                    paths["recovery"],
                    max_cube_displacement_m=float(
                        config["grasp"]["max_approach_cube_displacement_m"]
                    ),
                ),
            })
            for key in ("reach", "approach", "recovery"):
                _set_deterministic(policies[key], args.inference_seed)
            router = json.loads(paths["router"].read_text(encoding="utf-8"))
        robot = MujocoOpenArmWuji(
            args.model, args.synergies,
            arm_side=config["arm_side"], control_hz=30,
            image_height=240, image_width=320,
            front_camera=config["scene"]["front_camera_name"],
        )
        robot.connect()
        try:
            if args.mode == "conditional":
                conditional_rows = run_conditional(
                    robot=robot,
                    config=config,
                    grasp_policy=grasp,
                    probe_spec=probe_spec,
                    source_summary=args.source_summary,
                    limit=args.conditional_limit,
                    output=args.output,
                    grasp_timeout_frames=grasp_timeout,
                    preload_gate_rad=preload_gate,
                    regrasp_settle_frames=settle_frames,
                    resume=args.resume,
                )
            else:
                if not conditional_rows:
                    raise RuntimeError(
                        "run conditional evaluation before fresh full evaluation"
                    )
                full_rows = run_full(
                    robot=robot,
                    config=config,
                    policies=policies,
                    router=router,
                    probe_spec=probe_spec,
                    seed_start=args.full_seed_start,
                    rollouts=args.full_rollouts,
                    output=args.output,
                    grasp_timeout_frames=grasp_timeout,
                    preload_gate_rad=preload_gate,
                    regrasp_settle_frames=settle_frames,
                    resume=args.resume,
                )
        finally:
            robot.disconnect()

    if not conditional_rows:
        raise RuntimeError("conditional results are required for analysis")
    summary = build_summary(
        conditional_rows=conditional_rows,
        full_rows=full_rows,
        config_path=args.probe_config,
        output=args.output,
        report=args.report,
    )
    manifest = {
        "schema_version": 1,
        "frozen_components": paths,
        "probe_config": args.probe_config,
        "policy_training_performed": False,
        "scripted_lift_modified": False,
        "formal_graspsecure_gate_modified": False,
        "maximum_regrasp_retries": 1,
    }
    _write_json(args.output / "manifest.json", manifest)
    compact = {
        "conditional": {
            "terminal_states": summary["conditional"]["terminal_states"],
            "first_probe": summary["conditional"]["first_probe"],
            "retry": summary["conditional"]["retry"],
            "success_comparison": summary["conditional"]["success_comparison"],
        },
        "full": None if summary["full"] is None else {
            "rollouts": summary["full"]["rollouts"],
            "stage_counts": summary["full"]["stage_counts"],
            "stage_rates": summary["full"]["stage_rates"],
            "matched_comparison": summary["full"]["matched_comparison"],
        },
        "decision": summary["decision"],
    }
    print(json.dumps(_plain(compact), indent=2))


if __name__ == "__main__":
    main()
