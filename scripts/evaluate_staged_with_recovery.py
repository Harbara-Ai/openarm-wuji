"""Matched Reach->Approach baseline versus one-shot Recovery routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.policy import (
    ApproachPolicy,
    ReachPolicy,
    RecoveryPolicy,
    detect_near_failure,
    primary_trigger,
)
from openarm_wuji.policy.router_features import extract_router_features
from openarm_wuji.policy.router_candidates import candidate_fires
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import (
    capture_simulator_snapshot,
    restore_simulator_snapshot,
)
from openarm_wuji.tasks import ReachGraspLiftTask


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _set_deterministic(policy, seed: int) -> None:
    torch = policy.controller.torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    torch.backends.mkldnn.enabled = False


def _state(observation: dict) -> np.ndarray:
    return np.concatenate([
        observation["arm_joint_position"],
        observation["hand_joint_position"],
    ]).astype(np.float32)


def _clipping(predicted: np.ndarray, sent: np.ndarray) -> tuple[int, list[dict]]:
    difference = np.abs(predicted - sent)
    return int(np.count_nonzero(difference > 1e-9)), [
        {
            "index": index,
            "joint_name": JOINT_NAMES[index],
            "count": int(np.count_nonzero(difference[:, index] > 1e-9)),
            "max_magnitude_rad": float(difference[:, index].max(initial=0.0)),
        }
        for index in range(27)
    ]


def _run_episode(*, seed: int, rollout_index: int, robot, config: dict,
                 reach_policy: ReachPolicy, approach_policy: ApproachPolicy,
                 recovery_policy: RecoveryPolicy,
                 output: Path,
                 router_spec: dict[str, Any] | None = None,
                 frame_callback=None) -> dict[str, Any]:
    task = ReachGraspLiftTask(robot, config)
    task.reset(seed)
    reach_policy.reset()
    reset_cube = np.asarray(task.initial_cube_position, dtype=float)
    reach_errors = []
    for _ in range(reach_policy.timeout_frames):
        observation = robot.get_observation()
        predicted = reach_policy.select_action(observation)
        robot.send_controller_joint_target(predicted)
        telemetry = task.task_telemetry()
        error = float(np.linalg.norm(
            np.asarray(telemetry["grasp_center_position_m"], dtype=float)
            - np.asarray(task.target_position, dtype=float)
        ))
        cube_displacement = float(np.linalg.norm(
            np.asarray(telemetry["cube_position_m"], dtype=float) - reset_cube
        ))
        status = reach_policy.observe(
            position_error_m=error,
            orientation_error_deg=float(
                telemetry["palm_orientation_error_deg"]
            ),
            cube_displacement_m=cube_displacement,
        )
        if frame_callback is not None:
            frame_callback("REACH", robot.get_observation(), telemetry)
        reach_errors.append(error)
        if status.success or status.failure_reason is not None:
            break
    if not reach_policy.is_success():
        return {
            "rollout_index": rollout_index,
            "seed": seed,
            "reach_success": False,
            "reach_frames": reach_policy.status.frame,
            "reach_minimum_error_m": min(reach_errors),
            "reach_failure_reason": reach_policy.status.failure_reason,
            "approach_attempted": False,
            "triggered": False,
            "baseline_success": False,
            "baseline_outcome": "reach_failure",
            "new_success": False,
            "new_outcome": "reach_failure",
        }

    approach_policy.reset()
    approach_start_cube = np.asarray(
        task.task_telemetry()["cube_position_m"], dtype=float
    )
    target = approach_start_cube + np.asarray(
        config["grasp"]["target_offset_m"], dtype=float
    )
    errors: list[float] = []
    gate_frames_total = 0
    baseline_cube_displacements: list[float] = []
    baseline_terminal_errors: list[float] = []
    trigger_snapshot = None
    trigger_row = None
    trigger_prefix_cube: list[float] = []
    baseline_predicted: list[np.ndarray] = []
    baseline_sent: list[np.ndarray] = []
    cube_radial_velocities: list[float] = []
    arm_velocity_norms: list[float] = []
    arm_velocity_max_abs: list[float] = []
    grasp_velocity_norms: list[float] = []
    grasp_velocity_toward_target: list[float] = []
    gate_stall_flags: list[bool] = []

    for frame in range(approach_policy.timeout_frames):
        observation = robot.get_observation()
        predicted = approach_policy.select_action(observation)
        sent = robot.send_controller_joint_target(predicted)
        telemetry = task.task_telemetry()
        cube = np.asarray(telemetry["cube_position_m"], dtype=float)
        grasp = np.asarray(telemetry["grasp_center_position_m"], dtype=float)
        error_vector = target - grasp
        error = float(np.linalg.norm(error_vector))
        orientation = float(telemetry["palm_orientation_error_deg"])
        cube_vector = cube - approach_start_cube
        cube_displacement = float(np.linalg.norm(cube_vector))
        cube_velocity = np.asarray(
            telemetry["cube_linear_velocity_world_m_s"], dtype=float
        )
        grasp_velocity = np.asarray(
            telemetry["grasp_center_linear_velocity_world_m_s"], dtype=float
        )
        arm_velocity = np.asarray(
            robot._data.qvel[robot._arm_qvel_ids], dtype=float
        )
        status = approach_policy.observe(
            position_error_m=error,
            orientation_error_deg=orientation,
            cube_displacement_m=cube_displacement,
        )
        if frame_callback is not None:
            frame_callback("APPROACH", robot.get_observation(), telemetry)
        errors.append(error)
        baseline_terminal_errors.append(error)
        baseline_cube_displacements.append(cube_displacement)
        baseline_predicted.append(predicted)
        baseline_sent.append(sent)
        cube_radial_velocities.append(float(
            np.dot(cube_velocity, cube_vector / cube_displacement)
            if cube_displacement > 1e-12 else 0.0
        ))
        arm_velocity_norms.append(float(np.linalg.norm(arm_velocity)))
        arm_velocity_max_abs.append(float(np.max(np.abs(arm_velocity))))
        grasp_velocity_norms.append(float(np.linalg.norm(grasp_velocity)))
        grasp_velocity_toward_target.append(float(
            np.dot(grasp_velocity, error_vector / error)
            if error > 1e-12 else 0.0
        ))
        if error <= 0.012:
            gate_frames_total += 1
        gate_stall_flags.append(bool(
            not status.success
            and error < 0.020
            and gate_frames_total >= 4
            and status.consecutive_gate_frames < 3
        ))

        # A completed/unsafe stage takes precedence. Otherwise the first
        # unchanged near-failure detector atomically routes to Recovery.
        if (
            trigger_snapshot is None
            and not status.success
            and status.failure_reason is None
        ):
            triggers = detect_near_failure(
                errors=errors,
                cube_displacement_m=cube_displacement,
                gate_frames_total=gate_frames_total,
                gate_consecutive=status.consecutive_gate_frames,
                frame=frame,
                timeout_frames=approach_policy.timeout_frames,
                status_success=status.success,
            )
            features = None
            if triggers or router_spec is not None:
                features = extract_router_features(
                    errors_m=errors,
                    cube_displacements_m=baseline_cube_displacements,
                    cube_radial_velocities_m_s=cube_radial_velocities,
                    arm_velocity_norms_rad_s=arm_velocity_norms,
                    arm_velocity_max_abs_rad_s=arm_velocity_max_abs,
                    grasp_velocity_norms_m_s=grasp_velocity_norms,
                    grasp_velocity_toward_target_m_s=(
                        grasp_velocity_toward_target
                    ),
                    gate_stall_flags=gate_stall_flags,
                    gate_consecutive=status.consecutive_gate_frames,
                    gate_frames_total=gate_frames_total,
                    orientation_error_deg=orientation,
                    active_triggers=triggers,
                    frame=frame,
                    timeout_frames=approach_policy.timeout_frames,
                )
            should_route = (
                bool(triggers) if router_spec is None
                else candidate_fires(router_spec, features)
            )
            if should_route:
                trigger_snapshot = capture_simulator_snapshot(robot, task)
                primary = primary_trigger(triggers)
                if primary is None:
                    raise RuntimeError("active triggers have no primary label")
                trigger_row = {
                    "frame": frame,
                    "primary_type": primary,
                    "active_types": sorted(triggers),
                    "scores": triggers,
                    "terminal_error_m": error,
                    "cube_displacement_m": cube_displacement,
                    "features": features,
                    "router_candidate_name": (
                        router_spec["name"] if router_spec is not None
                        else "old_first_trigger"
                    ),
                }
                trigger_prefix_cube = baseline_cube_displacements.copy()
        if status.success or status.failure_reason is not None:
            break

    baseline_status = approach_policy.status
    baseline_success = bool(baseline_status.success)
    baseline_outcome = (
        "success" if baseline_success
        else baseline_status.failure_reason or "other"
    )
    baseline_clipped, baseline_clipping_by_joint = _clipping(
        np.asarray(baseline_predicted), np.asarray(baseline_sent)
    )
    baseline_metrics = {
        "success": baseline_success,
        "outcome": baseline_outcome,
        "frames": int(baseline_status.frame),
        "timeout": bool(baseline_status.timeout),
        "minimum_terminal_error_m": min(baseline_terminal_errors),
        "final_terminal_error_m": baseline_terminal_errors[-1],
        "maximum_cube_displacement_m": max(baseline_cube_displacements),
        "clipped_action_values": baseline_clipped,
        "clipping_by_joint": baseline_clipping_by_joint,
    }

    recovery_metrics = None
    recovery_arrays = None
    if trigger_snapshot is not None:
        restore_check = restore_simulator_snapshot(robot, task, trigger_snapshot)
        if not restore_check["passed"]:
            raise RuntimeError(
                f"trigger snapshot restoration failed for seed {seed}: "
                f"{restore_check}"
            )
        recovery_policy.reset()
        telemetry = task.task_telemetry()
        recovery_start_cube = np.asarray(
            telemetry["cube_position_m"], dtype=float
        )
        recovery_target = recovery_start_cube + np.asarray(
            config["grasp"]["target_offset_m"], dtype=float
        )
        recovery_values: dict[str, list] = {
            "observation_state": [],
            "predicted_action": [],
            "sent_action": [],
            "terminal_error_m": [],
            "cube_displacement_m": [],
        }
        for _ in range(recovery_policy.timeout_frames):
            observation = robot.get_observation()
            predicted = recovery_policy.select_action(observation)
            sent = robot.send_controller_joint_target(predicted)
            post = task.task_telemetry()
            cube = np.asarray(post["cube_position_m"], dtype=float)
            error = float(np.linalg.norm(
                np.asarray(post["grasp_center_position_m"], dtype=float)
                - recovery_target
            ))
            orientation = float(post["palm_orientation_error_deg"])
            cube_displacement = float(
                np.linalg.norm(cube - approach_start_cube)
            )
            status = recovery_policy.observe(
                position_error_m=error,
                orientation_error_deg=orientation,
                cube_displacement_m=cube_displacement,
            )
            if frame_callback is not None:
                frame_callback("RECOVERY", robot.get_observation(), post)
            recovery_values["observation_state"].append(_state(observation))
            recovery_values["predicted_action"].append(predicted)
            recovery_values["sent_action"].append(sent)
            recovery_values["terminal_error_m"].append(error)
            recovery_values["cube_displacement_m"].append(cube_displacement)
            if status.success or status.failure_reason is not None:
                break
        recovery_arrays = {
            key: np.asarray(value) for key, value in recovery_values.items()
        }
        recovery_clipped, recovery_clipping_by_joint = _clipping(
            recovery_arrays["predicted_action"],
            recovery_arrays["sent_action"],
        )
        recovery_status = recovery_policy.status
        recovery_success = bool(recovery_status.success)
        recovery_outcome = (
            "success" if recovery_success
            else recovery_status.failure_reason or "other"
        )
        recovery_metrics = {
            "success": recovery_success,
            "outcome": recovery_outcome,
            "frames": int(recovery_status.frame),
            "timeout": bool(recovery_status.timeout),
            "minimum_terminal_error_m": float(
                recovery_arrays["terminal_error_m"].min()
            ),
            "final_terminal_error_m": float(
                recovery_arrays["terminal_error_m"][-1]
            ),
            "maximum_cube_displacement_m": max(
                max(trigger_prefix_cube),
                float(recovery_arrays["cube_displacement_m"].max()),
            ),
            "clipped_action_values": recovery_clipped,
            "clipping_by_joint": recovery_clipping_by_joint,
            "restore_check": restore_check,
            "target_rebased_to_trigger_cube": True,
            "safety_gate_referenced_to_approach_start": True,
        }
        np.savez_compressed(
            output / f"recovery_seed_{seed:06d}.npz",
            **recovery_arrays,
            recovery_target_grasp_position_m=recovery_target,
            approach_start_cube_position_m=approach_start_cube,
            recovery_start_cube_position_m=recovery_start_cube,
        )

    if recovery_metrics is None:
        new_metrics = dict(baseline_metrics)
        direct_success = baseline_success
    else:
        new_metrics = dict(recovery_metrics)
        direct_success = False
    return {
        "rollout_index": rollout_index,
        "seed": seed,
        "reach_success": True,
        "reach_frames": reach_policy.status.frame,
        "reach_minimum_error_m": min(reach_errors),
        "reach_failure_reason": None,
        "approach_attempted": True,
        "triggered": trigger_row is not None,
        "trigger": trigger_row,
        "baseline_success": baseline_success,
        "baseline_outcome": baseline_outcome,
        "baseline": baseline_metrics,
        "approach_direct_success": direct_success,
        "recovery_attempted": recovery_metrics is not None,
        "recovery_success": bool(
            recovery_metrics is not None and recovery_metrics["success"]
        ),
        "recovery": recovery_metrics,
        "new_success": bool(new_metrics["success"]),
        "new_outcome": str(new_metrics["outcome"]),
        "new": new_metrics,
        "expert_action_supplied_to_policy": False,
        "maximum_recovery_attempts": 1,
    }


def _distribution(values: list[float], *, scale: float = 1.0) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float) * scale
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _summarize(rows: list[dict], *, requested: int, seed_start: int,
               reach_checkpoint: Path, approach_checkpoint: Path,
               recovery_checkpoint: Path,
               router_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    attempted = [item for item in rows if item["approach_attempted"]]
    triggered = [item for item in attempted if item["triggered"]]
    baseline_successes = sum(item["baseline_success"] for item in attempted)
    new_successes = sum(item["new_success"] for item in attempted)
    recovery_successes = sum(item["recovery_success"] for item in triggered)
    baseline_outcomes = {}
    new_outcomes = {}
    for item in attempted:
        baseline_outcomes[item["baseline_outcome"]] = (
            baseline_outcomes.get(item["baseline_outcome"], 0) + 1
        )
        new_outcomes[item["new_outcome"]] = (
            new_outcomes.get(item["new_outcome"], 0) + 1
        )
    trigger_by_primary = {}
    active_trigger_counts = {}
    for item in triggered:
        primary = item["trigger"]["primary_type"]
        bucket = trigger_by_primary.setdefault(primary, {
            "attempts": 0,
            "recovery_successes": 0,
            "baseline_successes": 0,
        })
        bucket["attempts"] += 1
        bucket["recovery_successes"] += int(item["recovery_success"])
        bucket["baseline_successes"] += int(item["baseline_success"])
        for trigger_type in item["trigger"]["active_types"]:
            active_trigger_counts[trigger_type] = (
                active_trigger_counts.get(trigger_type, 0) + 1
            )
    for bucket in trigger_by_primary.values():
        bucket["recovery_success_rate"] = (
            bucket["recovery_successes"] / bucket["attempts"]
        )
        bucket["baseline_success_rate"] = (
            bucket["baseline_successes"] / bucket["attempts"]
        )
    return {
        "requested_rollouts": requested,
        "completed_rollouts": len(rows),
        "seed_start": seed_start,
        "seed_end_inclusive": max(item["seed"] for item in rows) if rows else None,
        "reach_checkpoint": reach_checkpoint,
        "approach_checkpoint": approach_checkpoint,
        "recovery_checkpoint": recovery_checkpoint,
        "policies_modified": False,
        "expert_action_used": False,
        "router": {
            "type": (
                "hand-designed unchanged near-failure detectors"
                if router_spec is None
                else "hand-designed persistence/hysteresis candidate"
            ),
            "candidate_spec": router_spec,
            "maximum_recovery_attempts": 1,
            "success_precedes_trigger_on_same_frame": True,
            "primary_trigger_priority": [
                "cube_displacement", "moving_away", "terminal_plateau",
                "gate_stall", "near_timeout_terminal",
            ],
        },
        "reach_successes": sum(item["reach_success"] for item in rows),
        "approach_attempts": len(attempted),
        "approach_direct_successes": sum(
            item["approach_direct_success"] for item in attempted
        ),
        "recovery_triggers": len(triggered),
        "recovery_attempts": len(triggered),
        "recovery_successes": recovery_successes,
        "recovery_success_rate": (
            recovery_successes / len(triggered) if triggered else 0.0
        ),
        "mean_recovery_frames": (
            float(np.mean([item["recovery"]["frames"] for item in triggered]))
            if triggered else None
        ),
        "baseline": {
            "approach_successes": baseline_successes,
            "approach_conditional_success_rate": (
                baseline_successes / len(attempted) if attempted else 0.0
            ),
            "joint_successes": baseline_successes,
            "outcome_distribution": baseline_outcomes,
            "timeouts": baseline_outcomes.get("timeout", 0),
            "cube_safety_failures": baseline_outcomes.get(
                "cube_displacement", 0
            ),
            "maximum_cube_displacement_mm": _distribution([
                item["baseline"]["maximum_cube_displacement_m"]
                for item in attempted
            ], scale=1000.0),
            "triggered_episode_successes": sum(
                item["baseline_success"] for item in triggered
            ),
            "triggered_episode_success_rate": (
                sum(item["baseline_success"] for item in triggered)
                / len(triggered) if triggered else 0.0
            ),
        },
        "staged_with_recovery": {
            "approach_stage_successes": new_successes,
            "approach_conditional_success_rate": (
                new_successes / len(attempted) if attempted else 0.0
            ),
            "joint_successes": new_successes,
            "outcome_distribution": new_outcomes,
            "timeouts": new_outcomes.get("timeout", 0),
            "cube_safety_failures": new_outcomes.get(
                "cube_displacement", 0
            ),
            "maximum_cube_displacement_mm": _distribution([
                item["new"]["maximum_cube_displacement_m"]
                for item in attempted
            ], scale=1000.0),
        },
        "matched_switch_effect": {
            "rescued_failures": sum(
                (not item["baseline_success"]) and item["new_success"]
                for item in triggered
            ),
            "regressed_successes": sum(
                item["baseline_success"] and (not item["new_success"])
                for item in triggered
            ),
            "both_success": sum(
                item["baseline_success"] and item["new_success"]
                for item in triggered
            ),
            "both_failure": sum(
                (not item["baseline_success"]) and (not item["new_success"])
                for item in triggered
            ),
        },
        "trigger_by_primary_type": trigger_by_primary,
        "active_trigger_type_distribution": active_trigger_counts,
        "episodes_detail": rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recovery-checkpoint", type=Path,
        default=root / "outputs/staged_act_with_recovery/recovery_act_train/checkpoints/001500/pretrained_model",
    )
    parser.add_argument(
        "--reach-checkpoint", type=Path,
        default=root / "outputs/act_reach_only/act_train/checkpoints/002000/pretrained_model",
    )
    parser.add_argument(
        "--approach-checkpoint", type=Path,
        default=root / "outputs/act_staged/approach_only/act_train/checkpoints/002000/pretrained_model",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/staged_act_with_recovery/unseen_matched",
    )
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
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1400)
    parser.add_argument("--reach-timeout", type=int, default=160)
    parser.add_argument("--approach-timeout", type=int, default=90)
    parser.add_argument("--recovery-timeout", type=int, default=90)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument(
        "--router-spec", type=Path,
        default=root / "configs/recovery_router.json",
        help="JSON persistence/hysteresis router (defaults to the frozen router).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.rollouts < 1:
        raise ValueError("--rollouts must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.output.exists():
        import shutil
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    router_spec = (
        json.loads(args.router_spec.read_text(encoding="utf-8"))
        if args.router_spec is not None else None
    )
    safety_gate = float(config["grasp"]["max_approach_cube_displacement_m"])
    reach_policy = ReachPolicy(
        args.reach_checkpoint, timeout_frames=args.reach_timeout
    )
    approach_policy = ApproachPolicy(
        args.approach_checkpoint,
        timeout_frames=args.approach_timeout,
        max_cube_displacement_m=safety_gate,
    )
    recovery_policy = RecoveryPolicy(
        args.recovery_checkpoint,
        timeout_frames=args.recovery_timeout,
        max_cube_displacement_m=safety_gate,
    )
    for policy in (reach_policy, approach_policy, recovery_policy):
        _set_deterministic(policy, args.inference_seed)
    robot = MujocoOpenArmWuji(
        args.model,
        args.synergies,
        arm_side=config["arm_side"],
        control_hz=30,
        image_height=240,
        image_width=320,
        front_camera=config["scene"]["front_camera_name"],
    )
    robot.connect()
    rows = []
    try:
        for rollout_index in range(args.rollouts):
            seed = args.seed_start + rollout_index
            rows.append(_run_episode(
                seed=seed,
                rollout_index=rollout_index,
                robot=robot,
                config=config,
                reach_policy=reach_policy,
                approach_policy=approach_policy,
                recovery_policy=recovery_policy,
                output=args.output,
                router_spec=router_spec,
            ))
            if (rollout_index + 1) % 10 == 0 or rollout_index == 0:
                partial = _summarize(
                    rows,
                    requested=args.rollouts,
                    seed_start=args.seed_start,
                    reach_checkpoint=args.reach_checkpoint,
                    approach_checkpoint=args.approach_checkpoint,
                    recovery_checkpoint=args.recovery_checkpoint,
                    router_spec=router_spec,
                )
                (args.output / "summary.partial.json").write_text(
                    json.dumps(_plain(partial), indent=2), encoding="utf-8"
                )
                print(
                    f"rollouts={len(rows)}/{args.rollouts} "
                    f"reach={partial['reach_successes']} "
                    f"baseline={partial['baseline']['approach_successes']} "
                    f"recovery={partial['staged_with_recovery']['approach_stage_successes']} "
                    f"switches={partial['recovery_attempts']}",
                    flush=True,
                )
    finally:
        robot.disconnect()
    summary = _summarize(
        rows,
        requested=args.rollouts,
        seed_start=args.seed_start,
        reach_checkpoint=args.reach_checkpoint,
        approach_checkpoint=args.approach_checkpoint,
        recovery_checkpoint=args.recovery_checkpoint,
        router_spec=router_spec,
    )
    (args.output / "summary.json").write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        key: _plain(summary[key]) for key in (
            "completed_rollouts", "reach_successes", "approach_attempts",
            "approach_direct_successes", "recovery_attempts",
            "recovery_successes", "recovery_success_rate", "baseline",
            "staged_with_recovery", "matched_switch_effect",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
