"""Mine terminal Approach near-failure snapshots from frozen staged ACT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.policy import ApproachPolicy, ReachPolicy, detect_near_failure
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import capture_simulator_snapshot
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
    ]).astype(np.float64)


def _trigger_scores(*, errors: list[float], cube_displacement_m: float,
                    gate_frames_total: int, gate_consecutive: int,
                    frame: int, timeout_frames: int,
                    status_success: bool) -> dict[str, float]:
    """Compatibility wrapper around the deployment-shared detector."""
    return detect_near_failure(
        errors=errors,
        cube_displacement_m=cube_displacement_m,
        gate_frames_total=gate_frames_total,
        gate_consecutive=gate_consecutive,
        frame=frame,
        timeout_frames=timeout_frames,
        status_success=status_success,
    )


def _snapshot_payload(*, robot, task, seed: int,
                      rollout_index: int, frame: int, trigger_type: str,
                      trigger_score: float, target: np.ndarray,
                      approach_start_cube: np.ndarray,
                      terminal_error_vector: np.ndarray,
                      terminal_error_m: float, orientation_error_deg: float,
                      cube_displacement_vector: np.ndarray,
                      telemetry: dict) -> dict[str, np.ndarray]:
    # Render from the exact current MjData after telemetry/mj_forward instead
    # of reusing the wrapper's pre-telemetry framebuffer.
    payload = capture_simulator_snapshot(robot, task)
    payload.update({
        "source_seed": np.asarray(seed, dtype=np.int64),
        "rollout_index": np.asarray(rollout_index, dtype=np.int64),
        "source_frame": np.asarray(frame, dtype=np.int64),
        "act_phase": np.asarray("approach"),
        "trigger_type": np.asarray(trigger_type),
        "trigger_score": np.asarray(trigger_score, dtype=np.float64),
        "target_grasp_position_m": np.asarray(target, dtype=np.float64),
        "approach_start_cube_position_m": np.asarray(
            approach_start_cube, dtype=np.float64
        ),
        "terminal_error_vector_m": np.asarray(
            terminal_error_vector, dtype=np.float64
        ),
        "terminal_error_m": np.asarray(terminal_error_m, dtype=np.float64),
        "orientation_error_deg": np.asarray(
            orientation_error_deg, dtype=np.float64
        ),
        "cube_displacement_vector_m": np.asarray(
            cube_displacement_vector, dtype=np.float64
        ),
        "cube_displacement_m": np.asarray(
            np.linalg.norm(cube_displacement_vector), dtype=np.float64
        ),
        "cube_pose_world": np.r_[
            telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"]
        ].astype(np.float64),
        "cube_linear_velocity_world_m_s": np.asarray(
            telemetry["cube_linear_velocity_world_m_s"], dtype=np.float64
        ),
        "cube_angular_velocity_world_rad_s": np.asarray(
            telemetry["cube_angular_velocity_world_rad_s"], dtype=np.float64
        ),
        "contact_count": np.asarray(
            len(telemetry["contacts"]), dtype=np.int64
        ),
    })
    return payload


def _run_episode(*, seed: int, rollout_index: int, robot, config: dict,
                 reach_policy: ReachPolicy, approach_policy: ApproachPolicy,
                 snapshot_dir: Path | None) -> dict[str, Any]:
    task = ReachGraspLiftTask(robot, config)
    task.reset(seed)
    reach_policy.reset()
    reset_cube = np.asarray(task.initial_cube_position, dtype=float)
    reach_errors: list[float] = []
    for _ in range(reach_policy.timeout_frames):
        observation = robot.get_observation()
        predicted = reach_policy.select_action(observation)
        robot.send_controller_joint_target(predicted)
        telemetry = task.task_telemetry()
        error = float(np.linalg.norm(
            np.asarray(telemetry["grasp_center_position_m"])
            - np.asarray(task.target_position)
        ))
        cube_displacement = float(np.linalg.norm(
            np.asarray(telemetry["cube_position_m"]) - reset_cube
        ))
        status = reach_policy.observe(
            position_error_m=error,
            orientation_error_deg=float(
                telemetry["palm_orientation_error_deg"]
            ),
            cube_displacement_m=cube_displacement,
        )
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
            "approach_success": False,
            "outcome": "reach_failure",
            "triggers": [],
        }

    approach_policy.reset()
    approach_start_cube = np.asarray(
        task.task_telemetry()["cube_position_m"], dtype=float
    )
    target = approach_start_cube + np.asarray(
        config["grasp"]["target_offset_m"], dtype=float
    )
    errors: list[float] = []
    orientation_errors: list[float] = []
    cube_displacements: list[float] = []
    contacts: list[int] = []
    gate_frames_total = 0
    best_by_trigger: dict[str, tuple[float, dict[str, Any]]] = {}
    first_contact_frame = None
    for frame in range(approach_policy.timeout_frames):
        observation = robot.get_observation()
        predicted = approach_policy.select_action(observation)
        robot.send_controller_joint_target(predicted)
        telemetry = task.task_telemetry()
        grasp_position = np.asarray(
            telemetry["grasp_center_position_m"], dtype=float
        )
        terminal_vector = target - grasp_position
        error = float(np.linalg.norm(terminal_vector))
        orientation = float(telemetry["palm_orientation_error_deg"])
        cube_vector = np.asarray(
            telemetry["cube_position_m"], dtype=float
        ) - approach_start_cube
        cube_displacement = float(np.linalg.norm(cube_vector))
        status = approach_policy.observe(
            position_error_m=error,
            orientation_error_deg=orientation,
            cube_displacement_m=cube_displacement,
        )
        errors.append(error)
        orientation_errors.append(orientation)
        cube_displacements.append(cube_displacement)
        contacts.append(len(telemetry["contacts"]))
        if telemetry["contacts"] and first_contact_frame is None:
            first_contact_frame = frame
        if error <= 0.012:
            gate_frames_total += 1
        triggers = _trigger_scores(
            errors=errors,
            cube_displacement_m=cube_displacement,
            gate_frames_total=gate_frames_total,
            gate_consecutive=status.consecutive_gate_frames,
            frame=frame,
            timeout_frames=approach_policy.timeout_frames,
            status_success=status.success,
        )
        for trigger_type, score in triggers.items():
            previous = best_by_trigger.get(trigger_type)
            # For cube motion the first threshold crossing is the useful
            # pre-failure state. Keeping the maximum would often save the
            # already-ejected cube after the 25 mm safety failure.
            if trigger_type == "cube_displacement" and previous is not None:
                continue
            if previous is not None and previous[0] >= score:
                continue
            if snapshot_dir is None:
                payload = {
                    "source_frame": frame,
                    "terminal_error_m": error,
                    "terminal_error_vector_m": terminal_vector.copy(),
                    "cube_displacement_m": cube_displacement,
                }
            else:
                payload = _snapshot_payload(
                    robot=robot,
                    task=task,
                    seed=seed,
                    rollout_index=rollout_index,
                    frame=frame,
                    trigger_type=trigger_type,
                    trigger_score=score,
                    target=target,
                    approach_start_cube=approach_start_cube,
                    terminal_error_vector=terminal_vector,
                    terminal_error_m=error,
                    orientation_error_deg=orientation,
                    cube_displacement_vector=cube_vector,
                    telemetry=telemetry,
                )
            best_by_trigger[trigger_type] = (score, payload)
        if status.success or status.failure_reason is not None:
            break

    outcome = (
        "success" if approach_policy.is_success()
        else approach_policy.status.failure_reason or "other"
    )
    trigger_rows = []
    for trigger_type, (score, payload) in sorted(best_by_trigger.items()):
        row = {
            "trigger_type": trigger_type,
            "trigger_score": score,
            "source_frame": int(payload["source_frame"]),
            "terminal_error_m": float(payload["terminal_error_m"]),
            "terminal_error_vector_m": payload["terminal_error_vector_m"],
            "cube_displacement_m": float(payload["cube_displacement_m"]),
            "source_outcome": outcome,
        }
        if snapshot_dir is not None:
            payload["source_outcome"] = np.asarray(outcome)
            filename = (
                f"snapshot_rollout_{rollout_index:06d}_seed_{seed:06d}_"
                f"{trigger_type}.npz"
            )
            path = snapshot_dir / filename
            np.savez_compressed(path, **payload)
            row["snapshot"] = path.resolve()
        trigger_rows.append(row)
    return {
        "rollout_index": rollout_index,
        "seed": seed,
        "reach_success": True,
        "reach_frames": reach_policy.status.frame,
        "reach_minimum_error_m": min(reach_errors),
        "reach_failure_reason": None,
        "approach_attempted": True,
        "approach_success": approach_policy.is_success(),
        "outcome": outcome,
        "approach_frames": approach_policy.status.frame,
        "minimum_terminal_error_m": min(errors),
        "final_terminal_error_m": errors[-1],
        "minimum_orientation_error_deg": min(orientation_errors),
        "final_orientation_error_deg": orientation_errors[-1],
        "maximum_cube_displacement_m": max(cube_displacements),
        "first_contact_frame": first_contact_frame,
        "max_contact_count": max(contacts),
        "triggers": trigger_rows,
    }


def _summarize(rows: list[dict[str, Any]], *, requested: int,
               seed_start: int, reach_checkpoint: Path,
               approach_checkpoint: Path) -> dict[str, Any]:
    attempted = [row for row in rows if row["approach_attempted"]]
    triggers = [trigger for row in attempted for trigger in row["triggers"]]
    trigger_counts: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    for item in triggers:
        key = item["trigger_type"]
        trigger_counts[key] = trigger_counts.get(key, 0) + 1
    recoveries = [row for row in attempted if row["triggers"]]
    cube_motion = [row["maximum_cube_displacement_m"] for row in attempted]
    terminal_errors = [row["final_terminal_error_m"] for row in attempted]
    trigger_recovery_by_type = {}
    for trigger_type in sorted(trigger_counts):
        affected = [
            row for row in attempted
            if any(item["trigger_type"] == trigger_type for item in row["triggers"])
        ]
        trigger_recovery_by_type[trigger_type] = {
            "episodes": len(affected),
            "recovered_successes": sum(row["approach_success"] for row in affected),
            "recovery_rate": (
                sum(row["approach_success"] for row in affected) / len(affected)
                if affected else 0.0
            ),
        }
    return {
        "requested_rollouts": requested,
        "completed_rollouts": len(rows),
        "seed_start": seed_start,
        "seed_end_inclusive": max(row["seed"] for row in rows) if rows else None,
        "reach_checkpoint": reach_checkpoint,
        "approach_checkpoint": approach_checkpoint,
        "policies_modified": False,
        "expert_action_used_during_mining": False,
        "reach_successes": sum(row["reach_success"] for row in rows),
        "approach_attempts": len(attempted),
        "approach_successes": sum(row["approach_success"] for row in attempted),
        "approach_failures": sum(not row["approach_success"] for row in attempted),
        "outcome_distribution": outcomes,
        "candidate_snapshots": len(triggers),
        "rollouts_with_trigger": sum(bool(row["triggers"]) for row in attempted),
        "near_failure_episode_rate": len(recoveries) / len(attempted) if attempted else 0.0,
        "near_failure_recovered_successes": sum(
            row["approach_success"] for row in recoveries
        ),
        "near_failure_recovery_rate": (
            sum(row["approach_success"] for row in recoveries) / len(recoveries)
            if recoveries else 0.0
        ),
        "trigger_type_distribution": trigger_counts,
        "trigger_recovery_by_type": trigger_recovery_by_type,
        "timeout_failures": outcomes.get("timeout", 0),
        "cube_displacement_failures": outcomes.get("cube_displacement", 0),
        "cube_displacement_median_mm": (
            float(np.median(cube_motion) * 1000) if cube_motion else None
        ),
        "cube_displacement_p90_mm": (
            float(np.quantile(cube_motion, 0.9) * 1000) if cube_motion else None
        ),
        "final_terminal_error_median_mm": (
            float(np.median(terminal_errors) * 1000) if terminal_errors else None
        ),
        "final_terminal_error_p90_mm": (
            float(np.quantile(terminal_errors, 0.9) * 1000) if terminal_errors else None
        ),
        "thresholds": {
            "terminal_region_m": 0.030,
            "plateau_window_frames": 7,
            "plateau_max_improvement_m": 0.001,
            "moving_away_previous_entry_m": 0.020,
            "moving_away_growth_m": 0.005,
            "cube_displacement_trigger_m": 0.003,
            "near_timeout_last_frames": 10,
            "gate_stall_terminal_m": 0.020,
            "official_approach_position_gate_m": 0.012,
            "official_approach_orientation_gate_deg": 2.0,
            "official_hold_frames": 5,
            "official_cube_safety_gate_m": 0.025,
        },
        "rollouts": rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
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
        default=root / "outputs/approach_correction_demos/mining",
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
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--reach-timeout", type=int, default=160)
    parser.add_argument("--approach-timeout", type=int, default=90)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Compute trigger/recovery metrics without serializing simulator snapshots.",
    )
    args = parser.parse_args()
    if args.rollouts < 1:
        raise ValueError("--rollouts must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.output.exists():
        # Only clear files owned by this exact run directory.
        import shutil
        shutil.rmtree(args.output)
    snapshot_dir = None if args.metrics_only else args.output / "candidate_snapshots"
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True)
    else:
        args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reach_policy = ReachPolicy(
        args.reach_checkpoint, timeout_frames=args.reach_timeout
    )
    approach_policy = ApproachPolicy(
        args.approach_checkpoint,
        timeout_frames=args.approach_timeout,
        max_cube_displacement_m=float(
            config["grasp"]["max_approach_cube_displacement_m"]
        ),
    )
    _set_deterministic(reach_policy, args.inference_seed)
    _set_deterministic(approach_policy, args.inference_seed)
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
    rows: list[dict[str, Any]] = []
    try:
        for rollout_index in range(args.rollouts):
            seed = args.seed_start + rollout_index
            row = _run_episode(
                seed=seed,
                rollout_index=rollout_index,
                robot=robot,
                config=config,
                reach_policy=reach_policy,
                approach_policy=approach_policy,
                snapshot_dir=snapshot_dir,
            )
            rows.append(row)
            if (rollout_index + 1) % 10 == 0 or rollout_index == 0:
                partial = _summarize(
                    rows,
                    requested=args.rollouts,
                    seed_start=args.seed_start,
                    reach_checkpoint=args.reach_checkpoint,
                    approach_checkpoint=args.approach_checkpoint,
                )
                (args.output / "summary.partial.json").write_text(
                    json.dumps(_plain(partial), indent=2), encoding="utf-8"
                )
                print(
                    f"rollouts={len(rows)}/{args.rollouts} "
                    f"reach={partial['reach_successes']} "
                    f"approach={partial['approach_successes']} "
                    f"snapshots={partial['candidate_snapshots']}",
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
    )
    (args.output / "summary.json").write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_plain({
        key: summary[key] for key in (
            "completed_rollouts", "reach_successes", "approach_attempts",
            "approach_successes", "approach_failures", "candidate_snapshots",
            "rollouts_with_trigger", "trigger_type_distribution",
        )
    }), indent=2))


if __name__ == "__main__":
    main()
