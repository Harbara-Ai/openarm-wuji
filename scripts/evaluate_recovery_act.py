"""Evaluate an independent Recovery ACT from exact mined MuJoCo snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.policy import RecoveryPolicy
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import restore_simulator_snapshot
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


def _load_successful_starts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    starts = [item for item in payload["results"] if item["correction_success"]]
    starts.sort(key=lambda item: Path(item["raw_episode"]).name)
    if len(starts) != 30:
        raise ValueError(f"expected 30 successful correction starts, got {len(starts)}")
    snapshots = [Path(item["snapshot"]) for item in starts]
    missing = [path for path in snapshots if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing correction snapshots: {missing[:3]}")
    return starts


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


def _run_episode(*, episode_index: int, start: dict[str, Any], robot,
                 config: dict, policy: RecoveryPolicy,
                 output: Path) -> dict[str, Any]:
    snapshot_path = Path(start["snapshot"])
    with np.load(snapshot_path, allow_pickle=False) as loaded:
        snapshot = {key: loaded[key].copy() for key in loaded.files}
    seed = int(snapshot["source_seed"])
    task = ReachGraspLiftTask(robot, config)
    task.reset(seed)
    restore_check = restore_simulator_snapshot(robot, task, snapshot)
    if not restore_check["passed"]:
        raise RuntimeError(
            f"snapshot restoration failed: {snapshot_path} {restore_check}"
        )

    telemetry = task.task_telemetry()
    recovery_start_cube = np.asarray(telemetry["cube_position_m"], dtype=float)
    approach_start_cube = np.asarray(
        snapshot["approach_start_cube_position_m"], dtype=float
    )
    # Match the correction expert: the terminal target follows the cube pose
    # at recovery start, while the unchanged 25 mm safety gate remains tied to
    # the original Approach handoff cube pose.
    target = recovery_start_cube + np.asarray(
        config["grasp"]["target_offset_m"], dtype=float
    )
    policy.reset()
    values: dict[str, list] = {
        "observation_state": [],
        "predicted_action": [],
        "sent_action": [],
        "terminal_error_m": [],
        "orientation_error_deg": [],
        "cube_displacement_m": [],
        "additional_cube_displacement_m": [],
        "sim_time": [],
    }
    success_frame = None
    for frame in range(policy.timeout_frames):
        observation = robot.get_observation()
        predicted = policy.select_action(observation)
        sent = robot.send_controller_joint_target(predicted)
        post = task.task_telemetry()
        cube = np.asarray(post["cube_position_m"], dtype=float)
        error = float(np.linalg.norm(
            np.asarray(post["grasp_center_position_m"], dtype=float) - target
        ))
        orientation = float(post["palm_orientation_error_deg"])
        cube_displacement = float(np.linalg.norm(cube - approach_start_cube))
        additional_cube_displacement = float(
            np.linalg.norm(cube - recovery_start_cube)
        )
        status = policy.observe(
            position_error_m=error,
            orientation_error_deg=orientation,
            cube_displacement_m=cube_displacement,
        )
        values["observation_state"].append(_state(observation))
        values["predicted_action"].append(predicted)
        values["sent_action"].append(sent)
        values["terminal_error_m"].append(error)
        values["orientation_error_deg"].append(orientation)
        values["cube_displacement_m"].append(cube_displacement)
        values["additional_cube_displacement_m"].append(
            additional_cube_displacement
        )
        values["sim_time"].append(float(robot.data.time))
        if status.success:
            success_frame = frame
            break
        if status.failure_reason is not None:
            break

    arrays = {key: np.asarray(value) for key, value in values.items()}
    clipped, clipping_by_joint = _clipping(
        arrays["predicted_action"], arrays["sent_action"]
    )
    status = policy.status
    outcome = "success" if status.success else status.failure_reason or "other"
    metrics = {
        "episode_index": episode_index,
        "source_seed": seed,
        "source_rollout_index": int(start["source_rollout_index"]),
        "source_frame": int(start["source_frame"]),
        "trigger_type": str(start["trigger_type"]),
        "source_outcome": str(start["source_outcome"]),
        "snapshot": snapshot_path,
        "restore_check": restore_check,
        "recovery_success": bool(status.success),
        "success_frame": success_frame,
        "outcome": outcome,
        "timeout": bool(status.timeout),
        "recovery_frames": int(status.frame),
        "initial_terminal_error_m": float(
            np.linalg.norm(
                np.asarray(telemetry["grasp_center_position_m"], dtype=float)
                - target
            )
        ),
        "minimum_terminal_error_m": float(arrays["terminal_error_m"].min()),
        "final_terminal_error_m": float(arrays["terminal_error_m"][-1]),
        "final_orientation_error_deg": float(
            arrays["orientation_error_deg"][-1]
        ),
        "maximum_cube_displacement_m": float(
            arrays["cube_displacement_m"].max()
        ),
        "maximum_additional_cube_displacement_m": float(
            arrays["additional_cube_displacement_m"].max()
        ),
        "clipped_action_values": clipped,
        "clipping_by_joint": clipping_by_joint,
        "expert_action_supplied_to_policy": False,
        "target_rebased_to_recovery_cube": True,
        "safety_gate_referenced_to_approach_start": True,
    }
    np.savez_compressed(
        output / f"rollout_{episode_index:06d}_seed_{seed:06d}.npz",
        **arrays,
        target_grasp_position_m=target,
        approach_start_cube_position_m=approach_start_cube,
        recovery_start_cube_position_m=recovery_start_cube,
    )
    return metrics


def _distribution(values: list[float], *, scale: float = 1.0) -> dict:
    array = np.asarray(values, dtype=float) * scale
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _summarize(episodes: list[dict], *, checkpoint: Path,
               metadata: Path) -> dict[str, Any]:
    by_trigger = {}
    for trigger_type in sorted({item["trigger_type"] for item in episodes}):
        selected = [
            item for item in episodes if item["trigger_type"] == trigger_type
        ]
        successes = sum(item["recovery_success"] for item in selected)
        by_trigger[trigger_type] = {
            "episodes": len(selected),
            "successes": successes,
            "success_rate": successes / len(selected),
            "timeouts": sum(item["timeout"] for item in selected),
            "cube_safety_failures": sum(
                item["outcome"] == "cube_displacement" for item in selected
            ),
        }
    clipping_by_joint = []
    for index, name in enumerate(JOINT_NAMES):
        rows = [item["clipping_by_joint"][index] for item in episodes]
        clipping_by_joint.append({
            "index": index,
            "joint_name": name,
            "count": int(sum(item["count"] for item in rows)),
            "max_magnitude_rad": float(max(
                item["max_magnitude_rad"] for item in rows
            )),
        })
    successes = sum(item["recovery_success"] for item in episodes)
    outcomes = {}
    for item in episodes:
        outcomes[item["outcome"]] = outcomes.get(item["outcome"], 0) + 1
    return {
        "checkpoint": checkpoint,
        "correction_metadata": metadata,
        "episodes": len(episodes),
        "recovery_successes": successes,
        "recovery_success_rate": successes / len(episodes),
        "outcome_distribution": outcomes,
        "timeouts": sum(item["timeout"] for item in episodes),
        "cube_safety_failures": sum(
            item["outcome"] == "cube_displacement" for item in episodes
        ),
        "terminal_error_mm": _distribution([
            item["final_terminal_error_m"] for item in episodes
        ], scale=1000.0),
        "recovery_frames": _distribution([
            item["recovery_frames"] for item in episodes
        ]),
        "maximum_cube_displacement_mm": _distribution([
            item["maximum_cube_displacement_m"] for item in episodes
        ], scale=1000.0),
        "maximum_additional_cube_displacement_mm": _distribution([
            item["maximum_additional_cube_displacement_m"] for item in episodes
        ], scale=1000.0),
        "action_clipping_values": int(sum(
            item["clipped_action_values"] for item in episodes
        )),
        "action_clipping_by_joint": clipping_by_joint,
        "success_by_trigger": by_trigger,
        "restore_checks_all_passed": all(
            item["restore_check"]["passed"] for item in episodes
        ),
        "expert_action_supplied_to_policy": False,
        "episodes_detail": episodes,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--correction-metadata", type=Path,
        default=root / "outputs/approach_correction_demos/correction_results.json",
    )
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--recovery-timeout", type=int, default=90)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.output.exists():
        import shutil
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    starts = _load_successful_starts(args.correction_metadata)
    if args.limit is not None:
        starts = starts[:args.limit]
    policy = RecoveryPolicy(
        args.checkpoint,
        timeout_frames=args.recovery_timeout,
        max_cube_displacement_m=float(
            config["grasp"]["max_approach_cube_displacement_m"]
        ),
    )
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
    episodes = []
    try:
        for episode_index, start in enumerate(starts):
            episodes.append(_run_episode(
                episode_index=episode_index,
                start=start,
                robot=robot,
                config=config,
                policy=policy,
                output=args.output,
            ))
    finally:
        robot.disconnect()
    summary = _summarize(
        episodes,
        checkpoint=args.checkpoint,
        metadata=args.correction_metadata,
    )
    (args.output / "summary.json").write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        key: _plain(summary[key]) for key in (
            "episodes", "recovery_successes", "recovery_success_rate",
            "timeouts", "cube_safety_failures", "terminal_error_mm",
            "recovery_frames", "maximum_cube_displacement_mm",
            "action_clipping_values", "success_by_trigger",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
