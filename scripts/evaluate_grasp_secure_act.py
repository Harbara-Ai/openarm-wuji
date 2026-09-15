"""Closed-loop GraspSecure ACT evaluation from nominal or staged starts."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.policy import ApproachPolicy, ReachPolicy, RecoveryPolicy
from openarm_wuji.policy.grasp_secure_controller import GraspSecurePolicy
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import restore_simulator_snapshot
from openarm_wuji.tasks import ReachGraspLiftTask
from openarm_wuji.tasks.grasp_preload_stage import (
    FINGER_NAMES,
    finger_force_vector,
    grasp_window_metrics,
    initialize_task_from_handoff,
    window_passes,
)
from openarm_wuji.tasks.se3 import rotation_geodesic_angle_deg
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
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _state(observation: dict) -> np.ndarray:
    return np.concatenate([
        observation["arm_joint_position"], observation["hand_joint_position"]
    ]).astype(np.float32)


def _longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _set_grasp_deterministic(policy: GraspSecurePolicy, seed: int) -> None:
    torch = policy.controller.torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    torch.backends.mkldnn.enabled = False


def _run_grasp(*, robot, task, policy: GraspSecurePolicy,
               seed: int, source_kind: str, timeout_frames: int,
               preload_gate_rad: float, output: Path,
               episode_label: str, save_images: bool,
               frame_callback=None) -> dict[str, Any]:
    config = task.config
    minimum_fingers = int(config["grasp"]["min_finger_groups"])
    minimum_force = float(config["grasp"]["min_normal_force_n"])
    settle_window = int(config["grasp"]["contact_hold_frames"])
    baseline = config["external_baseline"]
    stage_start_cube = np.asarray(task.task_telemetry()["cube_position_m"])
    policy.reset()
    values: dict[str, list[Any]] = {
        "observation_state": [], "predicted_action": [], "sent_action": [],
        "next_state": [], "active_finger_mask": [],
        "finger_normal_force_n": [], "total_normal_force_n": [],
        "hand_target_actual_preload_rad": [], "hand_preload_l2_rad": [],
        "cube_pose_world": [], "cube_pose_relative_to_palm": [],
        "cube_displacement_m": [], "relative_translation_speed_m_s": [],
        "relative_rotation_speed_deg_s": [], "sim_time": [],
    }
    if save_images:
        values["observation.images.front"] = []
        values["observation.images.wrist"] = []
    telemetry_window: list[dict[str, Any]] = []
    contact_flags: list[bool] = []
    stable_contact_window_seen = False
    preload_window_seen = False
    first_contact_fingers: list[str] | None = None
    max_simultaneous = 0
    success_frame = None
    last_relative = None
    clip_counts = np.zeros(27, dtype=np.int64)
    clip_max = np.zeros(27, dtype=float)
    peak_forces = np.zeros(5, dtype=float)
    final_window_metrics = None

    for frame in range(timeout_frames):
        observation = robot.get_observation()
        predicted = policy.select_action(observation)
        sent = robot.send_controller_joint_target(predicted)
        post_observation = robot.latest_record
        next_state = _state(post_observation)
        telemetry = task.task_telemetry()
        forces = finger_force_vector(telemetry)
        active = forces >= minimum_force
        active_names = [
            name for name, flag in zip(FINGER_NAMES, active, strict=True) if flag
        ]
        if active_names and first_contact_fingers is None:
            first_contact_fingers = active_names
        max_simultaneous = max(max_simultaneous, len(active_names))
        peak_forces = np.maximum(peak_forces, forces)
        multi = len(active_names) >= minimum_fingers
        contact_flags.append(multi)
        preload_vector = sent[7:] - next_state[7:]
        preload_l2 = float(np.linalg.norm(preload_vector))
        cube = np.r_[
            telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"]
        ]
        relative = np.r_[
            telemetry["object_relative_position_m"],
            telemetry["object_relative_quaternion_wxyz"],
        ]
        if last_relative is None:
            translation_speed = rotation_speed = 0.0
        else:
            translation_speed = float(np.linalg.norm(
                relative[:3] - last_relative[:3]
            ) * robot.control_hz)
            rotation_speed = float(rotation_geodesic_angle_deg(
                last_relative[3:], relative[3:]
            ) * robot.control_hz)
        last_relative = relative
        clip = np.abs(predicted - sent)
        clip_counts += clip > 1e-9
        clip_max = np.maximum(clip_max, clip)
        values["observation_state"].append(_state(observation))
        values["predicted_action"].append(predicted)
        values["sent_action"].append(sent)
        values["next_state"].append(next_state)
        values["active_finger_mask"].append(active)
        values["finger_normal_force_n"].append(forces)
        values["total_normal_force_n"].append(float(np.sum(forces)))
        values["hand_target_actual_preload_rad"].append(preload_vector)
        values["hand_preload_l2_rad"].append(preload_l2)
        values["cube_pose_world"].append(cube)
        values["cube_pose_relative_to_palm"].append(relative)
        values["cube_displacement_m"].append(float(np.linalg.norm(
            np.asarray(telemetry["cube_position_m"]) - stage_start_cube
        )))
        values["relative_translation_speed_m_s"].append(translation_speed)
        values["relative_rotation_speed_deg_s"].append(rotation_speed)
        values["sim_time"].append(float(observation["sim_time"]))
        if save_images:
            values["observation.images.front"].append(
                np.asarray(observation["front_rgb"], dtype=np.uint8)
            )
            values["observation.images.wrist"].append(
                np.asarray(observation["wrist_rgb"], dtype=np.uint8)
            )
        if frame_callback is not None:
            frame_callback("GRASP_SECURE", robot.get_observation(), telemetry)
        telemetry_window.append(telemetry)
        if len(telemetry_window) >= settle_window:
            window = telemetry_window[-settle_window:]
            metrics = grasp_window_metrics(
                window, minimum_fingers=minimum_fingers,
                minimum_force_n=minimum_force,
            )
            stable_contact_window_seen = (
                stable_contact_window_seen or metrics["contacts_sustained"]
            )
            window_preload = float(np.mean(
                values["hand_preload_l2_rad"][-settle_window:]
            ))
            preload_window_seen = bool(
                preload_window_seen
                or (
                    metrics["contacts_sustained"]
                    and window_preload >= preload_gate_rad
                )
            )
            if window_passes(metrics, baseline=baseline) and (
                window_preload >= preload_gate_rad
            ):
                success_frame = frame
                final_window_metrics = metrics
                break

    arrays = {key: np.asarray(value) for key, value in values.items()}
    grasp_success = _longest_run(contact_flags) >= settle_window
    # Match the formal success gate: preload is the mean target-actual L2 over
    # a sustained-contact settle window.  Requiring every individual frame to
    # exceed the threshold can make a full success paradoxically report
    # preload_success=False.
    preload_success = preload_window_seen
    success = success_frame is not None
    if not any(contact_flags):
        failure_stage = "contact_acquisition"
    elif not grasp_success:
        failure_stage = "grasp_formation"
    elif not preload_success:
        failure_stage = "preload_formation"
    elif not success:
        failure_stage = "terminal_hold"
    else:
        failure_stage = None
    final_forces = arrays["finger_normal_force_n"][-1]
    metrics = {
        "seed": seed,
        "source_kind": source_kind,
        "grasp_preload_success": success,
        "grasp_success": grasp_success,
        "preload_success": preload_success,
        "terminal_hold_success": success,
        "failure_stage": failure_stage,
        "timeout": not success,
        "frames": len(arrays["sent_action"]),
        "success_frame": success_frame,
        "first_contact_fingers": first_contact_fingers or [],
        "max_simultaneous_contacts": max_simultaneous,
        "longest_persistent_multifinger_frames": _longest_run(contact_flags),
        "stable_contact_window_seen": stable_contact_window_seen,
        "preload_gate_rad": preload_gate_rad,
        "maximum_hand_preload_l2_rad": float(np.max(
            arrays["hand_preload_l2_rad"]
        )),
        "terminal_hand_preload_l2_rad": float(
            arrays["hand_preload_l2_rad"][-1]
        ),
        "peak_per_finger_normal_force_n": peak_forces,
        "final_per_finger_normal_force_n": final_forces,
        "maximum_total_normal_force_n": float(np.max(
            arrays["total_normal_force_n"]
        )),
        "maximum_cube_displacement_m": float(np.max(
            arrays["cube_displacement_m"]
        )),
        "maximum_relative_translation_speed_m_s": float(np.max(
            arrays["relative_translation_speed_m_s"]
        )),
        "maximum_relative_rotation_speed_deg_s": float(np.max(
            arrays["relative_rotation_speed_deg_s"]
        )),
        "final_window_metrics": final_window_metrics,
        "clipped_action_values": int(np.sum(clip_counts)),
        "clipping_by_joint": [
            {
                "index": index, "joint_name": JOINT_NAMES[index],
                "count": int(clip_counts[index]),
                "max_magnitude_rad": float(clip_max[index]),
            }
            for index in range(27)
        ],
        "unsupported_lift_executed": False,
        "expert_action_supplied_to_policy": False,
    }
    np.savez_compressed(
        output / f"{episode_label}.npz", **arrays,
        metrics_json=np.asarray(json.dumps(_plain(metrics))),
    )
    return metrics


def _load_snapshot(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def _summary(episodes: list[dict[str, Any]], *, checkpoint: Path,
             mode: str, preload_gate: float) -> dict[str, Any]:
    attempted = [item for item in episodes if item.get("grasp_attempted", True)]
    successes = sum(item["grasp_preload_success"] for item in attempted)
    failures: dict[str, int] = {}
    for item in attempted:
        if item["grasp_preload_success"]:
            continue
        stage = str(item["failure_stage"])
        failures[stage] = failures.get(stage, 0) + 1
    clipping_by_joint = []
    for index, name in enumerate(JOINT_NAMES):
        rows = [item["clipping_by_joint"][index] for item in attempted]
        clipping_by_joint.append({
            "index": index,
            "joint_name": name,
            "count": int(sum(row["count"] for row in rows)),
            "max_magnitude_rad": float(max(
                (row["max_magnitude_rad"] for row in rows), default=0.0
            )),
        })
    return {
        "checkpoint": checkpoint,
        "mode": mode,
        "episodes": len(episodes),
        "grasp_attempts": len(attempted),
        "upstream_successes": sum(
            item.get("grasp_attempted", True) for item in episodes
        ),
        "grasp_preload_successes": successes,
        "grasp_preload_success_rate": successes / len(attempted) if attempted else 0.0,
        "grasp_successes": sum(item["grasp_success"] for item in attempted),
        "preload_successes": sum(item["preload_success"] for item in attempted),
        "timeouts": sum(item["timeout"] for item in attempted),
        "failure_stage_distribution": failures,
        "preload_gate_rad": preload_gate,
        "cube_displacement_m": {
            "mean": float(np.mean([
                item["maximum_cube_displacement_m"] for item in attempted
            ])) if attempted else None,
            "median": float(np.median([
                item["maximum_cube_displacement_m"] for item in attempted
            ])) if attempted else None,
            "max": float(max(
                (item["maximum_cube_displacement_m"] for item in attempted),
                default=0.0,
            )),
        },
        "action_clipping_values": int(sum(
            item["clipped_action_values"] for item in attempted
        )),
        "action_clipping_by_joint": clipping_by_joint,
        "unsupported_lift_executed": False,
        "episodes_detail": episodes,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("standalone", "staged"), default="standalone")
    parser.add_argument(
        "--dataset-summary", type=Path,
        default=root / "outputs/grasp_preload_act/dataset/summary.json",
    )
    parser.add_argument(
        "--starts", type=Path,
        default=root / "outputs/grasp_preload_act/dataset/starts/nominal",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1800)
    parser.add_argument("--timeout-frames", type=int, default=80)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument(
        "--reach-checkpoint", type=Path,
        default=root / "outputs/act_reach_only/act_train/checkpoints/002000/pretrained_model",
    )
    parser.add_argument(
        "--approach-checkpoint", type=Path,
        default=root / "outputs/act_staged/approach_only/act_train/checkpoints/002000/pretrained_model",
    )
    parser.add_argument(
        "--recovery-checkpoint", type=Path,
        default=root / "outputs/staged_act_with_recovery/recovery_act_train/checkpoints/001500/pretrained_model",
    )
    parser.add_argument(
        "--router", type=Path, default=root / "configs/recovery_router.json"
    )
    parser.add_argument(
        "--model", type=Path,
        default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb",
    )
    parser.add_argument(
        "--config", type=Path, default=root / "configs/reach_grasp_lift.json"
    )
    parser.add_argument(
        "--synergies", type=Path,
        default=root / "configs/wuji_hand_left_synergies.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    dataset_summary = json.loads(args.dataset_summary.read_text(encoding="utf-8"))
    preload_gate = float(
        dataset_summary["preload_statistics"]["evaluation_preload_gate"]
        ["hand_preload_l2_min_rad"]
    )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = GraspSecurePolicy(args.checkpoint)
    _set_grasp_deterministic(policy, args.inference_seed)
    robot = MujocoOpenArmWuji(
        args.model, args.synergies,
        arm_side=config["arm_side"], control_hz=30,
        image_height=240, image_width=320,
        front_camera=config["scene"]["front_camera_name"],
    )
    robot.connect()
    episodes: list[dict[str, Any]] = []
    try:
        if args.mode == "standalone":
            starts = sorted(args.starts.glob("*.npz"))
            if args.limit is not None:
                starts = starts[:args.limit]
            for index, path in enumerate(starts):
                snapshot = _load_snapshot(path)
                seed = int(snapshot["source_seed"])
                task = ReachGraspLiftTask(robot, config)
                task.reset(seed)
                restore = restore_simulator_snapshot(robot, task, snapshot)
                if not restore["passed"]:
                    raise RuntimeError(f"snapshot restore failed: {path} {restore}")
                metrics = _run_grasp(
                    robot=robot, task=task, policy=policy, seed=seed,
                    source_kind=str(snapshot["source_kind"]),
                    timeout_frames=args.timeout_frames,
                    preload_gate_rad=preload_gate, output=args.output,
                    episode_label=f"rollout_{index:06d}_seed_{seed:06d}",
                    save_images=args.save_images,
                )
                metrics["start_snapshot"] = path
                metrics["restore_check"] = restore
                episodes.append(metrics)
                print(
                    f"standalone={index + 1}/{len(starts)} seed={seed} "
                    f"success={metrics['grasp_preload_success']}", flush=True,
                )
        else:
            router = json.loads(args.router.read_text(encoding="utf-8"))
            reach = ReachPolicy(args.reach_checkpoint)
            approach = ApproachPolicy(
                args.approach_checkpoint,
                max_cube_displacement_m=float(
                    config["grasp"]["max_approach_cube_displacement_m"]
                ),
            )
            recovery = RecoveryPolicy(
                args.recovery_checkpoint,
                max_cube_displacement_m=float(
                    config["grasp"]["max_approach_cube_displacement_m"]
                ),
            )
            for upstream_policy in (reach, approach, recovery):
                _set_deterministic(upstream_policy, args.inference_seed)
            upstream_dir = args.output / "upstream"
            upstream_dir.mkdir()
            for index in range(args.rollouts):
                seed = args.seed_start + index
                upstream = run_frozen_upstream(
                    seed=seed, rollout_index=index, robot=robot, config=config,
                    reach_policy=reach, approach_policy=approach,
                    recovery_policy=recovery, output=upstream_dir,
                    router_spec=router,
                )
                if not upstream["new_success"]:
                    episodes.append({
                        "seed": seed, "grasp_attempted": False,
                        "upstream": upstream,
                    })
                    print(
                        f"staged={index + 1}/{args.rollouts} seed={seed} "
                        f"upstream={upstream['new_outcome']}", flush=True,
                    )
                    continue
                task = ReachGraspLiftTask(robot, config)
                initialize_task_from_handoff(task)
                metrics = _run_grasp(
                    robot=robot, task=task, policy=policy, seed=seed,
                    source_kind="live_staged_handoff",
                    timeout_frames=args.timeout_frames,
                    preload_gate_rad=preload_gate, output=args.output,
                    episode_label=f"rollout_{index:06d}_seed_{seed:06d}",
                    save_images=args.save_images,
                )
                metrics["grasp_attempted"] = True
                metrics["upstream"] = upstream
                episodes.append(metrics)
                print(
                    f"staged={index + 1}/{args.rollouts} seed={seed} "
                    f"grasp={metrics['grasp_preload_success']}", flush=True,
                )
    finally:
        robot.disconnect()
    summary = _summary(
        episodes, checkpoint=args.checkpoint,
        mode=args.mode, preload_gate=preload_gate,
    )
    summary["closed_loop_observation_only"] = True
    summary["expert_state_or_action_injected"] = False
    summary["frozen_upstream"] = args.mode == "staged"
    (args.output / "summary.json").write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_plain({
        key: summary[key] for key in (
            "mode", "episodes", "grasp_attempts",
            "grasp_preload_successes", "grasp_preload_success_rate",
            "failure_stage_distribution", "cube_displacement_m",
            "action_clipping_values",
        )
    }), indent=2))


if __name__ == "__main__":
    main()
