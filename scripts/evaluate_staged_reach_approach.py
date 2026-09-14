"""Evaluate Approach-only or frozen Reach→Approach ACT stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.policy import ApproachPolicy, ReachPolicy
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


TRAINING_SEEDS = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22]


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


def _set_deterministic(policy, seed: int) -> None:
    torch = policy.controller.torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    torch.backends.mkldnn.enabled = False


def _expert_starts(raw_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    for path in sorted(raw_dir.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            seed = int(episode["episode_seed"])
            result[seed] = {
                "state": np.asarray(episode["observation.state"][0], dtype=float),
                "controller_target": np.asarray(
                    episode["initial_controller_target"], dtype=float
                ),
                "cube_pose": np.asarray(
                    episode["telemetry.cube_pose_world"][0], dtype=float
                ),
                "front": np.asarray(episode["observation.images.front"][0]),
                "wrist": np.asarray(episode["observation.images.wrist"][0]),
            }
    if sorted(result) != TRAINING_SEEDS:
        raise ValueError("Approach raw dataset does not contain the exact training seeds")
    return result


def _initialize_from_expert(task: ReachGraspLiftTask, robot: MujocoOpenArmWuji,
                            start: dict[str, np.ndarray]) -> float:
    import mujoco

    state = start["state"]
    controller_target = start["controller_target"]
    robot.data.qpos[robot.arm_qpos_ids] = state[:7]
    robot.data.qpos[robot.hand_qpos_ids] = state[7:]
    robot.data.qvel.fill(0.0)
    if robot.data.act.size:
        robot.data.act.fill(0.0)
    cube_address = int(robot.model.jnt_qposadr[task.cube_joint_id])
    robot.data.qpos[cube_address:cube_address + 7] = start["cube_pose"]
    actuator_ids = np.concatenate([robot.arm_actuator_ids, robot.hand_actuator_ids])
    robot.data.ctrl[actuator_ids] = controller_target
    mujoco.mj_forward(robot.model, robot.data)
    robot.synchronize_after_reset(controller_target[7:])
    task.initial_cube_position = start["cube_pose"][:3].copy()
    actual = _state(robot.get_observation())
    return float(np.max(np.abs(actual - state)))


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


def _run_approach(*, task: ReachGraspLiftTask, robot: MujocoOpenArmWuji,
                  policy: ApproachPolicy, target: np.ndarray,
                  start_cube: np.ndarray, initial_hand_target: np.ndarray,
                  save_images: bool) -> tuple[dict, dict[str, np.ndarray]]:
    policy.reset()
    values: dict[str, list] = {
        "observation_state": [], "predicted_action": [], "sent_action": [],
        "position_error_m": [], "orientation_error_deg": [],
        "cube_position_m": [], "cube_displacement_m": [],
        "contact_count": [], "sim_time": [],
    }
    if save_images:
        values["observation.images.front"] = []
        values["observation.images.wrist"] = []
    first_contact = None
    contact_fingers: set[str] = set()
    max_consecutive_position_gate = 0
    position_gate_run = 0
    success_frame = None
    for frame in range(policy.timeout_frames):
        observation = robot.get_observation()
        predicted = policy.select_action(observation)
        sent = robot.send_controller_joint_target(predicted)
        telemetry = task.task_telemetry()
        cube = np.asarray(telemetry["cube_position_m"], dtype=float)
        position_error = float(np.linalg.norm(
            np.asarray(telemetry["grasp_center_position_m"], dtype=float) - target
        ))
        orientation_error = float(telemetry["palm_orientation_error_deg"])
        cube_displacement = float(np.linalg.norm(cube - start_cube))
        status = policy.observe(
            position_error_m=position_error,
            orientation_error_deg=orientation_error,
            cube_displacement_m=cube_displacement,
        )
        fingers = sorted({str(contact["finger"]) for contact in telemetry["contacts"]})
        if fingers and first_contact is None:
            first_contact = frame
        contact_fingers.update(fingers)
        position_gate_run = position_gate_run + 1 if position_error <= 0.012 else 0
        max_consecutive_position_gate = max(max_consecutive_position_gate, position_gate_run)
        values["observation_state"].append(_state(observation))
        values["predicted_action"].append(predicted)
        values["sent_action"].append(sent)
        values["position_error_m"].append(position_error)
        values["orientation_error_deg"].append(orientation_error)
        values["cube_position_m"].append(cube)
        values["cube_displacement_m"].append(cube_displacement)
        values["contact_count"].append(len(fingers))
        values["sim_time"].append(float(robot.data.time))
        if save_images:
            values["observation.images.front"].append(observation["front_rgb"])
            values["observation.images.wrist"].append(observation["wrist_rgb"])
        if status.success:
            success_frame = frame
            break
        if status.failure_reason is not None:
            break
    arrays = {key: np.asarray(value) for key, value in values.items()}
    predicted = arrays["predicted_action"]
    sent = arrays["sent_action"]
    clipped, clipping_by_joint = _clipping(predicted, sent)
    status = policy.status
    metrics = {
        "approach_success": bool(status.success),
        "approach_success_frame": success_frame,
        "failure_reason": status.failure_reason,
        "timeout": bool(status.timeout),
        "frames_executed": int(len(predicted)),
        "minimum_grasp_pose_position_error_m": float(arrays["position_error_m"].min()),
        "final_grasp_pose_position_error_m": float(arrays["position_error_m"][-1]),
        "minimum_orientation_error_deg": float(arrays["orientation_error_deg"].min()),
        "final_orientation_error_deg": float(arrays["orientation_error_deg"][-1]),
        "longest_consecutive_frames_inside_12mm": int(max_consecutive_position_gate),
        "final_gate_dwell_frames": int(status.consecutive_gate_frames),
        "maximum_cube_displacement_m": float(arrays["cube_displacement_m"].max()),
        "final_cube_displacement_m": float(arrays["cube_displacement_m"][-1]),
        "early_contact": first_contact is not None,
        "first_contact_frame": first_contact,
        "contact_fingers_seen": sorted(contact_fingers),
        "max_simultaneous_contacts": int(arrays["contact_count"].max(initial=0)),
        "max_hand_target_change_rad": float(
            np.max(np.abs(sent[:, 7:] - initial_hand_target[None]))
        ),
        "clipped_action_values": clipped,
        "clipping_by_joint": clipping_by_joint,
        "expert_action_supplied_to_policy": False,
        "executed_stages": ["approach"],
        "excluded_stages": ["grasp", "preload", "lift", "hold"],
    }
    arrays["target_grasp_position_m"] = np.asarray(target)
    arrays["start_cube_position_m"] = np.asarray(start_cube)
    return metrics, arrays


def _aggregate(episodes: list[dict]) -> dict:
    attempted = [item for item in episodes if item.get("approach_attempted", True)]
    successes = [item for item in attempted if item["approach_success"]]
    return {
        "episodes": len(episodes),
        "reach_success_count": int(sum(bool(item.get("reach_success", True)) for item in episodes)),
        "approach_attempted_count": len(attempted),
        "approach_success_count": len(successes),
        "approach_conditional_success_rate": len(successes) / len(attempted) if attempted else 0.0,
        "joint_reach_approach_success_count": int(sum(
            bool(item.get("reach_success", True)) and bool(item["approach_success"])
            for item in episodes
        )),
        "timeout_count": int(sum(bool(item.get("timeout", False)) for item in attempted)),
        "minimum_position_error_mean_mm": float(np.mean([
            item["minimum_grasp_pose_position_error_m"] for item in attempted
        ]) * 1000) if attempted else None,
        "final_position_error_mean_mm": float(np.mean([
            item["final_grasp_pose_position_error_m"] for item in attempted
        ]) * 1000) if attempted else None,
        "maximum_cube_displacement_mean_mm": float(np.mean([
            item["maximum_cube_displacement_m"] for item in attempted
        ]) * 1000) if attempted else None,
        "maximum_cube_displacement_max_mm": float(np.max([
            item["maximum_cube_displacement_m"] for item in attempted
        ]) * 1000) if attempted else None,
        "early_contact_count": int(sum(bool(item["early_contact"]) for item in attempted)),
        "clipped_action_values": int(sum(item["clipped_action_values"] for item in attempted)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("approach-only", "reach-approach"), required=True)
    parser.add_argument("--approach-checkpoint", type=Path, required=True)
    parser.add_argument("--reach-checkpoint", type=Path)
    parser.add_argument("--approach-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb")
    parser.add_argument("--config", type=Path, default=root / "configs/reach_grasp_lift.json")
    parser.add_argument("--synergies", type=Path, default=root / "configs/wuji_hand_left_synergies.json")
    parser.add_argument("--seeds", default=",".join(map(str, TRAINING_SEEDS)))
    parser.add_argument("--reach-timeout", type=int, default=160)
    parser.add_argument("--approach-timeout", type=int, default=90)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--inference-seed", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "reach-approach" and args.reach_checkpoint is None:
        raise ValueError("--reach-checkpoint is required for reach-approach mode")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expert_starts = _expert_starts(args.approach_raw)

    approach_policy = ApproachPolicy(
        args.approach_checkpoint,
        timeout_frames=args.approach_timeout,
        max_cube_displacement_m=float(
            config["grasp"]["max_approach_cube_displacement_m"]
        ),
    )
    _set_deterministic(approach_policy, args.inference_seed)
    reach_policy = None
    if args.mode == "reach-approach":
        reach_policy = ReachPolicy(args.reach_checkpoint, timeout_frames=args.reach_timeout)
        _set_deterministic(reach_policy, args.inference_seed)

    robot = MujocoOpenArmWuji(
        args.model, args.synergies,
        arm_side=config["arm_side"], control_hz=30,
        image_height=240, image_width=320,
        front_camera=config["scene"]["front_camera_name"],
    )
    robot.connect()
    episodes = []
    training_start_matrix = np.stack([
        expert_starts[seed]["state"] for seed in TRAINING_SEEDS
    ])
    try:
        for seed in seeds:
            task = ReachGraspLiftTask(robot, config)
            task.reset(seed)
            reach_metrics = {}
            if args.mode == "approach-only":
                exact_error = _initialize_from_expert(
                    task, robot, expert_starts[seed]
                )
                observation = robot.get_observation()
                handoff_state = _state(observation)
                start_cube = np.asarray(
                    expert_starts[seed]["cube_pose"][:3], dtype=float
                )
                initial_hand_target = expert_starts[seed]["controller_target"][7:]
                reach_metrics = {
                    "approach_attempted": True,
                    "initial_state_max_abs_error_vs_expert": exact_error,
                }
            else:
                assert reach_policy is not None
                reach_policy.reset()
                reset_cube = np.asarray(task.initial_cube_position, dtype=float)
                reach_errors = []
                for _ in range(reach_policy.timeout_frames):
                    observation = robot.get_observation()
                    predicted = reach_policy.select_action(observation)
                    robot.send_controller_joint_target(predicted)
                    telemetry = task.task_telemetry()
                    position_error = float(np.linalg.norm(
                        np.asarray(telemetry["grasp_center_position_m"])
                        - np.asarray(task.target_position)
                    ))
                    cube_displacement = float(np.linalg.norm(
                        np.asarray(telemetry["cube_position_m"]) - reset_cube
                    ))
                    status = reach_policy.observe(
                        position_error_m=position_error,
                        orientation_error_deg=float(telemetry["palm_orientation_error_deg"]),
                        cube_displacement_m=cube_displacement,
                    )
                    reach_errors.append(position_error)
                    if status.success or status.failure_reason is not None:
                        break
                reach_metrics = {
                    "reach_success": reach_policy.is_success(),
                    "reach_frames": reach_policy.status.frame,
                    "reach_minimum_pregrasp_error_m": float(np.min(reach_errors)),
                    "reach_final_pregrasp_error_m": float(reach_errors[-1]),
                    "reach_failure_reason": reach_policy.status.failure_reason,
                    "approach_attempted": reach_policy.is_success(),
                }
                if not reach_policy.is_success():
                    episode = {
                        "seed": seed, **reach_metrics,
                        "approach_success": False,
                        "failure_stage": "reach",
                    }
                    episodes.append(episode)
                    continue
                observation = robot.get_observation()
                handoff_state = _state(observation)
                telemetry = task.task_telemetry()
                start_cube = np.asarray(telemetry["cube_position_m"], dtype=float)
                initial_hand_target = np.asarray(
                    observation["controller_joint_target"][7:], dtype=float
                )
                same_seed = expert_starts[seed]["state"]
                differences = training_start_matrix - handoff_state[None]
                nearest = int(np.argmin(np.linalg.norm(differences, axis=1)))
                reach_metrics["handoff"] = {
                    "state": handoff_state,
                    "cube_position_m": start_cube,
                    "same_seed_state_l2_rad": float(np.linalg.norm(handoff_state - same_seed)),
                    "same_seed_arm_l2_rad": float(np.linalg.norm(handoff_state[:7] - same_seed[:7])),
                    "same_seed_hand_l2_rad": float(np.linalg.norm(handoff_state[7:] - same_seed[7:])),
                    "same_seed_state_max_abs_rad": float(np.max(np.abs(handoff_state - same_seed))),
                    "nearest_expert_seed": TRAINING_SEEDS[nearest],
                    "nearest_expert_state_l2_rad": float(
                        np.linalg.norm(differences[nearest])
                    ),
                    "cube_position_error_vs_same_seed_m": float(np.linalg.norm(
                        start_cube - expert_starts[seed]["cube_pose"][:3]
                    )),
                }

            target = start_cube + np.asarray(
                config["grasp"]["target_offset_m"], dtype=float
            )
            approach_metrics, arrays = _run_approach(
                task=task, robot=robot, policy=approach_policy,
                target=target, start_cube=start_cube,
                initial_hand_target=initial_hand_target,
                save_images=args.save_images,
            )
            episode = {
                "seed": seed, **reach_metrics, **approach_metrics,
                "failure_stage": None if approach_metrics["approach_success"] else "approach",
            }
            episodes.append(episode)
            arrays["handoff_state"] = handoff_state
            arrays["expert_same_seed_start_state"] = expert_starts[seed]["state"]
            np.savez_compressed(
                args.output / f"rollout_seed_{seed:06d}.npz", **arrays
            )
    finally:
        robot.disconnect()

    summary = {
        "mode": args.mode,
        "task": "Reach ACT -> gate -> Approach ACT" if args.mode == "reach-approach" else "Approach ACT",
        "seeds": seeds,
        "reach_checkpoint": args.reach_checkpoint,
        "approach_checkpoint": args.approach_checkpoint,
        "state_dim": 27,
        "action_dim": 27,
        "execution_horizon": 1,
        "expert_action_supplied_to_policy": False,
        "gates": {
            "reach": {"position_m": 0.012, "hold_frames": 5},
            "approach": {
                "position_m": 0.012, "orientation_deg": 2.0,
                "hold_frames": 5,
                "max_cube_displacement_m": float(
                    config["grasp"]["max_approach_cube_displacement_m"]
                ),
                "early_contact_is_diagnostic_only": True,
            },
        },
        "episodes": episodes,
        "aggregate": _aggregate(episodes),
    }
    if args.mode == "reach-approach":
        handoffs = [item["handoff"] for item in episodes if "handoff" in item]
        summary["handoff_distribution"] = {
            "count": len(handoffs),
            "same_seed_arm_l2_mean_rad": float(np.mean([
                item["same_seed_arm_l2_rad"] for item in handoffs
            ])) if handoffs else None,
            "same_seed_arm_l2_max_rad": float(np.max([
                item["same_seed_arm_l2_rad"] for item in handoffs
            ])) if handoffs else None,
            "same_seed_state_max_abs_mean_rad": float(np.mean([
                item["same_seed_state_max_abs_rad"] for item in handoffs
            ])) if handoffs else None,
            "nearest_expert_state_l2_mean_rad": float(np.mean([
                item["nearest_expert_state_l2_rad"] for item in handoffs
            ])) if handoffs else None,
            "cube_position_error_mean_mm": float(np.mean([
                item["cube_position_error_vs_same_seed_m"] for item in handoffs
            ]) * 1000) if handoffs else None,
        }
    (args.output / "summary.json").write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_plain(summary["aggregate"]), indent=2))
    if "handoff_distribution" in summary:
        print(json.dumps(_plain(summary["handoff_distribution"]), indent=2))


if __name__ == "__main__":
    main()
