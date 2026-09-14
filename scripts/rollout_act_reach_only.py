"""Pure closed-loop Reach-only ACT evaluation in the existing MuJoCo scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.policy import ACTController
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


TASK = "Move OpenArm and the open Wuji hand to the cube pregrasp pose and hold it stable."
TRAINING_SEEDS = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22]


def _longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _plain(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def run_episode(*, robot: MujocoOpenArmWuji, controller: ACTController,
                config: dict, seed: int, timeout_frames: int,
                output_dir: Path, save_images: bool) -> dict:
    task = ReachGraspLiftTask(robot, config)
    task.reset(seed)
    controller.reset()
    target = np.asarray(task.target_position, dtype=float)
    reset_cube = np.asarray(task.initial_cube_position, dtype=float)
    gate_m = 0.012
    hold_frames = 5
    min_force = float(config["grasp"]["min_normal_force_n"])

    arrays: dict[str, list] = {
        "observation_state": [], "predicted_action": [], "sent_action": [],
        "cube_position_m": [], "grasp_center_position_m": [],
        "pregrasp_error_m": [], "palm_orientation_error_deg": [],
        "inside_12mm_gate": [], "contact_count": [], "sim_time": [],
    }
    if save_images:
        arrays["observation.images.front"] = []
        arrays["observation.images.wrist"] = []

    consecutive = 0
    first_entry = None
    success_frame = None
    first_contact_frame = None
    contact_fingers_seen: set[str] = set()
    for frame in range(timeout_frames):
        observation = robot.get_observation()
        state = np.concatenate([
            observation["arm_joint_position"], observation["hand_joint_position"]
        ]).astype(np.float32)
        # H_exec=1: discard the cached chunk and predict from the current frame.
        controller.reset()
        predicted = controller.predict(
            state=state,
            front_rgb=observation["front_rgb"],
            wrist_rgb=observation["wrist_rgb"],
        )
        sent = robot.send_controller_joint_target(predicted)
        telemetry = task.task_telemetry()
        cube = np.asarray(telemetry["cube_position_m"], dtype=float)
        grasp = np.asarray(telemetry["grasp_center_position_m"], dtype=float)
        error = float(np.linalg.norm(grasp - target))
        fingers = sorted({
            str(contact["finger"])
            for contact in telemetry["contacts"]
            if float(contact["normal_force_n"]) >= min_force
        })
        contact_fingers_seen.update(fingers)
        if fingers and first_contact_frame is None:
            first_contact_frame = frame
        inside = error <= gate_m
        if inside and first_entry is None:
            first_entry = frame
        consecutive = consecutive + 1 if inside else 0

        arrays["observation_state"].append(state)
        arrays["predicted_action"].append(predicted)
        arrays["sent_action"].append(sent)
        arrays["cube_position_m"].append(cube)
        arrays["grasp_center_position_m"].append(grasp)
        arrays["pregrasp_error_m"].append(error)
        arrays["palm_orientation_error_deg"].append(
            float(telemetry["palm_orientation_error_deg"])
        )
        arrays["inside_12mm_gate"].append(inside)
        arrays["contact_count"].append(len(fingers))
        arrays["sim_time"].append(float(robot.data.time))
        if save_images:
            arrays["observation.images.front"].append(
                np.asarray(observation["front_rgb"], dtype=np.uint8)
            )
            arrays["observation.images.wrist"].append(
                np.asarray(observation["wrist_rgb"], dtype=np.uint8)
            )
        if consecutive >= hold_frames:
            success_frame = frame
            break

    states = np.asarray(arrays["observation_state"], dtype=np.float32)
    predicted = np.asarray(arrays["predicted_action"], dtype=np.float32)
    sent = np.asarray(arrays["sent_action"], dtype=np.float32)
    cube = np.asarray(arrays["cube_position_m"], dtype=float)
    errors = np.asarray(arrays["pregrasp_error_m"], dtype=float)
    inside = np.asarray(arrays["inside_12mm_gate"], dtype=bool)
    clip = np.abs(predicted - sent)
    cube_displacement = np.linalg.norm(cube - reset_cube, axis=1)
    metrics = {
        "seed": seed,
        "frames_executed": int(len(states)),
        "timeout_frames": timeout_frames,
        "reach_success": success_frame is not None,
        "reach_success_frame": success_frame,
        "timeout": success_frame is None,
        "minimum_pregrasp_error_m": float(errors.min()),
        "frames_inside_12mm_gate": int(np.count_nonzero(inside)),
        "longest_consecutive_frames_inside_12mm": _longest_run(inside.tolist()),
        "first_frame_enter_12mm": first_entry,
        "last_frame_inside_12mm": int(np.flatnonzero(inside)[-1]) if np.any(inside) else None,
        "gate_m": gate_m,
        "required_hold_frames": hold_frames,
        "maximum_cube_displacement_m": float(cube_displacement.max()),
        "final_cube_displacement_m": float(cube_displacement[-1]),
        "first_contact_frame": first_contact_frame,
        "early_cube_contact": first_contact_frame is not None,
        "max_simultaneous_contacts": int(max(arrays["contact_count"], default=0)),
        "contact_fingers_seen": sorted(contact_fingers_seen),
        "clipped_action_values": int(np.count_nonzero(clip > 1e-9)),
        "max_abs_action_clip_rad": float(clip.max()),
        "clipping_by_joint": [
            {
                "index": index,
                "joint_name": JOINT_NAMES[index],
                "count": int(np.count_nonzero(clip[:, index] > 1e-9)),
                "max_magnitude_rad": float(clip[:, index].max()),
                "mean_magnitude_rad": float(clip[:, index].mean()),
            }
            for index in range(27)
        ],
        "expert_action_supplied_to_policy": False,
        "policy_reobserved_every_frame": True,
        "executed_stages": ["reach", "stable_pregrasp_hold_if_reached"],
        "excluded_stages": ["approach", "grasp", "preload", "lift", "hold"],
    }
    save_arrays = {
        key: np.asarray(value) for key, value in arrays.items()
    }
    save_arrays["reset_cube_position_m"] = reset_cube
    save_arrays["target_pregrasp_position_m"] = target
    np.savez_compressed(
        output_dir / f"rollout_seed_{seed:06d}.npz", **save_arrays
    )
    return metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb")
    parser.add_argument("--config", type=Path, default=root / "configs/reach_grasp_lift.json")
    parser.add_argument("--synergies", type=Path, default=root / "configs/wuji_hand_left_synergies.json")
    parser.add_argument("--seeds", default=",".join(map(str, TRAINING_SEEDS)))
    parser.add_argument("--timeout-frames", type=int, default=160)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--deterministic-inference", action="store_true")
    parser.add_argument("--inference-seed", type=int, default=0)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or args.timeout_frames < 5:
        raise ValueError("invalid seeds or timeout")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    controller = ACTController(args.checkpoint, device="cpu", task=TASK)
    if args.deterministic_inference:
        np.random.seed(args.inference_seed)
        controller.torch.manual_seed(args.inference_seed)
        controller.torch.set_num_threads(1)
        controller.torch.use_deterministic_algorithms(True)
        controller.torch.backends.mkldnn.deterministic = True
        controller.torch.backends.mkldnn.enabled = False
    robot = MujocoOpenArmWuji(
        args.model, args.synergies,
        arm_side=config["arm_side"], control_hz=30,
        image_height=240, image_width=320,
        front_camera=config["scene"]["front_camera_name"],
    )
    robot.connect()
    try:
        episodes = [
            run_episode(
                robot=robot, controller=controller, config=config, seed=seed,
                timeout_frames=args.timeout_frames, output_dir=args.output,
                save_images=args.save_images,
            )
            for seed in seeds
        ]
    finally:
        robot.disconnect()
    minimums = np.asarray([item["minimum_pregrasp_error_m"] for item in episodes])
    summary = {
        "checkpoint": args.checkpoint.resolve(),
        "task": TASK,
        "seeds": seeds,
        "state_dim": 27,
        "action_dim": 27,
        "camera_shape_hwc": [240, 320, 3],
        "fps": 30,
        "execution_horizon": 1,
        "timeout_frames": args.timeout_frames,
        "expert_action_supplied_to_policy": False,
        "closed_loop_observation_only": True,
        "images_saved": bool(args.save_images),
        "deterministic_inference": bool(args.deterministic_inference),
        "episodes": episodes,
        "aggregate": {
            "reach_success_count": int(sum(item["reach_success"] for item in episodes)),
            "reach_success_rate": float(np.mean([item["reach_success"] for item in episodes])),
            "timeout_count": int(sum(item["timeout"] for item in episodes)),
            "minimum_pregrasp_error_mean_m": float(minimums.mean()),
            "minimum_pregrasp_error_median_m": float(np.median(minimums)),
            "minimum_pregrasp_error_best_m": float(minimums.min()),
            "episodes_entering_12mm": int(sum(item["frames_inside_12mm_gate"] > 0 for item in episodes)),
            "total_frames_inside_12mm": int(sum(item["frames_inside_12mm_gate"] for item in episodes)),
            "max_longest_consecutive_dwell": int(max(item["longest_consecutive_frames_inside_12mm"] for item in episodes)),
            "early_cube_contact_count": int(sum(item["early_cube_contact"] for item in episodes)),
            "cube_displacement_over_25mm_count": int(sum(item["maximum_cube_displacement_m"] > 0.025 for item in episodes)),
            "clipped_action_values": int(sum(item["clipped_action_values"] for item in episodes)),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_plain(summary["aggregate"]), indent=2))


if __name__ == "__main__":
    main()
