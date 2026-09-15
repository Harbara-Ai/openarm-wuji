"""Roll out a trained 27-D LeRobot ACT policy in the Reach-Grasp-Lift scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openarm_wuji.policy import ACTController
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


def _longest_run(values: list[bool], start: int = 0) -> int:
    longest = current = 0
    for value in values[start:]:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _first_true(values: list[bool], start: int = 0) -> int | None:
    return next((index for index in range(start, len(values)) if values[index]), None)


def _first_run(values: list[bool], length: int, start: int = 0) -> int | None:
    """Return the first index of a contiguous qualifying run."""
    if length < 1:
        raise ValueError("run length must be positive")
    current = 0
    for index in range(start, len(values)):
        current = current + 1 if values[index] else 0
        if current >= length:
            return index - length + 1
    return None


def _target_motion_stats(deltas: np.ndarray) -> dict:
    """Summarize frame-to-frame absolute arm-target motion and reversals."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 2:
        raise ValueError("target deltas must be a 2-D array")
    if len(deltas) == 0:
        return {
            "transitions": 0,
            "mean_l2_delta_rad": 0.0,
            "median_l2_delta_rad": 0.0,
            "max_l2_delta_rad": 0.0,
            "mean_l2_second_delta_rad": 0.0,
            "max_l2_second_delta_rad": 0.0,
            "direction_reversal_fraction": 0.0,
            "direction_reversal_pairs": 0,
            "valid_direction_pairs": 0,
            "per_joint_mean_abs_delta_rad": [0.0] * int(deltas.shape[1]),
            "per_joint_max_abs_delta_rad": [0.0] * int(deltas.shape[1]),
        }
    norms = np.linalg.norm(deltas, axis=1)
    second = np.diff(deltas, axis=0)
    second_norms = np.linalg.norm(second, axis=1)
    if len(deltas) >= 2:
        previous_norm = norms[:-1]
        current_norm = norms[1:]
        valid = (previous_norm > 1e-6) & (current_norm > 1e-6)
        dots = np.sum(deltas[:-1] * deltas[1:], axis=1)
        reversals = valid & (dots < 0.0)
        reversal_count = int(np.count_nonzero(reversals))
        valid_count = int(np.count_nonzero(valid))
    else:
        reversal_count = valid_count = 0
    return {
        "transitions": int(len(deltas)),
        "mean_l2_delta_rad": float(norms.mean()),
        "median_l2_delta_rad": float(np.median(norms)),
        "max_l2_delta_rad": float(norms.max()),
        "mean_l2_second_delta_rad": float(second_norms.mean()) if len(second_norms) else 0.0,
        "max_l2_second_delta_rad": float(second_norms.max()) if len(second_norms) else 0.0,
        "direction_reversal_fraction": (
            reversal_count / valid_count if valid_count else 0.0
        ),
        "direction_reversal_pairs": reversal_count,
        "valid_direction_pairs": valid_count,
        "per_joint_mean_abs_delta_rad": np.mean(np.abs(deltas), axis=0).tolist(),
        "per_joint_max_abs_delta_rad": np.max(np.abs(deltas), axis=0).tolist(),
    }


def run_episode(*, robot: MujocoOpenArmWuji, controller: ACTController,
                config: dict, seed: int, steps: int, execution_horizon: int,
                output_dir: Path, save_images: bool = False) -> dict:
    task = ReachGraspLiftTask(robot, config)
    task.reset(seed)
    controller.reset()
    target_pregrasp = task.target_position.copy()
    target_approach_offset = np.asarray(config["grasp"]["target_offset_m"], dtype=float)
    reach_position_tolerance = float(config["reach"]["position_tolerance_m"])
    orientation_tolerance = float(config["reach"]["orientation_tolerance_deg"])
    lift_height = float(config["lift"]["success_height_m"])
    lift_hold_frames = int(config["lift"]["success_hold_frames"])
    min_force = float(config["grasp"]["min_normal_force_n"])
    reach_hold_frames = int(config["reach"]["hold_frames"])
    approach_hold_frames = int(config["grasp"]["approach_settle_frames"])
    grasp_hold_frames = int(config["grasp"]["contact_hold_frames"])
    grasp_contact_groups = int(config["grasp"]["min_finger_groups"])
    lift_contact_groups = int(config["lift"]["min_finger_groups"])
    max_approach_cube_displacement = float(
        config["grasp"]["max_approach_cube_displacement_m"]
    )

    states: list[np.ndarray] = []
    predicted_actions: list[np.ndarray] = []
    sent_actions: list[np.ndarray] = []
    cube_positions: list[np.ndarray] = []
    grasp_positions: list[np.ndarray] = []
    heights: list[float] = []
    reach_errors: list[float] = []
    approach_errors: list[float] = []
    orientation_errors: list[float] = []
    contact_counts: list[int] = []
    contact_fingers: list[list[str]] = []
    sim_times: list[float] = []
    policy_replan_flags: list[bool] = []
    front_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []

    for frame in range(steps):
        observation = robot.get_observation()
        state = np.concatenate([
            observation["arm_joint_position"],
            observation["hand_joint_position"],
        ]).astype(np.float32)
        # ACTPolicy.select_action() otherwise consumes its cached n_action_steps
        # queue without using the newly supplied observation. Clearing the queue
        # here makes the next call re-observe and predict a fresh action chunk.
        replan = frame % execution_horizon == 0
        if replan:
            controller.reset()
        predicted = controller.predict(
            state=state,
            front_rgb=observation["front_rgb"],
            wrist_rgb=observation["wrist_rgb"],
        )
        sent = robot.send_controller_joint_target(predicted)
        telemetry = task.task_telemetry()
        fingers = sorted({
            str(contact["finger"])
            for contact in telemetry["contacts"]
            if float(contact["normal_force_n"]) >= min_force
        })
        cube = np.asarray(telemetry["cube_position_m"], dtype=float)
        grasp = np.asarray(telemetry["grasp_center_position_m"], dtype=float)
        states.append(state)
        predicted_actions.append(predicted)
        sent_actions.append(sent)
        cube_positions.append(cube)
        grasp_positions.append(grasp)
        heights.append(float(telemetry["cube_height_m"]))
        reach_errors.append(float(np.linalg.norm(grasp - target_pregrasp)))
        approach_errors.append(float(np.linalg.norm(
            grasp - (cube + target_approach_offset)
        )))
        orientation_errors.append(float(telemetry["palm_orientation_error_deg"]))
        contact_counts.append(len(fingers))
        contact_fingers.append(fingers)
        sim_times.append(float(robot.data.time))
        policy_replan_flags.append(replan)
        if save_images:
            front_frames.append(np.asarray(observation["front_rgb"], dtype=np.uint8))
            wrist_frames.append(np.asarray(observation["wrist_rgb"], dtype=np.uint8))

    cube_array = np.asarray(cube_positions)
    reach_flags = [
        position <= reach_position_tolerance and orientation <= orientation_tolerance
        for position, orientation in zip(reach_errors, orientation_errors)
    ]
    approach_flags = [
        position <= reach_position_tolerance
        and orientation <= orientation_tolerance
        and float(np.linalg.norm(cube - task.initial_cube_position))
        <= max_approach_cube_displacement
        for position, orientation, cube in zip(
            approach_errors, orientation_errors, cube_array
        )
    ]
    multi_contact_flags = [
        count >= grasp_contact_groups for count in contact_counts
    ]
    lift_flags = [height >= lift_height for height in heights]
    held_lift_flags = [
        lifted and count >= lift_contact_groups
        for lifted, count in zip(lift_flags, contact_counts)
    ]

    # These are deliberately ordered, sustained milestones.  A single pose
    # threshold crossing, incidental collision, or tossed cube must not count
    # as successful Reach-Grasp-Lift reproduction.
    reach_frame = _first_run(reach_flags, reach_hold_frames)
    reach_end = None if reach_frame is None else reach_frame + reach_hold_frames
    approach_frame = (
        _first_run(approach_flags, approach_hold_frames, reach_end)
        if reach_end is not None else None
    )
    approach_end = (
        None if approach_frame is None else approach_frame + approach_hold_frames
    )
    grasp_frame = (
        _first_run(multi_contact_flags, grasp_hold_frames, approach_end)
        if approach_end is not None else None
    )
    grasp_end = None if grasp_frame is None else grasp_frame + grasp_hold_frames
    lift_threshold_frame = (
        _first_true(lift_flags, grasp_end) if grasp_end is not None else None
    )
    lift_frame = (
        _first_run(held_lift_flags, lift_hold_frames, grasp_end)
        if grasp_end is not None else None
    )
    lift_hold = (
        _longest_run(held_lift_flags, grasp_end)
        if grasp_end is not None else 0
    )
    first_contact_frame = _first_true(
        [count > 0 for count in contact_counts]
    )
    task_success = lift_frame is not None
    predicted_action_array = np.asarray(predicted_actions)
    sent_action_array = np.asarray(sent_actions)
    action_clip = np.abs(predicted_action_array - sent_action_array)
    reach_error_array = np.asarray(reach_errors, dtype=float)
    orientation_error_array = np.asarray(orientation_errors, dtype=float)
    reach_dwell_gate_m = 0.012
    inside_reach_dwell_gate = reach_error_array <= reach_dwell_gate_m
    inside_full_reach_gate = (
        inside_reach_dwell_gate
        & (orientation_error_array <= orientation_tolerance)
    )
    inside_frames = np.flatnonzero(inside_reach_dwell_gate)
    inside_full_frames = np.flatnonzero(inside_full_reach_gate)
    closest_approach_frame = int(np.argmin(reach_error_array))
    closest_window_start = max(0, closest_approach_frame - 10)
    closest_window_end = min(steps, closest_approach_frame + 11)
    arm_target_deltas = np.diff(predicted_action_array[:, :7], axis=0)
    consecutive_inside_transitions = (
        inside_reach_dwell_gate[:-1] & inside_reach_dwell_gate[1:]
    )
    closest_window_deltas = arm_target_deltas[
        closest_window_start:max(closest_window_start, closest_window_end - 1)
    ]
    cube_displacement_from_reset = np.linalg.norm(
        cube_array - np.asarray(task.initial_cube_position, dtype=float), axis=1
    )

    metrics = {
        "seed": seed,
        "frames": steps,
        "execution_horizon": execution_horizon,
        "policy_replans": int(sum(policy_replan_flags)),
        "policy_replan_frames": [
            index for index, value in enumerate(policy_replan_flags) if value
        ],
        "reobserve_repredict_every_frame": bool(
            execution_horizon == 1 and all(policy_replan_flags)
        ),
        "reach_frame": reach_frame,
        "approach_frame": approach_frame,
        "grasp_frame": grasp_frame,
        "lift_frame": lift_frame,
        "lift_threshold_frame": lift_threshold_frame,
        "milestone_semantics": {
            "reach_hold_frames": reach_hold_frames,
            "approach_hold_frames": approach_hold_frames,
            "approach_cube_displacement_limit_m": max_approach_cube_displacement,
            "grasp_contact_groups": grasp_contact_groups,
            "grasp_hold_frames": grasp_hold_frames,
            "lift_contact_groups": lift_contact_groups,
            "lift_height_m": lift_height,
            "lift_hold_frames": lift_hold_frames,
            "ordered_and_sustained": True,
        },
        "ordered_reach_approach_grasp_lift": bool(task_success),
        "ordered_lift_with_contacts_success": bool(task_success),
        "success_metric_note": (
            "ordered sustained Reach/Approach/Grasp followed by lift height "
            "and contact retention; SE(3) grasp stability is not evaluated"
        ),
        "grasp_stability_evaluated": False,
        "reach_success": reach_frame is not None,
        "approach_success": approach_frame is not None,
        "multi_finger_grasp": grasp_frame is not None,
        "task_success": bool(task_success),
        "first_contact_frame": first_contact_frame,
        "early_contact_before_valid_approach": bool(
            first_contact_frame is not None
            and (approach_end is None or first_contact_frame < approach_end)
        ),
        "max_simultaneous_contacts": int(max(contact_counts, default=0)),
        "longest_two_contact_frames": _longest_run(multi_contact_flags),
        "longest_lift_height_hold_frames": lift_hold,
        "longest_lift_with_contacts_frames": lift_hold,
        "minimum_pregrasp_error_m": float(min(reach_errors)),
        "closest_approach_frame": closest_approach_frame,
        "reach_dwell_gate_m": reach_dwell_gate_m,
        "reach_dwell_gate_semantics": "position error only; orientation is not included",
        "frames_inside_12mm_gate": int(len(inside_frames)),
        "longest_consecutive_frames_inside_12mm": _longest_run(
            inside_reach_dwell_gate.tolist()
        ),
        "first_frame_enter_12mm": (
            int(inside_frames[0]) if len(inside_frames) else None
        ),
        "last_frame_inside_12mm": (
            int(inside_frames[-1]) if len(inside_frames) else None
        ),
        "frames_inside_12mm_and_orientation_gate": int(len(inside_full_frames)),
        "longest_consecutive_frames_inside_12mm_and_orientation_gate": _longest_run(
            inside_full_reach_gate.tolist()
        ),
        "orientation_tolerance_deg": orientation_tolerance,
        "orientation_error_at_closest_approach_deg": float(
            orientation_error_array[closest_approach_frame]
        ),
        "pregrasp_error_trajectory_around_closest": [
            {"frame": int(frame), "error_m": float(reach_error_array[frame])}
            for frame in range(closest_window_start, closest_window_end)
        ],
        "arm_target_delta_trajectory_around_closest": [
            {
                "from_frame": int(frame),
                "to_frame": int(frame + 1),
                "l2_delta_rad": float(np.linalg.norm(arm_target_deltas[frame])),
            }
            for frame in range(closest_window_start, closest_window_end - 1)
        ],
        "arm_target_motion": {
            "source": "ACT predicted absolute target, OpenArm dimensions 0:7",
            "overall": _target_motion_stats(arm_target_deltas),
            "while_consecutively_inside_12mm": _target_motion_stats(
                arm_target_deltas[consecutive_inside_transitions]
            ),
            "around_closest_approach": {
                "frame_start_inclusive": closest_window_start,
                "frame_end_exclusive": closest_window_end,
                **_target_motion_stats(closest_window_deltas),
            },
        },
        "minimum_approach_error_m": float(min(approach_errors)),
        "maximum_cube_lift_m": float(max(heights)),
        "final_cube_lift_m": float(heights[-1]),
        # Keep the legacy key for old summary readers, but make its final-only
        # semantics explicit and also retain the transient maximum.
        "cube_displacement_m": float(cube_displacement_from_reset[-1]),
        "final_cube_displacement_m": float(cube_displacement_from_reset[-1]),
        "maximum_cube_displacement_m": float(cube_displacement_from_reset.max()),
        "max_abs_action_clip_rad": float(action_clip.max()),
        "clipped_action_values": int(np.count_nonzero(action_clip > 1e-9)),
        "contact_fingers_seen": sorted({
            finger for fingers in contact_fingers for finger in fingers
        }),
    }
    rollout_arrays = {
        "observation_state": np.asarray(states, dtype=np.float32),
        "predicted_action": np.asarray(predicted_actions, dtype=np.float32),
        "sent_action": np.asarray(sent_actions, dtype=np.float32),
        "cube_position_m": cube_array,
        "grasp_center_position_m": np.asarray(grasp_positions),
        "cube_height_m": np.asarray(heights),
        "reach_error_m": np.asarray(reach_errors),
        "approach_error_m": np.asarray(approach_errors),
        "palm_orientation_error_deg": np.asarray(orientation_errors),
        "contact_count": np.asarray(contact_counts),
        "sim_time": np.asarray(sim_times),
        "policy_replan": np.asarray(policy_replan_flags),
        "reach_inside_12mm_gate": inside_reach_dwell_gate,
        "reach_inside_full_gate": inside_full_reach_gate,
        "arm_target_delta_rad": arm_target_deltas.astype(np.float32),
    }
    if save_images:
        rollout_arrays["observation.images.front"] = np.asarray(
            front_frames, dtype=np.uint8
        )
        rollout_arrays["observation.images.wrist"] = np.asarray(
            wrist_frames, dtype=np.uint8
        )
    np.savez_compressed(
        output_dir / f"rollout_seed_{seed:06d}.npz", **rollout_arrays
    )
    return metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "outputs/act_e2e_smoke/act_train/checkpoints/000050/pretrained_model",
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
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/act_e2e_smoke/rollouts",
    )
    parser.add_argument("--seeds", default="0,7,11")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save front/wrist RGB only for explicit qualitative comparisons.",
    )
    parser.add_argument(
        "--deterministic-inference", action="store_true",
        help=(
            "Use a single PyTorch CPU thread and deterministic kernels so a "
            "closed-loop rollout can be reproduced for video capture."
        ),
    )
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument(
        "--execution-horizon", type=int, default=100,
        help=(
            "Number of cached ACT actions to execute before re-observing and "
            "predicting a fresh chunk (1 replans every control frame)."
        ),
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.execution_horizon < 1:
        raise ValueError("--execution-horizon must be positive")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds did not contain a seed")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    controller = ACTController(args.checkpoint, device="cpu")
    if args.deterministic_inference:
        np.random.seed(args.inference_seed)
        controller.torch.manual_seed(args.inference_seed)
        controller.torch.set_num_threads(1)
        controller.torch.use_deterministic_algorithms(True)
        controller.torch.backends.mkldnn.deterministic = True
        # MKLDNN can still introduce run-to-run floating-point differences on
        # Windows CPU even with deterministic kernels requested.  Disabling it
        # keeps a fixed RGB observation -> ACT action mapping bit-reproducible.
        controller.torch.backends.mkldnn.enabled = False
    checkpoint_horizon = int(controller.policy.config.n_action_steps)
    if args.execution_horizon > checkpoint_horizon:
        raise ValueError(
            "--execution-horizon cannot exceed the checkpoint's "
            f"n_action_steps ({checkpoint_horizon})"
        )
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
    try:
        episodes = []
        for seed in seeds:
            result = run_episode(
                robot=robot,
                controller=controller,
                config=config,
                seed=seed,
                steps=args.steps,
                execution_horizon=args.execution_horizon,
                output_dir=args.output,
                save_images=args.save_images,
            )
            episodes.append(result)
            print(json.dumps(result), flush=True)
    finally:
        robot.disconnect()

    summary = {
        "checkpoint": str(args.checkpoint),
        "policy": "ACT",
        "state_dim": 27,
        "action_dim": 27,
        "camera_shape_hwc": [240, 320, 3],
        "fps": 30,
        "execution_horizon": args.execution_horizon,
        "checkpoint_n_action_steps": checkpoint_horizon,
        "expert_action_supplied_to_policy": False,
        "closed_loop_observation_only": True,
        "images_saved_in_rollout_npz": bool(args.save_images),
        "deterministic_inference": bool(args.deterministic_inference),
        "inference_seed": int(args.inference_seed),
        "torch_num_threads": int(controller.torch.get_num_threads()),
        "seeds": seeds,
        "episodes": episodes,
        "successful_rollouts": sum(item["task_success"] for item in episodes),
        "success_rate": sum(item["task_success"] for item in episodes) / len(episodes),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
