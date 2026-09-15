"""Diagnose ACT compounding error with periodic expert-state resets.

This is deliberately an evaluation-only tool.  It restores the recorded arm,
hand, and cube pose, then executes the first action from a freshly predicted
ACT chunk.  No training data, controller limits, or task logic are changed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from openarm_wuji.policy import ACTController
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


def _state(observation: dict) -> np.ndarray:
    return np.concatenate([
        observation["arm_joint_position"],
        observation["hand_joint_position"],
    ]).astype(np.float64)


def _restore_expert_state(
    robot: MujocoOpenArmWuji,
    task: ReachGraspLiftTask,
    *,
    state_t: np.ndarray,
    state_tp1: np.ndarray,
    cube_pose_t: np.ndarray,
    cube_pose_tp1: np.ndarray,
    held_target: np.ndarray,
    sim_time: float,
    dt: float,
) -> None:
    """Restore observed qpos and a finite-difference estimate of qvel."""
    import mujoco

    qpos_t = robot.data.qpos.copy()
    qpos_tp1 = qpos_t.copy()
    qpos_t[robot.arm_qpos_ids] = state_t[:7]
    qpos_t[robot.hand_qpos_ids] = state_t[7:]
    qpos_tp1[robot.arm_qpos_ids] = state_tp1[:7]
    qpos_tp1[robot.hand_qpos_ids] = state_tp1[7:]
    cube_qpos = int(robot.model.jnt_qposadr[task.cube_joint_id])
    qpos_t[cube_qpos:cube_qpos + 7] = cube_pose_t
    qpos_tp1[cube_qpos:cube_qpos + 7] = cube_pose_tp1

    qvel = np.zeros(robot.model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(robot.model, qvel, dt, qpos_t, qpos_tp1)
    robot.data.qpos[:] = qpos_t
    robot.data.qvel[:] = qvel
    robot.data.time = float(sim_time)
    actuator_ids = np.concatenate([
        robot.arm_actuator_ids, robot.hand_actuator_ids,
    ])
    lower = robot.model.actuator_ctrlrange[actuator_ids, 0]
    upper = robot.model.actuator_ctrlrange[actuator_ids, 1]
    robot.data.ctrl[actuator_ids] = np.clip(held_target, lower, upper)
    mujoco.mj_forward(robot.model, robot.data)


def _group_by_age(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frames_since_reset"])].append(row)
    result = {}
    for age, values in sorted(grouped.items()):
        result[str(age)] = {
            "samples": len(values),
            "pre_state_mae_rad": float(np.mean([
                item["pre_state_mae_rad"] for item in values
            ])),
            "action_mae_rad": float(np.mean([
                item["action_mae_rad"] for item in values
            ])),
            "post_state_mae_rad": float(np.mean([
                item["post_state_mae_rad"] for item in values
            ])),
        }
    return result


def run_interval(
    *,
    robot: MujocoOpenArmWuji,
    controller: ACTController,
    config: dict,
    episode_path: Path,
    reset_interval: int,
    output_dir: Path,
) -> dict:
    with np.load(episode_path, allow_pickle=False) as episode:
        seed = int(episode["episode_seed"])
        expert_state = episode["observation.state"].astype(np.float64)
        expert_next_state = episode["next_observation.state"].astype(np.float64)
        expert_action = episode["action"].astype(np.float64)
        cube_pose = episode["telemetry.cube_pose_world"].astype(np.float64)
        front = episode["observation.images.front"]
        wrist = episode["observation.images.wrist"]
        sim_time = episode["sim_time"].astype(np.float64)
        phases = episode["phase"].astype(str)
        dt = 1.0 / float(episode["control_hz"])

    task = ReachGraspLiftTask(robot, config)
    task.reset(seed)
    controller.reset()
    rows: list[dict] = []
    predicted_actions = []
    sent_actions = []
    observed_states = []
    post_states = []
    image_mae_at_reset = []

    for frame in range(len(expert_action)):
        reset_now = frame % reset_interval == 0
        if reset_now:
            next_cube = cube_pose[min(frame + 1, len(cube_pose) - 1)]
            held = expert_action[frame - 1] if frame else expert_state[frame]
            _restore_expert_state(
                robot,
                task,
                state_t=expert_state[frame],
                state_tp1=expert_next_state[frame],
                cube_pose_t=cube_pose[frame],
                cube_pose_tp1=next_cube,
                held_target=held,
                sim_time=float(sim_time[frame]),
                dt=dt,
            )

        observation = robot.get_observation()
        actual_state = _state(observation)
        if reset_now:
            image_mae_at_reset.append({
                "frame": frame,
                "front_mae_uint8": float(np.mean(np.abs(
                    observation["front_rgb"].astype(np.int16)
                    - front[frame].astype(np.int16)
                ))),
                "wrist_mae_uint8": float(np.mean(np.abs(
                    observation["wrist_rgb"].astype(np.int16)
                    - wrist[frame].astype(np.int16)
                ))),
            })

        # H_exec=1: every prediction consumes a newly observed frame.
        controller.reset()
        predicted = controller.predict(
            state=actual_state.astype(np.float32),
            front_rgb=observation["front_rgb"],
            wrist_rgb=observation["wrist_rgb"],
        )
        sent = robot.send_controller_joint_target(predicted)
        post_state = _state(robot.latest_record)
        action_error = np.abs(predicted - expert_action[frame])
        row = {
            "frame": frame,
            "phase": str(phases[frame]),
            "reset": reset_now,
            "frames_since_reset": frame % reset_interval,
            "pre_state_mae_rad": float(np.mean(np.abs(
                actual_state - expert_state[frame]
            ))),
            "pre_arm_state_mae_rad": float(np.mean(np.abs(
                actual_state[:7] - expert_state[frame, :7]
            ))),
            "pre_hand_state_mae_rad": float(np.mean(np.abs(
                actual_state[7:] - expert_state[frame, 7:]
            ))),
            "action_mae_rad": float(np.mean(action_error)),
            "arm_action_mae_rad": float(np.mean(action_error[:7])),
            "hand_action_mae_rad": float(np.mean(action_error[7:])),
            "post_state_mae_rad": float(np.mean(np.abs(
                post_state - expert_next_state[frame]
            ))),
            "action_clip_max_rad": float(np.max(np.abs(predicted - sent))),
        }
        rows.append(row)
        predicted_actions.append(predicted)
        sent_actions.append(sent)
        observed_states.append(actual_state)
        post_states.append(post_state)

    reset_rows = [row for row in rows if row["reset"]]
    summary = {
        "seed": seed,
        "frames": len(rows),
        "reset_interval": reset_interval,
        "execution_horizon": 1,
        "expert_resets": len(reset_rows),
        "mean_pre_state_mae_rad": float(np.mean([
            row["pre_state_mae_rad"] for row in rows
        ])),
        "mean_action_mae_rad": float(np.mean([
            row["action_mae_rad"] for row in rows
        ])),
        "mean_arm_action_mae_rad": float(np.mean([
            row["arm_action_mae_rad"] for row in rows
        ])),
        "mean_hand_action_mae_rad": float(np.mean([
            row["hand_action_mae_rad"] for row in rows
        ])),
        "mean_post_state_mae_rad": float(np.mean([
            row["post_state_mae_rad"] for row in rows
        ])),
        "max_action_clip_rad": float(max(
            row["action_clip_max_rad"] for row in rows
        )),
        "reset_fidelity": {
            "state_mae_rad": float(np.mean([
                row["pre_state_mae_rad"] for row in reset_rows
            ])),
            "front_mae_uint8": float(np.mean([
                row["front_mae_uint8"] for row in image_mae_at_reset
            ])),
            "wrist_mae_uint8": float(np.mean([
                row["wrist_mae_uint8"] for row in image_mae_at_reset
            ])),
        },
        "error_by_frames_since_reset": _group_by_age(rows),
    }
    np.savez_compressed(
        output_dir / f"teacher_forced_seed_{seed:06d}_N{reset_interval:03d}.npz",
        observed_state=np.asarray(observed_states),
        post_state=np.asarray(post_states),
        expert_state=expert_state,
        expert_next_state=expert_next_state,
        predicted_action=np.asarray(predicted_actions),
        sent_action=np.asarray(sent_actions),
        expert_action=expert_action,
        phase=np.asarray(phases),
        reset=np.asarray([row["reset"] for row in rows]),
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "outputs/act_e2e_smoke/act_train/checkpoints/000500/pretrained_model",
    )
    parser.add_argument(
        "--episode", type=Path,
        default=root / (
            "outputs/act_e2e_smoke/coordinated_demos/raw/successful/"
            "episode_000000_seed_000000.npz"
        ),
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
    parser.add_argument("--reset-intervals", default="1,5,10,20")
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/act_e2e_smoke/diagnosis/teacher_forced",
    )
    args = parser.parse_args()
    intervals = [
        int(value.strip()) for value in args.reset_intervals.split(",")
        if value.strip()
    ]
    if not intervals or any(value < 1 for value in intervals):
        raise ValueError("--reset-intervals must contain positive integers")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    controller = ACTController(args.checkpoint, device="cpu")
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
        results = [
            run_interval(
                robot=robot,
                controller=controller,
                config=config,
                episode_path=args.episode,
                reset_interval=interval,
                output_dir=args.output,
            )
            for interval in intervals
        ]
    finally:
        robot.disconnect()
    output = {
        "checkpoint": str(args.checkpoint),
        "episode": str(args.episode),
        "diagnostic": (
            "H_exec=1 with periodic kinematic expert arm/hand/cube-pose resets; "
            "qvel estimated by MuJoCo finite differencing"
        ),
        "results": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
