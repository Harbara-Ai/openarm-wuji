from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_wuji.dataset import CausalEpisodeRecorder, replay_causal_episode
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


def make_robot(*, model: Path, synergies: Path, config: dict,
               image_height: int, image_width: int) -> MujocoOpenArmWuji:
    return MujocoOpenArmWuji(
        model,
        synergies,
        arm_side=config["arm_side"],
        control_hz=30,
        image_height=image_height,
        image_width=image_width,
        front_camera=config["scene"]["front_camera_name"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expected-outcome")
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    recorder = CausalEpisodeRecorder(
        task_name="reach_grasp_lift",
        control_hz=30,
        task_description="Pick up the cube and lift it at least 80 mm.",
        episode_index=0,
    )
    robot = make_robot(
        model=args.model,
        synergies=args.synergies,
        config=config,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    robot.connect()
    try:
        task = ReachGraspLiftTask(robot, config, recorder=recorder)
        result = task.run_lift(args.seed)
        recorder.finish(result.to_dict())
        episode_path = args.output / f"reach_grasp_lift_seed{args.seed:04d}.npz"
        recorder.save(episode_path)
        validation = CausalEpisodeRecorder.validate(episode_path)
    finally:
        robot.disconnect()

    replay_robot = make_robot(
        model=args.model,
        synergies=args.synergies,
        config=config,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    replay_robot.connect()
    try:
        replay_task = ReachGraspLiftTask(replay_robot, config)
        replay = replay_causal_episode(
            episode_path,
            robot=replay_robot,
            reset_fn=replay_task.reset,
            telemetry_fn=replay_task.task_telemetry,
        )
    finally:
        replay_robot.disconnect()

    state_replay_is_exact = all(
        replay[key] == 0
        for key in (
            "max_action_error",
            "max_state_error",
            "max_cube_position_error",
            "max_cube_rotation_error_deg",
            "max_grasp_center_position_error",
            "max_grasp_center_rotation_error_deg",
            "max_relative_position_error",
            "max_relative_rotation_error_deg",
            "max_contact_resultant_force_error_n",
            "max_contact_resultant_moment_error_nm",
            "max_contact_position_error_m",
            "max_contact_normal_error",
            "max_contact_force_error_n",
            "max_contact_moment_error_nm",
            "contact_mismatch_frames",
            "max_sim_time_error",
        )
    )
    image_replay_is_stable = (
        replay["max_front_pixel_error"] <= 2
        and replay["max_wrist_pixel_error"] <= 2
    )
    outcome_matches = (
        result.outcome == args.expected_outcome
        if args.expected_outcome is not None
        else result.task_success
    )
    report = {
        "episode_file": episode_path.name,
        "evaluation_summary": {
            "task_success": result.task_success,
            "grasp_stable": result.grasp_stable,
            "outcome": result.outcome,
            "expected_outcome": args.expected_outcome,
            "outcome_matches": outcome_matches,
            "max_relative_translation_drift_m": (
                result.max_relative_translation_drift_m
            ),
            "max_relative_rotation_drift_deg": (
                result.max_relative_rotation_drift_deg
            ),
            "establishment_translation_m": result.establishment_translation_m,
            "establishment_rotation_deg": result.establishment_rotation_deg,
        },
        "result": result.to_dict(),
        "validation": validation,
        "replay": replay,
        "checks": {
            "task_success": result.task_success,
            "grasp_stable": result.grasp_stable,
            "outcome": result.outcome,
            "state_replay_exact": state_replay_is_exact,
            "image_max_abs_error_threshold": 2,
            "image_replay_stable": image_replay_is_stable,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "episode_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not outcome_matches or not state_replay_is_exact or not image_replay_is_stable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
