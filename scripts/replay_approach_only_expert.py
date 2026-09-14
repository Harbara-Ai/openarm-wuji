"""Validate the Approach-only reset and gate with recorded expert targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_staged_reach_approach import (
    TRAINING_SEEDS,
    _expert_starts,
    _initialize_from_expert,
)
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb")
    parser.add_argument("--config", type=Path, default=root / "configs/reach_grasp_lift.json")
    parser.add_argument("--synergies", type=Path, default=root / "configs/wuji_hand_left_synergies.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    starts = _expert_starts(args.raw)
    by_seed = {}
    for path in args.raw.glob("episode_*.npz"):
        with np.load(path, allow_pickle=False) as episode:
            by_seed[int(episode["episode_seed"])] = path

    robot = MujocoOpenArmWuji(
        args.model, args.synergies,
        arm_side=config["arm_side"], control_hz=30,
        image_height=240, image_width=320,
        front_camera=config["scene"]["front_camera_name"],
    )
    robot.connect()
    results = []
    try:
        for seed in TRAINING_SEEDS:
            task = ReachGraspLiftTask(robot, config)
            task.reset(seed)
            initial_error = _initialize_from_expert(task, robot, starts[seed])
            start_cube = starts[seed]["cube_pose"][:3]
            target = start_cube + np.asarray(config["grasp"]["target_offset_m"])
            with np.load(by_seed[seed], allow_pickle=False) as episode:
                actions = np.asarray(episode["action"], dtype=float)
            dwell = 0
            maximum_dwell = 0
            minimum_error = float("inf")
            maximum_cube_displacement = 0.0
            first_contact = None
            success_frame = None
            for frame, action in enumerate(actions):
                robot.send_controller_joint_target(action)
                telemetry = task.task_telemetry()
                error = float(np.linalg.norm(
                    np.asarray(telemetry["grasp_center_position_m"]) - target
                ))
                orientation = float(telemetry["palm_orientation_error_deg"])
                displacement = float(np.linalg.norm(
                    np.asarray(telemetry["cube_position_m"]) - start_cube
                ))
                if telemetry["contacts"] and first_contact is None:
                    first_contact = frame
                inside = error <= 0.012 and orientation <= 2.0 and displacement <= 0.025
                dwell = dwell + 1 if inside else 0
                maximum_dwell = max(maximum_dwell, dwell)
                minimum_error = min(minimum_error, error)
                maximum_cube_displacement = max(maximum_cube_displacement, displacement)
                if dwell >= 5:
                    success_frame = frame
                    break
            results.append({
                "seed": seed,
                "success": success_frame is not None,
                "success_frame": success_frame,
                "initial_state_max_abs_error": initial_error,
                "minimum_position_error_m": minimum_error,
                "maximum_gate_dwell": maximum_dwell,
                "maximum_cube_displacement_m": maximum_cube_displacement,
                "early_contact": first_contact is not None,
                "first_contact_frame": first_contact,
            })
    finally:
        robot.disconnect()
    summary = {
        "purpose": "Approach-only expert-target replay gate validation",
        "expert_action_used_for_validation_only": True,
        "episodes": results,
        "aggregate": {
            "success_count": sum(item["success"] for item in results),
            "minimum_error_mean_mm": float(np.mean([
                item["minimum_position_error_m"] for item in results
            ]) * 1000),
            "maximum_cube_displacement_max_mm": float(np.max([
                item["maximum_cube_displacement_m"] for item in results
            ]) * 1000),
            "early_contact_count": sum(item["early_contact"] for item in results),
            "initial_state_max_abs_error": max(
                item["initial_state_max_abs_error"] for item in results
            ),
        },
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
