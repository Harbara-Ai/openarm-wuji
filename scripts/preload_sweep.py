"""Reproduce the seed-7 preload selection; this is tuning, not a seed benchmark."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--targets", type=float, nargs="+",
                        default=[0.74, 0.76, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93])
    parser.add_argument("--lift-compensation-m", type=float, default=0.02977)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "configs/reach_grasp_lift.json").read_text("utf-8"))
    base["lift"]["gravity_compensation_offset_m"] = [0, 0, args.lift_compensation_m]
    rows = []
    for target in args.targets:
        config = copy.deepcopy(base)
        config["grasp"]["preload_target_synergy"] = target
        robot = MujocoOpenArmWuji(
            root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb",
            root / "configs/wuji_hand_left_synergies.json",
            control_hz=30, image_height=48, image_width=64,
            front_camera=config["scene"]["front_camera_name"],
        )
        robot.connect()
        try:
            result = ReachGraspLiftTask(robot, config).run_lift(args.seed)
            row = {key: getattr(result, key) for key in (
                "task_success", "grasp_stable", "outcome", "peak_height_m",
                "final_height_m", "max_relative_translation_drift_m",
                "max_relative_rotation_drift_deg", "gripper_lift_within_baseline_window_m",
            )}
            row.update(target_synergy=target, grasp=result.grasp.to_dict(),
                       peak_force_n=result.contact_diagnostics["peak_resultant_force_magnitude_n"],
                       peak_moment_nm=result.contact_diagnostics["peak_resultant_moment_magnitude_nm"])
            rows.append(row)
            print(f"synergy={target:.2f}: {result.outcome}", flush=True)
        finally:
            robot.disconnect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "purpose": "single_seed_parameter_selection_not_success_rate",
        "seed": args.seed, "control_hz": 30, "base_config": base,
        "rows": rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
