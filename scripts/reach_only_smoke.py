from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
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
        task = ReachGraspLiftTask(robot, config)
        result = task.run_reach(args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        report_path = args.output / "reach_report.json"
        report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        frames = [record["front_rgb"] for record in robot.records]
        if frames:
            imageio.mimsave(args.output / "reach_demo.gif", frames[::2], duration=0.066, loop=0)
        print(json.dumps(result.to_dict(), indent=2))
        if not result.success:
            raise SystemExit(1)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
