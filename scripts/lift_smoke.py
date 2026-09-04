from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from openarm_wuji.dataset import CausalEpisodeRecorder
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expected-outcome")
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
        recorder = CausalEpisodeRecorder(
            task_name="reach_grasp_lift", control_hz=30
        )
        result = ReachGraspLiftTask(
            robot, config, recorder=recorder
        ).run_lift(args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "lift_report.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        frames = []
        for transition in recorder.transitions:
            observation = transition["observation_tp1"]
            telemetry = transition["telemetry_tp1"]
            frame = np.concatenate([
                observation["front_rgb"], observation["wrist_rgb"]
            ], axis=1)
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            lines = [
                f"phase={transition['phase']}  height={telemetry['cube_height_m'] * 1000:.1f} mm",
                f"task_success={result.task_success}  grasp_stable={result.grasp_stable}",
                f"outcome={result.outcome}",
            ]
            draw.rectangle((0, 0, 395, 39), fill=(0, 0, 0))
            for line_index, line in enumerate(lines):
                draw.text((5, 3 + 12 * line_index), line, fill=(255, 255, 255))
            frames.append(np.asarray(image))
        if frames:
            imageio.mimsave(
                args.output / "lift_demo.gif", frames[::2], duration=0.066, loop=0
            )
        print(json.dumps(result.to_dict(), indent=2))
        outcome_matches = (
            result.outcome == args.expected_outcome
            if args.expected_outcome is not None
            else result.task_success
        )
        if not outcome_matches:
            raise SystemExit(1)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
