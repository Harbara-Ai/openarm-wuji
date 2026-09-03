from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji


def validate_episode(path: Path, samples: int, image_shape: tuple[int, int, int]) -> None:
    with np.load(path) as episode:
        aligned = (
            "frame_index", "timestamp", "sim_time", "front_rgb", "wrist_rgb",
            "arm_joint_position", "arm_joint_velocity", "hand_joint_position",
            "hand_joint_velocity", "hand_joint_target", "hand_synergy_action", "sent_action",
        )
        if any(episode[key].shape[0] != samples for key in aligned):
            raise RuntimeError("recorded modalities are not frame-aligned")
        if (episode["front_rgb"].shape[1:] != image_shape
                or episode["wrist_rgb"].shape[1:] != image_shape):
            raise RuntimeError("recorded image shape changed")
        if episode["front_rgb"].dtype != np.uint8 or episode["wrist_rgb"].dtype != np.uint8:
            raise RuntimeError("RGB observations must use uint8")
        if not np.all(np.diff(episode["frame_index"]) == 1):
            raise RuntimeError("frame indices are not contiguous")
        if (not np.all(np.diff(episode["timestamp"]) > 0)
                or not np.all(np.diff(episode["sim_time"]) > 0)):
            raise RuntimeError("timestamps are not strictly increasing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recording = args.output / "combined_control_recording.npz"

    robot = MujocoOpenArmWuji(
        args.model, args.synergies, control_hz=30, image_height=240, image_width=320
    )
    robot.connect()
    initial = robot.get_observation()
    arm_target = initial["arm_joint_position"].copy()
    arm_target[0] += 0.08
    phases = ((0.0, 0.0, 0.0), (0.2, 1.0, 0.2), (1.0, 0.0, -0.2))
    for phase in phases:
        for alpha in np.linspace(0.0, 1.0, 30):
            robot.send_action(np.r_[arm_target, alpha * np.asarray(phase)])
    final = robot.get_observation()
    robot.save_recording(recording)
    validate_episode(recording, samples=90, image_shape=(240, 320, 3))
    animation = np.stack(
        [np.concatenate([row["front_rgb"], row["wrist_rgb"]], axis=1) for row in robot.records]
    )
    iio.imwrite(args.output / "combined_control_demo.gif", animation, duration=1000 / 30, loop=0)
    robot.disconnect()

    replay = MujocoOpenArmWuji(
        args.model, args.synergies, control_hz=30, image_height=240, image_width=320
    )
    replay.connect()
    replay_report = replay.replay_recording(recording)
    replay.disconnect()
    report = {
        "samples": 90,
        "expected_duration_s": 90 / 30,
        "simulation_duration_s": float(final["sim_time"] - initial["sim_time"]),
        "action_shape": list(final["sent_action"].shape),
        "front_rgb_shape": list(final["front_rgb"].shape),
        "wrist_rgb_shape": list(final["wrist_rgb"].shape),
        "arm_position_delta_max": float(np.max(np.abs(
            final["arm_joint_position"] - initial["arm_joint_position"]))),
        "hand_position_delta_max": float(np.max(np.abs(
            final["hand_joint_position"] - initial["hand_joint_position"]))),
        "finite": bool(all(np.isfinite(v).all() for v in final.values()
                           if isinstance(v, np.ndarray))),
        "replay": replay_report,
    }
    if (report["arm_position_delta_max"] <= 1e-4
            or report["hand_position_delta_max"] <= 1e-4
            or not report["finite"]
            or abs(report["simulation_duration_s"] - report["expected_duration_s"]) >= 0.0021
            or replay_report["max_arm_position_error"] >= 1e-4
            or replay_report["max_hand_position_error"] >= 1e-4):
        raise RuntimeError(f"combined control smoke failed: {report}")
    (args.output / "combined_control_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
