from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    robot = MujocoOpenArmWuji(args.model, args.synergies, control_hz=30)
    robot.connect()
    initial = robot.get_observation()
    arm_target = initial["arm_joint_position"].copy()
    arm_target[0] += 0.08
    for alpha in np.linspace(0.0, 1.0, 30):
        robot.send_action(np.r_[arm_target, alpha, 0.5 * alpha, 0.2 * alpha])
    final = robot.get_observation()
    args.output.mkdir(parents=True, exist_ok=True)
    robot.save_recording(args.output / "combined_control_recording.npz")
    report = {"samples": len(robot.records), "action_shape": list(final["sent_action"].shape),
              "arm_position_delta_max": float(np.max(np.abs(final["arm_joint_position"] - initial["arm_joint_position"]))),
              "hand_position_delta_max": float(np.max(np.abs(final["hand_joint_position"] - initial["hand_joint_position"]))),
              "final_synergy": final["hand_synergy_action"].tolist(),
              "finite": bool(all(np.isfinite(v).all() for v in final.values() if isinstance(v, np.ndarray)))}
    if report["arm_position_delta_max"] <= 1e-4 or report["hand_position_delta_max"] <= 1e-4 or not report["finite"]:
        raise RuntimeError(f"combined control smoke failed: {report}")
    (args.output / "combined_control_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    robot.disconnect()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
