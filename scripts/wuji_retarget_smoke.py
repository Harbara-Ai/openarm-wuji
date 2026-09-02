from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from wuji_retargeting import Retargeter


def names(model, kind, count):
    return [mujoco.mj_id2name(model, kind, i) or f"unnamed_{i}" for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Linux-native Wuji repository checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = args.source / "example/config/adaptive_analytical_avp.yaml"
    recording_path = args.source / "example/data/avp1.pkl"
    mjcf = args.source / f"wuji_retargeting/wuji-description/hand/body/mjcf/{args.hand}.xml"
    with recording_path.open("rb") as stream:
        recording = pickle.load(stream)
    retargeter = Retargeter.from_yaml(str(config), args.hand)
    joint_names = list(retargeter.optimizer.robot.dof_joint_names)
    limits = np.asarray(retargeter.optimizer.robot.joint_limits)

    frames, timestamps, qposes, costs = [], [], [], []
    started = time.perf_counter()
    for item in recording[: args.max_frames]:
        keypoints = np.asarray(item[f"{args.hand}_fingers"], dtype=np.float64)
        if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
            raise ValueError(f"invalid keypoints shape/value: {keypoints.shape}")
        qpos, verbose = retargeter.retarget_verbose(keypoints)
        if qpos.shape != (20,) or not np.isfinite(qpos).all():
            raise ValueError(f"invalid qpos shape/value: {qpos.shape}")
        frames.append(keypoints)
        timestamps.append(float(item["t"]))
        qposes.append(qpos)
        costs.append(float(verbose["cost"]))
    elapsed = time.perf_counter() - started
    frames = np.stack(frames)
    qposes = np.stack(qposes)
    timestamps = np.asarray(timestamps)
    max_step = float(np.abs(np.diff(qposes, axis=0)).max()) if len(qposes) > 1 else 0.0
    np.savez_compressed(
        args.output / "avp1_left_retarget.npz",
        timestamps=timestamps,
        keypoints=frames,
        qpos=qposes,
        joint_names=np.asarray(joint_names),
    )

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    actuator_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        for i in range(model.nu)
    ]
    permutation = np.asarray([joint_names.index(name) for name in actuator_joint_names])
    data.ctrl[:] = qposes[-1][permutation]
    for _ in range(200):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.05]
    camera.distance = 0.45
    camera.azimuth = 135
    camera.elevation = -20
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(args.output / "wuji_retarget_smoke.png")

    report = {
        "variant": "Wuji Hand (not Wuji Hand 2)",
        "hand": args.hand,
        "config": str(config),
        "mjcf": str(mjcf),
        "recording": str(recording_path),
        "recording_total_frames": len(recording),
        "processed_frames": len(qposes),
        "input_shape": list(frames.shape),
        "output_shape": list(qposes.shape),
        "finite": bool(np.isfinite(qposes).all()),
        "max_abs_frame_step_rad": max_step,
        "processing_seconds": elapsed,
        "mean_fps": len(qposes) / elapsed,
        "cost_min": float(np.min(costs)),
        "cost_max": float(np.max(costs)),
        "joint_names": joint_names,
        "joint_limits_rad": limits.tolist(),
        "actuator_joint_order": actuator_joint_names,
        "qpos_to_actuator_permutation": permutation.tolist(),
        "mujoco_dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu},
    }
    (args.output / "retarget_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
