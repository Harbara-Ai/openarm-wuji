from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np


def euler_xyz_to_quat(rpy):
    rx, ry, rz = rpy
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    return [cx * cy * cz + sx * sy * sz, sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz]


def object_names(model, kind, count):
    return [mujoco.mj_id2name(model, kind, i) or f"unnamed_{i}" for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    arm_path = project / cfg["arm_model"]
    hand_path = project / cfg["hand_model"]
    arm = mujoco.MjSpec.from_file(str(arm_path))
    hand = mujoco.MjSpec.from_file(str(hand_path))

    if cfg["remove_stock_gripper"]:
        for name in ("openarm_left_ee_inner_finger", "openarm_left_ee_outer_finger"):
            element = arm.body(name)
            if element is None:
                raise KeyError(f"stock gripper body missing: {name}")
            arm.delete(element)
        for element in (
            arm.actuator("left_finger1_ctrl"),
            arm.equality("openarm_left_ee_finger_joint_mimic"),
        ):
            if element is not None:
                arm.delete(element)

    site = arm.site(cfg["mount"]["parent_site"])
    if site is None:
        raise KeyError(f"mount site missing: {cfg['mount']['parent_site']}")
    frame = arm.attach(hand, prefix="wuji_", site=site)
    frame.pos = cfg["mount"]["xyz_m"]
    frame.quat = euler_xyz_to_quat(cfg["mount"]["rpy_rad"])
    model = arm.compile()
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    for _ in range(250):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        raise RuntimeError("combined simulation produced non-finite qpos")

    args.output.mkdir(parents=True, exist_ok=True)
    # MjSpec XML keeps source-relative mesh paths from two different repositories,
    # so it is not relocatable on its own. The compiled MJB embeds the resolved
    # assets and is the reproducible runtime artifact.
    mujoco.mj_saveModel(model, str(args.output / "openarm_v2_wuji_left.mjb"), None)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = model.stat.center
    camera.distance = model.stat.extent * 1.2
    camera.azimuth = 145
    camera.elevation = -20
    renderer.update_scene(data, camera=camera)
    iio.imwrite(args.output / "combined_smoke.png", renderer.render())

    joints = object_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuators = object_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        contacts.append([
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            float(contact.dist),
        ])
    report = {
        "config": cfg,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "runtime_model": "openarm_v2_wuji_left.mjb",
        "joint_names": joints,
        "actuator_names": actuators,
        "wuji_joint_count": len([n for n in joints if n.startswith("wuji_")]),
        "finite": bool(np.isfinite(data.qpos).all()),
        "simulation_time_s": float(data.time),
        "contacts_at_settle": contacts,
    }
    (args.output / "combined_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
