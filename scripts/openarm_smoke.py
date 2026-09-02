from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
import openarm_mujoco.v2 as openarm


def names(model, object_type, count):
    return [mujoco.mj_id2name(model, object_type, i) or f"unnamed_{i}" for i in range(count)]


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "openarm"
    output.mkdir(parents=True, exist_ok=True)
    model_path = openarm.openarm_bimanual_xml()
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    initial = data.qpos.copy()
    # Programmatic, bounded motion on the first position actuator.
    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    if model.nu:
        lo, hi = model.actuator_ctrlrange[0]
        data.ctrl[0] = np.clip(data.ctrl[0] + 0.10, lo, hi)
    for _ in range(500):
        mujoco.mj_step(model, data)
    movement = float(np.max(np.abs(data.qpos - initial)))
    if not np.isfinite(data.qpos).all() or movement <= 1e-5:
        raise RuntimeError(f"invalid simulation result: movement={movement}")
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = model.stat.center
    camera.distance = model.stat.extent * 1.25
    camera.azimuth = 145
    camera.elevation = -20
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    iio.imwrite(output / "headless_smoke_v2.png", image)
    report = {
        "model_path": model_path,
        "mujoco_version": mujoco.__version__,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "joint_names": names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "actuator_names": names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        "qpos_order": names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "max_qpos_movement": movement,
        "final_time_s": float(data.time),
        "finite": bool(np.isfinite(data.qpos).all()),
    }
    (output / "model_inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
