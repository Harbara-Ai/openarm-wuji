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


def require_element(element, description: str):
    if element is None:
        raise KeyError(f"model element missing: {description}")
    return element


def set_home_state(model, data, home):
    for side in ("left", "right"):
        for index, value in enumerate(home, start=1):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{index}"
            )
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_joint{index}_ctrl"
            )
            if joint_id >= 0:
                data.qpos[model.jnt_qposadr[joint_id]] = value
            if actuator_id >= 0:
                data.ctrl[actuator_id] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    task = json.loads(args.config.read_text(encoding="utf-8"))
    robot = json.loads((project / task["robot_config"]).read_text(encoding="utf-8"))

    arm = mujoco.MjSpec.from_file(str(project / task["arm_model"]))
    hand = mujoco.MjSpec.from_file(str(project / robot["hand_model"]))
    if robot["remove_stock_gripper"]:
        for name in ("openarm_left_ee_inner_finger", "openarm_left_ee_outer_finger"):
            arm.delete(require_element(arm.body(name), name))
        for element in (
            arm.actuator("left_finger1_ctrl"),
            arm.equality("openarm_left_ee_finger_joint_mimic"),
        ):
            if element is not None:
                arm.delete(element)

    mount_site = require_element(arm.site(robot["mount"]["parent_site"]), "mount site")
    frame = arm.attach(hand, prefix="wuji_", site=mount_site)
    frame.pos = robot["mount"]["xyz_m"]
    frame.quat = euler_xyz_to_quat(
        task.get("task_mount_rpy_rad", robot["mount"]["rpy_rad"])
    )

    scene = task["scene"]
    palm = require_element(arm.body("wuji_left_palm_link"), "Wuji palm")
    palm.add_site(
        name=scene["grasp_site_name"],
        pos=scene["grasp_site_offset_m"],
        size=[0.008],
        rgba=[0.1, 0.9, 0.2, 1.0],
        group=4,
    )
    world = arm.worldbody
    world.add_light(name="task_key_light", pos=[0, 0, 2], dir=[0, 0, -1])
    world.add_geom(
        name="task_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, 0], size=[0, 0, 0.05], rgba=[0.25, 0.28, 0.32, 1],
    )
    table = world.add_body(name="task_table", pos=scene["table_center_m"])
    table.add_geom(
        name="task_table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=scene["table_half_size_m"], rgba=[0.82, 0.71, 0.55, 1],
        friction=[1.0, 0.005, 0.0001],
    )
    cube = world.add_body(name=scene["cube_name"], pos=task["reset"]["cube_nominal_position_m"])
    cube.add_freejoint(name=scene["cube_joint_name"])
    cube.add_geom(
        name="task_cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=scene["cube_half_size_m"], rgba=[0.9, 0.2, 0.1, 1],
        mass=scene["cube_mass_kg"], friction=[1.0, 0.01, 0.001],
    )
    world.add_camera(
        name=scene["front_camera_name"], pos=scene["front_camera_position_m"],
        xyaxes=scene["front_camera_xyaxes"], fovy=55,
    )

    model = arm.compile()
    data = mujoco.MjData(model)
    set_home_state(model, data, task["reset"]["home_arm_joint_position_rad"])
    mujoco.mj_forward(model, data)
    for _ in range(task["reset"]["settle_physics_steps"]):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        raise RuntimeError("task simulation produced non-finite qpos")

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "reach_grasp_lift.mjb"
    mujoco.mj_saveModel(model, str(model_path), None)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=scene["front_camera_name"])
    iio.imwrite(args.output / "scene_smoke.png", renderer.render())
    renderer.close()

    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, scene["grasp_site_name"]
    )
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene["cube_name"])
    report = {
        "schema_version": task["schema_version"],
        "runtime_model": model_path.name,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "grasp_center_home_m": data.site_xpos[site_id].tolist(),
        "cube_settled_m": data.xpos[cube_id].tolist(),
        "finite": bool(np.isfinite(data.qpos).all()),
    }
    (args.output / "model_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
