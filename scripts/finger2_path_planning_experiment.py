from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from contact_acquisition_experiment import HeadlessRobot
from contact_grasp_experiment import hand_seed_library, plain
from openarm_wuji.tasks import ReachGraspLiftTask
from openarm_wuji.tasks.contact_grasp import ContactRegionGraspOptimizer
from openarm_wuji.tasks.finger_path_planner import (
    FINGER2_PATH_OUTCOMES,
    Finger2PathPlanner,
    execute_finger2_waypoint_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Finite seed-7 finger2 +X contact-region path search"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--optimization-config", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-dynamics", action="store_true")
    args = parser.parse_args()

    task_config = json.loads(args.task_config.read_text(encoding="utf-8"))
    config = json.loads(args.optimization_config.read_text(encoding="utf-8"))
    synergy_config = json.loads(args.synergies.read_text(encoding="utf-8"))
    open_pose = np.asarray(synergy_config["open_pose"], dtype=float)
    topology = next(
        item for item in config["topologies"]
        if item["name"] == "thumb_minus_x_fingers_plus_x"
    )
    initialization = next(
        item for item in topology["initializations"]
        if item["name"] == "current_grasp"
    )

    robot = HeadlessRobot(
        args.model, args.synergies, arm_side=task_config["arm_side"],
        control_hz=30,
    )
    robot.connect()
    try:
        task = ReachGraspLiftTask(robot, task_config)
        grasp = task.run_grasp(args.seed)
        if not grasp.success:
            raise RuntimeError(
                f"seed {args.seed} cannot initialize finger2 planning: "
                f"{grasp.failure_reason}"
            )
        optimizer = ContactRegionGraspOptimizer(
            robot.model, robot.data, config, topology=topology,
        )
        seeds = hand_seed_library(
            optimizer.nominal[len(optimizer.arm_joint_ids):], synergy_config
        )
        initial_q, initial_diag = optimizer.make_initial_q(
            initialization,
            hand_seed=seeds[initialization["hand_seed"]],
        )
        base_solution = optimizer.solve(
            initial_q=initial_q,
            initialization_name=initialization["name"],
            initialization_diagnostics=initial_diag,
        )
        if not base_solution.success:
            raise RuntimeError("the Stage-1 endpoint candidate is no longer valid")

        planner = Finger2PathPlanner(optimizer, config, open_hand=open_pose)
        candidates = planner.plan(base_solution.qpos, seed=args.seed)
        feasible = sorted(
            (item for item in candidates if item.path_feasible),
            key=lambda item: item.overall_non_tip_min_clearance_m,
            reverse=True,
        )
        kinematic_best = max(
            feasible,
            key=lambda item: item.overall_non_tip_min_clearance_m,
            default=None,
        )
        dynamic_validations = []
        if feasible and not args.no_dynamics:
            initial_state = {
                "qpos": robot.data.qpos.copy(),
                "qvel": robot.data.qvel.copy(),
                "ctrl": robot.data.ctrl.copy(),
                "time": float(robot.data.time),
            }
            for candidate in feasible:
                robot.data.qpos[:] = initial_state["qpos"]
                robot.data.qvel[:] = initial_state["qvel"]
                robot.data.ctrl[:] = initial_state["ctrl"]
                robot.data.time = initial_state["time"]
                mujoco.mj_forward(robot.model, robot.data)
                result = execute_finger2_waypoint_path(
                    robot, task, candidate, optimizer, config
                )
                dynamic_validations.append({
                    "candidate": candidate.name,
                    **result,
                })
        passing_names = {
            item["candidate"] for item in dynamic_validations
            if item["single_finger_path_pass"]
        }
        best = next(
            (item for item in feasible if item.name in passing_names),
            kinematic_best,
        )
        dynamic = next(
            (item for item in dynamic_validations
             if best is not None and item["candidate"] == best.name),
            None,
        )
        if args.no_dynamics and best is not None:
            conclusion = "path_feasible"
        elif dynamic_validations and passing_names:
            conclusion = "single_finger_path_pass"
        else:
            conclusion = "finger2_required_contact_rejected"
        report = plain({
            "schema_version": 1,
            "seed": args.seed,
            "scope": "finger2_path_feasibility_only",
            "outcomes_supported": list(FINGER2_PATH_OUTCOMES),
            "fixed": [
                "palm_pose", "cube_pose", "thumb_configuration",
                "middle_configuration", "ring_configuration",
                "little_configuration",
            ],
            "target": "finger2 designated fingertip -> cube +X face",
            "contact_region_candidates": [item.to_dict() for item in candidates],
            "best_candidate": None if best is None else best.to_dict(),
            "dynamic_validation": dynamic,
            "dynamic_validations": dynamic_validations,
            "conclusion": conclusion,
            "next_topology_if_rejected": (
                None if conclusion != "finger2_required_contact_rejected" else {
                    "required": ["finger1", "finger3", "finger4"],
                    "optional": ["finger2", "finger5"],
                }
            ),
            "notes": [
                "No Lift, preload, squeeze, force/wrench, RL, or ACT logic is used.",
                "Five semantic contact-region seeds are used; this is not a yaw/XY brute-force scan.",
                "All path clearance values are continuous diagnostics in meters.",
            ],
        })
        for item in candidates:
            print(
                f"{item.name:12s} endpoint={item.endpoint_feasible!s:5s} "
                f"endpoint_error={1000*item.endpoint_error_m:.3f}mm "
                f"tip_face={item.endpoint_diagnostics.get('tip_nearest_face')} "
                f"endpoint_non_tip={1000*item.endpoint_diagnostics.get('overall_non_tip_min_clearance_m', float('nan')):.3f}mm "
                f"edge={1000*item.endpoint_diagnostics.get('distance_to_nearest_edge_m', float('nan')):.2f}mm "
                f"self={1000*item.endpoint_diagnostics.get('self_collision_penetration_m', float('nan')):.3f}mm "
                f"geom53={item.geom53_min_clearance_m} "
                f"geom55={item.geom55_min_clearance_m} "
                f"global={item.overall_non_tip_min_clearance_m} "
                f"outcome={item.outcome}",
                flush=True,
            )
        print(f"CONCLUSION {conclusion}", flush=True)
        for item in dynamic_validations:
            print("DYNAMIC " + json.dumps({
                "candidate": item["candidate"],
                "outcome": item["outcome"],
                "first_contact": item["first_contact"],
                "max_cube_translation_m": item["max_cube_translation_m"],
                "max_cube_rotation_deg": item["max_cube_rotation_deg"],
            }), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
