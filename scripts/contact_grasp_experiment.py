from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask
from openarm_wuji.tasks.contact_grasp import (
    ContactGraspSolution,
    ContactRegionGraspOptimizer,
    execute_static_grasp_test,
)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def current_solution(optimizer, source_data, topology_name: str) -> ContactGraspSolution:
    q = source_data.qpos[optimizer.qpos_ids].copy()
    palm_quaternion = np.zeros(4)
    optimizer.mujoco.mju_mat2Quat(
        palm_quaternion, source_data.site_xmat[optimizer.palm_site_id]
    )
    return ContactGraspSolution(
        topology=f"baseline_for_{topology_name}",
        initialization="existing_grasp",
        success=True,
        iterations=0,
        initial_cost=0.0,
        final_cost=0.0,
        variable_joint_ids=tuple(optimizer.joint_ids),
        variable_joint_names=tuple(
            optimizer.mujoco.mj_id2name(
                optimizer.model,
                optimizer.mujoco.mjtObj.mjOBJ_JOINT,
                joint,
            ) or f"joint_{joint}" for joint in optimizer.joint_ids
        ),
        arm_joint_ids=tuple(optimizer.arm_joint_ids),
        hand_joint_ids=tuple(optimizer.hand_joint_ids),
        qpos=tuple(float(value) for value in q),
        palm_position_world_m=tuple(float(value) for value in
                                    source_data.site_xpos[optimizer.palm_site_id]),
        palm_quaternion_world_wxyz=tuple(float(value) for value in palm_quaternion),
        fingertip_diagnostics={},
        collision_diagnostics={},
        joint_limit_hits=(),
        required_fingers=(),
        required_fingers_satisfied=(),
        optional_fingers=(),
        initialization_diagnostics={},
        stage_history=(),
    )


def hand_seed_library(current_hand, synergy_config: dict) -> dict[str, np.ndarray]:
    open_pose = np.asarray(synergy_config["open_pose"], dtype=float)
    close_pose = np.asarray(synergy_config["close_pose"], dtype=float)
    pinch_pose = np.asarray(synergy_config["pinch_pose"], dtype=float)
    power_half = open_pose + 0.5 * (close_pose - open_pose)
    opposition = power_half.copy()
    opposition[:4] = pinch_pose[:4]
    opposition[4:16] = open_pose[4:16] + 0.55 * (
        close_pose[4:16] - open_pose[4:16]
    )
    opposition[16:20] = open_pose[16:20] + 0.35 * (
        close_pose[16:20] - open_pose[16:20]
    )
    return {
        "current": np.asarray(current_hand, dtype=float).copy(),
        "power_half": power_half,
        "opposition": opposition,
    }


def solution_rank(solution: ContactGraspSolution) -> tuple:
    required_errors = [
        solution.fingertip_diagnostics[finger]["position_error_m"]
        for finger in solution.required_fingers
    ]
    collision = solution.collision_diagnostics
    return (
        -len(solution.required_fingers_satisfied),
        collision["max_non_tip_actual_penetration_m"],
        collision["self_collision_penetration_sum_m"],
        collision["max_non_tip_clearance_violation_m"],
        max(required_errors, default=float("inf")),
        solution.final_cost,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--optimization-config", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    task_config = json.loads(args.task_config.read_text(encoding="utf-8"))
    optimization_config = json.loads(
        args.optimization_config.read_text(encoding="utf-8")
    )
    synergy_config = json.loads(args.synergies.read_text(encoding="utf-8"))
    close_direction = np.sign(
        np.asarray(synergy_config["close_pose"], dtype=float)
        - np.asarray(synergy_config["open_pose"], dtype=float)
    )
    robot = MujocoOpenArmWuji(
        args.model,
        args.synergies,
        arm_side=task_config["arm_side"],
        control_hz=30,
        image_height=32,
        image_width=32,
        front_camera=task_config["scene"]["front_camera_name"],
    )
    robot.connect()
    try:
        task = ReachGraspLiftTask(robot, task_config)
        grasp = task.run_grasp(args.seed)
        if not grasp.success:
            raise RuntimeError(
                f"seed {args.seed} cannot initialize optimization: "
                f"{grasp.failure_reason}"
            )
        attempted = []
        selected = None
        selected_optimizer = None
        for topology in optimization_config["topologies"]:
            optimizer = ContactRegionGraspOptimizer(
                robot.model, robot.data, optimization_config,
                topology=topology,
            )
            hand_seeds = hand_seed_library(
                optimizer.nominal[len(optimizer.arm_joint_ids):], synergy_config
            )
            topology_solutions = []
            for initialization in topology["initializations"]:
                initial_q, initial_diagnostics = optimizer.make_initial_q(
                    initialization,
                    hand_seed=hand_seeds[initialization["hand_seed"]],
                )
                solution = optimizer.solve(
                    initial_q=initial_q,
                    initialization_name=initialization["name"],
                    initialization_diagnostics=initial_diagnostics,
                )
                attempted.append(solution)
                topology_solutions.append(solution)
                print(
                    f"{solution.topology}/{solution.initialization}: "
                    f"success={solution.success} "
                    f"required={len(solution.required_fingers_satisfied)}/"
                    f"{len(solution.required_fingers)} "
                    f"cost={solution.initial_cost:.4f}->{solution.final_cost:.4f}",
                    flush=True,
                )
            successful = [item for item in topology_solutions if item.success]
            if successful:
                selected = min(successful, key=solution_rank)
                selected_optimizer = optimizer
                break
        if selected is None:
            selected = min(attempted, key=solution_rank)
            topology = next(item for item in optimization_config["topologies"]
                            if item["name"] == selected.topology)
            selected_optimizer = ContactRegionGraspOptimizer(
                robot.model, robot.data, optimization_config,
                topology=topology,
            )

        baseline_solution = current_solution(
            selected_optimizer, robot.data, selected.topology
        )
        baseline_static = execute_static_grasp_test(
            robot, task, baseline_solution, optimization_config,
            close_direction=close_direction,
            apply_candidate=False,
        )

        task = ReachGraspLiftTask(robot, task_config)
        repeated_grasp = task.run_grasp(args.seed)
        if not repeated_grasp.success:
            raise RuntimeError(
                f"deterministic re-initialization failed: {repeated_grasp.failure_reason}"
            )
        candidate_static = execute_static_grasp_test(
            robot, task, selected, optimization_config,
            close_direction=close_direction,
            apply_candidate=True,
        )
        fingertips = {
            finger: descriptor.to_dict()
            for finger, descriptor in selected_optimizer.tips.items()
        }
        q_by_finger = {}
        q = np.asarray(selected.qpos)
        arm_count = len(selected.arm_joint_ids)
        for index in range(5):
            start = arm_count + index * 4
            q_by_finger[f"finger{index + 1}"] = q[start:start + 4].tolist()
        report = plain({
            "schema_version": 1,
            "seed": args.seed,
            "fingertip_discovery": fingertips,
            "attempted_topologies": [item.to_dict() for item in attempted],
            "selected_topology": selected.topology,
            "selected_initialization": selected.initialization,
            "selected_kinematic_success": selected.success,
            "selected_palm_pose": {
                "position_world_m": selected.palm_position_world_m,
                "quaternion_world_wxyz": selected.palm_quaternion_world_wxyz,
            },
            "selected_finger_qpos_rad": q_by_finger,
            "baseline_static_test": baseline_static,
            "candidate_static_test": candidate_static,
            "accepted": bool(selected.success and candidate_static["static_success"]),
            "thresholds": optimization_config["static_test"],
            "notes": [
                "The policy action contract is unchanged; independent finger targets are experiment-only.",
                "No force-closure QP is included in this minimum version.",
                "The table is removed by disabling its collision during the gravity hold.",
            ],
        })
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({
            "selected_topology": report["selected_topology"],
            "selected_initialization": report["selected_initialization"],
            "kinematic_success": report["selected_kinematic_success"],
            "baseline_static_success": baseline_static["static_success"],
            "candidate_static_success": candidate_static["static_success"],
            "candidate_survival_s": candidate_static["survival_time_s"],
            "candidate_translation_drift_mm": 1000 * candidate_static[
                "max_relative_translation_drift_m"
            ],
            "candidate_rotation_drift_deg": candidate_static[
                "max_relative_rotation_drift_deg"
            ],
            "accepted": report["accepted"],
            "output": str(args.output) if args.output is not None else None,
        }, indent=2))
        if args.output is None:
            print("FULL_REPORT " + json.dumps(report))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
