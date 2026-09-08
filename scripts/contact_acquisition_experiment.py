from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import mujoco
from openarm_wuji.teleop.synergies import HandSynergyMapper
from openarm_wuji.tasks import ReachGraspLiftTask
from openarm_wuji.tasks.contact_grasp import (
    ContactRegionGraspOptimizer,
    execute_contact_acquisition,
    execute_single_finger_path_test,
)
from contact_grasp_experiment import hand_seed_library, plain


class HeadlessRobot:
    """Minimal backend for this contact-only run (no renderer allocation)."""
    def __init__(self, model_path, synergy_path, *, arm_side, control_hz):
        self.model_path = str(model_path)
        self.mapper = HandSynergyMapper.from_json(synergy_path)
        self.arm_side = arm_side
        self.control_hz = float(control_hz)
        self.control_dt = 1.0 / self.control_hz
        self.model = mujoco.MjModel.from_binary_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.arm_qpos_ids = np.asarray([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT,
                f"openarm_{arm_side}_joint{i}")]
            for i in range(1, 8)
        ], dtype=int)
        self.arm_qvel_ids = np.asarray([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT,
                f"openarm_{arm_side}_joint{i}")]
            for i in range(1, 8)
        ], dtype=int)
        self.arm_actuator_ids = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                              f"{arm_side}_joint{i}_ctrl")
            for i in range(1, 8)
        ], dtype=int)
        self.hand_qpos_ids = np.asarray([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"wuji_{name}")]
            for name in self.mapper.joint_names
        ], dtype=int)
        self.hand_qvel_ids = np.asarray([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"wuji_{name}")]
            for name in self.mapper.joint_names
        ], dtype=int)
        self.hand_actuator_ids = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                              f"wuji_{name}_actuator")
            for name in self.mapper.joint_names
        ], dtype=int)
        self._hand_target = self.mapper.open_pose.copy()
        self._last_synergy = np.zeros(3)
        self._frame_index = 0
        self._control_start_time = 0.0
        self.latest_record = None

    def connect(self):
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        self._control_start_time = float(self.data.time)
        mujoco.mj_forward(self.model, self.data)

    def synchronize_after_reset(self, hand_open):
        self._hand_target = np.asarray(hand_open, dtype=float).copy()
        self.data.ctrl[self.hand_actuator_ids] = self._hand_target
        self._control_start_time = float(self.data.time)
        self._frame_index = 0
        self.latest_record = self.get_observation()

    def send_action(self, action):
        action = np.asarray(action, dtype=float)
        arm = np.clip(action[:7],
                      self.model.actuator_ctrlrange[self.arm_actuator_ids, 0],
                      self.model.actuator_ctrlrange[self.arm_actuator_ids, 1])
        synergy = np.clip(action[7:], [0, 0, -1], [1, 1, 1])
        self._hand_target = self.mapper.next(
            synergy, previous=self._hand_target, dt=self.control_dt
        )
        self.data.ctrl[self.arm_actuator_ids] = arm
        self.data.ctrl[self.hand_actuator_ids] = self._hand_target
        steps = max(1, round((self._control_start_time +
                              (self._frame_index + 1) * self.control_dt -
                              self.data.time) / self.model.opt.timestep))
        mujoco.mj_step(self.model, self.data, nstep=steps)
        self._last_synergy = synergy.copy()
        self._frame_index += 1
        self.latest_record = self.get_observation()
        return synergy

    def get_observation(self):
        return {
            "frame_index": self._frame_index,
            "timestamp": time.perf_counter(),
            "sim_time": float(self.data.time),
            "front_rgb": np.zeros((1, 1, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((1, 1, 3), dtype=np.uint8),
            "arm_joint_position": self.data.qpos[self.arm_qpos_ids].copy(),
            "arm_joint_velocity": self.data.qvel[self.arm_qvel_ids].copy(),
            "hand_joint_position": self.data.qpos[self.hand_qpos_ids].copy(),
            "hand_joint_velocity": self.data.qvel[self.hand_qvel_ids].copy(),
            "hand_joint_target": self._hand_target.copy(),
            "hand_synergy_action": self._last_synergy.copy(),
            "sent_action": np.zeros(10),
        }

    def disconnect(self):
        self.data = None
        self.model = None


def main():
    parser = argparse.ArgumentParser(description="Seed-7 contact acquisition topology test")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--optimization-config", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--topology", default="thumb_minus_x_fingers_plus_x",
        help="named topology from the optimization config",
    )
    parser.add_argument(
        "--initialization", default="current_grasp",
        help="named initialization within the selected topology",
    )
    parser.add_argument(
        "--task-initialization", choices=("reach", "grasp"), default="grasp",
        help="initialize palm/cube state from reach-only or the legacy full grasp",
    )
    parser.add_argument(
        "--disable-squeeze", action="store_true",
        help="stop after contact hold; do not execute the optional squeeze phase",
    )
    parser.add_argument(
        "--lock-inactive-fingers", action="store_true",
        help="numerically hold optional and strategy-inactive fingers at their command",
    )
    parser.add_argument(
        "--smooth-control-substeps", action="store_true",
        help="linearly ramp each 30 Hz target across MuJoCo physics substeps",
    )
    parser.add_argument(
        "--single-fingers", default="finger2",
        help="comma-separated fingers for isolated path tests (default: finger2)",
    )
    args = parser.parse_args()
    task_config = json.loads(args.task_config.read_text(encoding="utf-8"))
    opt_config = json.loads(args.optimization_config.read_text(encoding="utf-8"))
    synergy_config = json.loads(args.synergies.read_text(encoding="utf-8"))
    open_pose = np.asarray(synergy_config["open_pose"], dtype=float)
    close_direction = np.sign(
        np.asarray(synergy_config["close_pose"], dtype=float) - open_pose
    )
    topology = next(item for item in opt_config["topologies"]
                    if item["name"] == args.topology)
    init = next(item for item in topology["initializations"]
                if item["name"] == args.initialization)
    results = []
    single_results = []
    robot = HeadlessRobot(
        args.model, args.synergies, arm_side=task_config["arm_side"],
        control_hz=30,
    )
    robot.connect()
    try:
        def initialize_task(task):
            return (task.run_reach(args.seed)
                    if args.task_initialization == "reach"
                    else task.run_grasp(args.seed))

        # First isolate the required fingers.  These tests run from the same
        # Stage-1 candidate but do not squeeze and keep the other fingers fixed.
        single_finger_names = tuple(
            item.strip() for item in args.single_fingers.split(",") if item.strip()
        )
        for single_finger in single_finger_names:
            task = ReachGraspLiftTask(robot, task_config)
            initialization_result = initialize_task(task)
            if not initialization_result.success:
                single_results.append({"single_finger": single_finger,
                                       "outcome": "initialization_failed",
                                       "failure_reason": initialization_result.failure_reason})
                continue
            optimizer = ContactRegionGraspOptimizer(
                robot.model, robot.data, opt_config, topology=topology,
            )
            seeds = hand_seed_library(
                optimizer.nominal[len(optimizer.arm_joint_ids):], synergy_config
            )
            initial_q, initial_diag = optimizer.make_initial_q(
                init, hand_seed=seeds[init["hand_seed"]],
            )
            candidate = optimizer.solve(
                initial_q=initial_q, initialization_name=init["name"],
                initialization_diagnostics=initial_diag,
            )
            if candidate.success:
                single_results.append(execute_single_finger_path_test(
                    robot, task, candidate, optimizer, opt_config,
                    finger=single_finger, open_hand=open_pose,
                    close_direction=close_direction,
                ))
            else:
                single_results.append({"single_finger": single_finger,
                                       "outcome": "kinematic_candidate_invalid"})
            del optimizer, task
            gc.collect()
        for strategy in ("synchronized", "thumb_first", "fingers_first"):
            task = ReachGraspLiftTask(robot, task_config)
            initialization_result = initialize_task(task)
            if not initialization_result.success:
                results.append({"strategy": strategy, "outcome": "initialization_failed",
                                "failure_reason": initialization_result.failure_reason})
                continue
            optimizer = ContactRegionGraspOptimizer(
                robot.model, robot.data, opt_config, topology=topology,
            )
            seeds = hand_seed_library(
                optimizer.nominal[len(optimizer.arm_joint_ids):], synergy_config
            )
            initial_q, initial_diag = optimizer.make_initial_q(
                init, hand_seed=seeds[init["hand_seed"]],
            )
            candidate = optimizer.solve(
                initial_q=initial_q,
                initialization_name=init["name"],
                initialization_diagnostics=initial_diag,
            )
            if not candidate.success:
                results.append({
                    "strategy": strategy,
                    "outcome": "kinematic_candidate_invalid",
                    "candidate": candidate.to_dict(),
                })
                continue
            result = execute_contact_acquisition(
                robot, task, candidate, optimizer, opt_config,
                open_hand=open_pose, close_direction=close_direction,
                strategy=strategy, allow_squeeze=not args.disable_squeeze,
                lock_inactive_fingers=args.lock_inactive_fingers,
                smooth_control_substeps=args.smooth_control_substeps,
            )
            result["candidate"] = candidate.to_dict()
            results.append(result)
            print(
                f"seed={args.seed} strategy={strategy} outcome={result['outcome']} "
                f"topology={result['final_required_contact_count']}/"
                f"{len(topology['required_fingers'])} "
                f"disp={1000*result['max_cube_translation_m']:.2f}mm "
                f"rot={result['max_cube_rotation_deg']:.2f}deg",
                flush=True,
            )
            del optimizer, task
            gc.collect()
    finally:
        robot.disconnect()
    report = plain({
            "schema_version": 1,
            "seed": args.seed,
            "topology": topology["name"],
            "initialization": init["name"],
            "task_initialization": args.task_initialization,
            "required_fingers": topology["required_fingers"],
            "optional_fingers": topology["optional_fingers"],
            "squeeze_enabled": not args.disable_squeeze,
            "inactive_fingers_locked": args.lock_inactive_fingers,
            "control_substeps_smoothed": args.smooth_control_substeps,
            "single_finger_tests": single_results,
            "strategies": results,
            "acquisition_config": opt_config["acquisition"],
            "notes": [
                "This round keeps the table enabled and does not perform lift or gravity-only testing.",
                "The 8 mm/6 deg external baseline is not used as a final dexterous-hand standard.",
                "Telemetry in time_series is diagnostic only and is not part of policy observation.",
            ],
        })
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except PermissionError:
            print(f"WARNING report path is not writable: {args.output}", flush=True)
    if args.output is None:
        print("FULL_REPORT " + json.dumps(report))


if __name__ == "__main__":
    main()
