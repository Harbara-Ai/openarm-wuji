"""Deduplicate mined states, run scripted recovery, and export LeRobot v3."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.dataset.lerobot_native import _features
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import restore_simulator_snapshot
from openarm_wuji.tasks import ReachGraspLiftTask


TASK = (
    "Correct a terminal Approach error, reach the cube grasp-start pose, "
    "and stop while keeping the Wuji hand open."
)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load_candidate(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as item:
        return {
            "path": path,
            "rollout_index": int(item["rollout_index"]),
            "seed": int(item["source_seed"]),
            "source_frame": int(item["source_frame"]),
            "trigger_type": str(item["trigger_type"]),
            "source_outcome": str(item["source_outcome"]),
            "trigger_score": float(item["trigger_score"]),
            "terminal_error_vector_m": np.asarray(
                item["terminal_error_vector_m"], dtype=float
            ),
            "terminal_error_m": float(item["terminal_error_m"]),
            "cube_displacement_vector_m": np.asarray(
                item["cube_displacement_vector_m"], dtype=float
            ),
            "cube_displacement_m": float(item["cube_displacement_m"]),
            "arm_qpos": np.asarray(item["observation.state"][:7], dtype=float),
        }


def _candidate_rank(item: dict[str, Any]) -> float:
    outcome_bonus = 2.0 if item["source_outcome"] != "success" else 0.0
    trigger_bonus = {
        "cube_displacement": 1.8,
        "moving_away": 1.6,
        "gate_stall": 1.4,
        "near_timeout_terminal": 1.2,
        "terminal_plateau": 1.0,
    }.get(item["trigger_type"], 0.5)
    terminal_bonus = max(0.0, 0.030 - item["terminal_error_m"]) / 0.030
    cube_bonus = min(item["cube_displacement_m"] / 0.025, 1.0)
    return outcome_bonus + trigger_bonus + terminal_bonus + cube_bonus


def _deduplicate(candidates: list[dict[str, Any]], target_count: int
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Keep recoverable, genuinely terminal snapshots. States already beyond
    # the official safety gate are diagnostics, not correction starting data.
    eligible = [
        item for item in candidates
        if item["terminal_error_m"] <= 0.050
        and item["cube_displacement_m"] < 0.025
    ]
    if not eligible:
        return [], {"eligible": 0, "excluded_unsafe_or_nonterminal": len(candidates)}
    raw = np.stack([
        np.r_[
            item["terminal_error_vector_m"],
            item["terminal_error_m"],
            item["arm_qpos"],
            item["cube_displacement_vector_m"],
            item["cube_displacement_m"],
        ]
        for item in eligible
    ])
    scale = np.std(raw, axis=0)
    scale = np.maximum(scale, np.asarray(
        [0.002] * 3 + [0.002] + [0.005] * 7 + [0.001] * 3 + [0.001]
    ))
    features = (raw - np.mean(raw, axis=0)) / scale
    for index, item in enumerate(eligible):
        item["rank"] = _candidate_rank(item)
        item["feature_index"] = index

    selected: list[dict[str, Any]] = []
    selected_rollouts: set[int] = set()
    # Seed the set with the strongest state in each trigger/outcome category.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in eligible:
        groups.setdefault(
            (item["trigger_type"], item["source_outcome"]), []
        ).append(item)
    for group in groups.values():
        choices = sorted(group, key=lambda row: row["rank"], reverse=True)
        choice = next(
            (row for row in choices if row["rollout_index"] not in selected_rollouts),
            None,
        )
        if choice is not None:
            selected.append(choice)
            selected_rollouts.add(choice["rollout_index"])
    selected = selected[:target_count]
    selected_rollouts = {item["rollout_index"] for item in selected}

    # Farthest-point fill balances error direction/magnitude, arm posture, and
    # cube displacement while limiting the first pass to one state per rollout.
    while len(selected) < min(target_count, len({
        item["rollout_index"] for item in eligible
    })):
        pool = [
            item for item in eligible
            if item["rollout_index"] not in selected_rollouts
        ]
        if not pool:
            break
        if selected:
            selected_features = features[[
                item["feature_index"] for item in selected
            ]]
            def score(item):
                vector = features[item["feature_index"]]
                distance = float(np.min(np.linalg.norm(
                    selected_features - vector[None], axis=1
                )))
                return distance + 0.05 * item["rank"]
        else:
            def score(item):
                return item["rank"]
        choice = max(pool, key=score)
        selected.append(choice)
        selected_rollouts.add(choice["rollout_index"])

    # If fewer unique rollouts than requested exist, permit a second diverse
    # trigger frame from a rollout rather than silently missing the target.
    while len(selected) < min(target_count, len(eligible)):
        remaining = [item for item in eligible if item not in selected]
        if not remaining:
            break
        selected_features = features[[item["feature_index"] for item in selected]]
        choice = max(
            remaining,
            key=lambda item: float(np.min(np.linalg.norm(
                selected_features
                - features[item["feature_index"]][None], axis=1
            ))) + 0.05 * item["rank"],
        )
        selected.append(choice)
    for item in selected:
        item.pop("feature_index", None)
    return selected, {
        "eligible": len(eligible),
        "excluded_unsafe_or_nonterminal": len(candidates) - len(eligible),
        "selected": len(selected),
        "unique_source_rollouts": len({item["rollout_index"] for item in selected}),
        "feature_dimensions": [
            "terminal_error_xyz", "terminal_error_magnitude", "arm_qpos_7d",
            "cube_displacement_xyz", "cube_displacement_magnitude",
        ],
        "one_per_rollout_first_pass": True,
    }


def _pose(telemetry: dict) -> np.ndarray:
    return np.r_[
        telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"]
    ].astype(np.float64)


def _relative_pose(telemetry: dict) -> np.ndarray:
    return np.r_[
        telemetry["object_relative_position_m"],
        telemetry["object_relative_quaternion_wxyz"],
    ].astype(np.float64)


def _contact_arrays(telemetry: dict) -> tuple[np.ndarray, np.ndarray]:
    forces = np.asarray([
        telemetry["finger_normal_forces_n"].get(f"finger{index}", 0.0)
        for index in range(1, 6)
    ], dtype=np.float64)
    return forces > 0.0, forces


def _run_correction(*, snapshot_path: Path, episode_index: int, robot,
                    config: dict, output_dir: Path, max_steps: int,
                    hold_frames: int) -> dict[str, Any]:
    with np.load(snapshot_path, allow_pickle=False) as loaded:
        snapshot = {key: loaded[key].copy() for key in loaded.files}
    seed = int(snapshot["source_seed"])
    task = ReachGraspLiftTask(robot, config)
    task.reset(seed)
    restore_check = restore_simulator_snapshot(robot, task, snapshot)
    if not restore_check["passed"]:
        raise RuntimeError(
            f"snapshot restoration failed exact check: {snapshot_path} {restore_check}"
        )
    telemetry = task.task_telemetry()
    correction_start_cube = np.asarray(
        telemetry["cube_position_m"], dtype=float
    )
    original_approach_cube = np.asarray(
        snapshot["approach_start_cube_position_m"], dtype=float
    )
    target = correction_start_cube + np.asarray(
        config["grasp"]["target_offset_m"], dtype=float
    )
    task.arm_command = np.asarray(
        snapshot["observation.controller_target"][:7], dtype=float
    ).copy()
    task._arm_command_initialized = True
    goal, ik_residual, ik_orientation_residual = task._solve_arm_goal(
        target, phase_config=config["grasp"]
    )
    if (
        ik_residual > float(config["grasp"]["ik_solve_tolerance_m"])
        or ik_orientation_residual
        > float(config["reach"]["ik_orientation_solve_tolerance_deg"])
    ):
        return {
            "correction_success": False,
            "failure_reason": "ik_unreachable",
            "source_seed": seed,
            "snapshot": snapshot_path,
            "restore_check": restore_check,
        }
    arm_target = task.arm_command.copy()
    hand_target = np.asarray(
        snapshot["observation.controller_target"][7:], dtype=float
    ).copy()
    max_step = float(config["grasp"]["max_action_step_rad"])
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    front: list[np.ndarray] = []
    wrist: list[np.ndarray] = []
    cube_poses: list[np.ndarray] = []
    relative_poses: list[np.ndarray] = []
    contacts: list[int] = []
    active_masks: list[np.ndarray] = []
    normal_forces: list[np.ndarray] = []
    terminal_errors: list[float] = []
    orientation_errors: list[float] = []
    cube_displacements: list[float] = []
    correction_cube_displacements: list[float] = []
    phases: list[str] = []
    gate_run = 0
    success_frame = None
    failure_reason = None

    def take_step(*, terminal_hold: bool) -> None:
        nonlocal arm_target, gate_run, success_frame, failure_reason
        observation = robot.get_observation()
        if not terminal_hold:
            arm_target += np.clip(goal - arm_target, -max_step, max_step)
        command = np.r_[arm_target, hand_target]
        sent = robot.send_controller_joint_target(command)
        post = task.task_telemetry()
        error = float(np.linalg.norm(
            np.asarray(post["grasp_center_position_m"]) - target
        ))
        orientation = float(post["palm_orientation_error_deg"])
        absolute_cube_displacement = float(np.linalg.norm(
            np.asarray(post["cube_position_m"]) - original_approach_cube
        ))
        local_cube_displacement = float(np.linalg.norm(
            np.asarray(post["cube_position_m"]) - correction_start_cube
        ))
        inside = (
            error <= 0.012
            and orientation <= 2.0
            and absolute_cube_displacement <= 0.025
        )
        gate_run = gate_run + 1 if inside else 0
        if gate_run >= 5 and success_frame is None:
            success_frame = len(actions)
        if absolute_cube_displacement > 0.025:
            failure_reason = "cube_displacement"
        active, force = _contact_arrays(post)
        states.append(np.concatenate([
            observation["arm_joint_position"],
            observation["hand_joint_position"],
        ]).astype(np.float32))
        actions.append(sent.astype(np.float32))
        front.append(np.asarray(observation["front_rgb"], dtype=np.uint8))
        wrist.append(np.asarray(observation["wrist_rgb"], dtype=np.uint8))
        cube_poses.append(_pose(post))
        relative_poses.append(_relative_pose(post))
        contacts.append(len(post["contacts"]))
        active_masks.append(active)
        normal_forces.append(force)
        terminal_errors.append(error)
        orientation_errors.append(orientation)
        cube_displacements.append(absolute_cube_displacement)
        correction_cube_displacements.append(local_cube_displacement)
        phases.append("approach_hold" if terminal_hold else "approach_correction")

    for _ in range(max_steps):
        take_step(terminal_hold=False)
        if failure_reason is not None or success_frame is not None:
            break
    if success_frame is not None and failure_reason is None:
        for _ in range(hold_frames):
            take_step(terminal_hold=True)
            if failure_reason is not None:
                break
    success = bool(
        success_frame is not None
        and failure_reason is None
        and gate_run >= 5 + hold_frames
    )
    if not success and failure_reason is None:
        failure_reason = "correction_timeout"
    result = {
        "correction_success": success,
        "failure_reason": failure_reason,
        "source_seed": seed,
        "source_rollout_index": int(snapshot["rollout_index"]),
        "source_frame": int(snapshot["source_frame"]),
        "trigger_type": str(snapshot["trigger_type"]),
        "source_outcome": str(snapshot["source_outcome"]),
        "snapshot": snapshot_path,
        "restore_check": restore_check,
        "initial_terminal_error_m": float(snapshot["terminal_error_m"]),
        "initial_rebased_terminal_error_m": float(np.linalg.norm(
            np.asarray(telemetry["grasp_center_position_m"]) - target
        )),
        "initial_terminal_error_vector_m": np.asarray(
            snapshot["terminal_error_vector_m"], dtype=float
        ),
        "initial_cube_displacement_m": float(snapshot["cube_displacement_m"]),
        "recovery_length": len(actions),
        "gate_success_frame": success_frame,
        "minimum_terminal_error_m": min(terminal_errors),
        "final_terminal_error_m": terminal_errors[-1],
        "maximum_cube_displacement_m": max(cube_displacements),
        "maximum_additional_cube_displacement_m": max(
            correction_cube_displacements
        ),
        "ik_residual_m": ik_residual,
        "ik_orientation_residual_deg": ik_orientation_residual,
        "correction_target_rebased_to_current_cube": True,
        "hand_target_max_change_rad": float(np.max(np.abs(
            np.asarray(actions)[:, 7:] - hand_target[None]
        ))),
    }
    if not success:
        return result
    raw_path = output_dir / (
        f"episode_{episode_index:06d}_seed_{seed:06d}.npz"
    )
    metadata = {
        **{key: _plain(value) for key, value in result.items()
           if key not in {"restore_check"}},
        "restore_check": restore_check,
        "target_grasp_position_m": target,
        "terminal_hold_frames": hold_frames,
        "excluded_phases": ["reach", "grasp", "preload", "lift", "hold"],
    }
    np.savez_compressed(
        raw_path,
        format_name=np.asarray("openarm_wuji_approach_correction"),
        schema_version=np.asarray(1, dtype=np.int64),
        task_description=np.asarray(TASK),
        source=np.asarray(snapshot_path.resolve().as_posix()),
        source_episode_index=np.asarray(
            int(snapshot["rollout_index"]), dtype=np.int64
        ),
        episode_index=np.asarray(episode_index, dtype=np.int64),
        episode_seed=np.asarray(seed, dtype=np.int64),
        control_hz=np.asarray(30.0, dtype=np.float64),
        joint_names=np.asarray(JOINT_NAMES),
        **{
            "observation.state": np.asarray(states, dtype=np.float32),
            "action": np.asarray(actions, dtype=np.float32),
            "observation.images.front": np.asarray(front, dtype=np.uint8),
            "observation.images.wrist": np.asarray(wrist, dtype=np.uint8),
            "telemetry.cube_pose_world": np.asarray(cube_poses),
            "telemetry.cube_pose_relative_to_palm": np.asarray(relative_poses),
            "telemetry.contact_count": np.asarray(contacts, dtype=np.int32),
            "telemetry.active_finger_mask": np.asarray(active_masks, dtype=bool),
            "telemetry.finger_normal_force_n": np.asarray(normal_forces),
        },
        initial_controller_target=np.asarray(
            snapshot["observation.controller_target"], dtype=np.float32
        ),
        timestamp=np.arange(len(actions), dtype=np.float64) / 30.0,
        frame_index=np.arange(len(actions), dtype=np.int64),
        phase=np.asarray(phases),
        source_frame_index=np.full(
            len(actions), int(snapshot["source_frame"]), dtype=np.int64
        ),
        synthetic_terminal_hold=np.asarray([
            phase == "approach_hold" for phase in phases
        ], dtype=bool),
        source_seed=np.asarray(seed, dtype=np.int64),
        source_frame=np.asarray(int(snapshot["source_frame"]), dtype=np.int64),
        trigger_type=np.asarray(str(snapshot["trigger_type"])),
        initial_terminal_error=np.asarray(
            float(snapshot["terminal_error_m"]), dtype=np.float64
        ),
        initial_cube_displacement=np.asarray(
            float(snapshot["cube_displacement_m"]), dtype=np.float64
        ),
        correction_success=np.asarray(True),
        recovery_length=np.asarray(len(actions), dtype=np.int64),
        terminal_error_m=np.asarray(terminal_errors, dtype=np.float64),
        orientation_error_deg=np.asarray(orientation_errors, dtype=np.float64),
        cube_displacement_m=np.asarray(cube_displacements, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(_plain(metadata))),
    )
    result["raw_episode"] = raw_path
    return result


def _export_native(*, raw_paths: list[Path], output: Path, cache_dir: Path,
                   repo_id: str) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader
    import datasets

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    datasets.config.HF_DATASETS_CACHE = str(cache_dir / "datasets")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=_features((240, 320, 3)),
        root=output,
        robot_type="openarm_wuji",
        use_videos=False,
        video_backend="pyav",
        image_writer_threads=0,
    )
    expected_states = []
    expected_actions = []
    frames = 0
    for path in raw_paths:
        with np.load(path, allow_pickle=False) as episode:
            states = np.asarray(episode["observation.state"], dtype=np.float32)
            actions = np.asarray(episode["action"], dtype=np.float32)
            for frame in range(len(actions)):
                dataset.add_frame({
                    "observation.state": states[frame],
                    "action": actions[frame],
                    "observation.images.front": episode[
                        "observation.images.front"
                    ][frame],
                    "observation.images.wrist": episode[
                        "observation.images.wrist"
                    ][frame],
                    "task": TASK,
                })
            dataset.save_episode(parallel_encoding=False)
            expected_states.append(states)
            expected_actions.append(actions)
            frames += len(actions)
    dataset.finalize()
    loaded = LeRobotDataset(repo_id=repo_id, root=output, video_backend="pyav")
    actual_states = np.stack([
        np.asarray(value) for value in loaded.hf_dataset["observation.state"]
    ])
    actual_actions = np.stack([
        np.asarray(value) for value in loaded.hf_dataset["action"]
    ])
    state_exact = np.array_equal(actual_states, np.concatenate(expected_states))
    action_exact = np.array_equal(actual_actions, np.concatenate(expected_actions))
    loader = DataLoader(
        loaded, batch_size=min(8, len(loaded)), shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    if not state_exact or not action_exact:
        raise RuntimeError("LeRobot correction export changed state/action values")
    return {
        "format": "LeRobotDataset",
        "codebase_version": str(loaded.meta.info.codebase_version),
        "repo_id": repo_id,
        "episodes": int(loaded.num_episodes),
        "frames": frames,
        "fps": 30,
        "state_dim": 27,
        "action_dim": 27,
        "image_shape_hwc": [240, 320, 3],
        "exact_round_trip": {"state": state_exact, "action": action_exact},
        "dataloader_smoke": {
            "state_shape": list(batch["observation.state"].shape),
            "action_shape": list(batch["action"].shape),
            "front_shape": list(batch["observation.images.front"].shape),
            "wrist_shape": list(batch["observation.images.wrist"].shape),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mining", type=Path,
        default=root / "outputs/approach_correction_demos/mining",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/approach_correction_demos",
    )
    parser.add_argument(
        "--model", type=Path,
        default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb",
    )
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/reach_grasp_lift.json",
    )
    parser.add_argument(
        "--synergies", type=Path,
        default=root / "configs/wuji_hand_left_synergies.json",
    )
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--minimum-count", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=root / ".hf_approach_corrections",
    )
    parser.add_argument("--repo-id", default="local/openarm-wuji-approach-corrections")
    args = parser.parse_args()
    if not 5 <= args.hold_frames <= 10:
        raise ValueError("--hold-frames must be between 5 and 10")
    correction_dir = args.output / "raw"
    native_dir = args.output / "lerobot_dataset"
    selection_path = args.output / "selection.json"
    results_path = args.output / "correction_results.json"
    for path in (correction_dir, native_dir, selection_path, results_path):
        if path.exists():
            raise FileExistsError(f"correction output already exists: {path}")
    candidates = [
        _load_candidate(path) for path in sorted(
            (args.mining / "candidate_snapshots").glob("snapshot_*.npz")
        )
    ]
    selected, dedup = _deduplicate(candidates, args.target_count)
    if len(selected) < args.minimum_count:
        raise RuntimeError(
            f"only {len(selected)} unique eligible snapshots; mine more rollouts"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    correction_dir.mkdir()
    selection = {
        "candidate_snapshots": len(candidates),
        "deduplication": dedup,
        "selected": selected,
    }
    selection_path.write_text(
        json.dumps(_plain(selection), indent=2), encoding="utf-8"
    )
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
    results: list[dict[str, Any]] = []
    success_index = 0
    try:
        for index, item in enumerate(selected):
            result = _run_correction(
                snapshot_path=item["path"],
                episode_index=success_index,
                robot=robot,
                config=config,
                output_dir=correction_dir,
                max_steps=args.max_steps,
                hold_frames=args.hold_frames,
            )
            results.append(result)
            if result["correction_success"]:
                success_index += 1
            print(
                f"correction={index + 1}/{len(selected)} "
                f"successes={success_index}", flush=True
            )
    finally:
        robot.disconnect()
    raw_paths = sorted(correction_dir.glob("episode_*.npz"))
    if not raw_paths:
        raise RuntimeError("no correction trajectory succeeded")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    native = _export_native(
        raw_paths=raw_paths,
        output=native_dir,
        cache_dir=args.cache_dir,
        repo_id=args.repo_id,
    )
    trigger_distribution: dict[str, int] = {}
    failure_distribution: dict[str, int] = {}
    for item in results:
        key = item["trigger_type"]
        trigger_distribution[key] = trigger_distribution.get(key, 0) + int(
            item["correction_success"]
        )
        if not item["correction_success"]:
            reason = item["failure_reason"]
            failure_distribution[reason] = failure_distribution.get(reason, 0) + 1
    summary = {
        "selected_snapshots": len(selected),
        "successful_correction_demos": len(raw_paths),
        "failed_corrections": len(selected) - len(raw_paths),
        "correction_failure_distribution": failure_distribution,
        "successful_trigger_type_distribution": trigger_distribution,
        "restore_checks_all_passed": all(
            item["restore_check"]["passed"] for item in results
            if "restore_check" in item
        ),
        "action_semantics": "27D absolute MuJoCo position-controller target",
        "policy_observation_compatible": True,
        "grasp_preload_lift_excluded": True,
        "native_dataset": native,
        "results": results,
    }
    results_path.write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(_plain({
            "mining_summary": args.mining / "summary.json",
            "selection": selection_path,
            "correction_results": results_path,
            "raw_directory": correction_dir,
            "lerobot_dataset": native_dir,
            **{key: value for key, value in summary.items() if key != "results"},
        }), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_plain({
        "selected_snapshots": len(selected),
        "successful_correction_demos": len(raw_paths),
        "failed_corrections": len(selected) - len(raw_paths),
        "native_dataset": native,
    }), indent=2))


if __name__ == "__main__":
    main()
