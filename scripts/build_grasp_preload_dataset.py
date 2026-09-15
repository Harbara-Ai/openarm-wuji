"""Build Grasp+Preload demonstrations from nominal and real staged starts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.dataset.lerobot_native import _features
from openarm_wuji.policy import ApproachPolicy, ReachPolicy, RecoveryPolicy
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import (
    capture_simulator_snapshot,
    restore_simulator_snapshot,
)
from openarm_wuji.tasks import ReachGraspLiftTask
from openarm_wuji.tasks.grasp_preload_stage import (
    initialize_task_from_handoff,
    run_scripted_grasp_preload,
)
from openarm_wuji.tasks.se3 import rotation_geodesic_angle_deg
from scripts.evaluate_staged_with_recovery import (
    _run_episode as run_frozen_upstream,
    _set_deterministic,
)


TASK = (
    "Close Wuji from a stable grasp-start pose, establish a multi-finger "
    "grasp and controller preload, then hold the load-bearing-ready state."
)
STAGE_PHASES = ("grasp_close", "preload", "preload_settle")


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


def _state(observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate([
        observation["arm_joint_position"],
        observation["hand_joint_position"],
    ]).astype(np.float32)


def _relative_speeds(relative_pose: np.ndarray, fps: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    translation = np.zeros(len(relative_pose), dtype=np.float64)
    rotation = np.zeros(len(relative_pose), dtype=np.float64)
    if len(relative_pose) > 1:
        translation[1:] = np.linalg.norm(
            np.diff(relative_pose[:, :3], axis=0), axis=1
        ) * fps
        rotation[1:] = np.asarray([
            rotation_geodesic_angle_deg(left, right)
            for left, right in zip(
                relative_pose[:-1, 3:], relative_pose[1:, 3:], strict=True
            )
        ]) * fps
    return translation, rotation


def extract_nominal_stage(source: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Extract causal Grasp+Preload frames from a successful full demo."""
    with np.load(source, allow_pickle=False) as episode:
        phases = np.asarray(episode["controller_phase"])
        mask = np.isin(phases, STAGE_PHASES)
        indices = np.flatnonzero(mask)
        if not len(indices) or not np.array_equal(
            indices, np.arange(indices[0], indices[-1] + 1)
        ):
            raise ValueError(f"non-contiguous Grasp+Preload phases in {source}")
        if not bool(episode["demonstration_success"]):
            raise ValueError(f"nominal source is not successful: {source}")
        selected = slice(int(indices[0]), int(indices[-1]) + 1)
        states = np.asarray(episode["observation.state"][selected], dtype=np.float32)
        next_states = np.asarray(
            episode["next_observation.state"][selected], dtype=np.float32
        )
        actions = np.asarray(episode["action"][selected], dtype=np.float32)
        relative = np.asarray(
            episode["telemetry.cube_pose_relative_to_palm"][selected],
            dtype=np.float64,
        )
        cube = np.asarray(
            episode["telemetry.cube_pose_world"][selected], dtype=np.float64
        )
        forces = np.asarray(
            episode["telemetry.finger_normal_force_n"][selected],
            dtype=np.float64,
        )
        active = forces >= 0.1
        speed, angular_speed = _relative_speeds(
            relative, float(episode["control_hz"])
        )
        arrays = {
            "observation.state": states,
            "next_observation.state": next_states,
            "action": actions,
            "raw_script_action": np.asarray(
                episode["raw_script_action"][selected], dtype=np.float32
            ),
            "observation.images.front": np.asarray(
                episode["observation.images.front"][selected], dtype=np.uint8
            ),
            "observation.images.wrist": np.asarray(
                episode["observation.images.wrist"][selected], dtype=np.uint8
            ),
            "phase": phases[selected],
            "telemetry.cube_pose_world": cube,
            "telemetry.cube_pose_relative_to_palm": relative,
            "telemetry.active_finger_mask": active,
            "telemetry.finger_normal_force_n": forces,
            "telemetry.total_normal_force_n": np.sum(forces, axis=1),
            "telemetry.hand_target_actual_preload_rad": (
                actions[:, 7:] - next_states[:, 7:]
            ),
            "telemetry.cube_displacement_m": np.linalg.norm(
                cube[:, :3] - cube[0, :3], axis=1
            ),
            "telemetry.relative_translation_speed_m_s": speed,
            "telemetry.relative_rotation_speed_deg_s": angular_speed,
            "sim_time": np.asarray(episode["sim_time"][selected]),
        }
        outcome = json.loads(str(episode["outcome_json"]))
        grasp = outcome["grasp"]
        terminal = phases[selected] == "preload_settle"
        preload = arrays["telemetry.hand_target_actual_preload_rad"]
        terminal_l2 = np.linalg.norm(preload[terminal], axis=1)
        result = {
            "success": True,
            "grasp_success": bool(grasp["synergy_frozen"]),
            "preload_success": bool(grasp["preload_reached"]),
            "terminal_hold_success": bool(grasp["settle_stable"]),
            "failure_stage": None,
            "seed": int(episode["episode_seed"]),
            "source_kind": "scripted_nominal",
            "source": source,
            "source_start_frame": int(indices[0]),
            "frames": len(actions),
            "first_contact_fingers": [],
            "max_simultaneous_contacts": int(np.max(np.sum(active, axis=1))),
            "longest_persistent_multifinger_frames": int(grasp["contact_hold_frames"]),
            "peak_per_finger_normal_force_n": np.max(forces, axis=0),
            "final_per_finger_normal_force_n": forces[-1],
            "terminal_hand_preload_l2_mean_rad": float(np.mean(terminal_l2)),
            "terminal_hand_preload_l2_min_rad": float(np.min(terminal_l2)),
            "terminal_hand_preload_abs_mean_rad": np.mean(
                np.abs(preload[terminal]), axis=0
            ),
            "maximum_cube_displacement_m": float(np.max(
                arrays["telemetry.cube_displacement_m"]
            )),
            "maximum_relative_translation_speed_m_s": float(np.max(speed)),
            "maximum_relative_rotation_speed_deg_s": float(np.max(angular_speed)),
            "load_bearing_ready_static_proxy": True,
            "unsupported_lift_tested": False,
        }
        prefix = {
            "seed": int(episode["episode_seed"]),
            "raw_script_action": np.asarray(
                episode["raw_script_action"][:indices[0]], dtype=np.float64
            ),
            "expected_state": states[0].astype(np.float64),
            "source_start_frame": int(indices[0]),
        }
    return {"result": result, "prefix": prefix}, arrays


def _save_snapshot(path: Path, snapshot: dict[str, np.ndarray], **metadata: Any) -> None:
    np.savez_compressed(
        path,
        **snapshot,
        **{
            key: np.asarray(json.dumps(_plain(value)))
            if isinstance(value, (dict, list, tuple)) else np.asarray(value)
            for key, value in metadata.items()
        },
    )


def _save_episode(path: Path, *, episode_index: int, seed: int,
                  source_kind: str, arrays: dict[str, np.ndarray],
                  result: dict[str, Any], start_snapshot: Path,
                  start_metadata: dict[str, Any]) -> None:
    frames = len(arrays["action"])
    if arrays["observation.state"].shape != (frames, 27):
        raise ValueError("Grasp+Preload state is not [T,27]")
    if arrays["action"].shape != (frames, 27):
        raise ValueError("Grasp+Preload action is not [T,27]")
    if not np.isfinite(arrays["observation.state"]).all():
        raise ValueError("Grasp+Preload state contains non-finite values")
    if not np.isfinite(arrays["action"]).all():
        raise ValueError("Grasp+Preload action contains non-finite values")
    payload = dict(arrays)
    payload.update({
        "format_name": np.asarray("openarm_wuji_grasp_preload_demo"),
        "schema_version": np.asarray(1, dtype=np.int64),
        "task_description": np.asarray(TASK),
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "episode_seed": np.asarray(seed, dtype=np.int64),
        "source_kind": np.asarray(source_kind),
        "control_hz": np.asarray(30.0, dtype=np.float64),
        "joint_names": np.asarray(JOINT_NAMES),
        "timestamp": np.arange(frames, dtype=np.float64) / 30.0,
        "frame_index": np.arange(frames, dtype=np.int64),
        "grasp_preload_success": np.asarray(bool(result["success"])),
        "start_snapshot": np.asarray(start_snapshot.resolve().as_posix()),
        "metadata_json": np.asarray(json.dumps(_plain({
            "result": result,
            "start_metadata": start_metadata,
            "policy_observation_fields": [
                "observation.state",
                "observation.images.front",
                "observation.images.wrist",
            ],
            "telemetry_excluded_from_policy_observation": True,
        }))),
    })
    np.savez_compressed(path, **payload)


def _replay_nominal_start(*, robot, config: dict, prefix: dict[str, Any],
                          snapshot_path: Path) -> dict[str, Any]:
    task = ReachGraspLiftTask(robot, config)
    task.reset(int(prefix["seed"]))
    for action in prefix["raw_script_action"]:
        robot.send_action(action)
    observation = robot.get_observation()
    replay_error = float(np.max(np.abs(
        _state(observation).astype(np.float64) - prefix["expected_state"]
    )))
    if replay_error > 1e-9:
        raise RuntimeError(
            f"nominal grasp-start replay mismatch: {replay_error:.3e}"
        )
    initialize_task_from_handoff(task)
    snapshot = capture_simulator_snapshot(robot, task, observation=observation)
    restore_check = restore_simulator_snapshot(robot, task, snapshot)
    if not restore_check["passed"]:
        raise RuntimeError(f"nominal snapshot restore failed: {restore_check}")
    _save_snapshot(
        snapshot_path, snapshot,
        source_kind="scripted_nominal",
        source_seed=int(prefix["seed"]),
        source_frame=int(prefix["source_start_frame"]),
        replay_state_max_abs_error=replay_error,
        restore_check=restore_check,
    )
    return {
        "snapshot": snapshot_path,
        "source_kind": "scripted_nominal",
        "source_seed": int(prefix["seed"]),
        "source_frame": int(prefix["source_start_frame"]),
        "replay_state_max_abs_error": replay_error,
        "restore_check": restore_check,
    }


def _export_native(*, raw_paths: list[Path], output: Path,
                   cache_dir: Path, repo_id: str) -> dict[str, Any]:
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
    source_rows = []
    for path in raw_paths:
        with np.load(path, allow_pickle=False) as episode:
            if not bool(episode["grasp_preload_success"]):
                continue
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
            source_rows.append({
                "source": path,
                "source_kind": str(episode["source_kind"]),
                "seed": int(episode["episode_seed"]),
                "frames": len(actions),
            })
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
    if not state_exact or not action_exact:
        raise RuntimeError("native Grasp+Preload export changed state/action")
    batch = next(iter(DataLoader(
        loaded, batch_size=min(8, len(loaded)), shuffle=False, num_workers=0
    )))
    return {
        "format": "LeRobotDataset",
        "codebase_version": str(loaded.meta.info.codebase_version),
        "repo_id": repo_id,
        "episodes": int(loaded.num_episodes),
        "frames": int(len(loaded)),
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
        "sources": source_rows,
    }


def _preload_statistics(paths: list[Path]) -> dict[str, Any]:
    per_episode_mean = []
    per_episode_min = []
    per_joint_abs = []
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            phases = np.asarray(episode["phase"])
            terminal = np.isin(phases, ["preload_settle", "terminal_hold"])
            preload = np.asarray(
                episode["telemetry.hand_target_actual_preload_rad"]
            )[terminal]
            norms = np.linalg.norm(preload, axis=1)
            per_episode_mean.append(float(np.mean(norms)))
            per_episode_min.append(float(np.min(norms)))
            per_joint_abs.append(np.mean(np.abs(preload), axis=0))
    means = np.asarray(per_episode_mean)
    minimums = np.asarray(per_episode_min)
    return {
        "terminal_hand_preload_l2_mean_rad": {
            "mean": float(np.mean(means)),
            "median": float(np.median(means)),
            "p10": float(np.quantile(means, 0.1)),
            "min": float(np.min(means)),
            "max": float(np.max(means)),
        },
        "terminal_hand_preload_l2_min_rad": {
            "mean": float(np.mean(minimums)),
            "median": float(np.median(minimums)),
            "p10": float(np.quantile(minimums, 0.1)),
            "min": float(np.min(minimums)),
        },
        "per_joint_terminal_abs_preload_mean_rad": np.mean(
            np.asarray(per_joint_abs), axis=0
        ),
        "evaluation_preload_gate": {
            "source": "10th percentile of successful expert episode terminal mean",
            "hand_preload_l2_min_rad": float(np.quantile(means, 0.1)),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nominal-source", type=Path,
        default=root / "outputs/act_e2e_smoke/coordinated_demos/raw/successful",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/grasp_preload_act/dataset",
    )
    parser.add_argument("--staged-successes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1600)
    parser.add_argument("--max-staged-attempts", type=int, default=80)
    parser.add_argument("--terminal-hold-frames", type=int, default=8)
    parser.add_argument(
        "--reach-checkpoint", type=Path,
        default=root / "outputs/act_reach_only/act_train/checkpoints/002000/pretrained_model",
    )
    parser.add_argument(
        "--approach-checkpoint", type=Path,
        default=root / "outputs/act_staged/approach_only/act_train/checkpoints/002000/pretrained_model",
    )
    parser.add_argument(
        "--recovery-checkpoint", type=Path,
        default=root / "outputs/staged_act_with_recovery/recovery_act_train/checkpoints/001500/pretrained_model",
    )
    parser.add_argument(
        "--router", type=Path, default=root / "configs/recovery_router.json"
    )
    parser.add_argument(
        "--model", type=Path,
        default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb",
    )
    parser.add_argument(
        "--config", type=Path, default=root / "configs/reach_grasp_lift.json"
    )
    parser.add_argument(
        "--synergies", type=Path,
        default=root / "configs/wuji_hand_left_synergies.json",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=root / ".hf_grasp_preload_act"
    )
    parser.add_argument("--repo-id", default="local/openarm-wuji-grasp-preload")
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.staged_successes < 0 or args.max_staged_attempts < args.staged_successes:
        raise ValueError("invalid staged success/attempt counts")
    if not 5 <= args.terminal_hold_frames <= 10:
        raise ValueError("terminal hold must be 5-10 frames")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}")
        shutil.rmtree(args.output)
    raw_success = args.output / "raw/successful"
    raw_diagnostics = args.output / "raw/diagnostics"
    starts_nominal = args.output / "starts/nominal"
    starts_staged = args.output / "starts/staged"
    upstream_output = args.output / "upstream_rollouts"
    for path in (
        raw_success, raw_diagnostics, starts_nominal,
        starts_staged, upstream_output,
    ):
        path.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    router = json.loads(args.router.read_text(encoding="utf-8"))
    robot = MujocoOpenArmWuji(
        args.model, args.synergies,
        arm_side=config["arm_side"], control_hz=30,
        image_height=240, image_width=320,
        front_camera=config["scene"]["front_camera_name"],
    )
    robot.connect()
    results: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    start_rows: list[dict[str, Any]] = []
    episode_index = 0
    nominal_sources = sorted(args.nominal_source.glob("*.npz"))
    try:
        for source in nominal_sources:
            extracted, arrays = extract_nominal_stage(source)
            prefix = extracted["prefix"]
            result = extracted["result"]
            snapshot_path = starts_nominal / (
                f"start_seed_{int(prefix['seed']):06d}.npz"
            )
            start = _replay_nominal_start(
                robot=robot, config=config, prefix=prefix,
                snapshot_path=snapshot_path,
            )
            raw_path = raw_success / (
                f"episode_{episode_index:06d}_seed_{int(prefix['seed']):06d}.npz"
            )
            _save_episode(
                raw_path, episode_index=episode_index,
                seed=int(prefix["seed"]), source_kind="scripted_nominal",
                arrays=arrays, result=result, start_snapshot=snapshot_path,
                start_metadata=start,
            )
            result["raw_episode"] = raw_path
            result["start_snapshot"] = snapshot_path
            results.append(result)
            start_rows.append(start)
            raw_paths.append(raw_path)
            episode_index += 1
        if args.staged_successes:
            reach = ReachPolicy(args.reach_checkpoint)
            approach = ApproachPolicy(
                args.approach_checkpoint,
                max_cube_displacement_m=float(
                    config["grasp"]["max_approach_cube_displacement_m"]
                ),
            )
            recovery = RecoveryPolicy(
                args.recovery_checkpoint,
                max_cube_displacement_m=float(
                    config["grasp"]["max_approach_cube_displacement_m"]
                ),
            )
            for policy in (reach, approach, recovery):
                _set_deterministic(policy, args.inference_seed)
            staged_successes = 0
            for rollout_index in range(args.max_staged_attempts):
                seed = args.seed_start + rollout_index
                upstream = run_frozen_upstream(
                    seed=seed, rollout_index=rollout_index,
                    robot=robot, config=config,
                    reach_policy=reach, approach_policy=approach,
                    recovery_policy=recovery, output=upstream_output,
                    router_spec=router,
                )
                if not upstream["new_success"]:
                    results.append({
                        "success": False,
                        "source_kind": "staged_handoff",
                        "seed": seed,
                        "failure_stage": "upstream",
                        "upstream": upstream,
                    })
                    print(
                        f"staged attempt={rollout_index + 1} seed={seed} "
                        f"upstream={upstream['new_outcome']} "
                        f"expert_successes={staged_successes}/{args.staged_successes}",
                        flush=True,
                    )
                    continue
                task = ReachGraspLiftTask(robot, config)
                initialize_task_from_handoff(task)
                observation = robot.get_observation()
                snapshot = capture_simulator_snapshot(
                    robot, task, observation=observation
                )
                snapshot_path = starts_staged / (
                    f"start_rollout_{rollout_index:06d}_seed_{seed:06d}.npz"
                )
                restore_check = restore_simulator_snapshot(robot, task, snapshot)
                if not restore_check["passed"]:
                    raise RuntimeError(
                        f"staged snapshot restore failed: {restore_check}"
                    )
                _save_snapshot(
                    snapshot_path, snapshot,
                    source_kind="staged_handoff",
                    source_seed=seed,
                    source_rollout_index=rollout_index,
                    upstream=upstream,
                    restore_check=restore_check,
                )
                start = {
                    "snapshot": snapshot_path,
                    "source_kind": "staged_handoff",
                    "source_seed": seed,
                    "source_rollout_index": rollout_index,
                    "upstream": upstream,
                    "restore_check": restore_check,
                }
                result, arrays = run_scripted_grasp_preload(
                    task, seed=seed, source_kind="staged_handoff",
                    terminal_hold_frames=args.terminal_hold_frames,
                )
                result["upstream"] = upstream
                result["start_snapshot"] = snapshot_path
                start_rows.append(start)
                if result["success"]:
                    raw_path = raw_success / (
                        f"episode_{episode_index:06d}_seed_{seed:06d}.npz"
                    )
                    _save_episode(
                        raw_path, episode_index=episode_index, seed=seed,
                        source_kind="staged_handoff", arrays=arrays,
                        result=result, start_snapshot=snapshot_path,
                        start_metadata=start,
                    )
                    result["raw_episode"] = raw_path
                    raw_paths.append(raw_path)
                    episode_index += 1
                    staged_successes += 1
                else:
                    diagnostic_path = raw_diagnostics / (
                        f"failed_seed_{seed:06d}.npz"
                    )
                    np.savez_compressed(
                        diagnostic_path, **arrays,
                        metadata_json=np.asarray(json.dumps(_plain(result))),
                    )
                    result["diagnostic_episode"] = diagnostic_path
                results.append(result)
                print(
                    f"staged attempt={rollout_index + 1} seed={seed} "
                    f"expert={result['success']} "
                    f"expert_successes={staged_successes}/{args.staged_successes}",
                    flush=True,
                )
                if staged_successes >= args.staged_successes:
                    break
    finally:
        robot.disconnect()

    staged_raw = []
    for path in raw_paths:
        with np.load(path, allow_pickle=False) as episode:
            if str(episode["source_kind"]) == "staged_handoff":
                staged_raw.append(path)
    if len(staged_raw) < args.staged_successes:
        raise RuntimeError(
            f"collected only {len(staged_raw)}/{args.staged_successes} "
            "successful staged expert episodes"
        )
    preload = _preload_statistics(raw_paths)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    native = _export_native(
        raw_paths=raw_paths,
        output=args.output / "lerobot_dataset",
        cache_dir=args.cache_dir,
        repo_id=args.repo_id,
    )
    successful_results = [item for item in results if item.get("success")]
    failure_distribution: dict[str, int] = {}
    for item in results:
        if item.get("success"):
            continue
        stage = str(item.get("failure_stage") or "unknown")
        failure_distribution[stage] = failure_distribution.get(stage, 0) + 1
    summary = {
        "task": TASK,
        "nominal_source_episodes": len(nominal_sources),
        "successful_episodes": len(raw_paths),
        "successful_frames": int(sum(item["frames"] for item in successful_results)),
        "source_distribution": {
            "scripted_nominal": len(raw_paths) - len(staged_raw),
            "staged_handoff": len(staged_raw),
        },
        "staged_attempts": sum(
            item.get("source_kind") == "staged_handoff" for item in results
        ),
        "failure_distribution": failure_distribution,
        "action_semantics": "27D absolute MuJoCo position-controller target",
        "policy_observation": [
            "front RGB 240x320", "wrist RGB 240x320", "27D actual qpos"
        ],
        "diagnostics_excluded_from_policy_observation": True,
        "unsupported_lift_executed": False,
        "preload_statistics": preload,
        "native_dataset": native,
        "start_snapshots": start_rows,
        "results": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_plain({
        key: summary[key] for key in (
            "successful_episodes", "successful_frames",
            "source_distribution", "staged_attempts",
            "failure_distribution", "preload_statistics",
            "native_dataset",
        )
    }), indent=2))


if __name__ == "__main__":
    main()
