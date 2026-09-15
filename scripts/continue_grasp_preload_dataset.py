"""Resume staged Grasp+Preload collection and finalize the native dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
from scripts.build_grasp_preload_dataset import (
    _export_native,
    _plain,
    _preload_statistics,
    _save_episode,
    _save_snapshot,
)
from scripts.evaluate_staged_with_recovery import (
    _run_episode as run_frozen_upstream,
    _set_deterministic,
)


def _source_kind(path: Path) -> str:
    with np.load(path, allow_pickle=False) as episode:
        return str(episode["source_kind"])


def _load_result(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as episode:
        metadata = json.loads(str(episode["metadata_json"]))
    result = metadata.get("result", metadata)
    result["raw_episode"] = path
    return result


def _episode_frames(path: Path) -> int:
    with np.load(path, allow_pickle=False) as episode:
        return len(episode["action"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/grasp_preload_act/dataset",
    )
    parser.add_argument("--target-staged-successes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1680)
    parser.add_argument("--max-new-attempts", type=int, default=30)
    parser.add_argument("--prior-staged-attempts", type=int, default=80)
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
    args = parser.parse_args()

    raw_success_dir = args.output / "raw/successful"
    diagnostics_dir = args.output / "raw/diagnostics"
    starts_dir = args.output / "starts/staged"
    upstream_dir = args.output / "upstream_rollouts"
    native_dir = args.output / "lerobot_dataset"
    if native_dir.exists():
        raise FileExistsError(f"native dataset already exists: {native_dir}")
    raw_paths = sorted(raw_success_dir.glob("*.npz"))
    if not raw_paths:
        raise FileNotFoundError("no partial successful Grasp+Preload episodes")
    staged_paths = [path for path in raw_paths if _source_kind(path) == "staged_handoff"]
    initial_staged = len(staged_paths)
    if initial_staged >= args.target_staged_successes:
        raise ValueError("partial output already satisfies the staged target")
    existing_staged_starts = len(list(starts_dir.glob("*.npz")))
    existing_diagnostics = len(list(diagnostics_dir.glob("*.npz")))
    episode_index = len(raw_paths)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    router = json.loads(args.router.read_text(encoding="utf-8"))
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
    robot = MujocoOpenArmWuji(
        args.model, args.synergies,
        arm_side=config["arm_side"], control_hz=30,
        image_height=240, image_width=320,
        front_camera=config["scene"]["front_camera_name"],
    )
    robot.connect()
    new_results = []
    new_staged_successes = 0
    progress_path = args.output / "continuation_progress.jsonl"
    try:
        for index in range(args.max_new_attempts):
            seed = args.seed_start + index
            upstream = run_frozen_upstream(
                seed=seed,
                rollout_index=args.prior_staged_attempts + index,
                robot=robot, config=config,
                reach_policy=reach, approach_policy=approach,
                recovery_policy=recovery, output=upstream_dir,
                router_spec=router,
            )
            if not upstream["new_success"]:
                result = {
                    "success": False, "source_kind": "staged_handoff",
                    "seed": seed, "failure_stage": "upstream",
                    "upstream": upstream,
                }
            else:
                task = ReachGraspLiftTask(robot, config)
                initialize_task_from_handoff(task)
                observation = robot.get_observation()
                snapshot = capture_simulator_snapshot(
                    robot, task, observation=observation
                )
                snapshot_path = starts_dir / (
                    f"start_rollout_{args.prior_staged_attempts + index:06d}_"
                    f"seed_{seed:06d}.npz"
                )
                restore_check = restore_simulator_snapshot(robot, task, snapshot)
                if not restore_check["passed"]:
                    raise RuntimeError(f"snapshot restore failed: {restore_check}")
                _save_snapshot(
                    snapshot_path, snapshot,
                    source_kind="staged_handoff", source_seed=seed,
                    source_rollout_index=args.prior_staged_attempts + index,
                    upstream=upstream, restore_check=restore_check,
                )
                start = {
                    "snapshot": snapshot_path,
                    "source_kind": "staged_handoff",
                    "source_seed": seed,
                    "source_rollout_index": args.prior_staged_attempts + index,
                    "upstream": upstream,
                    "restore_check": restore_check,
                }
                result, arrays = run_scripted_grasp_preload(
                    task, seed=seed, source_kind="staged_handoff",
                    terminal_hold_frames=args.terminal_hold_frames,
                )
                result["upstream"] = upstream
                result["start_snapshot"] = snapshot_path
                if result["success"]:
                    raw_path = raw_success_dir / (
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
                    new_staged_successes += 1
                else:
                    diagnostic_path = diagnostics_dir / (
                        f"failed_seed_{seed:06d}.npz"
                    )
                    np.savez_compressed(
                        diagnostic_path, **arrays,
                        metadata_json=np.asarray(json.dumps(_plain(result))),
                    )
                    result["diagnostic_episode"] = diagnostic_path
            new_results.append(result)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_plain(result)) + "\n")
            total_staged = initial_staged + new_staged_successes
            print(
                f"continuation={index + 1}/{args.max_new_attempts} seed={seed} "
                f"success={result['success']} "
                f"staged={total_staged}/{args.target_staged_successes}",
                flush=True,
            )
            if total_staged >= args.target_staged_successes:
                break
    finally:
        robot.disconnect()

    staged_paths = [path for path in raw_paths if _source_kind(path) == "staged_handoff"]
    if len(staged_paths) < args.target_staged_successes:
        raise RuntimeError(
            f"continuation reached only {len(staged_paths)}/"
            f"{args.target_staged_successes} staged successes"
        )
    preload = _preload_statistics(raw_paths)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    native = _export_native(
        raw_paths=raw_paths, output=native_dir,
        cache_dir=args.cache_dir, repo_id=args.repo_id,
    )
    loaded_successes = [_load_result(path) for path in raw_paths]
    diagnostic_results = []
    for path in sorted(diagnostics_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            diagnostic_results.append(json.loads(str(episode["metadata_json"])))
    failure_distribution = {
        "upstream": (
            args.prior_staged_attempts - existing_staged_starts
            + sum(item.get("failure_stage") == "upstream" for item in new_results)
        ),
    }
    for item in diagnostic_results:
        stage = str(item.get("failure_stage") or "unknown")
        failure_distribution[stage] = failure_distribution.get(stage, 0) + 1
    continuation_attempts = len(new_results)
    summary = {
        "task": (
            "GraspSecure: stable grasp-start to multi-finger grasp, controller "
            "preload, and terminal hold"
        ),
        "successful_episodes": len(raw_paths),
        "successful_frames": int(sum(_episode_frames(path) for path in raw_paths)),
        "source_distribution": {
            "scripted_nominal": len(raw_paths) - len(staged_paths),
            "staged_handoff": len(staged_paths),
        },
        "staged_attempts": args.prior_staged_attempts + continuation_attempts,
        "staged_upstream_success_states": len(list(starts_dir.glob("*.npz"))),
        "failure_distribution": failure_distribution,
        "failure_detail_limitation": (
            "The first 80 upstream failures were not checkpointed by subtype; "
            "their aggregate count is exact. Continuation subtypes are persisted."
        ),
        "action_semantics": "27D absolute MuJoCo position-controller target",
        "policy_observation": [
            "front RGB 240x320", "wrist RGB 240x320", "27D actual qpos"
        ],
        "diagnostics_excluded_from_policy_observation": True,
        "unsupported_lift_executed": False,
        "preload_statistics": preload,
        "native_dataset": native,
        "successful_results": loaded_successes,
        "diagnostic_results": diagnostic_results,
        "continuation_results": new_results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_plain({
        key: summary[key] for key in (
            "successful_episodes", "successful_frames",
            "source_distribution", "staged_attempts",
            "staged_upstream_success_states", "failure_distribution",
            "preload_statistics", "native_dataset",
        )
    }), indent=2))


if __name__ == "__main__":
    main()
