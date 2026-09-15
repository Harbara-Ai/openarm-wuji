"""Evaluate frozen staged ACT policies followed by the existing scripted Lift."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.policy import ApproachPolicy, ReachPolicy, RecoveryPolicy
from openarm_wuji.policy.grasp_secure_controller import GraspSecurePolicy
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.simulation.snapshot import capture_simulator_snapshot
from openarm_wuji.tasks import (
    ReachGraspLiftTask,
    cliffs_delta,
    initialize_task_from_handoff,
    run_scripted_lift_from_handoff,
)
from scripts.evaluate_grasp_secure_act import (
    _run_grasp,
    _set_grasp_deterministic,
)
from scripts.evaluate_staged_with_recovery import (
    _run_episode as run_frozen_upstream,
    _set_deterministic,
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _save_snapshot(path: Path, snapshot: dict[str, np.ndarray],
                   metadata: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        **snapshot,
        metadata_json=np.asarray(json.dumps(_plain(metadata))),
    )


def _map_upstream_failure(upstream: dict[str, Any]) -> str:
    if not upstream["reach_success"]:
        return "reach"
    if upstream.get("recovery_attempted"):
        return "recovery"
    return "approach"


def _map_grasp_failure(grasp: dict[str, Any]) -> str:
    return {
        "contact_acquisition": "grasp_formation",
        "grasp_formation": "grasp_formation",
        "preload_formation": "preload_formation",
        "terminal_hold": "terminal_hold",
    }.get(str(grasp.get("failure_stage")), "grasp_formation")


def run_one(*, seed: int, rollout_index: int, group: str, robot,
            config: dict[str, Any], reach, approach, recovery, grasp,
            router: dict[str, Any], preload_gate_rad: float,
            grasp_timeout_frames: int, output: Path,
            grasp_gate_mode: str = "required",
            frame_callback=None) -> dict[str, Any]:
    """Run one real closed-loop episode with no resets after stage handoff."""
    if grasp_gate_mode not in {"required", "diagnostic_only"}:
        raise ValueError(f"unsupported grasp gate mode: {grasp_gate_mode}")
    upstream_dir = output / group / "upstream"
    grasp_dir = output / group / "grasp"
    lift_dir = output / group / "lift"
    terminal_dir = output / group / "graspsecure_terminal_states"
    for directory in (upstream_dir, grasp_dir, lift_dir, terminal_dir):
        directory.mkdir(parents=True, exist_ok=True)

    upstream = run_frozen_upstream(
        seed=seed,
        rollout_index=rollout_index,
        robot=robot,
        config=config,
        reach_policy=reach,
        approach_policy=approach,
        recovery_policy=recovery,
        output=upstream_dir,
        router_spec=router,
        frame_callback=frame_callback,
    )
    row: dict[str, Any] = {
        "group": group,
        "rollout_index": rollout_index,
        "seed": seed,
        "reach_success": bool(upstream["reach_success"]),
        "approach_attempted": bool(upstream["approach_attempted"]),
        "approach_stage_success": bool(upstream["new_success"]),
        "recovery_attempted": bool(upstream.get("recovery_attempted", False)),
        "recovery_success": bool(upstream.get("recovery_success", False)),
        "graspsecure_attempted": False,
        "graspsecure_success": False,
        "graspsecure_gate_mode": grasp_gate_mode,
        "graspsecure_terminal_valid": False,
        "prelift_safety_pass": False,
        "lift_attempted": False,
        "lift_success": False,
        "full_task_success": False,
        "failure_stage": None,
        "failure_reason": None,
        "upstream": upstream,
        "expert_action_supplied_to_policy": False,
        "state_reset_between_stages": False,
    }
    if not upstream["new_success"]:
        row["failure_stage"] = _map_upstream_failure(upstream)
        row["failure_reason"] = upstream["new_outcome"]
        return row

    # The upstream runner leaves MuJoCo at its actual terminal state.  This
    # task object only initializes Python-side bookkeeping around that state.
    task = ReachGraspLiftTask(robot, config)
    initialize_task_from_handoff(task)
    row["graspsecure_attempted"] = True
    grasp_metrics = _run_grasp(
        robot=robot,
        task=task,
        policy=grasp,
        seed=seed,
        source_kind="live_staged_handoff",
        timeout_frames=grasp_timeout_frames,
        preload_gate_rad=preload_gate_rad,
        output=grasp_dir,
        episode_label=f"rollout_{rollout_index:06d}_seed_{seed:06d}",
        save_images=False,
        frame_callback=frame_callback,
    )
    row["graspsecure"] = grasp_metrics
    row["graspsecure_success"] = bool(grasp_metrics["grasp_preload_success"])
    row["graspsecure_gate_pass"] = row["graspsecure_success"]
    terminal_values = np.r_[
        robot.data.qpos, robot.data.qvel, robot.data.ctrl,
        task.task_telemetry()["cube_position_m"],
    ]
    row["graspsecure_terminal_valid"] = bool(np.isfinite(terminal_values).all())
    if not row["graspsecure_terminal_valid"]:
        row["failure_stage"] = "grasp_formation"
        row["failure_reason"] = "nonfinite_graspsecure_terminal"
        return row
    if grasp_gate_mode == "required" and not row["graspsecure_success"]:
        row["failure_stage"] = _map_grasp_failure(grasp_metrics)
        row["failure_reason"] = grasp_metrics["failure_stage"]
        return row

    safety_reference = float(config["grasp"]["max_approach_cube_displacement_m"])
    upstream_motion = float(upstream["new"]["maximum_cube_displacement_m"])
    grasp_motion = float(grasp_metrics["maximum_cube_displacement_m"])
    prelift_motion = max(upstream_motion, grasp_motion)
    row["prelift_cube_motion_m"] = prelift_motion
    row["prelift_cube_motion_components_m"] = {
        "upstream": upstream_motion,
        "graspsecure": grasp_motion,
    }
    row["prelift_safety_reference_m"] = safety_reference
    row["prelift_safety_pass"] = bool(prelift_motion <= safety_reference)
    row["maximum_cube_displacement_m"] = prelift_motion
    if not row["prelift_safety_pass"]:
        row["failure_stage"] = "terminal_hold"
        row["failure_reason"] = "prelift_cube_motion_safety"
        return row

    terminal_observation = robot.get_observation()
    snapshot = capture_simulator_snapshot(
        robot, task, observation=terminal_observation
    )
    terminal_path = terminal_dir / (
        f"terminal_{rollout_index:06d}_seed_{seed:06d}.npz"
    )
    _save_snapshot(terminal_path, snapshot, {
        "seed": seed,
        "rollout_index": rollout_index,
        "group": group,
        "graspsecure": grasp_metrics,
        "controller_target_semantics": "27D absolute position target",
    })
    row["graspsecure_terminal_snapshot"] = terminal_path
    row["lift_attempted"] = True
    lift_metrics, lift_arrays = run_scripted_lift_from_handoff(
        task, seed=seed, terminal_hold_s=1.0,
        frame_callback=frame_callback,
    )
    lift_path = lift_dir / f"lift_{rollout_index:06d}_seed_{seed:06d}.npz"
    np.savez_compressed(
        lift_path,
        **lift_arrays,
        metrics_json=np.asarray(json.dumps(_plain(lift_metrics))),
    )
    row["lift_trajectory"] = lift_path
    lift_cube_positions = np.asarray(lift_arrays["cube_pose_world"][:, :3])
    lift_cube_displacement = float(np.max(np.linalg.norm(
        lift_cube_positions - lift_cube_positions[0], axis=1
    ))) if len(lift_cube_positions) else 0.0
    lift_metrics["maximum_cube_displacement_world_m"] = lift_cube_displacement
    row["lift"] = lift_metrics
    row["lift_success"] = bool(lift_metrics["success"])
    row["full_task_success"] = row["lift_success"]
    if not row["lift_success"]:
        row["failure_stage"] = lift_metrics["failure_stage"]
        row["failure_reason"] = lift_metrics["failure_reason"]
    row["maximum_cube_displacement_m"] = max(
        prelift_motion, lift_cube_displacement
    )
    return row


def _preload_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [row for row in rows if row["lift_attempted"]]
    successful = [
        float(row["lift"]["preload_before_lift_l2_rad"])
        for row in attempts if row["lift_success"]
    ]
    failed = [
        float(row["lift"]["preload_before_lift_l2_rad"])
        for row in attempts if not row["lift_success"]
    ]
    success_distribution = _distribution(successful)
    failure_distribution = _distribution(failed)
    median_difference = None
    if successful and failed:
        median_difference = float(np.median(successful) - np.median(failed))
    return {
        "definition": "L2(controller hand target - actual hand qpos) at Lift handoff",
        "lift_success": success_distribution,
        "lift_failure": failure_distribution,
        "success_minus_failure_median_rad": median_difference,
        "cliffs_delta_success_minus_failure": cliffs_delta(successful, failed),
        "historical_reference_rad": {
            "graspsecure_success_mean": 1.353,
            "graspsecure_failure_mean": 0.726,
        },
    }


def summarize(rows: list[dict[str, Any]], *, group: str,
              requested_rollouts: int | None,
              target_conditional_lifts: int | None,
              seed_start: int) -> dict[str, Any]:
    total = len(rows)
    reach = sum(row["reach_success"] for row in rows)
    approach_attempts = sum(row["approach_attempted"] for row in rows)
    approach = sum(row["approach_stage_success"] for row in rows)
    grasp_attempts = sum(row["graspsecure_attempted"] for row in rows)
    grasp_success = sum(row["graspsecure_success"] for row in rows)
    safe_grasp = sum(row["prelift_safety_pass"] for row in rows)
    lift_attempts = sum(row["lift_attempted"] for row in rows)
    lift_success = sum(row["lift_success"] for row in rows)
    full_success = sum(row["full_task_success"] for row in rows)

    lift_rows = [row for row in rows if row["lift_attempted"]]
    stage_failures = Counter(
        str(row["failure_stage"]) for row in rows
        if row["failure_stage"] is not None
    )
    reasons = Counter(
        str(row["failure_reason"]) for row in rows
        if row["failure_reason"] is not None
    )
    topologies = Counter(
        str(row["lift"]["terminal_contact_topology"])
        for row in lift_rows
    )
    peak_forces = (
        np.asarray([
            row["lift"]["peak_per_finger_normal_force_n"] for row in lift_rows
        ], dtype=float)
        if lift_rows else np.zeros((0, 5), dtype=float)
    )
    product = (
        (reach / total if total else 0.0)
        * (approach / reach if reach else 0.0)
        * (grasp_success / approach if approach else 0.0)
        * (lift_success / grasp_success if grasp_success else 0.0)
    )
    return {
        "group": group,
        "requested_rollouts": requested_rollouts,
        "target_conditional_lift_attempts": target_conditional_lifts,
        "completed_rollouts": total,
        "seed_start": seed_start,
        "seed_end_inclusive": rows[-1]["seed"] if rows else None,
        "counts": {
            "reach_success": reach,
            "approach_attempts": approach_attempts,
            "approach_stage_success": approach,
            "graspsecure_attempts": grasp_attempts,
            "graspsecure_success": grasp_success,
            "prelift_safety_pass": safe_grasp,
            "prelift_safety_reject": grasp_success - safe_grasp,
            "lift_attempts": lift_attempts,
            "lift_success": lift_success,
            "full_task_success": full_success,
        },
        "probability_decomposition": {
            "P_full_success": full_success / total if total else 0.0,
            "P_Reach": reach / total if total else 0.0,
            "P_Approach_given_Reach": approach / reach if reach else 0.0,
            "P_GraspSecure_given_Approach": (
                grasp_success / approach if approach else 0.0
            ),
            # Safety rejects remain failures in this true conditional rate.
            "P_Lift_given_GraspSecure": (
                lift_success / grasp_success if grasp_success else 0.0
            ),
            "P_Lift_given_safe_GraspSecure_attempt": (
                lift_success / lift_attempts if lift_attempts else 0.0
            ),
            "stage_product": product,
        },
        "failure_stage_distribution": dict(stage_failures),
        "failure_reason_distribution": dict(reasons),
        "preload_vs_lift_success": _preload_analysis(rows),
        "lift_diagnostics": {
            "cube_palm_translation_drift_m": _distribution([
                float(row["lift"]["max_cube_palm_translation_drift_m"])
                for row in lift_rows
            ]),
            "cube_palm_rotation_drift_deg": _distribution([
                float(row["lift"]["max_cube_palm_rotation_drift_deg"])
                for row in lift_rows
            ]),
            "drop_count": sum(row["lift"]["drop"] for row in lift_rows),
            "drop_time_s": _distribution([
                float(row["lift"]["drop_time_s"])
                for row in lift_rows if row["lift"]["drop_time_s"] is not None
            ]),
            "terminal_hold_duration_s": _distribution([
                float(row["lift"]["terminal_hold_duration_s"])
                for row in lift_rows
            ]),
            "terminal_contact_topology_distribution": dict(topologies),
            "peak_per_finger_normal_force_n": {
                finger: _distribution(peak_forces[:, index].tolist())
                for index, finger in enumerate(
                    ("thumb", "index", "middle", "ring", "little")
                )
            },
            "maximum_hand_target_change_rad": max((
                float(row["lift"]["maximum_hand_target_change_rad"])
                for row in lift_rows
            ), default=0.0),
        },
        "gate_interpretation": {
            "graspsecure_gate_predictive_value": (
                lift_success / grasp_success if grasp_success else None
            ),
            "contact_topology_used_as_lift_gate": False,
            "force_used_as_lift_gate": False,
            "relative_drift_used_as_lift_gate": False,
            "prelift_cube_motion_safety_reference_m": 0.025,
        },
        "episodes_detail": rows,
    }


def _load_progress(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_plain(row)) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/staged_act_with_scripted_lift",
    )
    parser.add_argument(
        "--mode", choices=("conditional", "full", "both"), default="both"
    )
    parser.add_argument("--conditional-target", type=int, default=30)
    parser.add_argument("--conditional-max-rollouts", type=int, default=200)
    parser.add_argument("--conditional-seed-start", type=int, default=4000)
    parser.add_argument("--full-rollouts", type=int, default=100)
    parser.add_argument("--full-seed-start", type=int, default=5000)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
        "--grasp-config", type=Path, default=root / "configs/grasp_secure_stage.json"
    )
    parser.add_argument(
        "--dataset-summary", type=Path,
        default=root / "outputs/grasp_preload_act/dataset/summary.json",
    )
    args = parser.parse_args()
    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"output exists: {args.output}; use --resume or --overwrite"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    task_config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = json.loads(args.grasp_config.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_summary.read_text(encoding="utf-8"))
    preload_gate = float(
        dataset["preload_statistics"]["evaluation_preload_gate"]
        ["hand_preload_l2_min_rad"]
    )
    paths = {
        key: root / value for key, value in {
            "reach": frozen["frozen_upstream"]["reach_checkpoint"],
            "approach": frozen["frozen_upstream"]["approach_checkpoint"],
            "recovery": frozen["frozen_upstream"]["recovery_checkpoint"],
            "grasp": frozen["checkpoint"],
            "router": frozen["frozen_upstream"]["router"],
        }.items()
    }
    router = json.loads(paths["router"].read_text(encoding="utf-8"))
    reach = ReachPolicy(paths["reach"])
    approach = ApproachPolicy(
        paths["approach"],
        max_cube_displacement_m=float(
            task_config["grasp"]["max_approach_cube_displacement_m"]
        ),
    )
    recovery = RecoveryPolicy(
        paths["recovery"],
        max_cube_displacement_m=float(
            task_config["grasp"]["max_approach_cube_displacement_m"]
        ),
    )
    grasp = GraspSecurePolicy(paths["grasp"])
    for policy in (reach, approach, recovery):
        _set_deterministic(policy, args.inference_seed)
    _set_grasp_deterministic(grasp, args.inference_seed)
    robot = MujocoOpenArmWuji(
        args.model,
        args.synergies,
        arm_side=task_config["arm_side"],
        control_hz=30,
        image_height=240,
        image_width=320,
        front_camera=task_config["scene"]["front_camera_name"],
    )
    robot.connect()
    try:
        groups = []
        if args.mode in ("conditional", "both"):
            groups.append((
                "conditional", args.conditional_seed_start,
                args.conditional_max_rollouts, args.conditional_target,
            ))
        if args.mode in ("full", "both"):
            groups.append((
                "full", args.full_seed_start, args.full_rollouts, None,
            ))
        for group, seed_start, limit, target in groups:
            progress = args.output / group / "progress.jsonl"
            rows = _load_progress(progress) if args.resume else []
            lift_attempts = sum(row["lift_attempted"] for row in rows)
            for index in range(len(rows), limit):
                if target is not None and lift_attempts >= target:
                    break
                seed = seed_start + index
                row = run_one(
                    seed=seed,
                    rollout_index=index,
                    group=group,
                    robot=robot,
                    config=task_config,
                    reach=reach,
                    approach=approach,
                    recovery=recovery,
                    grasp=grasp,
                    router=router,
                    preload_gate_rad=preload_gate,
                    grasp_timeout_frames=int(frozen["gate"]["timeout_frames"]),
                    output=args.output,
                )
                rows.append(row)
                _append_progress(progress, row)
                lift_attempts += int(row["lift_attempted"])
                print(
                    f"{group} {index + 1}/{limit} seed={seed} "
                    f"reach={int(row['reach_success'])} "
                    f"approach={int(row['approach_stage_success'])} "
                    f"grasp={int(row['graspsecure_success'])} "
                    f"safe={int(row['prelift_safety_pass'])} "
                    f"lift={int(row['lift_success'])} "
                    f"conditional={lift_attempts}/{target or '-'}",
                    flush=True,
                )
                partial = summarize(
                    rows,
                    group=group,
                    requested_rollouts=limit if target is None else None,
                    target_conditional_lifts=target,
                    seed_start=seed_start,
                )
                _write_json(args.output / group / "summary.json", partial)
            if target is not None and lift_attempts < target:
                raise RuntimeError(
                    f"conditional run found only {lift_attempts}/{target} "
                    f"safe GraspSecure states in {limit} rollouts"
                )
    finally:
        robot.disconnect()

    manifest = {
        "schema_version": 1,
        "experiment": "frozen staged ACT with existing scripted Lift",
        "policies_trained": False,
        "lift_policy_trained": False,
        "frozen_components": _plain(paths),
        "graspsecure_preload_gate_rad": preload_gate,
        "lift_contract": {
            "trajectory": "existing two-segment quintic minimum-jerk S-curve",
            "trajectory_modified": False,
            "controller_gain_modified": False,
            "hand_target": "exact live GraspSecure 20-D controller target",
            "terminal_hold_s": 1.0,
            "contact_topology_is_diagnostic_only": True,
        },
    }
    _write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
