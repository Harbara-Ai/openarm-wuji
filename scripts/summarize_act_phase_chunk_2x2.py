"""Build a machine-readable ACT phase-balancing x chunk-size ablation summary."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


PHASE_ORDER = ("Reach", "Approach", "Grasp", "Preload", "Lift", "Hold")
RAW_PHASE_MAP = {
    "reach": "Reach",
    "approach": "Approach",
    "grasp_close": "Grasp",
    "freeze_synergy": "Grasp",
    "grasp_settle": "Grasp",
    "preload": "Preload",
    "preload_settle": "Preload",
    "lift": "Lift",
    "lift_s_curve": "Lift",
    "hold": "Hold",
}
CLIP_EPS_RAD = 1e-9


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    return float(value)


def _nominal_phase_timeline(raw_dir: Path, frames: int) -> tuple[list[str], list[str]]:
    paths = sorted(raw_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no raw demonstrations in {raw_dir}")
    trajectories: list[list[str]] = []
    joint_names: list[str] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            raw = [str(value) for value in episode["phase"]]
            unknown = sorted(set(raw) - set(RAW_PHASE_MAP))
            if unknown:
                raise ValueError(f"unmapped phases in {path}: {unknown}")
            trajectories.append([RAW_PHASE_MAP[value] for value in raw])
            names = [str(value) for value in episode["joint_names"]]
            if joint_names is None:
                joint_names = names
            elif names != joint_names:
                raise ValueError("joint order differs across raw demonstrations")
    timeline: list[str] = []
    for frame in range(frames):
        labels = [trajectory[frame] for trajectory in trajectories if frame < len(trajectory)]
        if not labels:
            timeline.append("Hold")
            continue
        counts = Counter(labels)
        timeline.append(max(PHASE_ORDER, key=lambda phase: (counts[phase], -PHASE_ORDER.index(phase))))
    assert joint_names is not None
    return timeline, joint_names


def _achieved_gate_stage(episode: dict[str, Any], frames: int) -> list[str]:
    """Label frames by milestones actually achieved, not intended policy phase."""
    semantics = episode["milestone_semantics"]
    reach = episode.get("reach_frame")
    approach = episode.get("approach_frame")
    grasp = episode.get("grasp_frame")
    lift = episode.get("lift_frame")
    reach_end = frames if reach is None else int(reach) + int(semantics["reach_hold_frames"])
    approach_end = frames if approach is None else int(approach) + int(semantics["approach_hold_frames"])
    grasp_end = frames if grasp is None else int(grasp) + int(semantics["grasp_hold_frames"])
    lift_end = frames if lift is None else int(lift) + int(semantics["lift_hold_frames"])
    labels: list[str] = []
    for frame in range(frames):
        if frame < reach_end:
            labels.append("Reach")
        elif frame < approach_end:
            labels.append("Approach")
        elif frame < grasp_end:
            labels.append("Grasp")
        elif frame < lift_end:
            labels.append("Lift")
        else:
            labels.append("Hold")
    return labels


def _magnitude_summary(values: np.ndarray) -> dict[str, float | int]:
    positive = values[values > CLIP_EPS_RAD]
    return {
        "count": int(positive.size),
        "sum_abs_rad": float(positive.sum()) if positive.size else 0.0,
        "mean_abs_when_clipped_rad": float(positive.mean()) if positive.size else 0.0,
        "p95_abs_when_clipped_rad": float(np.quantile(positive, 0.95)) if positive.size else 0.0,
        "max_abs_rad": float(positive.max()) if positive.size else 0.0,
    }


def _group_clipping(values: np.ndarray, labels: list[str], order: tuple[str, ...]) -> dict[str, Any]:
    label_array = np.asarray(labels)
    return {
        label: {
            "frames": int(np.count_nonzero(label_array == label)),
            **_magnitude_summary(values[label_array == label]),
        }
        for label in order
    }


def _rollout_and_clipping(
    summary_path: Path,
    raw_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _json(summary_path)
    episodes = source["episodes"]
    rollout_dir = summary_path.parent
    all_clips: list[np.ndarray] = []
    all_signed_corrections: list[np.ndarray] = []
    all_nominal: list[str] = []
    all_gate: list[str] = []
    joint_names: list[str] | None = None

    for episode in episodes:
        seed = int(episode["seed"])
        with np.load(rollout_dir / f"rollout_seed_{seed:06d}.npz", allow_pickle=False) as data:
            predicted = np.asarray(data["predicted_action"], dtype=np.float64)
            sent = np.asarray(data["sent_action"], dtype=np.float64)
        if predicted.shape != sent.shape or predicted.ndim != 2:
            raise ValueError(f"invalid action arrays for seed {seed}")
        nominal, names = _nominal_phase_timeline(raw_dir, predicted.shape[0])
        if joint_names is None:
            joint_names = names
        clip = np.abs(predicted - sent)
        all_clips.append(clip)
        all_signed_corrections.append(predicted - sent)
        all_nominal.extend(nominal)
        all_gate.extend(_achieved_gate_stage(episode, predicted.shape[0]))

    clips = np.concatenate(all_clips, axis=0)
    signed_corrections = np.concatenate(all_signed_corrections, axis=0)
    assert joint_names is not None
    if clips.shape[1] != len(joint_names):
        raise ValueError("rollout action dimension differs from raw joint names")
    per_joint = []
    for index, name in enumerate(joint_names):
        per_joint.append({
            "joint_index": index,
            "joint_name": name,
            **_magnitude_summary(clips[:, index]),
            "predicted_below_sent_count": int(np.count_nonzero(
                signed_corrections[:, index] < -CLIP_EPS_RAD
            )),
            "predicted_above_sent_count": int(np.count_nonzero(
                signed_corrections[:, index] > CLIP_EPS_RAD
            )),
        })
    per_joint.sort(key=lambda row: (-int(row["count"]), -float(row["sum_abs_rad"])))

    pregrasp = np.asarray([_float(row["minimum_pregrasp_error_m"]) for row in episodes])
    approach = np.asarray([_float(row["minimum_approach_error_m"]) for row in episodes])
    displacement = np.asarray([_float(row["maximum_cube_displacement_m"]) for row in episodes])
    rollout = {
        "episodes": len(episodes),
        "valid_reach": sum(bool(row["reach_success"]) for row in episodes),
        "valid_approach": sum(bool(row["approach_success"]) for row in episodes),
        "task_success": sum(bool(row["task_success"]) for row in episodes),
        "minimum_pregrasp_error_mm": {
            "mean": float(pregrasp.mean() * 1000),
            "median": float(np.median(pregrasp) * 1000),
            "best": float(pregrasp.min() * 1000),
        },
        "minimum_approach_error_mm": {
            "mean": float(approach.mean() * 1000),
            "median": float(np.median(approach) * 1000),
            "best": float(approach.min() * 1000),
        },
        "maximum_cube_displacement_mm": {
            "mean": float(displacement.mean() * 1000),
            "median": float(np.median(displacement) * 1000),
            "maximum": float(displacement.max() * 1000),
            "episodes_over_25mm": int(np.count_nonzero(displacement > 0.025)),
        },
        "any_contact": sum(bool(row["contact_fingers_seen"]) for row in episodes),
        "at_least_two_simultaneous_contacts": sum(
            int(row["max_simultaneous_contacts"]) >= 2 for row in episodes
        ),
        "persistent_multifinger_contact_ge_8_frames": sum(
            int(row["longest_two_contact_frames"]) >= 8 for row in episodes
        ),
        "maximum_simultaneous_contacts": max(
            int(row["max_simultaneous_contacts"]) for row in episodes
        ),
    }
    clipping = {
        "threshold_rad": CLIP_EPS_RAD,
        "global": {
            **_magnitude_summary(clips),
            "predicted_below_sent_count": int(np.count_nonzero(
                signed_corrections < -CLIP_EPS_RAD
            )),
            "predicted_above_sent_count": int(np.count_nonzero(
                signed_corrections > CLIP_EPS_RAD
            )),
        },
        "per_joint_sorted_by_count": per_joint,
        "by_nominal_expert_time_phase": _group_clipping(clips, all_nominal, PHASE_ORDER),
        "by_achieved_gate_stage": _group_clipping(
            clips, all_gate, ("Reach", "Approach", "Grasp", "Lift", "Hold")
        ),
        "phase_semantics": {
            "nominal_expert_time_phase": (
                "Per-frame modal phase of the 20 scripted expert demonstrations; diagnostic only."
            ),
            "achieved_gate_stage": (
                "Post-hoc task milestone stage. If Reach never passes, all frames remain Reach."
            ),
        },
    }
    return rollout, clipping


def _diagnostics(offline_path: Path, reset_path: Path) -> dict[str, Any]:
    offline = _json(offline_path)
    reset = _json(reset_path)
    reach = offline["phasewise"]["Reach"]["horizons"]["1"]
    approach = offline["phasewise"]["Approach"]["horizons"]["1"]
    paired = reset["policy_conditions"]["paired"]
    return {
        "offline_h1_mae_rad": {
            "reach_arm": _float(reach["arm_mae_rad"]),
            "reach_hand": _float(reach["hand_mae_rad"]),
            "approach_arm": _float(approach["arm_mae_rad"]),
            "approach_hand": _float(approach["hand_mae_rad"]),
        },
        "frame0": {
            "paired_arm_action_mae_rad": _float(
                paired["mae_vs_corresponding_expert_frame0_action"]["group_mae_rad"]["arm_7d"]
            ),
            "paired_arm_action_spread_rad": _float(
                paired["policy_action_statistics"]["rms_spread_rad"]["arm_7d"]
            ),
            "expert_arm_action_spread_rad": _float(
                reset["expert_frame0_action"]["statistics"]["rms_spread_rad"]["arm_7d"]
            ),
            "paired_arm_action_spread_ratio_vs_expert": _float(
                paired["policy_action_statistics"]["rms_spread_ratio_vs_expert"]["arm_7d"]
            ),
        },
        "padding_and_boundaries_valid": bool(
            offline["boundary_and_roundtrip_checks"][
                "padding_mask_matches_manual_episode_boundary"
            ]
            and not offline["boundary_and_roundtrip_checks"]["cross_episode_targets_in_mae"]
        ),
    }


def _factorial_effects(experiments: dict[str, Any], path: tuple[str, ...]) -> dict[str, float]:
    def value(name: str) -> float:
        node: Any = experiments[name]
        for key in path:
            node = node[key]
        return float(node)

    a, b, c, d = (value(name) for name in ("A", "B", "C", "D"))
    return {
        "A_normal_chunk100": a,
        "B_normal_chunk20": b,
        "C_phase_chunk100": c,
        "D_phase_chunk20": d,
        "chunk20_minus_chunk100_average": ((b + d) - (a + c)) / 2,
        "phase_minus_normal_average": ((c + d) - (a + b)) / 2,
        "interaction_D_minus_C_minus_B_plus_A": d - c - b + a,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment", action="append", nargs=6,
        metavar=("NAME", "SAMPLING", "CHUNK", "ROLLOUT", "OFFLINE", "RESET"),
        required=True,
    )
    args = parser.parse_args()
    if {row[0] for row in args.experiment} != {"A", "B", "C", "D"}:
        raise ValueError("experiments must be named exactly A, B, C, D")

    experiments: dict[str, Any] = {}
    for name, sampling, chunk, rollout, offline, reset in args.experiment:
        rollout_path, offline_path, reset_path = map(Path, (rollout, offline, reset))
        rollout_metrics, clipping = _rollout_and_clipping(rollout_path, args.raw_dir)
        experiments[name] = {
            "sampling": sampling,
            "chunk_size": int(chunk),
            "sources": {
                "rollout": rollout_path.resolve().as_posix(),
                "offline": offline_path.resolve().as_posix(),
                "reset_conditioning": reset_path.resolve().as_posix(),
            },
            "rollout": rollout_metrics,
            "diagnostics": _diagnostics(offline_path, reset_path),
            "clipping": clipping,
        }

    summary = {
        "schema_version": 1,
        "ablation": "phase balancing x ACT chunk size at step 500",
        "fixed": {
            "demonstrations": 20,
            "frames": 2804,
            "state_dim": 27,
            "action_dim": 27,
            "cameras": ["front", "wrist"],
            "image_shape_hwc": [240, 320, 3],
            "backbone": "ResNet18 ImageNet initialization",
            "batch_size": 8,
            "learning_rate": 1e-5,
            "training_seed": 1000,
            "training_steps": 500,
            "execution_horizon": 1,
            "rollout_seeds": list(range(20)),
        },
        "experiments": experiments,
        "factorial_effects": {
            "valid_reach_count": _factorial_effects(
                experiments, ("rollout", "valid_reach")
            ),
            "mean_minimum_pregrasp_error_mm": _factorial_effects(
                experiments, ("rollout", "minimum_pregrasp_error_mm", "mean")
            ),
            "reach_arm_h1_mae_rad": _factorial_effects(
                experiments, ("diagnostics", "offline_h1_mae_rad", "reach_arm")
            ),
            "mean_maximum_cube_displacement_mm": _factorial_effects(
                experiments, ("rollout", "maximum_cube_displacement_mm", "mean")
            ),
            "clipped_action_values": _factorial_effects(
                experiments, ("clipping", "global", "count")
            ),
            "clipping_sum_abs_rad": _factorial_effects(
                experiments, ("clipping", "global", "sum_abs_rad")
            ),
        },
    }
    best_name = min(
        experiments,
        key=lambda name: (
            -experiments[name]["rollout"]["valid_reach"],
            experiments[name]["diagnostics"]["offline_h1_mae_rad"]["reach_arm"],
            experiments[name]["rollout"]["minimum_pregrasp_error_mm"]["mean"],
            experiments[name]["rollout"]["maximum_cube_displacement_mm"]["mean"],
        ),
    )
    all_zero_reach = all(
        row["rollout"]["valid_reach"] == 0 for row in experiments.values()
    )
    summary["decision"] = {
        "case": (
            "E: none solves Reach; training duration/data coverage is primary suspect"
            if all_zero_reach
            else "not assigned automatically because at least one configuration reached"
        ),
        "best_experiment": best_name,
        "best_config": {
            "sampling": experiments[best_name]["sampling"],
            "chunk_size": experiments[best_name]["chunk_size"],
            "n_action_steps": experiments[best_name]["chunk_size"],
        },
        "recommended_next": (
            "continue best_config to step 2000; evaluate at steps 1000 and 2000"
            if all_zero_reach
            else "continue the successful configuration under the same evaluation gate"
        ),
        "stop_condition": (
            "if step 2000 remains 0/20 Reach or pregrasp plateaus, increase demonstrations"
            if all_zero_reach
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "experiments": {
            name: {
                "reach": row["rollout"]["valid_reach"],
                "pregrasp_mean_mm": row["rollout"]["minimum_pregrasp_error_mm"]["mean"],
                "reach_arm_h1_mae_rad": row["diagnostics"]["offline_h1_mae_rad"]["reach_arm"],
                "clipped_values": row["clipping"]["global"]["count"],
            }
            for name, row in experiments.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
