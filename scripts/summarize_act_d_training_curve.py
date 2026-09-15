"""Summarize checkpoint-wise closed-loop and offline ACT diagnostics.

The report keeps the configured Reach criterion separate from the position-only
12 mm dwell diagnostic.  This distinction is important: entering the position
basin is not a valid Reach unless the orientation gate is also held for five
consecutive frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


CLIP_EPS_RAD = 1e-9
STEP_PATTERN = re.compile(
    r"step:(?P<step>[0-9.]+[KMG]?).*?loss:(?P<loss>[0-9.]+).*?"
    r"l1_loss:(?P<l1>[0-9.]+).*?kld_loss:(?P<kld>[0-9.]+)"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: np.ndarray, *, scale: float = 1.0) -> dict[str, float]:
    values = np.asarray(values, dtype=float) * scale
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _motion_stats(deltas: np.ndarray) -> dict[str, Any]:
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        width = int(deltas.shape[1]) if deltas.ndim == 2 else 7
        return {
            "transitions": 0,
            "mean_l2_delta_rad": 0.0,
            "median_l2_delta_rad": 0.0,
            "max_l2_delta_rad": 0.0,
            "mean_l2_second_delta_rad": 0.0,
            "max_l2_second_delta_rad": 0.0,
            "direction_reversal_fraction": 0.0,
            "direction_reversal_pairs": 0,
            "valid_direction_pairs": 0,
            "per_joint_mean_abs_delta_rad": [0.0] * width,
            "per_joint_max_abs_delta_rad": [0.0] * width,
        }
    if deltas.ndim != 2:
        raise ValueError("arm target deltas must be a 2-D array")
    norms = np.linalg.norm(deltas, axis=1)
    second = np.diff(deltas, axis=0)
    second_norms = np.linalg.norm(second, axis=1)
    if len(deltas) > 1:
        valid = (norms[:-1] > 1e-6) & (norms[1:] > 1e-6)
        reversals = valid & (np.sum(deltas[:-1] * deltas[1:], axis=1) < 0.0)
        reversal_count = int(np.count_nonzero(reversals))
        valid_count = int(np.count_nonzero(valid))
    else:
        reversal_count = valid_count = 0
    return {
        "transitions": int(len(deltas)),
        "mean_l2_delta_rad": float(norms.mean()),
        "median_l2_delta_rad": float(np.median(norms)),
        "max_l2_delta_rad": float(norms.max()),
        "mean_l2_second_delta_rad": (
            float(second_norms.mean()) if len(second_norms) else 0.0
        ),
        "max_l2_second_delta_rad": (
            float(second_norms.max()) if len(second_norms) else 0.0
        ),
        "direction_reversal_fraction": (
            reversal_count / valid_count if valid_count else 0.0
        ),
        "direction_reversal_pairs": reversal_count,
        "valid_direction_pairs": valid_count,
        "per_joint_mean_abs_delta_rad": np.mean(np.abs(deltas), axis=0).tolist(),
        "per_joint_max_abs_delta_rad": np.max(np.abs(deltas), axis=0).tolist(),
    }


def _aggregate_motion_stats(segments: list[np.ndarray]) -> dict[str, Any]:
    """Aggregate motion without creating artificial cross-episode transitions."""
    clean = [np.asarray(segment, dtype=float) for segment in segments if len(segment)]
    if not clean:
        return _motion_stats(np.empty((0, 7)))
    deltas = np.concatenate(clean, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    second_parts = [np.diff(segment, axis=0) for segment in clean if len(segment) > 1]
    second = (
        np.concatenate(second_parts, axis=0)
        if second_parts else np.empty((0, deltas.shape[1]))
    )
    second_norms = np.linalg.norm(second, axis=1)
    reversal_count = 0
    valid_count = 0
    for segment in clean:
        if len(segment) < 2:
            continue
        segment_norms = np.linalg.norm(segment, axis=1)
        valid = (segment_norms[:-1] > 1e-6) & (segment_norms[1:] > 1e-6)
        reversals = valid & (np.sum(segment[:-1] * segment[1:], axis=1) < 0.0)
        reversal_count += int(np.count_nonzero(reversals))
        valid_count += int(np.count_nonzero(valid))
    return {
        "transitions": int(len(deltas)),
        "mean_l2_delta_rad": float(norms.mean()),
        "median_l2_delta_rad": float(np.median(norms)),
        "max_l2_delta_rad": float(norms.max()),
        "mean_l2_second_delta_rad": (
            float(second_norms.mean()) if len(second_norms) else 0.0
        ),
        "max_l2_second_delta_rad": (
            float(second_norms.max()) if len(second_norms) else 0.0
        ),
        "direction_reversal_fraction": (
            reversal_count / valid_count if valid_count else 0.0
        ),
        "direction_reversal_pairs": reversal_count,
        "valid_direction_pairs": valid_count,
        "per_joint_mean_abs_delta_rad": np.mean(np.abs(deltas), axis=0).tolist(),
        "per_joint_max_abs_delta_rad": np.max(np.abs(deltas), axis=0).tolist(),
    }


def _clip_stats(values: np.ndarray) -> dict[str, float | int]:
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > CLIP_EPS_RAD]
    return {
        "count": int(positive.size),
        "sum_abs_rad": float(positive.sum()) if positive.size else 0.0,
        "mean_abs_when_clipped_rad": float(positive.mean()) if positive.size else 0.0,
        "p95_abs_when_clipped_rad": (
            float(np.quantile(positive, 0.95)) if positive.size else 0.0
        ),
        "max_abs_rad": float(positive.max()) if positive.size else 0.0,
    }


def _load_joint_names(raw_dir: Path) -> list[str]:
    path = next(iter(sorted(raw_dir.glob("*.npz"))), None)
    if path is None:
        raise FileNotFoundError(f"no demonstration npz in {raw_dir}")
    with np.load(path, allow_pickle=False) as episode:
        return [str(value) for value in episode["joint_names"]]


def _offline_diagnostics(offline_path: Path, reset_path: Path) -> dict[str, Any]:
    offline = _json(offline_path)
    reset = _json(reset_path)
    reach = offline["phasewise"]["Reach"]["horizons"]["1"]
    approach = offline["phasewise"]["Approach"]["horizons"]["1"]
    paired = reset["policy_conditions"]["paired"]
    image_conditioned = reset["policy_conditions"]["vary_images_fixed_seed0_state"]
    front_conditioned = reset["policy_conditions"][
        "vary_front_fixed_seed0_state_and_wrist"
    ]
    wrist_conditioned = reset["policy_conditions"][
        "vary_wrist_fixed_seed0_state_and_front"
    ]
    paired_stats = paired["policy_action_statistics"]["rms_spread_rad"]
    expert_stats = reset["expert_frame0_action"]["statistics"]["rms_spread_rad"]
    return {
        "offline_h1_mae_rad": {
            "reach_arm": float(reach["arm_mae_rad"]),
            "reach_hand": float(reach["hand_mae_rad"]),
            "approach_arm": float(approach["arm_mae_rad"]),
            "approach_hand": float(approach["hand_mae_rad"]),
        },
        "frame0": {
            "paired_arm_action_mae_rad": float(
                paired["mae_vs_corresponding_expert_frame0_action"]
                ["group_mae_rad"]["arm_7d"]
            ),
            "paired_arm_action_spread_rad": float(paired_stats["arm_7d"]),
            "expert_arm_action_spread_rad": float(expert_stats["arm_7d"]),
            "paired_arm_action_spread_ratio_vs_expert": float(
                paired["policy_action_statistics"]
                ["rms_spread_ratio_vs_expert"]["arm_7d"]
            ),
            "image_conditioning_arm_spread_rad": float(
                image_conditioned["policy_action_statistics"]
                ["rms_spread_rad"]["arm_7d"]
            ),
            "image_conditioning_arm_spread_ratio_vs_expert": float(
                image_conditioned["policy_action_statistics"]
                ["rms_spread_ratio_vs_expert"]["arm_7d"]
            ),
            "front_only_arm_spread_ratio_vs_expert": float(
                front_conditioned["policy_action_statistics"]
                ["rms_spread_ratio_vs_expert"]["arm_7d"]
            ),
            "wrist_only_arm_spread_ratio_vs_expert": float(
                wrist_conditioned["policy_action_statistics"]
                ["rms_spread_ratio_vs_expert"]["arm_7d"]
            ),
        },
        "padding_and_boundaries_valid": bool(
            offline["boundary_and_roundtrip_checks"]
            ["padding_mask_matches_manual_episode_boundary"]
            and not offline["boundary_and_roundtrip_checks"]
            ["cross_episode_targets_in_mae"]
        ),
    }


def _training_loss(step: int, log_path: Path | None) -> dict[str, float] | None:
    if log_path is None:
        return None
    matches = list(STEP_PATTERN.finditer(
        log_path.read_text(encoding="utf-8", errors="replace")
    ))
    match_for_step = None
    for match in matches:
        token = match.group("step")
        multiplier = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}.get(
            token[-1], 1
        )
        number = token[:-1] if multiplier != 1 else token
        if int(float(number) * multiplier) == step:
            match_for_step = match
    if match_for_step is None:
        raise ValueError(f"step {step} not found in {log_path}")
    return {
        "loss": float(match_for_step.group("loss")),
        "l1_loss": float(match_for_step.group("l1")),
        "kld_loss": float(match_for_step.group("kld")),
    }


def _summarize_checkpoint(
    *,
    step: int,
    rollout_path: Path,
    offline_path: Path,
    reset_path: Path,
    raw_dir: Path,
    log_path: Path | None,
) -> dict[str, Any]:
    rollout = _json(rollout_path)
    episodes = rollout["episodes"]
    rollout_dir = rollout_path.parent
    joint_names = _load_joint_names(raw_dir)
    pregrasp = np.asarray([row["minimum_pregrasp_error_m"] for row in episodes])
    approach = np.asarray([row["minimum_approach_error_m"] for row in episodes])
    displacement = np.asarray([row["maximum_cube_displacement_m"] for row in episodes])
    dwell_frames: list[int] = []
    dwell_longest: list[int] = []
    full_gate_frames: list[int] = []
    full_gate_longest: list[int] = []
    all_clips: list[np.ndarray] = []
    overall_deltas: list[np.ndarray] = []
    inside_delta_segments: list[np.ndarray] = []
    closest_deltas: list[np.ndarray] = []
    early_contacts: list[bool] = []
    per_seed = []

    for row in episodes:
        seed = int(row["seed"])
        with np.load(
            rollout_dir / f"rollout_seed_{seed:06d}.npz", allow_pickle=False
        ) as data:
            predicted = np.asarray(data["predicted_action"], dtype=float)
            sent = np.asarray(data["sent_action"], dtype=float)
            reach_error = np.asarray(data["reach_error_m"], dtype=float)
            orientation = np.asarray(data["palm_orientation_error_deg"], dtype=float)
            contact_count = np.asarray(data["contact_count"], dtype=int)
        deltas = np.diff(predicted[:, :7], axis=0)
        inside = reach_error <= 0.012
        full_gate = inside & (orientation <= 2.0)
        consecutive_inside = inside[:-1] & inside[1:]
        closest = int(np.argmin(reach_error))
        window_start = max(0, closest - 10)
        window_end = min(len(reach_error), closest + 11)
        all_clips.append(np.abs(predicted - sent))
        overall_deltas.append(deltas)
        inside_transition_indices = np.flatnonzero(consecutive_inside)
        if len(inside_transition_indices):
            split_points = np.flatnonzero(np.diff(inside_transition_indices) > 1) + 1
            for indices in np.split(inside_transition_indices, split_points):
                inside_delta_segments.append(deltas[indices])
        closest_deltas.append(deltas[window_start:max(window_start, window_end - 1)])
        local_dwell_longest = 0
        local_full_longest = 0
        current_dwell = 0
        current_full = 0
        for inside_value, full_value in zip(inside, full_gate):
            current_dwell = current_dwell + 1 if inside_value else 0
            current_full = current_full + 1 if full_value else 0
            local_dwell_longest = max(local_dwell_longest, current_dwell)
            local_full_longest = max(local_full_longest, current_full)
        dwell_frames.append(int(np.count_nonzero(inside)))
        dwell_longest.append(local_dwell_longest)
        full_gate_frames.append(int(np.count_nonzero(full_gate)))
        full_gate_longest.append(local_full_longest)
        contact_indices = np.flatnonzero(contact_count > 0)
        first_contact = int(contact_indices[0]) if len(contact_indices) else None
        approach_frame = row.get("approach_frame")
        approach_end = (
            None if approach_frame is None else int(approach_frame)
            + int(row["milestone_semantics"]["approach_hold_frames"])
        )
        early_contacts.append(bool(
            first_contact is not None
            and (approach_end is None or first_contact < approach_end)
        ))
        inside_indices = np.flatnonzero(inside)
        per_seed.append({
            "seed": seed,
            "valid_reach": bool(row["reach_success"]),
            "minimum_pregrasp_error_mm": float(reach_error[closest] * 1000),
            "closest_approach_frame": closest,
            "frames_inside_12mm_gate": int(np.count_nonzero(inside)),
            "longest_consecutive_frames_inside_12mm": local_dwell_longest,
            "first_frame_enter_12mm": (
                int(inside_indices[0]) if len(inside_indices) else None
            ),
            "last_frame_inside_12mm": (
                int(inside_indices[-1]) if len(inside_indices) else None
            ),
            "orientation_error_at_closest_approach_deg": float(orientation[closest]),
            "pregrasp_error_trajectory_around_closest": [
                {"frame": int(frame), "error_m": float(reach_error[frame])}
                for frame in range(window_start, window_end)
            ],
            "arm_target_delta_trajectory_around_closest": [
                {
                    "from_frame": int(frame),
                    "to_frame": int(frame + 1),
                    "l2_delta_rad": float(np.linalg.norm(deltas[frame])),
                }
                for frame in range(window_start, window_end - 1)
            ],
        })

    clips = np.concatenate(all_clips, axis=0)
    dwell_frames_array = np.asarray(dwell_frames, dtype=int)
    dwell_longest_array = np.asarray(dwell_longest, dtype=int)
    full_gate_frames_array = np.asarray(full_gate_frames, dtype=int)
    full_gate_longest_array = np.asarray(full_gate_longest, dtype=int)
    per_joint = [
        {
            "joint_index": index,
            "joint_name": joint_names[index],
            **_clip_stats(clips[:, index]),
        }
        for index in range(clips.shape[1])
    ]
    per_joint.sort(key=lambda row: (-int(row["count"]), -float(row["sum_abs_rad"])))
    return {
        "step": step,
        "checkpoint": rollout["checkpoint"],
        "training_loss_at_checkpoint": _training_loss(step, log_path),
        "rollout": {
            "episodes": len(episodes),
            "valid_reach": int(sum(bool(row["reach_success"]) for row in episodes)),
            "valid_approach": int(sum(bool(row["approach_success"]) for row in episodes)),
            "task_success": int(sum(bool(row["task_success"]) for row in episodes)),
            "minimum_pregrasp_error_mm": _stats(pregrasp, scale=1000),
            "minimum_approach_error_mm": _stats(approach, scale=1000),
            "reach_dwell_position_only_12mm": {
                "episodes_entering": int(np.count_nonzero(dwell_frames_array > 0)),
                "episodes_with_at_least_5_consecutive_frames": int(
                    np.count_nonzero(dwell_longest_array >= 5)
                ),
                "frames_per_episode": _stats(dwell_frames_array),
                "longest_consecutive_frames_per_episode": _stats(dwell_longest_array),
            },
            "reach_dwell_position_and_orientation": {
                "orientation_tolerance_deg": 2.0,
                "episodes_entering": int(np.count_nonzero(full_gate_frames_array > 0)),
                "episodes_with_at_least_5_consecutive_frames": int(
                    np.count_nonzero(full_gate_longest_array >= 5)
                ),
                "frames_per_episode": _stats(full_gate_frames_array),
                "longest_consecutive_frames_per_episode": _stats(full_gate_longest_array),
            },
            "cube_displacement_mm": {
                **_stats(displacement, scale=1000),
                "episodes_over_25mm": int(np.count_nonzero(displacement > 0.025)),
            },
            "early_contact_before_valid_approach": int(sum(early_contacts)),
            "any_contact": int(sum(bool(row["contact_fingers_seen"]) for row in episodes)),
            "at_least_two_simultaneous_contacts": int(sum(
                int(row["max_simultaneous_contacts"]) >= 2 for row in episodes
            )),
            "per_seed": per_seed,
        },
        "arm_target_motion": {
            "source": "predicted absolute OpenArm target dimensions 0:7",
            "overall": _aggregate_motion_stats(overall_deltas),
            "while_consecutively_inside_12mm": _aggregate_motion_stats(
                inside_delta_segments
            ),
            "within_plus_minus_10_frames_of_closest_approach": (
                _aggregate_motion_stats(closest_deltas)
            ),
        },
        "action_clipping": {
            "threshold_rad": CLIP_EPS_RAD,
            "global": _clip_stats(clips),
            "per_joint_sorted_by_count": per_joint,
        },
        "diagnostics": _offline_diagnostics(offline_path, reset_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", nargs=6, required=True,
        metavar=("STEP", "ROLLOUT", "OFFLINE", "RESET", "LOG", "SAMPLING_REPORT"),
    )
    args = parser.parse_args()
    checkpoints = []
    sampling_reports: dict[str, Any] = {}
    for values in args.checkpoint:
        step_text, rollout, offline, reset, log, sampling = values
        step = int(step_text)
        checkpoints.append(_summarize_checkpoint(
            step=step,
            rollout_path=Path(rollout),
            offline_path=Path(offline),
            reset_path=Path(reset),
            raw_dir=args.raw_dir,
            log_path=Path(log),
        ))
        sampling_reports[str(step)] = _json(Path(sampling))
    checkpoints.sort(key=lambda row: int(row["step"]))
    result = {
        "schema_version": 1,
        "experiment": "ACT best config D exact-resume training curve",
        "fixed_configuration": {
            "demonstrations": 20,
            "frames": 2804,
            "state_dim": 27,
            "action_dim": 27,
            "cameras": ["front 240x320 RGB", "wrist 240x320 RGB"],
            "backbone": "ResNet18 ImageNet initialization",
            "sampling": "phase-balanced",
            "chunk_size": 20,
            "n_action_steps": 20,
            "batch_size": 8,
            "optimizer": "AdamW",
            "learning_rate": 1e-5,
            "training_seed": 1000,
            "execution_horizon": 1,
            "evaluation_seeds": list(range(20)),
        },
        "reach_semantics": {
            "valid_reach": "position <= 12 mm AND orientation <= 2 deg for 5 consecutive frames",
            "dwell_diagnostic": "position <= 12 mm only; not itself a valid Reach",
        },
        "checkpoints": checkpoints,
        "phase_sampling_reports": sampling_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "steps": [row["step"] for row in checkpoints],
    }, indent=2))


if __name__ == "__main__":
    main()
