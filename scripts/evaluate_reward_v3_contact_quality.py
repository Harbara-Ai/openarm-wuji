"""Paired deterministic contact-quality evaluation for Reward V2 and V3.

This script never trains.  It loads two independently trained PCA5 SAC
checkpoints and evaluates both with the same reset seeds.  Contact-quality
telemetry remains outside the policy observation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from openarm_wuji.rl import WujiStaticGraspEnv


FINGERS = tuple(f"finger{i}" for i in range(1, 6))


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(value)]


def _stats(values: list[float]) -> dict[str, float | None]:
    values = _finite(values)
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _max_run(flags: list[bool]) -> int:
    current = maximum = 0
    for flag in flags:
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return maximum


def _first_drop(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    first_three = next(
        (index for index, row in enumerate(rows) if row["contact_count"] >= 3),
        None,
    )
    if first_three is None:
        return None
    established = set(rows[first_three]["contact_fingers"])
    for index in range(first_three + 1, len(rows)):
        current = set(rows[index]["contact_fingers"])
        lost = sorted(established - current)
        if lost:
            before = rows[index - 1]
            after = rows[index]
            return {
                "first_three_step": rows[first_three]["step"],
                "first_three_fingers": sorted(established),
                "drop_step": after["step"],
                "dropped_fingers": lost,
                "contact_fingers_before": before["contact_fingers"],
                "contact_fingers_after": after["contact_fingers"],
                "finger_before": {
                    finger: before["per_finger"][finger] for finger in lost
                },
                "finger_after": {
                    finger: after["per_finger"][finger] for finger in lost
                },
            }
    return None


def evaluate_one(
    model: SAC,
    config: Path,
    root: Path,
    seed: int,
    trace_path: Path,
) -> dict[str, Any]:
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    observation, _ = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    reward_sums: dict[str, float] = {}
    total_return = 0.0
    success = False
    failure_reason = None
    try:
        for step in range(env.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            quality = info["contact_quality"]
            per_finger = quality["per_finger"]
            contact_fingers = [
                finger for finger in FINGERS
                if per_finger[finger]["valid_contact"]
            ]
            for name, value in info["reward_terms"].items():
                reward_sums[name] = reward_sums.get(name, 0.0) + float(value)
            rows.append({
                "step": step + 1,
                "time_s": float((step + 1) * env.control_dt),
                "latent_action": np.asarray(action, dtype=float).tolist(),
                "contact_count": int(quality["valid_contact_count"]),
                "contact_fingers": contact_fingers,
                "interior_contact_count": int(quality["interior_contact_count"]),
                "interior_contact_fraction": float(quality["interior_contact_fraction"]),
                "mean_edge_margin_m": quality["mean_edge_margin_m"],
                "min_edge_margin_m": quality["min_edge_margin_m"],
                "mean_tangential_speed_m_s": float(quality["mean_tangential_speed_m_s"]),
                "max_tangential_speed_m_s": float(quality["max_tangential_speed_m_s"]),
                "persistence_score": float(quality["persistence_score"]),
                "contact_interior_reward": float(quality["contact_interior_reward"]),
                "contact_slip_penalty": float(quality["contact_slip_penalty"]),
                "per_finger": per_finger,
                "relative_linear_speed_m_s": float(info["relative_linear_speed_m_s"]),
                "relative_angular_speed_deg_s": float(info["relative_angular_speed_deg_s"]),
                "window_translation_drift_m": float(info["window_translation_drift_m"]),
                "window_rotation_drift_deg": float(info["window_rotation_drift_deg"]),
                "cube_displacement_m": float(info["cube_displacement_m"]),
                "deepest_penetration_m": float(info["deepest_finger_cube_penetration_m"]),
                "reward": float(reward),
            })
            success = bool(info["static_grasp_success"])
            failure_reason = info["failure_reason"]
            if terminated or truncated:
                break
    finally:
        env.close()

    contact_counts = [row["contact_count"] for row in rows]
    ge2 = [count >= 2 for count in contact_counts]
    ge3 = [count >= 3 for count in contact_counts]
    run2 = _max_run(ge2)
    run3 = _max_run(ge3)
    all_margins: list[float] = []
    all_slips: list[float] = []
    all_regions: list[str] = []
    finger_summary: dict[str, Any] = {}
    for finger in FINGERS:
        valid = [
            row["per_finger"][finger] for row in rows
            if row["per_finger"][finger]["valid_contact"]
        ]
        flags = [row["per_finger"][finger]["valid_contact"] for row in rows]
        margins = [float(item["edge_margin_m"]) for item in valid]
        slips = [float(item["tangential_speed_m_s"]) for item in valid]
        all_margins.extend(margins)
        all_slips.extend(slips)
        all_regions.extend(item["contact_region"] for item in valid)
        finger_summary[finger] = {
            "total_contact_duration_s": float(sum(flags) * env.control_dt),
            "max_contiguous_contact_duration_s": float(_max_run(flags) * env.control_dt),
            "edge_margin_m": _stats(margins),
            "tangential_speed_m_s": _stats(slips),
            "interior_fraction": (
                float(np.mean([item["contact_region"] == "face_interior" for item in valid]))
                if valid else 0.0
            ),
            "max_normal_force_n": (
                max((float(item["normal_force_n"]) for item in valid), default=0.0)
            ),
        }
    frame_thresholds = {
        name: int(np.ceil(seconds * env.control_hz))
        for name, seconds in (("0p1s", 0.1), ("0p3s", 0.3), ("0p5s", 0.5))
    }
    summary = {
        "seed": seed,
        "steps": len(rows),
        "return": float(total_return),
        "any_contact": any(count >= 1 for count in contact_counts),
        "contacts_ge2": any(ge2),
        "contacts_ge3": any(ge3),
        "max_contact_fingers": max(contact_counts, default=0),
        "duration_ge2_s": float(sum(ge2) * env.control_dt),
        "duration_ge3_s": float(sum(ge3) * env.control_dt),
        "max_contiguous_ge2_s": float(run2 * env.control_dt),
        "max_contiguous_ge3_s": float(run3 * env.control_dt),
        "hold": {
            f"ge2_{name}": bool(run2 >= frames)
            for name, frames in frame_thresholds.items()
        } | {
            f"ge3_{name}": bool(run3 >= frames)
            for name, frames in frame_thresholds.items()
        },
        "edge_margin_m": _stats(all_margins),
        "interior_contact_fraction": (
            float(np.mean([region == "face_interior" for region in all_regions]))
            if all_regions else 0.0
        ),
        "contact_regions": {
            region: all_regions.count(region)
            for region in ("face_interior", "edge", "corner")
        },
        "tangential_speed_m_s": _stats(all_slips),
        "per_finger": finger_summary,
        "first_drop_after_three": _first_drop(rows),
        "relative_linear_speed_m_s": _stats([
            row["relative_linear_speed_m_s"] for row in rows
            if row["contact_count"] >= 2
        ]),
        "relative_angular_speed_deg_s": _stats([
            row["relative_angular_speed_deg_s"] for row in rows
            if row["contact_count"] >= 2
        ]),
        "max_window_translation_drift_m": max(_finite([
            row["window_translation_drift_m"] for row in rows
        ]), default=None),
        "max_window_rotation_drift_deg": max(_finite([
            row["window_rotation_drift_deg"] for row in rows
        ]), default=None),
        "stable_grasp_success": success,
        "failure_reason": failure_reason,
        "deepest_penetration_m": max(
            (row["deepest_penetration_m"] for row in rows), default=0.0
        ),
        "penetration_exploit_count": sum(
            row["deepest_penetration_m"] > 0.008 for row in rows
        ),
        "max_cube_displacement_m": max(
            (row["cube_displacement_m"] for row in rows), default=0.0
        ),
        "reward_component_sums": reward_sums,
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps({
        "summary": summary, "timeseries": rows,
    }, indent=2), encoding="utf-8")
    return summary


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    holds = sorted(items[0]["hold"]) if items else []
    all_regions = {
        region: sum(item["contact_regions"][region] for item in items)
        for region in ("face_interior", "edge", "corner")
    }
    total_regions = sum(all_regions.values())
    reward_terms: dict[str, list[float]] = {}
    for item in items:
        for name, value in item["reward_component_sums"].items():
            reward_terms.setdefault(name, []).append(float(value))
    return {
        "n": len(items),
        "counts": {
            "any_contact": sum(item["any_contact"] for item in items),
            "contacts_ge2": sum(item["contacts_ge2"] for item in items),
            "contacts_ge3": sum(item["contacts_ge3"] for item in items),
            "stable_grasp_success": sum(item["stable_grasp_success"] for item in items),
            **{name: sum(item["hold"][name] for item in items) for name in holds},
        },
        "max_contact_fingers": [item["max_contact_fingers"] for item in items],
        "return": _stats([item["return"] for item in items]),
        "max_contiguous_ge2_s": _stats([item["max_contiguous_ge2_s"] for item in items]),
        "max_contiguous_ge3_s": _stats([item["max_contiguous_ge3_s"] for item in items]),
        "edge_margin_m": _stats([
            item["edge_margin_m"]["mean"] for item in items
            if item["edge_margin_m"]["mean"] is not None
        ]),
        "tangential_speed_m_s": _stats([
            item["tangential_speed_m_s"]["mean"] for item in items
            if item["tangential_speed_m_s"]["mean"] is not None
        ]),
        "contact_regions": all_regions,
        "interior_contact_fraction": (
            float(all_regions["face_interior"] / total_regions)
            if total_regions else 0.0
        ),
        "per_finger": {
            finger: {
                "total_contact_duration_s": _stats([
                    item["per_finger"][finger]["total_contact_duration_s"]
                    for item in items
                ]),
                "max_contiguous_contact_duration_s": _stats([
                    item["per_finger"][finger]["max_contiguous_contact_duration_s"]
                    for item in items
                ]),
                "edge_margin_m": _stats([
                    item["per_finger"][finger]["edge_margin_m"]["mean"]
                    for item in items
                    if item["per_finger"][finger]["edge_margin_m"]["mean"] is not None
                ]),
                "tangential_speed_m_s": _stats([
                    item["per_finger"][finger]["tangential_speed_m_s"]["mean"]
                    for item in items
                    if item["per_finger"][finger]["tangential_speed_m_s"]["mean"] is not None
                ]),
            } for finger in FINGERS
        },
        "relative_linear_speed_m_s": _stats([
            item["relative_linear_speed_m_s"]["mean"] for item in items
            if item["relative_linear_speed_m_s"]["mean"] is not None
        ]),
        "relative_angular_speed_deg_s": _stats([
            item["relative_angular_speed_deg_s"]["mean"] for item in items
            if item["relative_angular_speed_deg_s"]["mean"] is not None
        ]),
        "max_window_translation_drift_m": _stats([
            item["max_window_translation_drift_m"] for item in items
            if item["max_window_translation_drift_m"] is not None
        ]),
        "max_window_rotation_drift_deg": _stats([
            item["max_window_rotation_drift_deg"] for item in items
            if item["max_window_rotation_drift_deg"] is not None
        ]),
        "deepest_penetration_m": _stats([item["deepest_penetration_m"] for item in items]),
        "penetration_exploit_count": sum(item["penetration_exploit_count"] for item in items),
        "max_cube_displacement_m": _stats([item["max_cube_displacement_m"] for item in items]),
        "reward_component_sums": {
            name: _stats(values) for name, values in reward_terms.items()
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--v2-config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5.json"))
    parser.add_argument("--v3-config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5_reward_v3.json"))
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(7, 12)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    configs = {"v2": resolve(args.v2_config), "v3": resolve(args.v3_config)}
    checkpoints = {
        "v2": resolve(args.v2_checkpoint), "v3": resolve(args.v3_checkpoint),
    }
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {"v2": [], "v3": []}
    for version in ("v2", "v3"):
        model = SAC.load(checkpoints[version], device="cpu")
        for seed in args.seeds:
            results[version].append(evaluate_one(
                model, configs[version], root, seed,
                output / "contact_quality_timeseries" / version / f"seed_{seed}.json",
            ))
    payload = {
        "method": "independently trained fresh 5K PCA5 SAC policies; deterministic paired seeds",
        "seeds": args.seeds,
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "configs": {key: str(value) for key, value in configs.items()},
        "observation_dimension": 228,
        "contact_quality_in_policy_observation": False,
        "aggregate": {version: aggregate(items) for version, items in results.items()},
        "per_seed": results,
    }
    (output / "deterministic_seed_7_11.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    compact = {
        "output": str(output),
        "v2": payload["aggregate"]["v2"],
        "v3": payload["aggregate"]["v3"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
