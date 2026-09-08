"""No-training thumb-index pinch reachability search.

Only finger1 (thumb) and finger2 (index) targets vary.  Finger3--5 remain at
the configured open posture.  The fixed palm, rigid cube, reset, contact
model, PD controller, rate limits, penetration rule, and formal success gate
come directly from the existing Stage-1 environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openarm_wuji.rl import WujiStaticGraspEnv
from openarm_wuji.tasks.se3 import pose_drift, rotation_geodesic_angle_deg


FINGERS = tuple(f"finger{i}" for i in range(1, 6))
ACTIVE_FINGERS = ("finger1", "finger2")
PASSIVE_FINGERS = ("finger3", "finger4", "finger5")
MANUAL_PINCH = np.asarray([
    1.30, 0.45, 1.10, 1.10,
    1.20, -0.08, 1.15, 1.10,
], dtype=float)


def latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty((count, dimensions), dtype=float)
    for dimension in range(dimensions):
        samples[:, dimension] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    return 2.0 * samples - 1.0


def max_run(flags: list[bool]) -> int:
    current = maximum = 0
    for flag in flags:
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return maximum


def stats(values: list[float]) -> dict[str, float | None]:
    values = [float(value) for value in values if np.isfinite(value)]
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)), "median": float(np.median(array)),
        "min": float(np.min(array)), "max": float(np.max(array)),
    }


def aggregate_contact_normals(
    env: WujiStaticGraspEnv, contacts: list[dict[str, Any]], finger: str,
) -> dict[str, Any]:
    threshold = float(env.config["success"]["min_normal_force_n"])
    items = [
        item for item in contacts
        if item["finger"] == finger
        and float(item["normal_force_n"]) >= threshold
    ]
    if not items:
        return {
            "valid": False, "normal_cube": None, "normal_world": None,
            "normal_force_n": 0.0, "faces": [], "geoms": [],
        }
    forces = np.asarray([float(item["normal_force_n"]) for item in items])
    weights = forces / max(float(np.sum(forces)), 1e-12)

    def weighted_normal(key: str) -> list[float]:
        normal = np.sum(np.asarray([item[key] for item in items]) * weights[:, None], axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        return normal.tolist()

    return {
        "valid": True,
        "normal_cube": weighted_normal("normal_on_cube_cube"),
        "normal_world": weighted_normal("normal_on_cube_world"),
        "normal_force_n": float(np.sum(forces)),
        "faces": sorted({str(item["cube_face"]) for item in items}),
        "geoms": sorted({
            item.get("other_geom_name") or str(item["other_geom_id"])
            for item in items
        }),
    }


def opposition_angle(thumb: dict[str, Any], index: dict[str, Any]) -> float | None:
    if not thumb["valid"] or not index["valid"]:
        return None
    dot = float(np.clip(np.dot(thumb["normal_cube"], index["normal_cube"]), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def pinch_rank(item: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(item["strong_two_finger_pinch"]),
        float(item["formal_success_with_pure_thumb_index"]),
        # Once no full success exists, refine geometrically plausible pinches
        # before longer same-face co-contact (which is not force opposition).
        float(item["opposing_normals_over_90deg"]),
        float(item["longest_thumb_index_contact_s"] >= 0.5),
        float(item["longest_thumb_index_contact_s"]),
        float(item["thumb_index_contact_fraction"]),
        float(item["mean_opposition_angle_deg"] or 0.0),
        float(item["diagnostic_objective"]),
    )


class PinchSearch:
    def __init__(self, config: Path, root: Path, seed: int):
        self.env = WujiStaticGraspEnv.from_json(config, project_root=root)
        self.seed = seed
        self.dt = self.env.control_dt
        self.open_pose = self.env.robot.mapper.open_pose.copy()
        # Broad but mechanically plausible two-finger closing ranges.  The
        # final values are intersected with the actual model/actuator limits.
        proposed_lower = np.asarray([
            0.60, -0.10, 0.30, 0.30,
            0.40, -0.37, 0.30, 0.30,
        ])
        proposed_upper = np.asarray([
            1.603, 0.90, 1.55, 1.55,
            1.55, 0.37, 1.54, 1.55,
        ])
        self.lower = np.maximum(proposed_lower, self.env.hand_lower[:8])
        self.upper = np.minimum(proposed_upper, self.env.hand_upper[:8])
        self.center = 0.5 * (self.lower + self.upper)
        self.half = 0.5 * (self.upper - self.lower)

    def close(self) -> None:
        self.env.close()

    def target_from_search(self, search_u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(search_u, dtype=float), -1.0, 1.0)
        target = self.open_pose.copy()
        target[:8] = np.clip(self.center + self.half * u, self.lower, self.upper)
        return target

    def search_from_target(self, target_8d: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(target_8d) - self.center) / self.half, -1.0, 1.0)

    def evaluate(
        self,
        search_u: np.ndarray,
        *,
        reset_seed: int | None = None,
        terminal_hold_s: float = 0.5,
        keep_timeseries: bool = False,
    ) -> dict[str, Any]:
        u = np.clip(np.asarray(search_u, dtype=float), -1.0, 1.0)
        target = self.target_from_search(u)
        seed = self.seed if reset_seed is None else int(reset_seed)
        self.env.reset(seed=seed)
        initial_cube_position = self.env.data.xpos[self.env.cube_body_id].copy()
        initial_cube_quaternion = self.env.data.xquat[self.env.cube_body_id].copy()
        approach_frames = int(round(1.0 * self.env.control_hz))
        hold_frames = int(round(terminal_hold_s * self.env.control_hz))
        max_steps = approach_frames + hold_frames
        rows = []
        target_reached_step = None
        held_frames = 0
        success = False
        failure_reason = None
        for step in range(max_steps):
            _, _, terminated, truncated, info = self.env.step_absolute_target(
                target, reward_action=np.zeros(5)
            )
            contacts, _ = self.env._contacts()
            thumb_normal = aggregate_contact_normals(self.env, contacts, "finger1")
            index_normal = aggregate_contact_normals(self.env, contacts, "finger2")
            angle = opposition_angle(thumb_normal, index_normal)
            relative = self.env._relative_state()
            quality = info["contact_quality"]
            applied = np.asarray(info["applied_hand_target_rad"], dtype=float)
            target_reached = bool(np.max(np.abs(applied - target)) <= 1e-6)
            if target_reached and target_reached_step is None:
                target_reached_step = step + 1
            if target_reached_step is not None:
                held_frames += 1
            active = {
                finger: bool(quality["per_finger"][finger]["valid_contact"])
                for finger in FINGERS
            }
            rows.append({
                "step": step + 1,
                "time_s": float((step + 1) * self.dt),
                "target_reached": target_reached,
                "qpos": self.env.data.qpos[self.env.hand_qpos_ids].copy(),
                "applied_target": applied,
                "contact_fingers": [finger for finger in FINGERS if active[finger]],
                "thumb_index_contact": active["finger1"] and active["finger2"],
                "passive_contact": any(active[finger] for finger in PASSIVE_FINGERS),
                "opposition_angle_deg": angle,
                "thumb_contact": thumb_normal,
                "index_contact": index_normal,
                "contact_quality": quality,
                "relative_position": relative["position"].copy(),
                "relative_quaternion": relative["quaternion"].copy(),
                "relative_linear_speed_m_s": float(info["relative_linear_speed_m_s"]),
                "relative_angular_speed_deg_s": float(info["relative_angular_speed_deg_s"]),
                "cube_position": self.env.data.xpos[self.env.cube_body_id].copy(),
                "cube_quaternion": self.env.data.xquat[self.env.cube_body_id].copy(),
                "cube_displacement_m": float(info["cube_displacement_m"]),
                "deepest_penetration_m": float(info["deepest_finger_cube_penetration_m"]),
                "formal_success": bool(info["static_grasp_success"]),
            })
            success = success or bool(info["static_grasp_success"])
            failure_reason = info["failure_reason"]
            if terminated or truncated:
                break
            if target_reached_step is not None and held_frames >= hold_frames:
                break
        # If rigid contact prevents exact joint-target convergence, the
        # requested target was still constant throughout; evaluate the final
        # complete hold window.
        window = rows[-min(hold_frames, len(rows)):]
        both = [row["thumb_index_contact"] for row in window]
        passive = [row["passive_contact"] for row in window]
        angles = [
            float(row["opposition_angle_deg"]) for row in window
            if row["opposition_angle_deg"] is not None
        ]
        thumb_items = [
            row["contact_quality"]["per_finger"]["finger1"] for row in window
            if row["contact_quality"]["per_finger"]["finger1"]["valid_contact"]
        ]
        index_items = [
            row["contact_quality"]["per_finger"]["finger2"] for row in window
            if row["contact_quality"]["per_finger"]["finger2"]["valid_contact"]
        ]
        active_items = thumb_items + index_items
        slips = [float(item["tangential_speed_m_s"]) for item in active_items]
        margins = [float(item["edge_margin_m"]) for item in active_items]
        both_rows = [row for row in window if row["thumb_index_contact"]]
        motion_rows = both_rows or window
        relative_reference = window[0]
        relative_drifts = [pose_drift(
            relative_reference["relative_position"],
            relative_reference["relative_quaternion"],
            row["relative_position"], row["relative_quaternion"],
        ) for row in window]
        cube_rotations = [rotation_geodesic_angle_deg(
            initial_cube_quaternion, row["cube_quaternion"]
        ) for row in rows]
        longest = float(max_run(both) * self.dt)
        penetration_limit = float(self.env.config["failure"]["max_penetration_m"])
        mean_slip = float(np.mean(slips)) if slips else None
        translation_drift = max((value[0] for value in relative_drifts), default=0.0)
        rotation_drift = max((value[1] for value in relative_drifts), default=0.0)
        penetration = max((row["deepest_penetration_m"] for row in rows), default=0.0)
        cube_displacement = max((row["cube_displacement_m"] for row in rows), default=0.0)
        pure_two_finger = not any(passive)
        opposing = bool(angles and float(np.mean(angles)) > 90.0)
        strong = bool(
            longest >= 0.5
            and mean_slip is not None and mean_slip < 0.010
            and translation_drift < 0.008
            and rotation_drift < 6.0
            and penetration <= penetration_limit
            and pure_two_finger
            and opposing
        )
        formal_pure = bool(success and pure_two_finger and longest >= 0.5)
        contact_score = float(np.mean(both)) if both else 0.0
        opposition_score = float(np.clip((np.mean(angles) if angles else 0.0) / 180.0, 0.0, 1.0))
        slip_penalty = float(1.0 - np.exp(-((mean_slip or 0.0) / 0.010) ** 2)) if slips else 0.0
        drift_penalty = 0.5 * float(np.clip(translation_drift / 0.008, 0.0, 1.0)) + 0.5 * float(np.clip(rotation_drift / 6.0, 0.0, 1.0))
        penetration_penalty = float(np.clip(max(0.0, penetration - 0.001) / 0.007, 0.0, 1.0) ** 2)
        objective_components = {
            "pinch_persistence": 4.0 * float(np.clip(longest / 0.5, 0.0, 1.0)),
            "pinch_fraction": 2.0 * contact_score,
            "opposition": opposition_score,
            "interior": 0.5 * float(np.clip((np.mean(margins) if margins else 0.0) / 0.003, 0.0, 1.0)),
            "slip": -slip_penalty,
            "drift": -0.5 * drift_penalty,
            "cube_motion": -0.5 * float(np.clip(cube_displacement / 0.050, 0.0, 1.0)),
            "penetration": -2.0 * penetration_penalty,
            "strong_priority": 3.0 * float(strong),
        }
        result: dict[str, Any] = {
            "search_normalized_u": u.tolist(),
            "q_target_20d": target.tolist(),
            "thumb_q_target": target[:4].tolist(),
            "index_q_target": target[4:8].tolist(),
            "passive_q_target": target[8:].tolist(),
            "reset_seed": seed,
            "steps": len(rows),
            "terminal_window_s": float(len(window) * self.dt),
            "target_reached": target_reached_step is not None,
            "target_reached_step": target_reached_step,
            "longest_thumb_index_contact_s": longest,
            "thumb_index_contact_fraction": contact_score,
            "passive_contact_fraction": float(np.mean(passive)) if passive else 0.0,
            "pure_thumb_index": pure_two_finger,
            "thumb_contact_duration_s": float(len(thumb_items) * self.dt),
            "index_contact_duration_s": float(len(index_items) * self.dt),
            "thumb_contact_faces": sorted({face for row in window for face in row["thumb_contact"]["faces"]}),
            "index_contact_faces": sorted({face for row in window for face in row["index_contact"]["faces"]}),
            "thumb_contact_geoms": sorted({geom for row in window for geom in row["thumb_contact"]["geoms"]}),
            "index_contact_geoms": sorted({geom for row in window for geom in row["index_contact"]["geoms"]}),
            "mean_thumb_normal_cube": mean_normal([row["thumb_contact"]["normal_cube"] for row in both_rows]),
            "mean_index_normal_cube": mean_normal([row["index_contact"]["normal_cube"] for row in both_rows]),
            "opposition_angle_deg": stats(angles),
            "mean_opposition_angle_deg": float(np.mean(angles)) if angles else None,
            "mean_edge_margin_m": float(np.mean(margins)) if margins else None,
            "min_edge_margin_m": float(np.min(margins)) if margins else None,
            "mean_tangential_slip_m_s": mean_slip,
            "max_tangential_slip_m_s": float(np.max(slips)) if slips else None,
            "relative_linear_speed_m_s": stats([row["relative_linear_speed_m_s"] for row in motion_rows]),
            "relative_angular_speed_deg_s": stats([row["relative_angular_speed_deg_s"] for row in motion_rows]),
            "se3_translation_drift_m": translation_drift,
            "se3_rotation_drift_deg": rotation_drift,
            "cube_displacement_m": cube_displacement,
            "cube_rotation_deg": max(cube_rotations, default=0.0),
            "deepest_penetration_m": penetration,
            "penetration_exploit": penetration > penetration_limit,
            "formal_stable_grasp_success": bool(success),
            "formal_success_with_pure_thumb_index": formal_pure,
            "opposing_normals_over_90deg": opposing,
            "strong_two_finger_pinch": strong,
            "failure_reason": failure_reason,
            "diagnostic_objective": float(sum(objective_components.values())),
            "diagnostic_objective_components": objective_components,
        }
        if keep_timeseries:
            result["initial_cube_position_m"] = initial_cube_position.tolist()
            result["initial_cube_quaternion"] = initial_cube_quaternion.tolist()
            result["timeseries"] = [plain_row(row) for row in rows]
        return result


def mean_normal(values: list[list[float] | None]) -> list[float] | None:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(array):
        return None
    result = np.mean(array, axis=0)
    result /= max(float(np.linalg.norm(result)), 1e-12)
    return result.tolist()


def plain_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in row.items()
    }


def flat_row(candidate_id: str, stage: str, item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate_id": candidate_id, "stage": stage}
    row.update({f"u{i + 1}": item["search_normalized_u"][i] for i in range(8)})
    row.update({f"thumb_q{i + 1}": item["thumb_q_target"][i] for i in range(4)})
    row.update({f"index_q{i + 1}": item["index_q_target"][i] for i in range(4)})
    for key in (
        "diagnostic_objective", "strong_two_finger_pinch",
        "formal_stable_grasp_success", "formal_success_with_pure_thumb_index",
        "pure_thumb_index", "opposing_normals_over_90deg",
        "longest_thumb_index_contact_s", "thumb_index_contact_fraction",
        "passive_contact_fraction", "thumb_contact_duration_s",
        "index_contact_duration_s", "mean_opposition_angle_deg",
        "mean_edge_margin_m", "min_edge_margin_m",
        "mean_tangential_slip_m_s", "max_tangential_slip_m_s",
        "se3_translation_drift_m", "se3_rotation_drift_deg",
        "cube_displacement_m", "cube_rotation_deg", "deepest_penetration_m",
        "penetration_exploit", "target_reached", "target_reached_step",
        "failure_reason",
    ):
        row[key] = item[key]
    row["thumb_faces_json"] = json.dumps(item["thumb_contact_faces"])
    row["index_faces_json"] = json.dumps(item["index_contact_faces"])
    row["objective_components_json"] = json.dumps(item["diagnostic_objective_components"])
    return row


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5_reward_v3.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/two_finger_pinch"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--global-candidates", type=int, default=1600)
    parser.add_argument("--local-top-k", type=int, default=30)
    parser.add_argument("--local-per-scale", type=int, default=4)
    parser.add_argument("--full-top-k", type=int, default=30)
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    search = PinchSearch(config, root, args.seed)
    rng = np.random.default_rng(args.seed + 2000)
    candidates = []
    rows = []
    try:
        manual_u = search.search_from_target(MANUAL_PINCH)
        manual = search.evaluate(manual_u, terminal_hold_s=0.5)
        manual["candidate_id"] = "manual_reference"
        candidates.append(manual)
        rows.append(flat_row("manual_reference", "manual_reference", manual))
        samples = latin_hypercube(args.global_candidates, 8, args.seed)
        for index, u in enumerate(samples):
            item = search.evaluate(u, terminal_hold_s=0.5)
            item["candidate_id"] = f"global_{index:04d}"
            candidates.append(item)
            rows.append(flat_row(item["candidate_id"], "global_lhs", item))
            if (index + 1) % 100 == 0:
                best = max(candidates, key=pinch_rank)
                print(json.dumps({
                    "stage": "global", "complete": index + 1,
                    "total": args.global_candidates,
                    "best_longest_pinch_s": best["longest_thumb_index_contact_s"],
                    "strong_found": any(value["strong_two_finger_pinch"] for value in candidates),
                }), flush=True)
        seeds = sorted(candidates, key=pinch_rank, reverse=True)[:args.local_top_k]
        local = []
        for seed_index, seed_item in enumerate(seeds):
            incumbent = seed_item
            for scale in (0.15, 0.075, 0.035):
                proposals = []
                for local_index in range(args.local_per_scale):
                    u = np.clip(
                        np.asarray(incumbent["search_normalized_u"])
                        + rng.normal(0.0, scale, 8), -1.0, 1.0,
                    )
                    item = search.evaluate(u, terminal_hold_s=0.5)
                    item["candidate_id"] = f"local_{seed_index:02d}_{scale:.3f}_{local_index:02d}"
                    item["parent_candidate_id"] = incumbent["candidate_id"]
                    proposals.append(item)
                    local.append(item)
                    row = flat_row(item["candidate_id"], f"local_sigma_{scale:.3f}", item)
                    row["parent_candidate_id"] = item["parent_candidate_id"]
                    rows.append(row)
                incumbent = max([incumbent, *proposals], key=pinch_rank)
            print(json.dumps({
                "stage": "local", "complete": seed_index + 1,
                "total": len(seeds),
                "best_longest_pinch_s": incumbent["longest_thumb_index_contact_s"],
                "strong": incumbent["strong_two_finger_pinch"],
            }), flush=True)
        pq.write_table(pa.Table.from_pylist(rows), output / "candidates.parquet", compression="zstd")

        combined = candidates + local
        unique = {}
        for item in sorted(combined, key=pinch_rank, reverse=True):
            unique.setdefault(tuple(np.round(item["search_normalized_u"], 8)), item)
        full = []
        for index, item in enumerate(list(unique.values())[:args.full_top_k]):
            evaluated = search.evaluate(
                item["search_normalized_u"], terminal_hold_s=1.0,
                keep_timeseries=index < 10,
            )
            evaluated["candidate_id"] = f"full_{index:02d}_from_{item['candidate_id']}"
            full.append(evaluated)
        full.sort(key=pinch_rank, reverse=True)
        best = full[0]
        top_payload = {
            "method": "no training; direct bounded 8D thumb-index joint target search",
            "active_fingers": ["finger1/thumb", "finger2/index"],
            "passive_fingers": "finger3-5 held at configured open_pose",
            "search_bounds_rad": {
                "lower": search.lower.tolist(), "upper": search.upper.tolist(),
            },
            "manual_reference": manual,
            "global_candidate_count": args.global_candidates,
            "local_candidate_count": len(local),
            "full_candidate_count": len(full),
            "strong_candidate_count": sum(item["strong_two_finger_pinch"] for item in full),
            "formal_pure_pinch_count": sum(item["formal_success_with_pure_thumb_index"] for item in full),
            "top_candidates": full,
        }
        (output / "top_candidates.json").write_text(
            json.dumps(top_payload, indent=2), encoding="utf-8"
        )
        validation = []
        for seed in range(7, 12):
            item = search.evaluate(
                best["search_normalized_u"], reset_seed=seed,
                terminal_hold_s=1.0,
            )
            validation.append(item)
        validation_payload = {
            "fixed_q_target_20d": best["q_target_20d"],
            "thumb_q_target": best["thumb_q_target"],
            "index_q_target": best["index_q_target"],
            "passive_q_target": best["passive_q_target"],
            "candidate_reoptimized_per_seed": False,
            "seeds": list(range(7, 12)),
            "strong_success_count": sum(item["strong_two_finger_pinch"] for item in validation),
            "formal_pure_pinch_count": sum(item["formal_success_with_pure_thumb_index"] for item in validation),
            "results": validation,
        }
        (output / "multi_seed_validation.json").write_text(
            json.dumps(validation_payload, indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "output": str(output), "training": "none",
            "candidate_count": len(rows),
            "strong_candidate_count": top_payload["strong_candidate_count"],
            "formal_pure_pinch_count": top_payload["formal_pure_pinch_count"],
            "best": {
                key: best[key] for key in (
                    "thumb_q_target", "index_q_target",
                    "longest_thumb_index_contact_s",
                    "mean_tangential_slip_m_s", "mean_opposition_angle_deg",
                    "se3_translation_drift_m", "se3_rotation_drift_deg",
                    "cube_displacement_m", "deepest_penetration_m",
                    "strong_two_finger_pinch",
                )
            },
            "multi_seed_strong_success_count": validation_payload["strong_success_count"],
        }, indent=2), flush=True)
    finally:
        search.close()


if __name__ == "__main__":
    main()
