"""Search the current PCA5 latent domain for a stable fixed-palm grasp.

No policy is loaded or trained.  Every candidate is a constant normalized
latent vector mapped through the existing expert PCA5 absolute-posture prior,
then executed through the existing rate-limited PD controller from the same
Stage-1 reset.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openarm_wuji.rl import WujiStaticGraspEnv
from openarm_wuji.tasks.se3 import pose_drift, rotation_geodesic_angle_deg


FINGERS = tuple(f"finger{i}" for i in range(1, 6))
PRIMES = (2, 3, 5, 7, 11)


def latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Reproducible centered-strata Latin hypercube in [-1, 1]."""
    if count < 1 or dimensions < 1:
        raise ValueError("count and dimensions must be positive")
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


def finite_stats(values: list[float]) -> dict[str, float | None]:
    values = [float(value) for value in values if np.isfinite(value)]
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)), "median": float(np.median(array)),
        "min": float(np.min(array)), "max": float(np.max(array)),
    }


def diagnostic_objective(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Ranking-only objective; formal success remains authoritative."""
    contact = float(np.clip(metrics["mean_contact_count_after_settle"] / 3.0, 0.0, 1.0))
    persistence = 0.5 * float(np.clip(
        metrics["longest_2_contact_duration_s"] / 0.30, 0.0, 1.0
    )) + 0.5 * float(np.clip(
        metrics["longest_3_contact_duration_s"] / 0.30, 0.0, 1.0
    ))
    interior = 0.5 * float(np.clip(
        (metrics["mean_edge_margin_m"] or 0.0) / 0.003, 0.0, 1.0
    )) + 0.5 * float(metrics["interior_contact_fraction"])
    slip = float(1.0 - np.exp(-(
        (metrics["mean_tangential_slip_m_s"] or 0.0) / 0.010
    ) ** 2)) if metrics["valid_contact_samples"] else 0.0
    drift = 0.5 * float(np.clip(
        metrics["se3_translation_drift_m"] / 0.012, 0.0, 1.0
    )) + 0.5 * float(np.clip(
        metrics["se3_rotation_drift_deg"] / 10.0, 0.0, 1.0
    ))
    cube_motion = float(np.clip(metrics["cube_displacement_m"] / 0.050, 0.0, 1.0))
    penetration = float(np.clip(
        max(0.0, metrics["deepest_penetration_m"] - 0.001) / 0.007,
        0.0, 1.0,
    ) ** 2)
    components = {
        "contact": 2.0 * contact,
        "persistence": 2.0 * persistence,
        "interior": 1.0 * interior,
        "slip": -1.0 * slip,
        "drift": -1.0 * drift,
        "cube_motion": -0.5 * cube_motion,
        "penetration": -2.0 * penetration,
        "formal_success_priority": 3.0 * float(metrics["stable_grasp_success"]),
    }
    return float(sum(components.values())), components


def stability_level(metrics: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    checks = {
        "level0_contact_reachability": metrics["max_contacts_after_settle"] >= 2,
        "level1_three_contacts_0p10s": metrics["longest_3_contact_duration_s"] >= 0.10,
        "level2_persistent_low_slip": bool(
            metrics["longest_2_contact_duration_s"] >= 0.30
            and metrics["valid_contact_samples"] > 0
            and metrics["mean_tangential_slip_m_s"] < 0.010
            and not metrics["penetration_exploit"]
        ),
        "level3_formal_stable_grasp": bool(metrics["stable_grasp_success"]),
        "level4_strong_candidate": bool(
            metrics["stable_grasp_success"]
            and metrics["longest_2_contact_duration_s"] >= 0.50
            and metrics["mean_tangential_slip_m_s"] < 0.010
            and metrics["se3_translation_drift_m"] < 0.008
            and metrics["se3_rotation_drift_deg"] < 6.0
        ),
    }
    level = max(
        (index for index, name in enumerate(checks) if checks[name]),
        default=-1,
    )
    # Levels are diagnostic milestones, not logically cumulative in arbitrary
    # data, so report the highest consecutively satisfied level.
    consecutive = -1
    for index, passed in enumerate(checks.values()):
        if not passed:
            break
        consecutive = index
    return consecutive, checks


def candidate_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Prioritize physical milestones before the scalar diagnostic score."""
    return (
        float(metrics["stability_level"]),
        float(metrics["longest_3_contact_duration_s"]),
        float(metrics["longest_2_contact_duration_s"]),
        float(metrics["max_contacts_after_settle"]),
        float(metrics["mean_contact_count_after_settle"]),
        float(metrics["diagnostic_objective"]),
    )


class ReachabilitySearch:
    def __init__(self, config: Path, root: Path, seed: int):
        self.env = WujiStaticGraspEnv.from_json(config, project_root=root)
        self.seed = seed
        self.control_dt = self.env.control_dt
        self.mean = self.env.expert_pca5_mean.copy()
        self.basis = self.env.expert_pca5_basis.copy()
        self.center = self.env.expert_pca5_latent_center.copy()
        self.half = self.env.expert_pca5_latent_half_range.copy()

    def close(self) -> None:
        self.env.close()

    def target(self, normalized_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.clip(np.asarray(normalized_z, dtype=float), -1.0, 1.0)
        physical = self.center + self.half * z
        unclipped = self.mean + self.basis @ physical
        clipped = np.clip(unclipped, self.env.hand_lower, self.env.hand_upper)
        return physical, unclipped, clipped

    def evaluate(
        self,
        normalized_z: np.ndarray,
        *,
        mode: str,
        reset_seed: int | None = None,
        keep_timeseries: bool = False,
    ) -> dict[str, Any]:
        if mode not in {"quick", "full"}:
            raise ValueError("mode must be quick or full")
        z = np.clip(np.asarray(normalized_z, dtype=float), -1.0, 1.0)
        physical, unclipped, target = self.target(z)
        self.env.reset(seed=self.seed if reset_seed is None else reset_seed)
        initial_cube_position = self.env.data.xpos[self.env.cube_body_id].copy()
        initial_cube_quaternion = self.env.data.xquat[self.env.cube_body_id].copy()
        initial_relative = self.env._relative_state()
        quick_steps = int(round(0.5 * self.env.control_hz))
        approach_limit = int(round(1.0 * self.env.control_hz))
        hold_frames = int(round(1.0 * self.env.control_hz))
        max_steps = quick_steps if mode == "quick" else approach_limit + hold_frames
        rows: list[dict[str, Any]] = []
        reached_index: int | None = None
        held_after_reach = 0
        failure_reason = None
        success = False
        rigid_success = False
        for step in range(max_steps):
            _, _, terminated, truncated, info = self.env.step_absolute_target(
                target, reward_action=z
            )
            relative = self.env._relative_state()
            quality = info["contact_quality"]
            reached = bool(np.max(np.abs(
                np.asarray(info["applied_hand_target_rad"]) - target
            )) <= 1e-6)
            if reached and reached_index is None:
                reached_index = step
            if reached_index is not None:
                held_after_reach += 1
            rows.append({
                "step": step + 1,
                "time_s": float((step + 1) * self.env.control_dt),
                "target_reached": reached,
                "qpos": self.env.data.qpos[self.env.hand_qpos_ids].copy(),
                "applied_target": np.asarray(info["applied_hand_target_rad"], dtype=float),
                "contact_count": int(quality["valid_contact_count"]),
                "contact_quality": quality,
                "relative_position": relative["position"].copy(),
                "relative_quaternion": relative["quaternion"].copy(),
                "relative_linear_speed_m_s": float(info["relative_linear_speed_m_s"]),
                "relative_angular_speed_deg_s": float(info["relative_angular_speed_deg_s"]),
                "window_translation_drift_m": float(info["window_translation_drift_m"]),
                "window_rotation_drift_deg": float(info["window_rotation_drift_deg"]),
                "cube_position": self.env.data.xpos[self.env.cube_body_id].copy(),
                "cube_quaternion": self.env.data.xquat[self.env.cube_body_id].copy(),
                "cube_displacement_m": float(info["cube_displacement_m"]),
                "deepest_penetration_m": float(info["deepest_finger_cube_penetration_m"]),
                "success_hold_frames": int(info["success_hold_frames"]),
            })
            failure_reason = info["failure_reason"]
            success = success or bool(info["static_grasp_success"])
            rigid_success = rigid_success or bool(info["rigid_success_diagnostic"])
            if terminated or truncated:
                break
            if mode == "full" and reached_index is not None and held_after_reach >= hold_frames:
                break

        if reached_index is not None:
            evaluation_rows = rows[reached_index:]
            evaluation_window_reason = "post_command_target_reached"
        elif mode == "full":
            # A collision-loaded hand may never make the rate-limited applied
            # target numerically equal to the requested posture.  The request
            # is nevertheless constant for the whole rollout, so use the
            # complete final one-second terminal window rather than silently
            # reducing persistence to five frames.
            evaluation_rows = rows[-min(hold_frames, len(rows)):]
            evaluation_window_reason = "terminal_1s_contact_blocked_or_not_converged"
        else:
            evaluation_rows = rows
            evaluation_window_reason = "complete_0p5s_quick_screen"
        contact_counts = [row["contact_count"] for row in evaluation_rows]
        ge2 = [value >= 2 for value in contact_counts]
        ge3 = [value >= 3 for value in contact_counts]
        valid_finger_samples: list[dict[str, Any]] = []
        contact_geoms = {finger: set() for finger in FINGERS}
        contact_regions = {finger: set() for finger in FINGERS}
        contact_faces = {finger: set() for finger in FINGERS}
        finger_metrics = {}
        for finger in FINGERS:
            flags = []
            finger_samples = []
            for row in evaluation_rows:
                item = row["contact_quality"]["per_finger"][finger]
                flags.append(bool(item["valid_contact"]))
                if item["valid_contact"]:
                    finger_samples.append(item)
                    valid_finger_samples.append(item)
                    contact_geoms[finger].update(item["contact_geom"])
                    contact_regions[finger].add(item["contact_region"])
                    if item["nearest_face"] is not None:
                        contact_faces[finger].add(item["nearest_face"])
            finger_metrics[finger] = {
                "total_contact_duration_s": float(sum(flags) * self.control_dt),
                "longest_contact_duration_s": float(max_run(flags) * self.control_dt),
                "mean_edge_margin_m": (
                    float(np.mean([sample["edge_margin_m"] for sample in finger_samples]))
                    if finger_samples else None
                ),
                "mean_tangential_slip_m_s": (
                    float(np.mean([sample["tangential_speed_m_s"] for sample in finger_samples]))
                    if finger_samples else None
                ),
                "max_normal_force_n": max(
                    (float(sample["normal_force_n"]) for sample in finger_samples),
                    default=0.0,
                ),
            }
        margins = [float(sample["edge_margin_m"]) for sample in valid_finger_samples]
        slips = [float(sample["tangential_speed_m_s"]) for sample in valid_finger_samples]
        interior_fraction = float(np.mean([
            margin >= float(self.env.config["reward"].get(
                "interior_margin_threshold_m", 0.003
            )) for margin in margins
        ])) if margins else 0.0
        relative_reference = evaluation_rows[0]
        relative_drifts = [pose_drift(
            relative_reference["relative_position"],
            relative_reference["relative_quaternion"],
            row["relative_position"], row["relative_quaternion"],
        ) for row in evaluation_rows]
        cube_rotations = [rotation_geodesic_angle_deg(
            initial_cube_quaternion, row["cube_quaternion"]
        ) for row in rows]
        contact_rows = [row for row in evaluation_rows if row["contact_count"] >= 2]
        speed_rows = contact_rows or evaluation_rows
        last_window = evaluation_rows[-min(15, len(evaluation_rows)):]
        success_cfg = self.env.config["success"]
        metrics: dict[str, Any] = {
            "mode": mode,
            "reset_seed": int(self.seed if reset_seed is None else reset_seed),
            "latent_normalized_z": z.tolist(),
            "latent_physical": physical.tolist(),
            "q_target_20d": target.tolist(),
            "q_target_unclipped_20d": unclipped.tolist(),
            "joint_limit_clip_count": int(np.sum(np.abs(target - unclipped) > 1e-12)),
            "steps": len(rows),
            "target_reached": reached_index is not None,
            "target_reached_step": None if reached_index is None else reached_index + 1,
            "hold_after_target_s": float(held_after_reach * self.control_dt),
            "terminal_evaluation_window_s": float(len(evaluation_rows) * self.control_dt),
            "evaluation_window_reason": evaluation_window_reason,
            "max_simultaneous_contacts": max((row["contact_count"] for row in rows), default=0),
            "max_contacts_after_settle": max(contact_counts, default=0),
            "mean_contact_count_after_settle": float(np.mean(contact_counts)) if contact_counts else 0.0,
            "min_contact_count_after_settle": min(contact_counts, default=0),
            "longest_2_contact_duration_s": float(max_run(ge2) * self.control_dt),
            "longest_3_contact_duration_s": float(max_run(ge3) * self.control_dt),
            "contact_finger_identity": [
                finger for finger in FINGERS
                if any(row["contact_quality"]["per_finger"][finger]["valid_contact"] for row in evaluation_rows)
            ],
            "contact_geom": {finger: sorted(values) for finger, values in contact_geoms.items()},
            "contact_region": {finger: sorted(values) for finger, values in contact_regions.items()},
            "contact_face": {finger: sorted(values) for finger, values in contact_faces.items()},
            "valid_contact_samples": len(valid_finger_samples),
            "mean_edge_margin_m": float(np.mean(margins)) if margins else None,
            "min_edge_margin_m": float(np.min(margins)) if margins else None,
            "interior_contact_fraction": interior_fraction,
            "mean_tangential_slip_m_s": float(np.mean(slips)) if slips else None,
            "max_tangential_slip_m_s": float(np.max(slips)) if slips else None,
            "relative_linear_speed_m_s": finite_stats([
                row["relative_linear_speed_m_s"] for row in speed_rows
            ]),
            "relative_angular_speed_deg_s": finite_stats([
                row["relative_angular_speed_deg_s"] for row in speed_rows
            ]),
            "se3_translation_drift_m": max((item[0] for item in relative_drifts), default=0.0),
            "se3_rotation_drift_deg": max((item[1] for item in relative_drifts), default=0.0),
            "cube_displacement_m": max((row["cube_displacement_m"] for row in rows), default=0.0),
            "cube_rotation_deg": max(cube_rotations, default=0.0),
            "deepest_penetration_m": max((row["deepest_penetration_m"] for row in rows), default=0.0),
            "penetration_exploit": bool(any(
                row["deepest_penetration_m"] > float(self.env.config["failure"]["max_penetration_m"])
                for row in rows
            )),
            "stable_grasp_success": bool(success),
            "rigid_success_diagnostic": bool(rigid_success),
            "failure_reason": failure_reason,
            "per_finger": finger_metrics,
            "formal_conditions_final_window": {
                "enough_contacts_all_frames": bool(last_window and all(
                    row["contact_count"] >= int(success_cfg["min_contact_fingers"])
                    for row in last_window
                )),
                "full_sustain_window": len(last_window) >= int(np.ceil(
                    float(success_cfg["sustain_s"]) * self.env.control_hz
                )),
                "linear_velocity_all_frames": bool(last_window and all(
                    row["relative_linear_speed_m_s"] <= float(success_cfg["max_relative_linear_speed_m_s"])
                    for row in last_window
                )),
                "angular_velocity_all_frames": bool(last_window and all(
                    row["relative_angular_speed_deg_s"] <= float(success_cfg["max_relative_angular_speed_deg_s"])
                    for row in last_window
                )),
                "translation_drift": bool(metrics_value(relative_drifts, 0) <= float(success_cfg["max_window_translation_drift_m"])),
                "rotation_drift": bool(metrics_value(relative_drifts, 1) <= float(success_cfg["max_window_rotation_drift_deg"])),
                "cube_displacement": bool(max((row["cube_displacement_m"] for row in rows), default=0.0) <= float(success_cfg["max_cube_displacement_m"])),
            },
        }
        objective, components = diagnostic_objective(metrics)
        level, level_checks = stability_level(metrics)
        metrics["diagnostic_objective"] = objective
        metrics["diagnostic_objective_components"] = components
        metrics["stability_level"] = level
        metrics["stability_level_checks"] = level_checks
        if keep_timeseries:
            metrics["timeseries"] = [serializable_row(row) for row in rows]
            metrics["initial_cube_position_m"] = initial_cube_position.tolist()
            metrics["initial_cube_quaternion"] = initial_cube_quaternion.tolist()
            metrics["initial_relative_position_m"] = initial_relative["position"].tolist()
            metrics["initial_relative_quaternion"] = initial_relative["quaternion"].tolist()
        return metrics


def metrics_value(relative_drifts: list[tuple[float, float]], index: int) -> float:
    return max((item[index] for item in relative_drifts), default=0.0)


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in row.items()
    }


def flat_candidate(candidate_id: str, source: str, item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate_id": candidate_id, "source": source}
    row.update({f"z{i + 1}": item["latent_normalized_z"][i] for i in range(5)})
    row.update({f"latent_physical_{i + 1}": item["latent_physical"][i] for i in range(5)})
    row.update({f"q_target_{i + 1}": item["q_target_20d"][i] for i in range(20)})
    for key in (
        "diagnostic_objective", "stability_level", "max_simultaneous_contacts",
        "max_contacts_after_settle", "mean_contact_count_after_settle",
        "min_contact_count_after_settle", "longest_2_contact_duration_s",
        "longest_3_contact_duration_s", "valid_contact_samples",
        "mean_edge_margin_m", "min_edge_margin_m", "interior_contact_fraction",
        "mean_tangential_slip_m_s", "max_tangential_slip_m_s",
        "se3_translation_drift_m", "se3_rotation_drift_deg",
        "cube_displacement_m", "cube_rotation_deg", "deepest_penetration_m",
        "penetration_exploit", "stable_grasp_success", "rigid_success_diagnostic",
        "target_reached", "target_reached_step", "hold_after_target_s",
        "joint_limit_clip_count", "failure_reason",
    ):
        row[key] = item[key]
    row["contact_finger_identity_json"] = json.dumps(item["contact_finger_identity"])
    row["contact_geom_json"] = json.dumps(item["contact_geom"])
    row["contact_region_json"] = json.dumps(item["contact_region"])
    row["per_finger_json"] = json.dumps(item["per_finger"])
    row["objective_components_json"] = json.dumps(item["diagnostic_objective_components"])
    return row


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def load_expert_postures(source: Path, phase_path: Path) -> dict[str, Any]:
    table = pq.read_table(source, columns=[
        "episode_index", "frame_index", "q_hand"
    ]).to_pydict()
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    phase_by_id = {int(item["episode_id"]): item for item in phase["episodes"]}
    episodes = np.asarray(table["episode_index"], dtype=int)
    frames = np.asarray(table["frame_index"], dtype=int)
    q_values = []
    metadata = []
    for index, (episode, frame) in enumerate(zip(episodes, frames)):
        if episode not in phase_by_id:
            continue
        interval = phase_by_id[episode]
        if int(interval["grasp_established_frame"]) <= frame < int(interval["end_frame_exclusive"]):
            q_values.append(np.asarray(table["q_hand"][index], dtype=float))
            metadata.append({"episode_id": int(episode), "frame_index": int(frame)})
    return {"q": np.asarray(q_values), "metadata": metadata}


def attach_expert_distance(
    item: dict[str, Any], expert: dict[str, Any],
    mean: np.ndarray, basis: np.ndarray, center: np.ndarray, half: np.ndarray,
) -> None:
    q = np.asarray(item["q_target_20d"], dtype=float)
    expert_q = expert["q"]
    q_distances = np.linalg.norm(expert_q - q, axis=1)
    q_index = int(np.argmin(q_distances))
    expert_latent = (expert_q - mean) @ basis
    expert_normalized = (expert_latent - center) / half
    z = np.asarray(item["latent_normalized_z"])
    latent_distances = np.linalg.norm(expert_normalized - z, axis=1)
    latent_index = int(np.argmin(latent_distances))
    item["nearest_expert_q_distance_l2_rad"] = float(q_distances[q_index])
    item["nearest_expert_q_metadata"] = expert["metadata"][q_index]
    item["nearest_expert_latent_distance_l2"] = float(latent_distances[latent_index])
    item["nearest_expert_latent_metadata"] = expert["metadata"][latent_index]


def boundary_report(top: list[dict[str, Any]]) -> dict[str, Any]:
    z = np.asarray([item["latent_normalized_z"] for item in top], dtype=float)
    return {
        "candidate_count": len(top),
        "exact_boundary_definition": "abs(z) >= 0.999",
        "near_boundary_definition": "abs(z) >= 0.95",
        "exact_boundary_hit_frequency": np.mean(np.abs(z) >= 0.999, axis=0).tolist(),
        "near_boundary_hit_frequency": np.mean(np.abs(z) >= 0.95, axis=0).tolist(),
        "per_dimension": {
            f"PC{index + 1}": finite_stats(z[:, index].tolist())
            for index in range(5)
        },
        "all_normalized_z": z.tolist(),
        "any_exact_boundary_fraction": float(np.mean(np.any(np.abs(z) >= 0.999, axis=1))),
        "any_near_boundary_fraction": float(np.mean(np.any(np.abs(z) >= 0.95, axis=1))),
    }


def sensitivity_report(
    search: ReachabilitySearch,
    bases: list[dict[str, Any]],
) -> dict[str, Any]:
    records = []
    for base_index, base in enumerate(bases):
        base_z = np.asarray(base["latent_normalized_z"], dtype=float)
        for pc in range(5):
            for delta in (-0.10, -0.05, 0.05, 0.10):
                perturbed = base_z.copy()
                perturbed[pc] = np.clip(perturbed[pc] + delta, -1.0, 1.0)
                actual_delta = float(perturbed[pc] - base_z[pc])
                if abs(actual_delta) < 1e-12:
                    continue
                item = search.evaluate(perturbed, mode="full")
                records.append({
                    "base_rank": base_index + 1,
                    "pc": pc + 1,
                    "requested_delta": delta,
                    "actual_delta": actual_delta,
                    "latent_normalized_z": perturbed.tolist(),
                    "diagnostic_objective": item["diagnostic_objective"],
                    "delta_objective": item["diagnostic_objective"] - base["diagnostic_objective"],
                    "delta_longest_2_contact_duration_s": item["longest_2_contact_duration_s"] - base["longest_2_contact_duration_s"],
                    "delta_longest_3_contact_duration_s": item["longest_3_contact_duration_s"] - base["longest_3_contact_duration_s"],
                    "delta_mean_edge_margin_m": (item["mean_edge_margin_m"] or 0.0) - (base["mean_edge_margin_m"] or 0.0),
                    "delta_mean_tangential_slip_m_s": (item["mean_tangential_slip_m_s"] or 0.0) - (base["mean_tangential_slip_m_s"] or 0.0),
                    "delta_se3_translation_drift_m": item["se3_translation_drift_m"] - base["se3_translation_drift_m"],
                    "delta_se3_rotation_drift_deg": item["se3_rotation_drift_deg"] - base["se3_rotation_drift_deg"],
                    "delta_max_contacts": item["max_contacts_after_settle"] - base["max_contacts_after_settle"],
                    "topology_collapsed": item["longest_3_contact_duration_s"] + 1e-12 < base["longest_3_contact_duration_s"],
                    "per_finger": {
                        finger: {
                            "delta_contact_duration_s": item["per_finger"][finger]["total_contact_duration_s"] - base["per_finger"][finger]["total_contact_duration_s"],
                            "delta_edge_margin_m": (item["per_finger"][finger]["mean_edge_margin_m"] or 0.0) - (base["per_finger"][finger]["mean_edge_margin_m"] or 0.0),
                            "delta_slip_m_s": (item["per_finger"][finger]["mean_tangential_slip_m_s"] or 0.0) - (base["per_finger"][finger]["mean_tangential_slip_m_s"] or 0.0),
                        } for finger in FINGERS
                    },
                })
    summary = {}
    basis_influence = {}
    for pc in range(5):
        selected = [item for item in records if item["pc"] == pc + 1]
        summary[f"PC{pc + 1}"] = {
            "sample_count": len(selected),
            "mean_abs_delta_objective": float(np.mean(np.abs([item["delta_objective"] for item in selected]))),
            "mean_abs_delta_longest_3_contact_duration_s": float(np.mean(np.abs([item["delta_longest_3_contact_duration_s"] for item in selected]))),
            "mean_abs_delta_edge_margin_m": float(np.mean(np.abs([item["delta_mean_edge_margin_m"] for item in selected]))),
            "mean_abs_delta_tangential_slip_m_s": float(np.mean(np.abs([item["delta_mean_tangential_slip_m_s"] for item in selected]))),
            "mean_abs_delta_se3_translation_drift_m": float(np.mean(np.abs([item["delta_se3_translation_drift_m"] for item in selected]))),
            "mean_abs_delta_se3_rotation_drift_deg": float(np.mean(np.abs([item["delta_se3_rotation_drift_deg"] for item in selected]))),
            "topology_collapse_fraction": float(np.mean([item["topology_collapsed"] for item in selected])),
            "per_finger_mean_abs_effect": {
                finger: {
                    "contact_duration_s": float(np.mean(np.abs([
                        item["per_finger"][finger]["delta_contact_duration_s"] for item in selected
                    ]))),
                    "edge_margin_m": float(np.mean(np.abs([
                        item["per_finger"][finger]["delta_edge_margin_m"] for item in selected
                    ]))),
                    "slip_m_s": float(np.mean(np.abs([
                        item["per_finger"][finger]["delta_slip_m_s"] for item in selected
                    ]))),
                } for finger in FINGERS
            },
        }
        basis_influence[f"PC{pc + 1}"] = {
            finger: float(np.linalg.norm(search.basis[4 * index:4 * (index + 1), pc]))
            for index, finger in enumerate(FINGERS)
        }
    return {
        "perturbations": "each of the final top-3 candidates, PC_i +/-0.05 and +/-0.10, clipped to [-1,1]",
        "basis_joint_space_l2_influence_per_finger": basis_influence,
        "summary": summary,
        "records": records,
    }


def _scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if high <= low + 1e-12:
        return 0.5 * (out_low + out_high)
    return out_low + (value - low) * (out_high - out_low) / (high - low)


def svg_parallel(top: list[dict[str, Any]], path: Path) -> None:
    width, height = 900, 520
    xs = np.linspace(100, 800, 5)
    durations = [item["longest_3_contact_duration_s"] for item in top]
    dmax = max(durations, default=1.0) or 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="30" y="30" font-size="20">Top candidates: normalized PCA5 latent coordinates</text>']
    for index, x in enumerate(xs):
        parts.append(f'<line x1="{x}" y1="70" x2="{x}" y2="450" stroke="#999"/>')
        parts.append(f'<text x="{x-16}" y="478" font-size="14">PC{index+1}</text>')
    for rank, item in reversed(list(enumerate(top, start=1))):
        z = item["latent_normalized_z"]
        points = " ".join(f"{xs[i]:.1f},{_scale(z[i],-1,1,450,70):.1f}" for i in range(5))
        intensity = int(80 + 175 * item["longest_3_contact_duration_s"] / dmax)
        color = f"rgb({255-intensity//2},{80},{intensity})"
        stroke = 3 if item["stable_grasp_success"] else 1.2
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke}" opacity="0.65"/>')
    parts.append('<text x="100" y="500" font-size="12">Color intensity = longest continuous >=3-contact duration; thick = formal success</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_scatter(items: list[dict[str, Any]], path: Path, kind: str) -> None:
    width, height = 850, 520
    if kind == "edge_slip":
        xs = [(item["mean_edge_margin_m"] or 0.0) * 1000 for item in items]
        ys = [(item["mean_tangential_slip_m_s"] or 0.0) * 1000 for item in items]
        xlabel, ylabel = "Mean edge margin (mm)", "Mean tangential slip (mm/s)"
        title = "Contact placement vs tangential slip"
    else:
        xs = [item["se3_translation_drift_m"] * 1000 for item in items]
        ys = [item["longest_3_contact_duration_s"] for item in items]
        xlabel, ylabel = "SE(3) translation drift (mm)", "Longest >=3-contact duration (s)"
        title = "Relative drift vs multi-contact persistence"
    xmin, xmax = min(xs, default=0.0), max(xs, default=1.0)
    ymin, ymax = min(ys, default=0.0), max(ys, default=1.0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="30" y="30" font-size="20">{html.escape(title)}</text>',
             '<line x1="90" y1="450" x2="800" y2="450" stroke="black"/>',
             '<line x1="90" y1="60" x2="90" y2="450" stroke="black"/>',
             f'<text x="330" y="500" font-size="14">{html.escape(xlabel)}</text>',
             f'<text x="12" y="255" font-size="14" transform="rotate(-90 12 255)">{html.escape(ylabel)}</text>']
    for item, x, y in zip(items, xs, ys):
        px = _scale(x, xmin, xmax, 105, 785)
        py = _scale(y, ymin, ymax, 435, 75)
        color = ["#999", "#4c78a8", "#f58518", "#54a24b", "#b279a2"][max(0, item["stability_level"])]
        radius = 7 if item["stable_grasp_success"] else 4
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" opacity="0.75"/>')
    parts.extend([
        f'<text x="90" y="470" font-size="11">{xmin:.3g}</text>',
        f'<text x="760" y="470" font-size="11">{xmax:.3g}</text>',
        f'<text x="45" y="445" font-size="11">{ymin:.3g}</text>',
        f'<text x="45" y="80" font-size="11">{ymax:.3g}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_sensitivity(report: dict[str, Any], path: Path) -> None:
    metrics = [
        ("3-contact duration", "mean_abs_delta_longest_3_contact_duration_s"),
        ("edge margin", "mean_abs_delta_edge_margin_m"),
        ("tangential slip", "mean_abs_delta_tangential_slip_m_s"),
        ("SE3 translation", "mean_abs_delta_se3_translation_drift_m"),
        ("SE3 rotation", "mean_abs_delta_se3_rotation_drift_deg"),
    ]
    array = np.asarray([[report["summary"][f"PC{pc+1}"][key] for _, key in metrics] for pc in range(5)])
    maxima = np.maximum(np.max(array, axis=0), 1e-12)
    normalized = array / maxima
    width, height = 900, 520
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="30" y="30" font-size="20">PCA5 local sensitivity (column-normalized mean absolute change)</text>']
    for pc in range(5):
        parts.append(f'<text x="35" y="{115+pc*65}" font-size="14">PC{pc+1}</text>')
        for metric in range(5):
            value = normalized[pc, metric]
            shade = int(245 - 180 * value)
            x, y = 140 + metric * 140, 75 + pc * 65
            parts.append(f'<rect x="{x}" y="{y}" width="125" height="52" fill="rgb({shade},{shade},255)" stroke="white"/>')
            parts.append(f'<text x="{x+35}" y="{y+31}" font-size="12">{value:.2f}</text>')
    for metric, (label, _) in enumerate(metrics):
        parts.append(f'<text x="{145+metric*140}" y="440" font-size="11">{html.escape(label)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5_reward_v3.json"))
    parser.add_argument("--expert-source", type=Path, default=Path("outputs/wuji_teleop_analysis/cube_60_hand_trajectories.parquet"))
    parser.add_argument("--phase", type=Path, default=Path("outputs/expert_pca5/phase_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pca5_reachability"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--global-candidates", type=int, default=2500)
    parser.add_argument("--local-top-k", type=int, default=30)
    parser.add_argument("--local-per-scale", type=int, default=8)
    parser.add_argument("--full-top-k", type=int, default=50)
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    config, source, phase, output = map(resolve, (
        args.config, args.expert_source, args.phase, args.output
    ))
    output.mkdir(parents=True, exist_ok=True)
    (output / "top10_timeseries").mkdir(parents=True, exist_ok=True)
    search = ReachabilitySearch(config, root, args.seed)
    rng = np.random.default_rng(args.seed + 1000)
    try:
        global_rows = []
        global_nested = []
        samples = latin_hypercube(args.global_candidates, 5, args.seed)
        for index, z in enumerate(samples):
            item = search.evaluate(z, mode="quick")
            item["candidate_id"] = f"global_{index:04d}"
            global_nested.append(item)
            global_rows.append(flat_candidate(item["candidate_id"], "global_lhs", item))
            if (index + 1) % 250 == 0:
                print(json.dumps({
                    "stage": "global", "complete": index + 1,
                    "total": args.global_candidates,
                    "best_objective": max(value["diagnostic_objective"] for value in global_nested),
                }), flush=True)
        write_parquet(output / "global_search_candidates.parquet", global_rows)

        seeds = sorted(global_nested, key=candidate_rank, reverse=True)[
            :args.local_top_k
        ]
        local_nested = []
        local_rows = []
        for seed_index, seed_item in enumerate(seeds):
            incumbent = seed_item
            for scale in (0.20, 0.10, 0.05):
                proposals = []
                for local_index in range(args.local_per_scale):
                    z = np.clip(
                        np.asarray(incumbent["latent_normalized_z"])
                        + rng.normal(0.0, scale, 5), -1.0, 1.0,
                    )
                    item = search.evaluate(z, mode="quick")
                    item["candidate_id"] = (
                        f"local_{seed_index:02d}_{scale:.2f}_{local_index:02d}"
                    )
                    item["parent_candidate_id"] = incumbent["candidate_id"]
                    proposals.append(item)
                    local_nested.append(item)
                    row = flat_candidate(item["candidate_id"], f"local_sigma_{scale:.2f}", item)
                    row["parent_candidate_id"] = item["parent_candidate_id"]
                    local_rows.append(row)
                incumbent = max([incumbent, *proposals], key=candidate_rank)
            print(json.dumps({
                "stage": "local", "seed_complete": seed_index + 1,
                "total_seeds": len(seeds),
                "incumbent_objective": incumbent["diagnostic_objective"],
            }), flush=True)
        write_parquet(output / "local_refinement_candidates.parquet", local_rows)

        combined = global_nested + local_nested
        unique = {}
        for item in sorted(combined, key=candidate_rank, reverse=True):
            key = tuple(np.round(item["latent_normalized_z"], 8))
            unique.setdefault(key, item)
        full_inputs = list(unique.values())[:args.full_top_k]
        full = []
        for index, quick in enumerate(full_inputs):
            item = search.evaluate(quick["latent_normalized_z"], mode="full")
            item["candidate_id"] = f"full_{index:02d}_from_{quick['candidate_id']}"
            item["quick_objective"] = quick["diagnostic_objective"]
            full.append(item)
        full.sort(key=candidate_rank, reverse=True)

        expert = load_expert_postures(source, phase)
        for item in full:
            attach_expert_distance(
                item, expert, search.mean, search.basis, search.center, search.half
            )
        top10 = full[:10]
        for rank, item in enumerate(top10, start=1):
            detailed = search.evaluate(
                item["latent_normalized_z"], mode="full", keep_timeseries=True
            )
            detailed["rank"] = rank
            detailed["candidate_id"] = item["candidate_id"]
            attach_expert_distance(
                detailed, expert, search.mean, search.basis, search.center, search.half
            )
            (output / "top10_timeseries" / f"rank_{rank:02d}.json").write_text(
                json.dumps(detailed, indent=2), encoding="utf-8"
            )

        sensitivity = sensitivity_report(search, top10[:3])
        (output / "latent_sensitivity.json").write_text(
            json.dumps(sensitivity, indent=2), encoding="utf-8"
        )
        boundary = boundary_report(full[:20])
        (output / "boundary_saturation.json").write_text(
            json.dumps(boundary, indent=2), encoding="utf-8"
        )

        best = top10[0]
        multi_seed = []
        for seed in range(7, 12):
            item = search.evaluate(
                best["latent_normalized_z"], mode="full", reset_seed=seed
            )
            item["candidate_id"] = best["candidate_id"]
            multi_seed.append(item)
        multi_seed_payload = {
            "fixed_candidate_z": best["latent_normalized_z"],
            "selection_seed": args.seed,
            "validation_seeds": list(range(7, 12)),
            "candidate_reoptimized_per_seed": False,
            "formal_success_count": sum(item["stable_grasp_success"] for item in multi_seed),
            "level_counts": {
                str(level): sum(item["stability_level"] >= level for item in multi_seed)
                for level in range(5)
            },
            "results": multi_seed,
        }
        (output / "multi_seed_validation.json").write_text(
            json.dumps(multi_seed_payload, indent=2), encoding="utf-8"
        )
        top_payload = {
            "method": "no training; PCA5 static absolute-posture deterministic search",
            "search_space": "normalized latent z in [-1,1]^5, mapped with the current dataset min/max scaling",
            "config": str(config),
            "global_search": {
                "method": "Latin hypercube", "candidate_count": args.global_candidates,
                "quick_horizon_s": 0.5, "reset_seed": args.seed,
            },
            "local_refinement": {
                "seed_count": args.local_top_k,
                "scales": [0.20, 0.10, 0.05],
                "proposals_per_scale": args.local_per_scale,
                "candidate_count": len(local_nested),
            },
            "full_validation": {
                "candidate_count": len(full),
                "constant_target_hold_after_reach_s": 1.0,
            },
            "objective_is_training_reward": False,
            "objective_definition": "2 contact + 2 persistence + interior - slip - drift - 0.5 cube motion - 2 penetration + 3 formal-success priority",
            "stability_levels": {
                "0": "max simultaneous contact >=2",
                "1": ">=3 contacts continuously >=0.10 s",
                "2": ">=2 contacts continuously >=0.30 s, mean contact slip <0.010 m/s, no penetration exploit",
                "3": "formal stable_grasp_success",
                "4": "formal success + >=0.5 s hold + <8 mm/<6 deg relative drift + <0.010 m/s slip",
            },
            "formal_success_count": sum(item["stable_grasp_success"] for item in full),
            "level_counts": {
                str(level): sum(item["stability_level"] >= level for item in full)
                for level in range(5)
            },
            "top10": top10,
            "full_candidate_summaries": full,
        }
        (output / "top_candidates.json").write_text(
            json.dumps(top_payload, indent=2), encoding="utf-8"
        )
        svg_parallel(top10, output / "top_latent_vs_contact_duration.svg")
        svg_scatter(full, output / "edge_margin_vs_tangential_slip.svg", "edge_slip")
        svg_scatter(full, output / "se3_drift_vs_contact_duration.svg", "drift_contact")
        svg_sensitivity(sensitivity, output / "pc_sensitivity.svg")
        print(json.dumps({
            "output": str(output),
            "training": "none",
            "global_candidates": len(global_nested),
            "local_candidates": len(local_nested),
            "full_candidates": len(full),
            "formal_success_count": top_payload["formal_success_count"],
            "level_counts": top_payload["level_counts"],
            "best": {
                key: best[key] for key in (
                    "latent_normalized_z", "diagnostic_objective",
                    "stability_level", "longest_2_contact_duration_s",
                    "longest_3_contact_duration_s", "mean_edge_margin_m",
                    "mean_tangential_slip_m_s", "stable_grasp_success",
                )
            },
        }, indent=2), flush=True)
    finally:
        search.close()


if __name__ == "__main__":
    main()
