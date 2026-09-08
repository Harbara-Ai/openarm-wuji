"""No-training palm/object geometry diagnosis for thumb-index pinching.

This script keeps the current MuJoCo model, rigid free cube, contact/friction,
PD controller, joint-rate limit, open passive fingers, and all formal task
gates unchanged.  It performs two diagnostics:

1. A kinematic map of thumb/index tip and distal-pad closest points on the
   cube over the existing bounded 8-D posture space.
2. A deterministic bounded search over small palm-target translations and
   cube reset-position translations.  At every geometry it re-runs a smaller
   version of the same dynamic thumb-index posture search.

No policy is loaded and no learning takes place.  Geometry perturbations are
applied only to the in-memory environment configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from two_finger_pinch_reachability import (
    MANUAL_PINCH,
    PinchSearch,
    flat_row,
    latin_hypercube,
    pinch_rank,
)


FACE_NORMALS = {
    "+X": np.asarray([1.0, 0.0, 0.0]),
    "-X": np.asarray([-1.0, 0.0, 0.0]),
    "+Y": np.asarray([0.0, 1.0, 0.0]),
    "-Y": np.asarray([0.0, -1.0, 0.0]),
    "+Z": np.asarray([0.0, 0.0, 1.0]),
    "-Z": np.asarray([0.0, 0.0, -1.0]),
}
ROLES = ("thumb_tip", "thumb_pad", "index_tip", "index_pad")


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def cube_surface_faces(point_cube: np.ndarray, half_size: np.ndarray,
                       tolerance_m: float = 2e-5) -> list[str]:
    """Return every box face incident to a closest point (edge/corner aware)."""
    point = np.asarray(point_cube, dtype=float)
    half = np.asarray(half_size, dtype=float)
    faces = []
    for axis, label in enumerate("XYZ"):
        if abs(abs(point[axis]) - half[axis]) <= tolerance_m:
            faces.append(("+" if point[axis] >= 0.0 else "-") + label)
    if faces:
        return faces
    # Numerical fallback for a point slightly off the box surface.
    scaled = np.abs(point) / np.maximum(half, 1e-12)
    axis = int(np.argmax(scaled))
    return [("+" if point[axis] >= 0.0 else "-") + "XYZ"[axis]]


def contains_opposing_faces(first: Iterable[str], second: Iterable[str]) -> bool:
    return any(
        float(np.dot(FACE_NORMALS[a], FACE_NORMALS[b])) < -0.999
        for a in first for b in second
    )


def geometry_variants(bound_m: float, cube_bound_m: float) -> list[dict[str, Any]]:
    """Small, auditable pose set; all translations are expressed in world XYZ."""
    variants = [{
        "geometry_id": "baseline",
        "palm_offset_world_m": [0.0, 0.0, 0.0],
        "cube_reset_offset_world_m": [0.0, 0.0, 0.0],
    }]
    for axis, label in enumerate("xyz"):
        for sign, word in ((-1.0, "neg"), (1.0, "pos")):
            delta = np.zeros(3)
            delta[axis] = sign * bound_m
            variants.append({
                "geometry_id": f"palm_d{label}_{word}",
                "palm_offset_world_m": delta.tolist(),
                "cube_reset_offset_world_m": [0.0, 0.0, 0.0],
            })
    corner = bound_m / np.sqrt(2.0)
    for axes, name in (((0, 1), "xy"), ((0, 2), "xz"), ((1, 2), "yz")):
        for first in (-1.0, 1.0):
            for second in (-1.0, 1.0):
                delta = np.zeros(3)
                delta[axes[0]] = first * corner
                delta[axes[1]] = second * corner
                variants.append({
                    "geometry_id": (
                        f"palm_d{name}_{'p' if first > 0 else 'n'}"
                        f"{'p' if second > 0 else 'n'}"
                    ),
                    "palm_offset_world_m": delta.tolist(),
                    "cube_reset_offset_world_m": [0.0, 0.0, 0.0],
                })
    for axis, label in enumerate("xy"):
        for sign, word in ((-1.0, "neg"), (1.0, "pos")):
            delta = np.zeros(3)
            delta[axis] = sign * cube_bound_m
            variants.append({
                "geometry_id": f"cube_d{label}_{word}",
                "palm_offset_world_m": [0.0, 0.0, 0.0],
                "cube_reset_offset_world_m": delta.tolist(),
            })
    return variants


class GeometryDiagnosis:
    def __init__(self, search: PinchSearch):
        self.search = search
        self.env = search.env
        self.base_palm_offset = np.asarray(
            self.env.config["reset"]["pregrasp_extra_offset_m"], dtype=float
        )
        self.base_cube_nominal = np.asarray(
            self.env.task.config["reset"]["cube_nominal_position_m"], dtype=float
        )

    def apply_geometry(self, variant: dict[str, Any]) -> None:
        self.env.config["reset"]["pregrasp_extra_offset_m"] = (
            self.base_palm_offset
            + np.asarray(variant["palm_offset_world_m"], dtype=float)
        ).tolist()
        self.env.task.config["reset"]["cube_nominal_position_m"] = (
            self.base_cube_nominal
            + np.asarray(variant["cube_reset_offset_world_m"], dtype=float)
        ).tolist()

    def restore_geometry(self) -> None:
        self.env.config["reset"]["pregrasp_extra_offset_m"] = (
            self.base_palm_offset.tolist()
        )
        self.env.task.config["reset"]["cube_nominal_position_m"] = (
            self.base_cube_nominal.tolist()
        )

    def reset_geometry(self, variant: dict[str, Any], seed: int) -> dict[str, Any]:
        self.apply_geometry(variant)
        _, info = self.env.reset(seed=seed)
        palm_position = self.env.data.site_xpos[self.env.palm_site_id].copy()
        cube_position = self.env.data.xpos[self.env.cube_body_id].copy()
        relative = self.env._relative_state()
        return {
            **variant,
            "reset_seed": seed,
            "achieved_palm_position_world_m": palm_position.tolist(),
            "achieved_cube_position_world_m": cube_position.tolist(),
            "achieved_cube_minus_palm_world_m": (
                cube_position - palm_position
            ).tolist(),
            "achieved_cube_position_palm_m": relative["position"].tolist(),
            "pregrasp_ik_residual_m": float(info["pregrasp_ik_residual_m"]),
            "pregrasp_orientation_residual_deg": float(
                info["pregrasp_orientation_residual_deg"]
            ),
            "reset_cube_displacement_m": float(info["reset_cube_displacement_m"]),
        }

    def closest_point(self, geom_id: int) -> dict[str, Any]:
        import mujoco

        segment = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(
            self.env.model, self.env.data, geom_id, self.env.cube_geom_id,
            0.25, segment,
        ))
        cube_rotation = np.asarray(
            self.env.data.xmat[self.env.cube_body_id]
        ).reshape(3, 3)
        cube_position = self.env.data.xpos[self.env.cube_body_id]
        cube_point = cube_rotation.T @ (segment[3:] - cube_position)
        faces = cube_surface_faces(
            cube_point, self.env.model.geom_size[self.env.cube_geom_id]
        )
        margins = []
        half = self.env.model.geom_size[self.env.cube_geom_id]
        for face in faces:
            axis = "XYZ".index(face[1])
            margins.append(min(
                half[other] - abs(cube_point[other])
                for other in range(3) if other != axis
            ))
        return {
            "distance_m": distance,
            "cube_point_m": cube_point.tolist(),
            "faces": faces,
            "edge_margin_m": float(max(margins)) if margins else None,
        }

    def kinematic_scan(self, variant: dict[str, Any], seed: int,
                       samples: np.ndarray, contact_band_m: float,
                       output_path: Path) -> dict[str, Any]:
        import mujoco

        reset = self.reset_geometry(variant, seed)
        role_geoms = {
            "thumb_tip": self.env.fingertips["finger1"].tip_geom_id,
            "thumb_pad": self.env.pad_geom_ids["finger1"],
            "index_tip": self.env.fingertips["finger2"].tip_geom_id,
            "index_pad": self.env.pad_geom_ids["finger2"],
        }
        rows = []
        for index, u in enumerate(samples):
            target = self.search.target_from_search(u)
            self.env.data.qpos[self.env.hand_qpos_ids] = target
            self.env.data.qvel[self.env.hand_qvel_ids] = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            closest = {
                role: self.closest_point(geom) for role, geom in role_geoms.items()
            }
            thumb_role = min(
                ("thumb_tip", "thumb_pad"),
                key=lambda role: closest[role]["distance_m"],
            )
            index_role = min(
                ("index_tip", "index_pad"),
                key=lambda role: closest[role]["distance_m"],
            )
            thumb_near = closest[thumb_role]["distance_m"] <= contact_band_m
            index_near = closest[index_role]["distance_m"] <= contact_band_m
            pair_opposing = bool(
                thumb_near and index_near and contains_opposing_faces(
                    closest[thumb_role]["faces"], closest[index_role]["faces"]
                )
            )
            row: dict[str, Any] = {
                "sample_id": index,
                **{f"u{i + 1}": float(u[i]) for i in range(8)},
                **{f"q{i + 1}_rad": float(target[i]) for i in range(8)},
                "thumb_nearest_role": thumb_role,
                "index_nearest_role": index_role,
                "thumb_near": thumb_near,
                "index_near": index_near,
                "simultaneous_near": thumb_near and index_near,
                "opposing_face_pair_near": pair_opposing,
            }
            for role, item in closest.items():
                row[f"{role}_distance_m"] = item["distance_m"]
                row[f"{role}_faces_json"] = json.dumps(item["faces"])
                row[f"{role}_cube_point_json"] = json.dumps(item["cube_point_m"])
                row[f"{role}_edge_margin_m"] = item["edge_margin_m"]
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), output_path, compression="zstd")
        return {
            "geometry": reset,
            "samples": len(rows),
            "contact_band_m": contact_band_m,
            "roles": {
                role: summarize_role(rows, role, contact_band_m) for role in ROLES
            },
            "simultaneous_thumb_index_near_count": sum(
                row["simultaneous_near"] for row in rows
            ),
            "opposing_face_pair_near_count": sum(
                row["opposing_face_pair_near"] for row in rows
            ),
            "nearest_face_pair_counts": count_kinematic_face_pairs(
                rows, only_near=True
            ),
            "sample_table": output_path.name,
        }


def summarize_role(rows: list[dict[str, Any]], role: str,
                   contact_band_m: float) -> dict[str, Any]:
    distances = np.asarray([row[f"{role}_distance_m"] for row in rows])
    near = [row for row in rows if row[f"{role}_distance_m"] <= contact_band_m]
    faces = Counter()
    regions: dict[str, list[np.ndarray]] = {}
    for row in near:
        point = np.asarray(json.loads(row[f"{role}_cube_point_json"]), dtype=float)
        for face in json.loads(row[f"{role}_faces_json"]):
            faces[face] += 1
            regions.setdefault(face, []).append(point)
    return {
        "minimum_distance_m": float(np.min(distances)),
        "distance_quantiles_m": {
            str(q): float(np.quantile(distances, q)) for q in (0.01, 0.05, 0.5)
        },
        "within_contact_band_count": len(near),
        "within_contact_band_fraction": float(len(near) / max(len(rows), 1)),
        "reachable_face_counts": dict(sorted(faces.items())),
        "reachable_regions_cube_m": {
            face: {
                "min": np.min(points, axis=0).tolist(),
                "max": np.max(points, axis=0).tolist(),
            }
            for face, points in sorted(regions.items())
        },
    }


def count_kinematic_face_pairs(rows: list[dict[str, Any]], *,
                               only_near: bool) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if only_near and not row["simultaneous_near"]:
            continue
        thumb = json.loads(row[f"{row['thumb_nearest_role']}_faces_json"])
        index = json.loads(row[f"{row['index_nearest_role']}_faces_json"])
        for first in thumb:
            for second in index:
                counts[f"{first} / {second}"] += 1
    return dict(counts.most_common())


def prior_search_points(path: Path, search: PinchSearch,
                        limit: int) -> list[np.ndarray]:
    points = [search.search_from_target(MANUAL_PINCH)]
    if not path.exists():
        return points
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("top_candidates", [])[:limit]:
        if "search_normalized_u" in item:
            points.append(np.asarray(item["search_normalized_u"], dtype=float))
        elif "q_target_20d" in item:
            points.append(search.search_from_target(item["q_target_20d"][:8]))
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for point in points:
        unique.setdefault(tuple(np.round(point, 8)), point)
    return list(unique.values())


def dynamic_pose_search(diagnosis: GeometryDiagnosis,
                        variants: list[dict[str, Any]], *, seed: int,
                        posture_samples: np.ndarray,
                        warm_points: list[np.ndarray],
                        output: Path, full_top_k: int) -> dict[str, Any]:
    rows = []
    candidates: list[dict[str, Any]] = []
    geometry_resets = []
    points = warm_points + [np.asarray(item) for item in posture_samples]
    for pose_index, variant in enumerate(variants):
        try:
            geometry_resets.append(diagnosis.reset_geometry(variant, seed))
        except RuntimeError as exc:
            geometry_resets.append({**variant, "error": str(exc)})
            print(json.dumps({
                "stage": "geometry", "geometry_id": variant["geometry_id"],
                "error": str(exc),
            }), flush=True)
            continue
        for posture_index, u in enumerate(points):
            diagnosis.apply_geometry(variant)
            item = diagnosis.search.evaluate(u, terminal_hold_s=0.5)
            item["candidate_id"] = (
                f"{variant['geometry_id']}_posture_{posture_index:04d}"
            )
            item["geometry_id"] = variant["geometry_id"]
            item["palm_offset_world_m"] = variant["palm_offset_world_m"]
            item["cube_reset_offset_world_m"] = (
                variant["cube_reset_offset_world_m"]
            )
            candidates.append(item)
            row = flat_row(item["candidate_id"], "geometry_quick", item)
            row.update({
                "geometry_id": item["geometry_id"],
                "palm_offset_world_json": json.dumps(
                    item["palm_offset_world_m"]
                ),
                "cube_reset_offset_world_json": json.dumps(
                    item["cube_reset_offset_world_m"]
                ),
            })
            rows.append(row)
        pose_items = candidates[-len(points):]
        best = max(pose_items, key=pinch_rank)
        print(json.dumps({
            "stage": "geometry", "complete": pose_index + 1,
            "total": len(variants), "geometry_id": variant["geometry_id"],
            "best_longest_pinch_s": best["longest_thumb_index_contact_s"],
            "best_opposition_deg": best["mean_opposition_angle_deg"],
            "strong": best["strong_two_finger_pinch"],
        }), flush=True)
    pq.write_table(
        pa.Table.from_pylist(rows), output / "dynamic_candidates.parquet",
        compression="zstd",
    )
    candidates.sort(key=pinch_rank, reverse=True)
    selected = []
    seen = set()
    # Keep the best from every pose eligible before filling globally.  This
    # prevents one geometry with many near-duplicate postures dominating QA.
    for variant in variants:
        pose_items = [
            item for item in candidates
            if item["geometry_id"] == variant["geometry_id"]
        ]
        if pose_items:
            selected.append(pose_items[0])
            seen.add(pose_items[0]["candidate_id"])
    for item in candidates:
        if len(selected) >= full_top_k:
            break
        if item["candidate_id"] not in seen:
            selected.append(item)
            seen.add(item["candidate_id"])
    selected.sort(key=pinch_rank, reverse=True)
    selected = selected[:full_top_k]
    full = []
    variant_map = {item["geometry_id"]: item for item in variants}
    for index, item in enumerate(selected):
        variant = variant_map[item["geometry_id"]]
        diagnosis.apply_geometry(variant)
        evaluated = diagnosis.search.evaluate(
            item["search_normalized_u"], terminal_hold_s=1.0,
            keep_timeseries=index < 10,
        )
        evaluated.update({
            "candidate_id": f"full_{index:02d}_from_{item['candidate_id']}",
            "geometry_id": item["geometry_id"],
            "palm_offset_world_m": item["palm_offset_world_m"],
            "cube_reset_offset_world_m": item["cube_reset_offset_world_m"],
        })
        full.append(evaluated)
    full.sort(key=pinch_rank, reverse=True)
    pose_summaries = []
    for variant in variants:
        pose_items = [
            item for item in candidates
            if item["geometry_id"] == variant["geometry_id"]
        ]
        if not pose_items:
            continue
        best = pose_items[0]
        pose_summaries.append({
            **variant,
            "candidate_count": len(pose_items),
            "any_dual_contact_count": sum(
                item["longest_thumb_index_contact_s"] > 0.0 for item in pose_items
            ),
            "contact_over_0p1s_count": sum(
                item["longest_thumb_index_contact_s"] >= 0.1 for item in pose_items
            ),
            "contact_over_0p2s_count": sum(
                item["longest_thumb_index_contact_s"] >= 0.2 for item in pose_items
            ),
            "contact_over_0p5s_count": sum(
                item["longest_thumb_index_contact_s"] >= 0.5 for item in pose_items
            ),
            "opposing_normals_count": sum(
                item["opposing_normals_over_90deg"] for item in pose_items
            ),
            "strong_count": sum(
                item["strong_two_finger_pinch"] for item in pose_items
            ),
            "best": {
                key: best[key] for key in (
                    "candidate_id", "thumb_q_target", "index_q_target",
                    "longest_thumb_index_contact_s",
                    "thumb_contact_faces", "index_contact_faces",
                    "mean_opposition_angle_deg", "mean_tangential_slip_m_s",
                    "se3_translation_drift_m", "se3_rotation_drift_deg",
                    "cube_displacement_m", "deepest_penetration_m",
                    "strong_two_finger_pinch",
                )
            },
        })
    pose_summaries.sort(
        key=lambda item: pinch_rank({
            **next(value for value in candidates
                   if value["candidate_id"] == item["best"]["candidate_id"])
        }), reverse=True,
    )
    payload = {
        "method": (
            "no training; bounded in-memory palm/cube translation sweep with "
            "direct 8-D thumb-index target search"
        ),
        "controller_and_task_configuration_modified": False,
        "geometry_perturbations_persisted_to_config": False,
        "reset_seed": seed,
        "postures_per_geometry": len(points),
        "geometry_count": len(variants),
        "candidate_count": len(candidates),
        "strong_candidate_count_quick": sum(
            item["strong_two_finger_pinch"] for item in candidates
        ),
        "strong_candidate_count_full": sum(
            item["strong_two_finger_pinch"] for item in full
        ),
        "geometry_resets": geometry_resets,
        "geometry_summaries": pose_summaries,
        "top_full_window_candidates": full,
    }
    (output / "dynamic_summary.json").write_text(
        json.dumps(_plain(payload), indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/rl/grasp_stage1_expert_pca5_reward_v3.json"),
    )
    parser.add_argument(
        "--previous-top", type=Path,
        default=Path("outputs/two_finger_pinch/top_candidates.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/palm_object_geometry"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--kinematic-samples", type=int, default=4096)
    parser.add_argument("--postures-per-geometry", type=int, default=64)
    parser.add_argument("--warm-starts", type=int, default=12)
    parser.add_argument("--palm-translation-bound-m", type=float, default=0.012)
    parser.add_argument("--cube-translation-bound-m", type=float, default=0.015)
    parser.add_argument("--kinematic-contact-band-m", type=float, default=0.002)
    parser.add_argument("--full-top-k", type=int, default=30)
    parser.add_argument(
        "--kinematic-only", action="store_true",
        help="write the baseline tip/pad reachability map without dynamics",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    previous = (
        args.previous_top if args.previous_top.is_absolute()
        else root / args.previous_top
    )
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    search = PinchSearch(config, root, args.seed)
    diagnosis = GeometryDiagnosis(search)
    try:
        kinematic_u = latin_hypercube(
            args.kinematic_samples, 8, args.seed + 3100
        )
        baseline = geometry_variants(
            args.palm_translation_bound_m, args.cube_translation_bound_m
        )[0]
        kinematic = diagnosis.kinematic_scan(
            baseline, args.seed, kinematic_u,
            args.kinematic_contact_band_m,
            output / "kinematic_samples.parquet",
        )
        dynamic = None
        if not args.kinematic_only:
            variants = geometry_variants(
                args.palm_translation_bound_m,
                args.cube_translation_bound_m,
            )
            posture_u = latin_hypercube(
                args.postures_per_geometry, 8, args.seed + 4100
            )
            warm = prior_search_points(previous, search, args.warm_starts)
            dynamic = dynamic_pose_search(
                diagnosis, variants, seed=args.seed,
                posture_samples=posture_u, warm_points=warm,
                output=output, full_top_k=args.full_top_k,
            )
        summary = {
            "training": "none",
            "policy_loaded": False,
            "reward_success_lift_act_observation_modified": False,
            "kinematic_reachability": kinematic,
            "dynamic_search": None if dynamic is None else {
                key: dynamic[key] for key in (
                    "reset_seed", "postures_per_geometry", "geometry_count",
                    "candidate_count", "strong_candidate_count_quick",
                    "strong_candidate_count_full",
                )
            },
        }
        (output / "summary.json").write_text(
            json.dumps(_plain(summary), indent=2), encoding="utf-8"
        )
        print(json.dumps(_plain({
            "output": str(output),
            "kinematic": {
                "samples": kinematic["samples"],
                "simultaneous_near": (
                    kinematic["simultaneous_thumb_index_near_count"]
                ),
                "opposing_face_pair_near": (
                    kinematic["opposing_face_pair_near_count"]
                ),
            },
            "dynamic": summary["dynamic_search"],
        }), indent=2), flush=True)
    finally:
        diagnosis.restore_geometry()
        search.close()


if __name__ == "__main__":
    main()
