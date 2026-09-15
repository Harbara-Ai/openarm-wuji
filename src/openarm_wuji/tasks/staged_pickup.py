"""Portable manifest and reporting helpers for the frozen staged pickup."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np


REQUIRED_PHASES = (
    "REACH",
    "APPROACH",
    "RECOVERY",
    "GRASP_SECURE",
    "LIFT",
    "HOLD",
)


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _path_is_absolute(value: str) -> bool:
    return bool(
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )


def load_pipeline_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the single frozen-pipeline source of truth."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported staged pipeline manifest schema")
    if manifest.get("name") != "openarm_wuji_staged_act_pickup_v1":
        raise ValueError("unexpected staged pipeline name")
    stages = manifest.get("stages", {})
    for name in ("reach", "approach", "recovery", "grasp_secure", "lift"):
        if name not in stages:
            raise ValueError(f"manifest is missing stage: {name}")
    portable_values = [
        manifest["assets"]["mujoco_model"]["path"],
        manifest["assets"]["task_config"],
        manifest["assets"]["hand_synergies"],
        manifest["assets"]["grasp_stage_config"],
        stages["reach"]["checkpoint"],
        stages["approach"]["checkpoint"],
        stages["recovery"]["checkpoint"],
        stages["recovery"]["router"],
        stages["grasp_secure"]["checkpoint"],
        *[
            item["path"]
            for item in manifest.get("artifact_integrity", {}).get("files", [])
        ],
    ]
    absolute = [value for value in portable_values if _path_is_absolute(value)]
    if absolute:
        raise ValueError(f"manifest contains absolute paths: {absolute}")
    if manifest["lift_admission"]["require_graspsecure_static_gate"]:
        raise ValueError("frozen v1 uses the static GraspSecure gate as telemetry only")
    return manifest


def verify_artifact_integrity(
        manifest: Mapping[str, Any], repo_root: str | Path
        ) -> list[dict[str, Any]]:
    """Verify the exact large local assets named by the frozen manifest."""
    root = Path(repo_root).resolve()
    results: list[dict[str, Any]] = []
    for expected in manifest.get("artifact_integrity", {}).get("files", []):
        path = root / expected["path"]
        actual_size = path.stat().st_size if path.exists() else None
        digest = None
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
        results.append({
            "path": expected["path"],
            "exists": path.is_file(),
            "bytes_match": actual_size == expected["bytes"],
            "sha256_match": digest == expected["sha256"],
            "actual_bytes": actual_size,
            "actual_sha256": digest,
        })
    return results


def resolve_pipeline_paths(manifest: Mapping[str, Any], repo_root: str | Path
                           ) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    stages = manifest["stages"]
    assets = manifest["assets"]
    return {
        "model": root / assets["mujoco_model"]["path"],
        "task_config": root / assets["task_config"],
        "synergies": root / assets["hand_synergies"],
        "grasp_config": root / assets["grasp_stage_config"],
        "reach": root / stages["reach"]["checkpoint"],
        "approach": root / stages["approach"]["checkpoint"],
        "recovery": root / stages["recovery"]["checkpoint"],
        "router": root / stages["recovery"]["router"],
        "grasp": root / stages["grasp_secure"]["checkpoint"],
    }


def missing_pipeline_paths(paths: Mapping[str, Path]) -> list[str]:
    return [name for name, path in paths.items() if not path.exists()]


def phase_outcomes(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create one explicit stage timeline ending in SUCCESS or FAILURE."""
    phases: list[dict[str, Any]] = [{
        "phase": "REACH",
        "status": "success" if row["reach_success"] else "failure",
    }]
    if not row["reach_success"]:
        phases.append({"phase": "FAILURE", "status": row["failure_reason"]})
        return phases
    phases.append({
        "phase": "APPROACH",
        "status": "success" if row["approach_stage_success"] else "failure",
    })
    if row.get("recovery_attempted"):
        phases.append({
            "phase": "RECOVERY",
            "status": "success" if row.get("recovery_success") else "failure",
        })
    if not row["approach_stage_success"]:
        phases.append({"phase": "FAILURE", "status": row["failure_reason"]})
        return phases
    phases.append({
        "phase": "GRASP_SECURE",
        "status": (
            "terminal_valid" if row.get("graspsecure_terminal_valid")
            else "failure"
        ),
        "static_gate_pass": bool(row.get("graspsecure_success", False)),
        "static_gate_semantics": "diagnostic_only",
    })
    if not row.get("graspsecure_terminal_valid") or not row.get(
            "prelift_safety_pass"):
        phases.append({"phase": "FAILURE", "status": row["failure_reason"]})
        return phases
    lift_complete = bool(row.get("lift", {}).get("trajectory_complete", False))
    phases.append({
        "phase": "LIFT",
        "status": "complete" if lift_complete else "failure",
    })
    phases.append({
        "phase": "HOLD",
        "status": (
            "success" if row.get("lift_success")
            else "failure" if lift_complete
            else "not_reached"
        ),
    })
    phases.append({
        "phase": "SUCCESS" if row.get("full_task_success") else "FAILURE",
        "status": row.get("failure_reason") or "success",
    })
    return phases


def _is_timeout(row: Mapping[str, Any]) -> bool:
    if "timeout" in str(row.get("failure_reason", "")).lower():
        return True
    upstream = row.get("upstream") or {}
    if upstream.get("reach_failure_reason") == "timeout":
        return True
    stage = upstream.get("new") or {}
    return bool(stage.get("timeout"))


def summarize_staged_pickup(rows: Sequence[Mapping[str, Any]], *,
                            seed_start: int, requested_runs: int,
                            pipeline_name: str) -> dict[str, Any]:
    total = len(rows)
    reach = sum(bool(row["reach_success"]) for row in rows)
    approach = sum(bool(row["approach_stage_success"]) for row in rows)
    recovery_attempts = sum(bool(row.get("recovery_attempted")) for row in rows)
    recovery_success = sum(bool(row.get("recovery_success")) for row in rows)
    grasp_attempts = sum(bool(row.get("graspsecure_attempted")) for row in rows)
    terminal_valid = sum(
        bool(row.get("graspsecure_terminal_valid")) for row in rows
    )
    gate_pass = sum(bool(row.get("graspsecure_success")) for row in rows)
    safety_pass = sum(bool(row.get("prelift_safety_pass")) for row in rows)
    lift_attempts = sum(bool(row.get("lift_attempted")) for row in rows)
    lift_success = sum(bool(row.get("lift_success")) for row in rows)
    full_success = sum(bool(row.get("full_task_success")) for row in rows)
    lift_rows = [row for row in rows if row.get("lift_attempted")]
    failure_stages = Counter(
        str(row["failure_stage"]) for row in rows
        if row.get("failure_stage") is not None
    )
    failure_reasons = Counter(
        str(row["failure_reason"]) for row in rows
        if row.get("failure_reason") is not None
    )
    lift_failures = Counter(
        str(row["lift"]["failure_reason"]) for row in lift_rows
        if not row.get("lift_success")
    )
    cube_displacements = [
        float(row["maximum_cube_displacement_m"])
        for row in rows if row.get("maximum_cube_displacement_m") is not None
    ]
    prelift_displacements = [
        float(row["prelift_cube_motion_m"])
        for row in rows if row.get("prelift_cube_motion_m") is not None
    ]
    return {
        "schema_version": 1,
        "pipeline": pipeline_name,
        "requested_runs": requested_runs,
        "completed_runs": total,
        "seed_start": seed_start,
        "seed_end_inclusive": rows[-1]["seed"] if rows else None,
        "fresh_seed_claim": (
            "fresh relative to policy training/selection; deterministic MuJoCo seeds"
        ),
        "counts": {
            "reach_success": reach,
            "approach_stage_success": approach,
            "recovery_triggered": recovery_attempts,
            "recovery_success": recovery_success,
            "graspsecure_attempted": grasp_attempts,
            "graspsecure_terminal_valid": terminal_valid,
            "graspsecure_static_gate_pass_diagnostic": gate_pass,
            "prelift_safety_pass": safety_pass,
            "lift_attempted": lift_attempts,
            "lift_success": lift_success,
            "full_task_success": full_success,
        },
        "probabilities": {
            "P_Reach": _rate(reach, total),
            "P_Approach_stage_given_Reach": _rate(approach, reach),
            "P_GraspSecure_terminal_given_Approach": _rate(
                terminal_valid, approach
            ),
            "P_static_gate_PASS_given_Approach_diagnostic": _rate(
                gate_pass, approach
            ),
            "P_Lift_success_given_GraspSecure_terminal": _rate(
                lift_success, terminal_valid
            ),
            "P_Lift_success_given_Lift_attempt": _rate(
                lift_success, lift_attempts
            ),
            "P_full_task_success": _rate(full_success, total),
        },
        "failure_stage_distribution": dict(failure_stages),
        "failure_reason_distribution": dict(failure_reasons),
        "lift_failure_distribution": dict(lift_failures),
        "diagnostics": {
            "drop": sum(reason == "drop" for reason in (
                row.get("lift", {}).get("failure_reason") for row in lift_rows
            )),
            "environment_support_failure": sum(
                row.get("lift", {}).get("failure_reason") == "environment_support"
                for row in lift_rows
            ),
            "height_hold_failure": sum(
                row.get("lift", {}).get("failure_reason") == "height_not_held"
                for row in lift_rows
            ),
            "timeout": sum(_is_timeout(row) for row in rows),
            "cube_displacement_m": _distribution(cube_displacements),
            "prelift_cube_motion_m": _distribution(prelift_displacements),
            "recovery_trigger_rate": _rate(recovery_attempts, reach),
            "recovery_success_rate": _rate(recovery_success, recovery_attempts),
        },
        "semantics": {
            "graspsecure_static_gate": "diagnostic_only_for_lift_admission",
            "P_GraspSecure_definition": "finite GraspSecure terminal state",
            "prelift_25mm_cube_motion_guard": "hard safety admission",
            "retry_enabled": False,
            "micro_lift_probe_enabled": False,
            "lift_policy": "scripted",
        },
    }


def select_representative_rollouts(
        rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    success = next((row for row in rows if row.get("full_task_success")), None)
    upstream_failure = next((
        row for row in rows
        if row.get("failure_stage") in {"reach", "approach", "recovery"}
    ), None)
    grasp_lift_failure = next((
        row for row in rows
        if row.get("lift_attempted") and not row.get("lift_success")
    ), None)
    if grasp_lift_failure is None:
        grasp_lift_failure = next((
            row for row in rows
            if row.get("failure_stage") not in {
                None, "reach", "approach", "recovery"
            }
        ), None)
    selected = {
        "success": success,
        "reach_or_approach_failure": upstream_failure,
        "grasp_or_lift_failure": grasp_lift_failure,
    }
    return {name: row for name, row in selected.items() if row is not None}


__all__ = [
    "REQUIRED_PHASES",
    "load_pipeline_manifest",
    "missing_pipeline_paths",
    "phase_outcomes",
    "resolve_pipeline_paths",
    "select_representative_rollouts",
    "summarize_staged_pickup",
    "verify_artifact_integrity",
]
