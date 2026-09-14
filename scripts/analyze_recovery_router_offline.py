"""Diagnose first-trigger labels and rank interpretable router candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.policy.router_candidates import (
    candidate_fires,
    default_candidate_catalog,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _matched_type(row: dict) -> str:
    approach = bool(row["baseline_success"])
    recovery = bool(row["recovery_success"])
    if not approach and recovery:
        return "A_useful_recovery"
    if approach and not recovery:
        return "B_harmful_recovery"
    if approach and recovery:
        return "C_both_success"
    return "D_both_fail"


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.quantile(array, 0.1)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _evaluate(spec: dict, attempted: list[dict]) -> dict[str, Any]:
    switched = []
    chosen_success = []
    chosen_outcome = []
    chosen_cube = []
    for row in attempted:
        fires = bool(row["triggered"] and candidate_fires(
            spec, row["trigger"]["features"]
        ))
        if fires:
            switched.append(row)
            chosen_success.append(bool(row["recovery_success"]))
            chosen_outcome.append(row["recovery"]["outcome"])
            chosen_cube.append(row["recovery"]["maximum_cube_displacement_m"])
        else:
            chosen_success.append(bool(row["baseline_success"]))
            chosen_outcome.append(row["baseline_outcome"])
            chosen_cube.append(row["baseline"]["maximum_cube_displacement_m"])
    useful = sum(
        (not row["baseline_success"]) and row["recovery_success"]
        for row in switched
    )
    harmful = sum(
        row["baseline_success"] and (not row["recovery_success"])
        for row in switched
    )
    both_success = sum(
        row["baseline_success"] and row["recovery_success"]
        for row in switched
    )
    both_fail = len(switched) - useful - harmful - both_success
    cube_array = np.asarray(chosen_cube, dtype=float) * 1000.0
    return {
        "name": spec["name"],
        "description": spec["description"],
        "would_switch_count": len(switched),
        "useful_switches": int(useful),
        "harmful_switches": int(harmful),
        "neutral_both_success": int(both_success),
        "neutral_both_fail": int(both_fail),
        "precision_useful_per_switch": useful / len(switched) if switched else 0.0,
        "net_benefit": int(useful - harmful),
        "predicted_successes": int(sum(chosen_success)),
        "predicted_success_rate_given_reach": sum(chosen_success) / len(attempted),
        "predicted_timeouts": chosen_outcome.count("timeout"),
        "predicted_cube_safety_failures": chosen_outcome.count("cube_displacement"),
        "predicted_cube_displacement_median_mm": float(np.median(cube_array)),
        "predicted_cube_displacement_p90_mm": float(np.quantile(cube_array, 0.9)),
        "spec": spec,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matched", type=Path,
        default=root / "outputs/recovery_router_ablation/first_trigger_replicate/summary.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/recovery_router_ablation/offline_summary.json",
    )
    parser.add_argument(
        "--catalog", type=Path,
        default=root / "outputs/recovery_router_ablation/candidate_catalog.json",
    )
    args = parser.parse_args()
    matched = json.loads(args.matched.read_text(encoding="utf-8"))
    attempted = [row for row in matched["episodes_detail"] if row["approach_attempted"]]
    triggered = [row for row in attempted if row["triggered"]]
    for row in triggered:
        if "features" not in row["trigger"]:
            raise ValueError("matched data does not contain instrumented features")

    types: dict[str, int] = {}
    by_trigger: dict[str, dict[str, Any]] = {}
    for row in triggered:
        label = _matched_type(row)
        types[label] = types.get(label, 0) + 1
        key = row["trigger"]["primary_type"]
        bucket = by_trigger.setdefault(key, {
            "triggers": 0, "approach_successes": 0,
            "recovery_successes": 0, "useful_recoveries": 0,
            "harmful_recoveries": 0,
        })
        bucket["triggers"] += 1
        bucket["approach_successes"] += int(row["baseline_success"])
        bucket["recovery_successes"] += int(row["recovery_success"])
        bucket["useful_recoveries"] += int(label == "A_useful_recovery")
        bucket["harmful_recoveries"] += int(label == "B_harmful_recovery")
    for bucket in by_trigger.values():
        count = bucket["triggers"]
        bucket["approach_self_recovery_rate"] = bucket["approach_successes"] / count
        bucket["recovery_success_rate"] = bucket["recovery_successes"] / count
        bucket["trigger_precision_for_useful_recovery"] = (
            bucket["useful_recoveries"] / count
        )
        bucket["net_benefit"] = (
            bucket["useful_recoveries"] - bucket["harmful_recoveries"]
        )

    feature_names = sorted(
        key for key, value in triggered[0]["trigger"]["features"].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    feature_distributions = {}
    for feature in feature_names:
        feature_distributions[feature] = {
            label: _distribution([
                float(row["trigger"]["features"][feature])
                for row in triggered if _matched_type(row) == label
            ])
            for label in ("A_useful_recovery", "B_harmful_recovery")
        }
    boolean_feature_rates = {}
    for feature in ("entered_12mm_basin",):
        boolean_feature_rates[feature] = {
            label: {
                "count": sum(
                    _matched_type(row) == label for row in triggered
                ),
                "true_count": sum(
                    _matched_type(row) == label
                    and bool(row["trigger"]["features"][feature])
                    for row in triggered
                ),
            }
            for label in ("A_useful_recovery", "B_harmful_recovery")
        }
        for values in boolean_feature_rates[feature].values():
            values["true_rate"] = (
                values["true_count"] / values["count"]
                if values["count"] else 0.0
            )

    catalog = default_candidate_catalog()
    evaluations = [_evaluate(spec, attempted) for spec in catalog]
    evaluations.sort(key=lambda row: (
        -row["net_benefit"], row["harmful_switches"],
        -row["useful_switches"], row["would_switch_count"], row["name"],
    ))
    payload = {
        "source_matched": args.matched,
        "approach_attempts": len(attempted),
        "first_trigger_states": len(triggered),
        "type_counts": types,
        "trigger_diagnosis": by_trigger,
        "feature_distributions_type_a_vs_b": feature_distributions,
        "boolean_feature_rates_type_a_vs_b": boolean_feature_rates,
        "candidate_ranking": evaluations,
        "legacy_baseline": {
            "approach_successes": matched["baseline"]["approach_successes"],
            "first_trigger_recovery_successes": matched[
                "staged_with_recovery"
            ]["approach_stage_successes"],
            "rescued": matched["matched_switch_effect"]["rescued_failures"],
            "regressed": matched["matched_switch_effect"]["regressed_successes"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    args.catalog.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(json.dumps({
        "type_counts": types,
        "trigger_diagnosis": by_trigger,
        "top_candidates": [{
            key: row[key] for key in (
                "name", "would_switch_count", "useful_switches",
                "harmful_switches", "net_benefit", "predicted_successes",
            )
        } for row in evaluations[:8]],
    }, indent=2))


if __name__ == "__main__":
    main()
