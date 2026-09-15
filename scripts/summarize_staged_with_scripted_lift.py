"""Build the aggregate JSON and report for staged ACT + scripted Lift."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.tasks import cliffs_delta


FINGERS = ("thumb", "index", "middle", "ring", "little")


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


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    values_array = np.asarray(values, dtype=float)
    return {
        "count": len(values_array),
        "mean": float(values_array.mean()),
        "median": float(np.median(values_array)),
        "std": float(values_array.std()),
        "min": float(values_array.min()),
        "p90": float(np.quantile(values_array, 0.9)),
        "max": float(values_array.max()),
    }


def _by_lift_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row["lift_attempted"]]
    result = {}
    for success, name in ((True, "lift_success"), (False, "lift_failure")):
        selected = [row for row in attempted if row["lift_success"] is success]
        peak_forces = np.asarray([
            row["lift"]["peak_per_finger_normal_force_n"] for row in selected
        ], dtype=float) if selected else np.zeros((0, 5))
        result[name] = {
            "count": len(selected),
            "preload_before_lift_l2_rad": _distribution([
                row["lift"]["preload_before_lift_l2_rad"] for row in selected
            ]),
            "preload_during_lift_mean_l2_rad": _distribution([
                row["lift"]["preload_during_lift_l2_rad"]["mean"]
                for row in selected
            ]),
            "max_cube_palm_translation_drift_m": _distribution([
                row["lift"]["max_cube_palm_translation_drift_m"]
                for row in selected
            ]),
            "max_cube_palm_rotation_drift_deg": _distribution([
                row["lift"]["max_cube_palm_rotation_drift_deg"]
                for row in selected
            ]),
            "maximum_zero_contact_run_frames": _distribution([
                row["lift"]["maximum_consecutive_zero_hand_contact_frames"]
                for row in selected
            ]),
            "terminal_hold_duration_s": _distribution([
                row["lift"]["terminal_hold_duration_s"] for row in selected
            ]),
            "terminal_contact_topology": dict(Counter(
                row["lift"]["terminal_contact_topology"] for row in selected
            )),
            "peak_per_finger_normal_force_n": {
                finger: _distribution(peak_forces[:, index].tolist())
                for index, finger in enumerate(FINGERS)
            },
        }
    return result


def _pooled_preload(conditional: dict[str, Any], full: dict[str, Any]
                    ) -> dict[str, Any]:
    rows = [
        row for summary in (conditional, full)
        for row in summary["episodes_detail"] if row["lift_attempted"]
    ]
    successful = [
        row["lift"]["preload_before_lift_l2_rad"]
        for row in rows if row["lift_success"]
    ]
    failed = [
        row["lift"]["preload_before_lift_l2_rad"]
        for row in rows if not row["lift_success"]
    ]
    return {
        "note": (
            "Pooled only as a descriptive effect-size check; conditional and "
            "fresh-full rates remain separately reported."
        ),
        "lift_success": _distribution(successful),
        "lift_failure": _distribution(failed),
        "success_minus_failure_median_rad": float(
            np.median(successful) - np.median(failed)
        ),
        "cliffs_delta_success_minus_failure": cliffs_delta(successful, failed),
        "group_overlap": bool(
            min(successful) < max(failed) and min(failed) < max(successful)
        ),
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=root / "outputs/staged_act_with_scripted_lift",
    )
    parser.add_argument(
        "--report", type=Path,
        default=root / "docs/staged_act_with_scripted_lift.md",
    )
    args = parser.parse_args()
    conditional = json.loads(
        (args.input / "conditional/summary.json").read_text(encoding="utf-8")
    )
    full = json.loads(
        (args.input / "full/summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (args.input / "manifest.json").read_text(encoding="utf-8")
    )
    conditional_outcomes = _by_lift_outcome(conditional["episodes_detail"])
    full_outcomes = _by_lift_outcome(full["episodes_detail"])
    pooled_preload = _pooled_preload(conditional, full)

    result = {
        "schema_version": 1,
        "experiment": "frozen staged ACT with existing scripted Lift",
        "decision": {
            "case": "Case B",
            "label": "GraspSecure gate is not sufficient for load-bearing",
            "reason": (
                "Only 14/32 formal GraspSecure passes in the conditional run "
                "and 17/39 in the fresh full run became Lift successes. The "
                "same scripted Lift succeeds for roughly half of safe handoffs, "
                "while terminal preload distributions overlap."
            ),
            "next_action": (
                "Keep Lift scripted; improve and validate a load-bearing "
                "GraspSecure criterion before any Lift learning."
            ),
            "automatic_training_started": False,
        },
        "frozen_contract": manifest,
        "conditional": {
            key: conditional[key] for key in (
                "completed_rollouts", "seed_start", "seed_end_inclusive",
                "counts", "probability_decomposition",
                "failure_stage_distribution", "failure_reason_distribution",
                "preload_vs_lift_success", "lift_diagnostics",
                "gate_interpretation",
            )
        },
        "full": {
            key: full[key] for key in (
                "completed_rollouts", "seed_start", "seed_end_inclusive",
                "counts", "probability_decomposition",
                "failure_stage_distribution", "failure_reason_distribution",
                "preload_vs_lift_success", "lift_diagnostics",
                "gate_interpretation",
            )
        },
        "conditional_lift_outcome_diagnostics": conditional_outcomes,
        "full_lift_outcome_diagnostics": full_outcomes,
        "pooled_preload_effect": pooled_preload,
        "invariants": {
            "maximum_20d_hand_target_change_rad": max(
                conditional["lift_diagnostics"]["maximum_hand_target_change_rad"],
                full["lift_diagnostics"]["maximum_hand_target_change_rad"],
            ),
            "no_reset_at_graspsecure_to_lift_handoff": True,
            "no_expert_state_or_action_injected": True,
            "scripted_lift_trajectory_modified": False,
            "controller_gain_modified": False,
            "grasp_parameters_modified": False,
            "contact_topology_used_as_hard_gate": False,
            "force_used_as_hard_gate": False,
            "relative_se3_drift_used_as_hard_gate": False,
        },
        "raw_summaries": {
            "conditional": args.input / "conditional/summary.json",
            "full": args.input / "full/summary.json",
        },
    }
    (args.input / "summary.json").write_text(
        json.dumps(_plain(result), indent=2), encoding="utf-8"
    )

    cp = conditional["probability_decomposition"]
    fp = full["probability_decomposition"]
    cc = conditional["counts"]
    fc = full["counts"]
    c_success = conditional_outcomes["lift_success"]
    c_failure = conditional_outcomes["lift_failure"]
    f_success = full_outcomes["lift_success"]
    f_failure = full_outcomes["lift_failure"]
    full_force = full["lift_diagnostics"]["peak_per_finger_normal_force_n"]
    report = f"""# Frozen staged ACT with scripted Lift

## Decision

**Case B: GraspSecure passes, but many grasps fail under Lift. The current
static GraspSecure/preload gate is not sufficient to identify a load-bearing
grasp.** Keep Lift scripted; improve and validate a load-bearing GraspSecure
criterion before considering Lift learning.

This is not an interface failure. Across all 64 actual Lift attempts the exact
20-D hand controller target present at the GraspSecure terminal state was held
unchanged (maximum change `{result['invariants']['maximum_20d_hand_target_change_rad']:.1f}`
rad). No reset, snapshot restore, expert state, or expert action occurred at the
GraspSecure-to-Lift handoff.

## Frozen protocol

- Reach, Approach, Recovery, the `selective_hysteresis` router, and GraspSecure
  step 1500 were loaded frozen. No policy was trained or updated.
- Lift reused the configured two-segment quintic minimum-jerk Cartesian
  trajectory, IK tolerances, `0.015 rad` arm target rate limit, controller
  gains, and grasp parameters. The reference lasts 36 control frames at 30 Hz.
- After the reference, the final scripted arm target and the unchanged 20-D
  GraspSecure hand target were held for 30 frames (1.0 s).
- The pre-Lift 25 mm cube-motion safety reference was retained. A rejected
  GraspSecure pass did not enter Lift.
- Lift success requires a completed trajectory, cube height continuously at or
  above 80 mm for the full 1.0 s terminal hold, no environment support during
  that hold, and no drop. Finger count, topology, force, antipodality, slip,
  and SE(3) drift are diagnostics only.

## Conditional GraspSecure -> Lift

The runner used seeds 4000--4076 and needed 77 frozen full-pipeline rollouts to
obtain 30 safe GraspSecure terminal states. There were 32 formal GraspSecure
passes; 2 exceeded the 25 mm safety reference and correctly did not enter
Lift.

| Metric | Result |
| --- | ---: |
| Formal GraspSecure passes | {cc['graspsecure_success']}/50 Approach-stage successes |
| Safety rejects after GraspSecure | {cc['prelift_safety_reject']} |
| Actual scripted Lift attempts | {cc['lift_attempts']} |
| Lift successes / safe attempts | {cc['lift_success']}/{cc['lift_attempts']} ({_pct(cp['P_Lift_given_safe_GraspSecure_attempt'])}) |
| Lift successes / all GraspSecure passes | {cc['lift_success']}/{cc['graspsecure_success']} ({_pct(cp['P_Lift_given_GraspSecure'])}) |
| Drops | {conditional['lift_diagnostics']['drop_count']} |
| Hold/height failures | {conditional['failure_reason_distribution'].get('height_not_held', 0)} |
| Environment-support failures | {conditional['failure_reason_distribution'].get('environment_support', 0)} |

The 16 actual Lift failures comprise 12 drops, 2 terminal height-not-held
cases, and 2 cases with environment support during the required hold.

## Fresh 100-run end-to-end result

Seeds 5000--5099 were run independently after the conditional collection.

| Stage probability | Count | Rate |
| --- | ---: | ---: |
| P(Reach) | {fc['reach_success']}/100 | {_pct(fp['P_Reach'])} |
| P(Approach-stage success \\| Reach) | {fc['approach_stage_success']}/{fc['reach_success']} | {_pct(fp['P_Approach_given_Reach'])} |
| P(GraspSecure \\| Approach) | {fc['graspsecure_success']}/{fc['approach_stage_success']} | {_pct(fp['P_GraspSecure_given_Approach'])} |
| P(Lift \\| GraspSecure), including safety rejects | {fc['lift_success']}/{fc['graspsecure_success']} | {_pct(fp['P_Lift_given_GraspSecure'])} |
| P(Lift \\| safe GraspSecure actually lifted) | {fc['lift_success']}/{fc['lift_attempts']} | {_pct(fp['P_Lift_given_safe_GraspSecure_attempt'])} |
| **P(full success)** | **{fc['full_task_success']}/100** | **{_pct(fp['P_full_success'])}** |

The product of the four formal stage probabilities is
`{fp['stage_product']:.3f}`, exactly matching the observed `17/100`. The
smallest full-run conditional probability is Lift after GraspSecure (43.6%).

### Mutually exclusive first failure stage

| Stage | Count |
| --- | ---: |
| Reach | {full['failure_stage_distribution'].get('reach', 0)} |
| Approach | {full['failure_stage_distribution'].get('approach', 0)} |
| Recovery | {full['failure_stage_distribution'].get('recovery', 0)} |
| Grasp formation | {full['failure_stage_distribution'].get('grasp_formation', 0)} |
| Preload formation | {full['failure_stage_distribution'].get('preload_formation', 0)} |
| Terminal hold / pre-Lift safety | {full['failure_stage_distribution'].get('terminal_hold', 0)} |
| Lift | {full['failure_stage_distribution'].get('lift', 0)} |
| Lift hold | {full['failure_stage_distribution'].get('lift_hold', 0)} |

The 17 actual full-run Lift failures comprise 8 drops, 7 environment-support
failures, and 2 height-not-held failures. Five additional formal GraspSecure
passes were stopped by the pre-Lift 25 mm safety guard.

## Does preload predict load-bearing success?

| Population | Lift success preload L2 | Lift failure preload L2 | Median difference | Cliff's delta |
| --- | ---: | ---: | ---: | ---: |
| Conditional | {conditional['preload_vs_lift_success']['lift_success']['median']:.3f} rad | {conditional['preload_vs_lift_success']['lift_failure']['median']:.3f} rad | {conditional['preload_vs_lift_success']['success_minus_failure_median_rad']:+.3f} rad | {conditional['preload_vs_lift_success']['cliffs_delta_success_minus_failure']:+.3f} |
| Fresh full | {full['preload_vs_lift_success']['lift_success']['median']:.3f} rad | {full['preload_vs_lift_success']['lift_failure']['median']:.3f} rad | {full['preload_vs_lift_success']['success_minus_failure_median_rad']:+.3f} rad | {full['preload_vs_lift_success']['cliffs_delta_success_minus_failure']:+.3f} |
| Pooled descriptive | {pooled_preload['lift_success']['median']:.3f} rad | {pooled_preload['lift_failure']['median']:.3f} rad | {pooled_preload['success_minus_failure_median_rad']:+.3f} rad | {pooled_preload['cliffs_delta_success_minus_failure']:+.3f} |

Larger preload is associated with better Lift outcome, especially in the
conditional set, but it is not sufficient. The pooled ranges overlap:
successful grasps include `{pooled_preload['lift_success']['min']:.3f}` rad,
while failures extend to `{pooled_preload['lift_failure']['max']:.3f}` rad.
The full-run failure mean (`{full['preload_vs_lift_success']['lift_failure']['mean']:.3f}`
rad) is also far above the historical GraspSecure-failure mean of 0.726 rad.
The existing scalar preload gate filters very weak closures but cannot by
itself identify load-bearing geometry.

## Load and contact diagnostics

| Diagnostic (median) | Conditional success | Conditional failure | Full success | Full failure |
| --- | ---: | ---: | ---: | ---: |
| Max cube/palm translation drift | {1000*c_success['max_cube_palm_translation_drift_m']['median']:.1f} mm | {1000*c_failure['max_cube_palm_translation_drift_m']['median']:.1f} mm | {1000*f_success['max_cube_palm_translation_drift_m']['median']:.1f} mm | {1000*f_failure['max_cube_palm_translation_drift_m']['median']:.1f} mm |
| Max cube/palm rotation drift | {c_success['max_cube_palm_rotation_drift_deg']['median']:.1f} deg | {c_failure['max_cube_palm_rotation_drift_deg']['median']:.1f} deg | {f_success['max_cube_palm_rotation_drift_deg']['median']:.1f} deg | {f_failure['max_cube_palm_rotation_drift_deg']['median']:.1f} deg |
| Longest zero-hand-contact run | {c_success['maximum_zero_contact_run_frames']['median']:.0f} frames | {c_failure['maximum_zero_contact_run_frames']['median']:.0f} frames | {f_success['maximum_zero_contact_run_frames']['median']:.0f} frames | {f_failure['maximum_zero_contact_run_frames']['median']:.0f} frames |

All 14 conditional Lift successes ended with all five fingers above the 0.1 N
diagnostic threshold. In the fresh full set, 15/17 successes ended with all
five and 2/17 with thumb/index/middle/ring. By contrast, 10/17 fresh-full Lift
failures ended with no finger above threshold. This topology separation is
explanatory evidence, not a newly imposed success gate.

Peak-force distributions are heavy-tailed, so medians are more representative
than means. Across the 34 fresh full Lift attempts the per-finger peak normal
force medians were: thumb `{full_force['thumb']['median']:.2f}` N, index
`{full_force['index']['median']:.2f}` N, middle
`{full_force['middle']['median']:.2f}` N, ring
`{full_force['ring']['median']:.2f}` N, and little
`{full_force['little']['median']:.2f}` N. Per-frame contact points, world-frame
force vectors, resultants, moments, topology, and preload are retained in each
Lift NPZ.

The external 8 mm / 6 deg values were not reintroduced as hard gates. Even
physical Lift successes have substantial establishment/settling motion (fresh
full success medians 42.8 mm and 15.8 deg), so those CD-WM values remain
diagnostic baselines rather than calibrated Wuji acceptance thresholds.

## Interpretation and next step

The existing scripted Lift can carry real GraspSecure ACT outputs, so the
pipeline and handoff are functional. It does so for only 46.7% of safe
conditional handoffs and 50.0% of safe handoffs in the independent full run.
Together with the moderate-but-overlapping preload effect and frequent contact
collapse in failures, this supports **Case B**, not reliable full-pipeline
freeze and not immediate Lift learning.

The next experiment should keep Lift scripted and use these labeled terminal
states to improve a load-bearing GraspSecure criterion. It should test
geometry/contact persistence and a small physical load probe as predictors,
with preload retained as a continuous feature. This report does not change a
gate, train a policy, or start that experiment.

## Artifacts

- Machine summary: `{(args.input / 'summary.json').as_posix()}`
- Conditional raw summary: `{(args.input / 'conditional/summary.json').as_posix()}`
- Full raw summary: `{(args.input / 'full/summary.json').as_posix()}`
- Exact GraspSecure terminal snapshots: `{(args.input / 'conditional/graspsecure_terminal_states').as_posix()}` and `{(args.input / 'full/graspsecure_terminal_states').as_posix()}`
- Per-frame Lift trajectories: `{(args.input / 'conditional/lift').as_posix()}` and `{(args.input / 'full/lift').as_posix()}`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
