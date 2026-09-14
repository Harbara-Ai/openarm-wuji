"""Aggregate the Approach correction-data experiment and write JSON/Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


STEPS = (500, 1000, 1500, 2000)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _distribution(values: list[float], scale: float = 1.0) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("count", "mean", "median", "p90", "max")}
    array = np.asarray(values, dtype=float) * scale
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _evaluation(source: dict[str, Any]) -> dict[str, Any]:
    episodes = source["episodes"]
    attempted = [row for row in episodes if row.get("approach_attempted", True)]
    success = sum(bool(row["approach_success"]) for row in attempted)
    failure_counts: dict[str, int] = {}
    for row in attempted:
        key = "success" if row["approach_success"] else str(row.get("failure_reason") or "other")
        failure_counts[key] = failure_counts.get(key, 0) + 1
    return {
        "episodes": len(episodes),
        "reach_successes": sum(bool(row.get("reach_success", True)) for row in episodes),
        "approach_attempts": len(attempted),
        "approach_successes": success,
        "approach_conditional_success_rate": success / len(attempted) if attempted else 0.0,
        "joint_reach_approach_successes": sum(
            bool(row.get("reach_success", True)) and bool(row["approach_success"])
            for row in episodes
        ),
        "failure_distribution": failure_counts,
        "timeouts": sum(bool(row.get("timeout", False)) for row in attempted),
        "cube_motion_safety_failures": sum(
            row.get("failure_reason") == "cube_displacement" for row in attempted
        ),
        "early_contacts": sum(bool(row.get("early_contact", False)) for row in attempted),
        "action_clipping_values": sum(int(row.get("clipped_action_values", 0)) for row in attempted),
        "minimum_terminal_error_mm": _distribution([
            row["minimum_grasp_pose_position_error_m"] for row in attempted
        ], 1000.0),
        "final_terminal_error_mm": _distribution([
            row["final_grasp_pose_position_error_m"] for row in attempted
        ], 1000.0),
        "maximum_cube_displacement_mm": _distribution([
            row["maximum_cube_displacement_m"] for row in attempted
        ], 1000.0),
    }


def _recovery(source: dict[str, Any]) -> dict[str, Any]:
    attempted = [row for row in source["rollouts"] if row["approach_attempted"]]
    triggered = [row for row in attempted if row["triggers"]]
    return {
        "episodes": len(source["rollouts"]),
        "reach_successes": int(source["reach_successes"]),
        "approach_attempts": int(source["approach_attempts"]),
        "approach_successes": int(source["approach_successes"]),
        "approach_conditional_success_rate": (
            source["approach_successes"] / source["approach_attempts"]
            if source["approach_attempts"] else 0.0
        ),
        "joint_reach_approach_successes": int(source["approach_successes"]),
        "failure_distribution": source["outcome_distribution"],
        "timeouts": int(source["timeout_failures"]),
        "cube_motion_safety_failures": int(source["cube_displacement_failures"]),
        "near_failure_trigger_events": int(source["candidate_snapshots"]),
        "rollouts_with_near_failure": int(source["rollouts_with_trigger"]),
        "near_failure_episode_rate": float(source["near_failure_episode_rate"]),
        "near_failure_recovered_successes": int(source["near_failure_recovered_successes"]),
        "near_failure_recovery_rate": float(source["near_failure_recovery_rate"]),
        "trigger_type_distribution": source["trigger_type_distribution"],
        "trigger_recovery_by_type": source["trigger_recovery_by_type"],
        "early_contacts": sum(row.get("first_contact_frame") is not None for row in attempted),
        "final_terminal_error_mm": _distribution([
            row["final_terminal_error_m"] for row in attempted
        ], 1000.0),
        "maximum_cube_displacement_mm": _distribution([
            row["maximum_cube_displacement_m"] for row in attempted
        ], 1000.0),
    }


def _checkpoint_key(item: dict[str, Any]) -> tuple:
    unseen = item["unseen_staged"]
    regression = item["training_seed_staged"]
    return (
        unseen["approach_conditional_success_rate"],
        -unseen["cube_motion_safety_failures"],
        -unseen["maximum_cube_displacement_mm"]["p90"],
        -unseen["timeouts"],
        regression["approach_conditional_success_rate"],
    )


def _case(
    selected: dict[str, Any], baseline: dict[str, Any],
    baseline_training: dict[str, Any], correction_fraction: float,
) -> tuple[str, str]:
    unseen = selected["unseen_staged"]
    training = selected["training_seed_staged"]
    delta = (
        unseen["approach_conditional_success_rate"]
        - baseline["approach_conditional_success_rate"]
    )
    nominal_degraded = (
        training["approach_conditional_success_rate"] + 0.025
        < baseline_training["approach_conditional_success_rate"]
    )
    recovery_better = (
        unseen["near_failure_recovery_rate"]
        > baseline["near_failure_recovery_rate"] + 0.03
        or unseen["timeouts"] < baseline["timeouts"]
        or unseen["cube_motion_safety_failures"] < baseline["cube_motion_safety_failures"]
    )
    safety_not_worse = (
        unseen["cube_motion_safety_failures"] <= baseline["cube_motion_safety_failures"]
        and unseen["maximum_cube_displacement_mm"]["p90"]
        <= baseline["maximum_cube_displacement_mm"]["p90"] + 0.5
    )
    if delta >= 0.05 and safety_not_worse and not nominal_degraded:
        return (
            "Case A",
            "Correction demos significantly improve unseen Approach robustness without hurting nominal performance.",
        )
    if recovery_better and nominal_degraded:
        if correction_fraction <= 0.205:
            return (
                "Case B",
                "At 20% correction, cube safety improves but overall recovery and staged nominal performance still regress; keep the original baseline and inspect the sampling/target conflict before another mix retry.",
            )
        return (
            "Case B",
            "Correction demos improve cube safety but not overall recovery, while staged nominal Approach regresses; retry 20% corrections.",
        )
    worse_success = delta < -0.025
    worse_recovery = (
        unseen["near_failure_recovery_rate"] < baseline["near_failure_recovery_rate"]
        and unseen["timeouts"] >= baseline["timeouts"]
        and unseen["cube_motion_safety_failures"] >= baseline["cube_motion_safety_failures"]
    )
    if worse_success and worse_recovery and nominal_degraded:
        return (
            "Case D",
            "Correction demos worsen both nominal and recovery behavior; revert and diagnose sampling mismatch.",
        )
    return (
        "Case C",
        "Correction demos produce little or no robust matched improvement; inspect representation or terminal formulation before collecting more of the same data.",
    )


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=Path,
        default=root / "outputs/approach_act_with_correction",
    )
    parser.add_argument(
        "--baseline-training-nominal", type=Path,
        default=root / "outputs/act_staged/approach_only/rollouts_step_002000/summary.json",
    )
    parser.add_argument(
        "--baseline-training-staged", type=Path,
        default=root / "outputs/act_staged/reach_approach_step_002000/summary.json",
    )
    parser.add_argument(
        "--baseline-unseen", type=Path,
        default=root / "outputs/approach_act_with_correction/baseline_unseen/summary.json",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=root / "outputs/approach_act_with_correction/summary.json",
    )
    parser.add_argument(
        "--output-doc", type=Path,
        default=root / "docs/approach_act_with_correction_demos.md",
    )
    parser.add_argument(
        "--previous-correction-summary", type=Path, default=None,
        help="Optional prior correction-mix summary for a matched comparison row.",
    )
    args = parser.parse_args()

    baseline_nominal = _evaluation(_json(args.baseline_training_nominal))
    baseline_training = _evaluation(_json(args.baseline_training_staged))
    baseline_unseen = _recovery(_json(args.baseline_unseen))
    checkpoints = []
    for step in STEPS:
        checkpoints.append({
            "step": step,
            "checkpoint": (
                args.experiment / "act_train" / "checkpoints"
                / f"{step:06d}" / "pretrained_model"
            ).resolve().as_posix(),
            "training_seed_nominal": _evaluation(_json(
                args.experiment / f"nominal_step_{step:06d}" / "summary.json"
            )),
            "training_seed_staged": _evaluation(_json(
                args.experiment / f"staged_training_step_{step:06d}" / "summary.json"
            )),
            "unseen_staged": _recovery(_json(
                args.experiment / f"unseen_step_{step:06d}" / "summary.json"
            )),
        })
    sampling = _json(args.experiment / "sampling_report.json")
    correction_fraction = float(sampling["actual_source_fractions"]["correction"])
    previous = None
    previous_selected = None
    if args.previous_correction_summary is not None:
        previous = _json(args.previous_correction_summary)
        previous_selected = next(
            item for item in previous["checkpoints"]
            if int(item["step"]) == int(previous["selected_step"])
        )
    selected = max(checkpoints, key=_checkpoint_key)
    case, conclusion = _case(
        selected, baseline_unseen, baseline_training, correction_fraction
    )
    proceed = case == "Case A"
    summary = {
        "experiment": (
            "Approach ACT with targeted near-failure correction demonstrations "
            f"({100 * correction_fraction:.1f}% correction actual)"
        ),
        "model_and_interface_unchanged": True,
        "training": {
            "original_dataset": {"episodes": 20, "frames": 957},
            "correction_dataset": {"episodes": 30, "frames": 559},
            "sampling": sampling,
            "fresh_training": True,
            "steps": list(STEPS),
            "state_dim": 27,
            "action_dim": 27,
            "chunk_size": 20,
            "n_action_steps": 20,
            "execution_horizon": 1,
            "vision_backbone": "ResNet18 ImageNet",
        },
        "baselines": {
            "training_seed_nominal": baseline_nominal,
            "training_seed_staged": baseline_training,
            "unseen_1200_1399_original_approach": baseline_unseen,
            "historical_mining_1000_1199": {
                "reach_successes": 160,
                "approach_attempts": 160,
                "approach_successes": 115,
                "conditional_success_rate": 115 / 160,
                "note": "historical context only; not the matched unseen comparison",
            },
        },
        "previous_correction_experiment": previous,
        "checkpoints": checkpoints,
        "selection_priority": [
            "unseen staged conditional Approach success",
            "cube safety",
            "timeout rate",
            "original-seed regression",
        ],
        "selected_step": selected["step"],
        "selected_checkpoint": selected["checkpoint"],
        "decision": case,
        "conclusion": conclusion,
        "proceed_to_grasp_preload_act": proceed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for item in checkpoints:
        nominal = item["training_seed_nominal"]
        staged = item["training_seed_staged"]
        unseen = item["unseen_staged"]
        rows.append(
            f"| {item['step']} | {nominal['approach_successes']}/20 | "
            f"{staged['reach_successes']}/20 | {staged['approach_successes']}/{staged['approach_attempts']} "
            f"({_pct(staged['approach_conditional_success_rate'])}) | "
            f"{unseen['reach_successes']}/200 | {unseen['approach_successes']}/{unseen['approach_attempts']} "
            f"({_pct(unseen['approach_conditional_success_rate'])}) | "
            f"{unseen['cube_motion_safety_failures']} | {unseen['timeouts']} | "
            f"{_pct(unseen['near_failure_recovery_rate'])} |"
        )
    actual = sampling["actual_source_fractions"]
    stratum_values = list(sampling["actual_correction_stratum_counts"].values())
    stratum_range = f"{min(stratum_values)}--{max(stratum_values)}"
    selected_unseen = selected["unseen_staged"]
    base = baseline_unseen
    previous_row = ""
    if previous_selected is not None:
        previous_unseen = previous_selected["unseen_staged"]
        previous_actual = previous["training"]["sampling"]["actual_source_fractions"]
        previous_row = (
            f"| Prior {100 * previous_actual['correction']:.1f}% correction "
            f"(step {previous_selected['step']}) | "
            f"{previous_unseen['reach_successes']}/200 | "
            f"{previous_unseen['approach_successes']}/{previous_unseen['approach_attempts']} "
            f"({_pct(previous_unseen['approach_conditional_success_rate'])}) | "
            f"{previous_unseen['joint_reach_approach_successes']}/200 | "
            f"{previous_unseen['cube_motion_safety_failures']} | "
            f"{previous_unseen['timeouts']} | "
            f"{_pct(previous_unseen['near_failure_episode_rate'])} | "
            f"{_pct(previous_unseen['near_failure_recovery_rate'])} | "
            f"{previous_unseen['maximum_cube_displacement_mm']['median']:.2f} / "
            f"{previous_unseen['maximum_cube_displacement_mm']['p90']:.2f} mm |"
        )
    failure_rows = []
    for outcome in ("success", "timeout", "cube_displacement", "reach_failure"):
        failure_rows.append(
            f"| {outcome} | {base['failure_distribution'].get(outcome, 0)} | "
            f"{selected_unseen['failure_distribution'].get(outcome, 0)} |"
        )
    trigger_rows = []
    for trigger in (
        "moving_away", "terminal_plateau", "gate_stall",
        "near_timeout_terminal", "cube_displacement",
    ):
        base_trigger = base["trigger_recovery_by_type"].get(
            trigger, {"episodes": 0, "recovery_rate": 0.0}
        )
        selected_trigger = selected_unseen["trigger_recovery_by_type"].get(
            trigger, {"episodes": 0, "recovery_rate": 0.0}
        )
        trigger_rows.append(
            f"| {trigger} | {base_trigger['episodes']} / "
            f"{_pct(base_trigger['recovery_rate'])} | "
            f"{selected_trigger['episodes']} / "
            f"{_pct(selected_trigger['recovery_rate'])} |"
        )
    staged_success_delta = (
        selected["training_seed_staged"]["approach_successes"]
        - baseline_training["approach_successes"]
    )
    if staged_success_delta > 0:
        staged_comparison = f"improves by {staged_success_delta} successes"
    elif staged_success_delta < 0:
        staged_comparison = f"regresses by {-staged_success_delta} successes"
    else:
        staged_comparison = "is unchanged"
    doc = f"""# Approach ACT with targeted correction demonstrations ({100 * correction_fraction:.1f}% actual)

## Outcome

**{case}.** {conclusion}

Selected checkpoint: `{selected['checkpoint']}`.  Proceed to Grasp/Preload ACT:
`{str(proceed).lower()}`.

## Controlled setup

- Original Approach: 20 episodes / 957 frames.
- Successful corrections: 30 episodes / 559 frames.
- Actual training mix: original `{_pct(actual['original'])}`, correction
  `{_pct(actual['correction'])}` ({sampling['samples']} samples total).
- Sampling is episode-first; correction slots are balanced by
  `(trigger_type, source_outcome)` before sampling an episode and frame.
- The ten correction strata received {stratum_range} samples each; no raw dataset
  file was modified.
- Fresh ACT; ImageNet ResNet-18; chunk20; `n_action_steps=20`; `H_exec=1`;
  27D actual state to 27D absolute controller target; front+wrist 240x320.
- Reach ACT, controllers, gates, MuJoCo, grasp, reward and action semantics were unchanged.

## Checkpoint results

| step | nominal Approach | train Reach | train Approach given Reach | unseen Reach | unseen Approach given Reach | unseen cube failures | unseen timeouts | recovery after trigger |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Checkpoint choice follows the requested priority: unseen conditional success,
then cube safety, timeout rate, and training-seed regression—not training loss.

## Matched unseen comparison: seeds 1200-1399

| policy | Reach | Approach given Reach | joint success | cube failures | timeouts | near-failure rate | recovery after near-failure | cube displacement median / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original Approach ACT | {base['reach_successes']}/200 | {base['approach_successes']}/{base['approach_attempts']} ({_pct(base['approach_conditional_success_rate'])}) | {base['joint_reach_approach_successes']}/200 | {base['cube_motion_safety_failures']} | {base['timeouts']} | {_pct(base['near_failure_episode_rate'])} | {_pct(base['near_failure_recovery_rate'])} | {base['maximum_cube_displacement_mm']['median']:.2f} / {base['maximum_cube_displacement_mm']['p90']:.2f} mm |
{previous_row}
| New step {selected['step']} | {selected_unseen['reach_successes']}/200 | {selected_unseen['approach_successes']}/{selected_unseen['approach_attempts']} ({_pct(selected_unseen['approach_conditional_success_rate'])}) | {selected_unseen['joint_reach_approach_successes']}/200 | {selected_unseen['cube_motion_safety_failures']} | {selected_unseen['timeouts']} | {_pct(selected_unseen['near_failure_episode_rate'])} | {_pct(selected_unseen['near_failure_recovery_rate'])} | {selected_unseen['maximum_cube_displacement_mm']['median']:.2f} / {selected_unseen['maximum_cube_displacement_mm']['p90']:.2f} mm |

The 1000-1199 mining result (115/160 conditional, 71.9%) is retained only as
historical context.  The table above is the valid matched comparison because
both policies see exactly the same unseen seeds 1200-1399.

## Nominal regression and safety

The original step-2000 baseline achieved
`{baseline_nominal['approach_successes']}/20` from exact expert Approach starts
and `{baseline_training['approach_successes']}/{baseline_training['approach_attempts']}`
after frozen Reach.  The selected correction model achieves
`{selected['training_seed_nominal']['approach_successes']}/20` and
`{selected['training_seed_staged']['approach_successes']}/{selected['training_seed_staged']['approach_attempts']}`
respectively.  Thus exact expert-start nominal performance improves, but the
real frozen-Reach handoff {staged_comparison}.  Per-checkpoint
terminal-error, cube-motion, early-contact, and action-clipping distributions
are preserved in `summary.json`.

## Failure-mode comparison

| outcome | Original ACT | New step {selected['step']} |
|---|---:|---:|
{chr(10).join(failure_rows)}

The correction model reduces formal cube-motion safety failures from
`{base['cube_motion_safety_failures']}` to
`{selected_unseen['cube_motion_safety_failures']}` and cube-displacement p90
from `{base['maximum_cube_displacement_mm']['p90']:.2f}` to
`{selected_unseen['maximum_cube_displacement_mm']['p90']:.2f}` mm.  This comes
with more timeouts (`{base['timeouts']}` to `{selected_unseen['timeouts']}`)
and a higher median cube displacement
(`{base['maximum_cube_displacement_mm']['median']:.2f}` to
`{selected_unseen['maximum_cube_displacement_mm']['median']:.2f}` mm).

## Recovery interpretation

`near_failure` uses the same five detectors used for mining: moving away,
terminal plateau, gate stall, near-timeout terminal, and cube displacement.
A recovery means a rollout triggered at least one detector and nevertheless
passed the unchanged Approach gate.  This is stricter and more informative
than looking only at aggregate success, while still treating overlapping
trigger types as one episode for the overall recovery rate.

| trigger | Original: episodes / recovery | New: episodes / recovery |
|---|---:|---:|
{chr(10).join(trigger_rows)}

Overall trigger-after-recovery is `{_pct(base['near_failure_recovery_rate'])}`
for the original policy and
`{_pct(selected_unseen['near_failure_recovery_rate'])}` for the selected model:
there is no aggregate recovery gain.  Cube-displacement triggers become much
more frequent (`{base['trigger_type_distribution'].get('cube_displacement', 0)}`
to `{selected_unseen['trigger_type_distribution'].get('cube_displacement', 0)}`),
but fewer cross the unchanged 25 mm safety-failure gate.  The learned trade is
therefore "smaller pushes and more timeouts", not "more reliable recovery".

## Decision

{conclusion}
"""
    args.output_doc.parent.mkdir(parents=True, exist_ok=True)
    args.output_doc.write_text(doc, encoding="utf-8")
    print(json.dumps({
        "selected_step": selected["step"],
        "decision": case,
        "proceed_to_grasp_preload_act": proceed,
        "unseen_conditional_success": selected_unseen["approach_conditional_success_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
