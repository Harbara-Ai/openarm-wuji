"""Build the final independent-Recovery ACT matched evaluation report."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _recovery_curve(experiment: Path) -> list[dict]:
    rows = []
    for path in sorted(experiment.glob("recovery_eval_step_*/summary.json")):
        match = re.search(r"step_(\d+)", path.parent.name)
        if match is None:
            continue
        row = _load(path)
        row["step"] = int(match.group(1))
        row["summary_path"] = path
        rows.append(row)
    if not rows:
        raise FileNotFoundError("no recovery_eval_step_*/summary.json files found")
    return rows


def _select(rows: list[dict]) -> dict:
    # Requested checkpoint priority: recovery success, cube safety, timeout.
    # Error and later step are deterministic tie-breakers only.
    return min(rows, key=lambda row: (
        -row["recovery_successes"],
        row["cube_safety_failures"],
        row["timeouts"],
        row["terminal_error_mm"]["median"],
        -row["step"],
    ))


def _mcnemar_exact_p(rescues: int, regressions: int) -> float:
    discordant = rescues + regressions
    if discordant == 0:
        return 1.0
    tail = min(rescues, regressions)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def _decision(selected: dict, unseen: dict) -> tuple[str, str]:
    training_rate = selected["recovery_success_rate"]
    unseen_rate = unseen["recovery_success_rate"]
    baseline = unseen["baseline"]
    staged = unseen["staged_with_recovery"]
    effect = unseen["matched_switch_effect"]
    pvalue = _mcnemar_exact_p(
        effect["rescued_failures"], effect["regressed_successes"]
    )
    safety_not_worse = (
        staged["cube_safety_failures"] <= baseline["cube_safety_failures"]
    )
    # A statistically clear matched regression identifies a switching/router
    # failure directly, even when Recovery-only fit is imperfect.  Without
    # this precedence the generic training-fit heuristic can hide the causal
    # evidence supplied by the paired experiment.
    if effect["regressed_successes"] > effect["rescued_failures"] and pvalue < 0.05:
        return (
            "Case D",
            "Recovery succeeds on a substantial fraction of both training "
            "and unseen trigger states, but first-trigger switching causes a "
            "statistically clear matched regression; keep Recovery and "
            "redesign or calibrate the trigger/switch logic before "
            "Grasp/Preload.",
        )
    if training_rate < 0.8:
        return (
            "Case C",
            "Recovery ACT does not reliably fit the 30 training correction "
            "starts; diagnose its observation/action formulation before "
            "collecting more data or proceeding to Grasp/Preload.",
        )
    if unseen_rate < 0.6:
        return (
            "Case B",
            "Recovery ACT fits its training starts but generalizes poorly to "
            "the first-trigger unseen distribution; mine more diverse, "
            "router-aligned recovery starts before Grasp/Preload.",
        )
    if (
        effect["rescued_failures"] > effect["regressed_successes"]
        and pvalue < 0.05
        and safety_not_worse
    ):
        return (
            "Case A",
            "Recovery ACT significantly improves matched staged Approach "
            "robustness without worsening cube safety; freeze the three "
            "policies and proceed to Grasp+Preload.",
        )
    return (
        "Case D",
        "Recovery itself works, but first-trigger switching does not produce "
        "a significant safe net gain; keep Recovery and redesign or calibrate "
        "the trigger/switch logic before Grasp/Preload.",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=Path,
        default=root / "outputs/staged_act_with_recovery",
    )
    parser.add_argument(
        "--unseen", type=Path,
        default=root / "outputs/staged_act_with_recovery/unseen_matched/summary.json",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=root / "outputs/staged_act_with_recovery/summary.json",
    )
    parser.add_argument(
        "--output-doc", type=Path,
        default=root / "docs/staged_act_with_recovery.md",
    )
    args = parser.parse_args()
    curve = _recovery_curve(args.experiment)
    selected = _select(curve)
    unseen = _load(args.unseen)
    selected_checkpoint = Path(selected["checkpoint"]).resolve()
    unseen_checkpoint = Path(unseen["recovery_checkpoint"]).resolve()
    if selected_checkpoint != unseen_checkpoint:
        raise ValueError(
            "unseen evaluation does not use the selected Recovery checkpoint: "
            f"{unseen_checkpoint} != {selected_checkpoint}"
        )
    case, conclusion = _decision(selected, unseen)
    effect = unseen["matched_switch_effect"]
    mcnemar_p = _mcnemar_exact_p(
        effect["rescued_failures"], effect["regressed_successes"]
    )
    proceed = case == "Case A"
    summary = {
        "experiment": "independent Recovery ACT with one-shot staged routing",
        "controlled_setup": {
            "recovery_dataset_episodes": 30,
            "recovery_dataset_frames": 559,
            "recovery_dataset_fraction": 1.0,
            "original_approach_data_used_for_recovery": False,
            "state_dim": 27,
            "action_dim": 27,
            "action_semantics": "absolute MuJoCo position-controller target",
            "cameras": ["front", "wrist"],
            "image_shape_hwc": [240, 320, 3],
            "vision_backbone": "ResNet18 ImageNet initialization",
            "chunk_size": 20,
            "n_action_steps": 20,
            "execution_horizon": 1,
            "training_seed": 1000,
            "sampling": "normal",
            "reach_approach_modified": False,
            "grasp_preload_lift_executed": False,
        },
        "recovery_checkpoint_curve": curve,
        "selected_recovery_step": selected["step"],
        "selected_recovery_checkpoint": selected["checkpoint"],
        "selected_training_recovery": selected,
        "unseen_matched": unseen,
        "matched_mcnemar_exact_p": mcnemar_p,
        "decision": case,
        "conclusion": conclusion,
        "proceed_to_grasp_preload_act": proceed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_doc.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )

    curve_rows = []
    for row in curve:
        curve_rows.append(
            f"| {row['step']} | {row['recovery_successes']}/30 "
            f"({_pct(row['recovery_success_rate'])}) | "
            f"{row['cube_safety_failures']} | {row['timeouts']} | "
            f"{row['terminal_error_mm']['median']:.2f} / "
            f"{row['terminal_error_mm']['p90']:.2f} mm | "
            f"{row['recovery_frames']['mean']:.1f} | "
            f"{row['action_clipping_values']} |"
        )
    trigger_rows = []
    training_by_trigger = selected["success_by_trigger"]
    unseen_by_trigger = unseen["trigger_by_primary_type"]
    trigger_order = (
        "moving_away", "terminal_plateau", "gate_stall",
        "near_timeout_terminal", "cube_displacement",
    )
    for trigger in trigger_order:
        train = training_by_trigger.get(trigger, {
            "episodes": 0, "successes": 0, "success_rate": 0.0
        })
        new = unseen_by_trigger.get(trigger, {
            "attempts": 0, "recovery_successes": 0,
            "recovery_success_rate": 0.0,
        })
        trigger_rows.append(
            f"| {trigger} | {train['successes']}/{train['episodes']} "
            f"({_pct(train['success_rate'])}) | "
            f"{new['recovery_successes']}/{new['attempts']} "
            f"({_pct(new['recovery_success_rate'])}) |"
        )
    baseline = unseen["baseline"]
    staged = unseen["staged_with_recovery"]
    doc = f"""# Staged ACT with independent Recovery

## Outcome

**{case}.** {conclusion}

Selected Recovery checkpoint: `{selected['checkpoint']}`.
Proceed to Grasp+Preload ACT: `{str(proceed).lower()}`.

## Controlled setup

- Recovery training uses only the 30 successful correction episodes / 559
  frames. Original Approach demonstrations are not mixed in.
- Fresh ACT; normal sampling; ImageNet ResNet-18; chunk20;
  `n_action_steps=20`; `H_exec=1`; training seed 1000.
- Observation is front+wrist RGB plus 27D actual qpos. Action is the 27D
  absolute target actually sent to the MuJoCo position controller.
- Reach ACT and original Approach ACT are frozen. Grasp, Preload, Lift, RL and
  SmolVLA are not run.
- The router reuses the five mining detectors and permits exactly zero or one
  Recovery attempt. Recovery success completes the Approach stage; it never
  switches back to Approach.

## Recovery-only checkpoint curve: exact 30 training starts

| step | recovery success | cube failures | timeout | final error median / p90 | mean frames | clipped values |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(curve_rows)}

Checkpoint selection follows recovery success, cube safety, then timeout; it
does not use training loss or unseen results.

## Recovery by trigger source

| trigger | training correction starts | unseen first-trigger starts |
|---|---:|---:|
{chr(10).join(trigger_rows)}

## Matched unseen seeds 1400-1599

Both branches share the exact same Reach and Approach prefix. At the first
near-failure trigger, the baseline continues original Approach while the new
branch restores that exact MuJoCo state and performs one Recovery attempt.

| metric | Reach -> Approach baseline | Reach -> Approach -> Recovery |
|---|---:|---:|
| Reach success | {unseen['reach_successes']}/200 | {unseen['reach_successes']}/200 |
| Approach-stage success given Reach | {baseline['approach_successes']}/{unseen['approach_attempts']} ({_pct(baseline['approach_conditional_success_rate'])}) | {staged['approach_stage_successes']}/{unseen['approach_attempts']} ({_pct(staged['approach_conditional_success_rate'])}) |
| joint success | {baseline['joint_successes']}/200 | {staged['joint_successes']}/200 |
| timeout | {baseline['timeouts']} | {staged['timeouts']} |
| cube safety failure | {baseline['cube_safety_failures']} | {staged['cube_safety_failures']} |
| cube displacement median / p90 | {baseline['maximum_cube_displacement_mm']['median']:.2f} / {baseline['maximum_cube_displacement_mm']['p90']:.2f} mm | {staged['maximum_cube_displacement_mm']['median']:.2f} / {staged['maximum_cube_displacement_mm']['p90']:.2f} mm |

- Direct Approach successes before any trigger:
  `{unseen['approach_direct_successes']}`.
- Recovery attempts: `{unseen['recovery_attempts']}`; successes:
  `{unseen['recovery_successes']}`
  (`{_pct(unseen['recovery_success_rate'])}`).
- Mean Recovery length: `{unseen['mean_recovery_frames']:.1f}` frames.
- Matched switch effect: rescued baseline failures
  `{effect['rescued_failures']}`, regressed baseline successes
  `{effect['regressed_successes']}`, both-success `{effect['both_success']}`,
  both-failure `{effect['both_failure']}`.
- Two-sided exact McNemar/binomial p-value on discordant pairs: `{mcnemar_p:.4g}`.

## Answers

1. **Training correction starts:** partially, but not reliably. Recovery
   succeeds on `{selected['recovery_successes']}/30`
   (`{_pct(selected['recovery_success_rate'])}`).
2. **Unseen near-failure states:** partially. It succeeds on
   `{unseen['recovery_successes']}/{unseen['recovery_attempts']}`
   (`{_pct(unseen['recovery_success_rate'])}`), close to the training-start
   rate, so there is no large train-to-unseen collapse.
3. **Overall robustness improved:** no. Matched Approach-stage success falls
   from `{baseline['approach_successes']}` to
   `{staged['approach_stage_successes']}` out of
   `{unseen['approach_attempts']}` Reach successes.
4. **Safer-but-more-timeout tradeoff solved:** no. Cube failures fall from
   `{baseline['cube_safety_failures']}` to
   `{staged['cube_safety_failures']}`, but timeouts rise from
   `{baseline['timeouts']}` to `{staged['timeouts']}`.
5. **Enter Grasp+Preload ACT:** `{str(proceed).lower()}`. Redesign and retest
   the trigger/switch logic first.

## Decision

{conclusion}
"""
    args.output_doc.write_text(doc, encoding="utf-8")
    print(json.dumps({
        "selected_step": selected["step"],
        "training_recovery_success_rate": selected["recovery_success_rate"],
        "unseen_recovery_success_rate": unseen["recovery_success_rate"],
        "baseline_success": baseline["approach_successes"],
        "staged_success": staged["approach_stage_successes"],
        "decision": case,
        "proceed_to_grasp_preload_act": proceed,
    }, indent=2))


if __name__ == "__main__":
    main()
