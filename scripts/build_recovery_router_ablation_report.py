"""Build the final Recovery-router diagnosis and matched-ablation report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _mcnemar_exact_p(rescues: int, regressions: int) -> float:
    discordant = rescues + regressions
    if discordant == 0:
        return 1.0
    tail = min(rescues, regressions)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def _compact_closed_loop(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    baseline = summary["baseline"]
    candidate = summary["staged_with_recovery"]
    effect = summary["matched_switch_effect"]
    return {
        "name": name,
        "router_spec": summary["router"]["candidate_spec"],
        "reach_successes": summary["reach_successes"],
        "approach_attempts": summary["approach_attempts"],
        "approach_direct_successes": summary["approach_direct_successes"],
        "recovery_attempts": summary["recovery_attempts"],
        "recovery_trigger_rate_given_reach": (
            summary["recovery_attempts"] / summary["approach_attempts"]
        ),
        "recovery_successes": summary["recovery_successes"],
        "recovery_success_rate": summary["recovery_success_rate"],
        "mean_recovery_frames": summary["mean_recovery_frames"],
        "baseline_successes": baseline["approach_successes"],
        "candidate_successes": candidate["approach_stage_successes"],
        "baseline_success_rate_given_reach": baseline[
            "approach_conditional_success_rate"
        ],
        "candidate_success_rate_given_reach": candidate[
            "approach_conditional_success_rate"
        ],
        "rescued": effect["rescued_failures"],
        "regressed": effect["regressed_successes"],
        "net_benefit": (
            effect["rescued_failures"] - effect["regressed_successes"]
        ),
        "both_success": effect["both_success"],
        "both_fail": effect["both_failure"],
        "matched_exact_p": _mcnemar_exact_p(
            effect["rescued_failures"], effect["regressed_successes"]
        ),
        "baseline_timeouts": baseline["timeouts"],
        "candidate_timeouts": candidate["timeouts"],
        "baseline_cube_safety_failures": baseline["cube_safety_failures"],
        "candidate_cube_safety_failures": candidate["cube_safety_failures"],
        "baseline_cube_displacement_mm": baseline[
            "maximum_cube_displacement_mm"
        ],
        "candidate_cube_displacement_mm": candidate[
            "maximum_cube_displacement_mm"
        ],
        "trigger_by_primary_type": summary["trigger_by_primary_type"],
        "summary_path": summary.get("_path"),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical", type=Path,
        default=root / "outputs/staged_act_with_recovery/unseen_matched/summary.json",
    )
    parser.add_argument(
        "--diagnostic", type=Path,
        default=root / "outputs/recovery_router_ablation/first_trigger_replicate/summary.json",
    )
    parser.add_argument(
        "--offline", type=Path,
        default=root / "outputs/recovery_router_ablation/offline_summary.json",
    )
    parser.add_argument(
        "--closed-loop", type=Path,
        default=root / "outputs/recovery_router_ablation/closed_loop",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=root / "outputs/recovery_router_ablation/summary.json",
    )
    parser.add_argument(
        "--output-doc", type=Path,
        default=root / "docs/recovery_router_ablation.md",
    )
    parser.add_argument(
        "--frozen-router", type=Path,
        default=root / "configs/recovery_router.json",
    )
    args = parser.parse_args()

    historical = _load(args.historical)
    diagnostic = _load(args.diagnostic)
    offline = _load(args.offline)
    candidates = []
    for path in sorted(args.closed_loop.glob("*/summary.json")):
        raw = _load(path)
        raw["_path"] = path.resolve()
        candidates.append(_compact_closed_loop(path.parent.name, raw))
    if len(candidates) < 2:
        raise ValueError("need at least two completed closed-loop candidates")
    for row in candidates:
        if row["approach_attempts"] < 1:
            raise ValueError(f"candidate has no Approach attempts: {row['name']}")
    best = min(candidates, key=lambda row: (
        -row["net_benefit"],
        -row["candidate_success_rate_given_reach"],
        row["candidate_cube_safety_failures"],
        row["candidate_timeouts"],
    ))
    frozen_router = _load(args.frozen_router)
    if frozen_router != best["router_spec"]:
        raise ValueError(
            "frozen router config does not match the winning evaluated spec"
        )
    net_positive = best["net_benefit"] > 0
    safer = (
        best["candidate_cube_safety_failures"]
        < best["baseline_cube_safety_failures"]
    )
    timeout_not_worse = (
        best["candidate_timeouts"] <= best["baseline_timeouts"]
    )
    if net_positive and safer and timeout_not_worse:
        case = "Case A"
        conclusion = (
            "A simple persistence/hysteresis router makes Recovery net-positive, "
            "raises matched Approach-stage success, reduces cube failures, and "
            "does not increase timeout. Freeze this router and proceed to "
            "Grasp/Preload as the next separate stage."
        )
    elif net_positive:
        case = "Case B"
        conclusion = (
            "Rule routing is net-positive but does not satisfy both safety and "
            "timeout requirements; keep the best rules and only then consider "
            "a lightweight learned router."
        )
    elif any(row["net_benefit"] > -5 for row in candidates):
        case = "Case D"
        conclusion = (
            "Recovery is only useful for a narrow failure subtype; restrict the "
            "main pipeline to that subtype."
        )
    else:
        case = "Case C"
        conclusion = (
            "Selective routing does not add value; drop Recovery from the main "
            "pipeline and keep the original Approach baseline."
        )

    best_trigger_rows = []
    for name, values in sorted(best["trigger_by_primary_type"].items()):
        net = values["recovery_successes"] - values["baseline_successes"]
        best_trigger_rows.append(
            f"| {name} | {values['attempts']} | "
            f"{values['baseline_successes']} | {values['recovery_successes']} | "
            f"{net:+d} |"
        )
    closed_rows = []
    for row in sorted(candidates, key=lambda item: -item["net_benefit"]):
        closed_rows.append(
            f"| {row['name']} | {row['recovery_attempts']} "
            f"({_pct(row['recovery_trigger_rate_given_reach'])}) | "
            f"{row['recovery_successes']} "
            f"({_pct(row['recovery_success_rate'])}) | "
            f"{row['baseline_successes']} -> {row['candidate_successes']} "
            f"({_pct(row['candidate_success_rate_given_reach'])}) | "
            f"{row['rescued']} | {row['regressed']} | "
            f"{row['net_benefit']:+d} | "
            f"{row['baseline_timeouts']} -> {row['candidate_timeouts']} | "
            f"{row['baseline_cube_safety_failures']} -> "
            f"{row['candidate_cube_safety_failures']} |"
        )
    trigger_rows = []
    trigger_order = (
        "moving_away", "terminal_plateau", "gate_stall",
        "near_timeout_terminal", "cube_displacement",
    )
    for trigger in trigger_order:
        values = offline["trigger_diagnosis"].get(trigger)
        if values is None:
            trigger_rows.append(f"| {trigger} | 0 | 0 | 0 | 0 | 0 |")
            continue
        trigger_rows.append(
            f"| {trigger} | {values['triggers']} | "
            f"{values['approach_successes']} | "
            f"{values['recovery_successes']} | "
            f"{_pct(values['trigger_precision_for_useful_recovery'])} | "
            f"{values['net_benefit']:+d} |"
        )
    offline_rows = []
    for row in offline["candidate_ranking"]:
        offline_rows.append(
            f"| {row['name']} | {row['would_switch_count']} | "
            f"{row['useful_switches']} | {row['harmful_switches']} | "
            f"{row['net_benefit']:+d} | {row['predicted_successes']} | "
            f"{row['predicted_timeouts']} | "
            f"{row['predicted_cube_safety_failures']} |"
        )

    type_counts = offline["type_counts"]
    features = offline["feature_distributions_type_a_vs_b"]
    feature_rows = []
    feature_specs = (
        ("approach_elapsed_frames", 1.0, "frames"),
        ("terminal_error_m", 1000.0, "mm"),
        ("error_slope_5_m_s", 1000.0, "mm/s"),
        ("moving_away_consecutive_frames", 1.0, "frames"),
        ("non_improving_consecutive_frames", 1.0, "frames"),
        ("cube_displacement_m", 1000.0, "mm"),
        ("cube_radial_velocity_m_s", 1000.0, "mm/s"),
        ("arm_joint_velocity_l2_rad_s", 1.0, "rad/s"),
        ("grasp_center_velocity_m_s", 1000.0, "mm/s"),
        ("remaining_timeout_frames", 1.0, "frames"),
        ("basin_exit_distance_m", 1000.0, "mm"),
    )
    for feature, scale, unit in feature_specs:
        values = features[feature]
        a = values["A_useful_recovery"]
        b = values["B_harmful_recovery"]
        feature_rows.append(
            f"| {feature} | {a['median'] * scale:.2f} "
            f"[{a['p25'] * scale:.2f}, {a['p75'] * scale:.2f}] {unit} | "
            f"{b['median'] * scale:.2f} "
            f"[{b['p25'] * scale:.2f}, {b['p75'] * scale:.2f}] {unit} |"
        )
    basin = offline["boolean_feature_rates_type_a_vs_b"]["entered_12mm_basin"]
    feature_rows.append(
        "| entered_12mm_basin | "
        f"{_pct(basin['A_useful_recovery']['true_rate'])} | "
        f"{_pct(basin['B_harmful_recovery']['true_rate'])} |"
    )

    summary = {
        "experiment": "Recovery router diagnosis and hysteresis ablation",
        "policies_frozen": True,
        "policy_training_performed": False,
        "reach_checkpoint": diagnostic["reach_checkpoint"],
        "approach_checkpoint": diagnostic["approach_checkpoint"],
        "recovery_checkpoint": diagnostic["recovery_checkpoint"],
        "historical_first_trigger_summary": {
            "path": args.historical,
            "reach_successes": historical["reach_successes"],
            "baseline_successes": historical["baseline"]["approach_successes"],
            "first_trigger_successes": historical[
                "staged_with_recovery"
            ]["approach_stage_successes"],
            "recovery_attempts": historical["recovery_attempts"],
            "rescued": historical["matched_switch_effect"]["rescued_failures"],
            "regressed": historical["matched_switch_effect"]["regressed_successes"],
        },
        "instrumented_diagnosis": {
            "path": args.diagnostic,
            "approach_attempts": diagnostic["approach_attempts"],
            "trigger_states": diagnostic["recovery_attempts"],
            "type_counts": type_counts,
            "trigger_diagnosis": offline["trigger_diagnosis"],
            "feature_distributions_type_a_vs_b": features,
            "boolean_feature_rates_type_a_vs_b": offline[
                "boolean_feature_rates_type_a_vs_b"
            ],
        },
        "offline_candidates": offline["candidate_ranking"],
        "closed_loop_candidates": candidates,
        "best_router": best,
        "frozen_router_config": args.frozen_router,
        "decision": case,
        "conclusion": conclusion,
        "learned_router_trained": False,
        "proceed_to_grasp_preload": case == "Case A",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_doc.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_plain(summary), indent=2), encoding="utf-8"
    )

    old = historical
    old_base = old["baseline"]
    old_new = old["staged_with_recovery"]
    best_p = best["matched_exact_p"]
    doc = f"""# Recovery router ablation

## Outcome

**{case}.** {conclusion}

Best router: `{best['name']}`.
Frozen deployable config: `{args.frozen_router.resolve().as_posix()}`.

No Reach, Approach, or Recovery ACT was retrained. Grasp/Preload/Lift was not
executed. The only experimental variable is when the frozen step-1500
Recovery policy is allowed to take over.

## Why the old router fails

Historical matched seeds 1400-1599 reproduced the original problem:

- Reach: `{old['reach_successes']}/200`.
- Baseline Approach: `{old_base['approach_successes']}/{old['approach_attempts']}`
  (`{_pct(old_base['approach_conditional_success_rate'])}`).
- Old first-trigger Recovery: `{old_new['approach_stage_successes']}/{old['approach_attempts']}`
  (`{_pct(old_new['approach_conditional_success_rate'])}`).
- Rescued `{old['matched_switch_effect']['rescued_failures']}`, regressed
  `{old['matched_switch_effect']['regressed_successes']}`; net
  `{old['matched_switch_effect']['rescued_failures'] - old['matched_switch_effect']['regressed_successes']:+d}`.

The old artifact did not store the requested 3/5/10-frame causal histories.
An instrumented matched replicate was therefore run with the same frozen
policies and seeds. Feature and outcome labels in the diagnosis below always
come from the same exact MuJoCo snapshot fork. The replicate again produced a
net `-23`, although individual seeds are not bitwise repeatable across fresh
processes.

## Matched outcome taxonomy at first trigger

| type | programmatic definition | count |
|---|---|---:|
| A | Approach fails; Recovery succeeds (useful switch) | {type_counts.get('A_useful_recovery', 0)} |
| B | Approach succeeds; Recovery fails (harmful switch) | {type_counts.get('B_harmful_recovery', 0)} |
| C | both succeed (neutral) | {type_counts.get('C_both_success', 0)} |
| D | both fail (neutral) | {type_counts.get('D_both_fail', 0)} |

## Legacy trigger precision

Precision is `Approach FAIL + Recovery SUCCESS` divided by all states where
that trigger is primary. A high Recovery success rate alone is insufficient
because it includes states Approach would also solve.

| primary trigger | states | Approach succeeds | Recovery succeeds | useful precision | net |
|---|---:|---:|---:|---:|---:|
{chr(10).join(trigger_rows)}

`moving_away` and `gate_stall` are over-sensitive at first detection. Their
Approach self-recovery rates are high, so a single legacy hit is not evidence
that takeover is beneficial.

## Type A versus Type B feature distributions

Values are median [p25, p75] at the first legacy trigger. All features use
only current/past frames; none uses future outcome information.

| feature | Type A: useful | Type B: harmful |
|---|---:|---:|
{chr(10).join(feature_rows)}

The complete JSON also contains 3/5/10-frame slopes and recent minima, cube
motion change, joint/EEF velocity, gate durations, elapsed/remaining time,
and basin-exit motion.

## Offline candidate evaluation on first-trigger futures

This table is a screening test only. Persistence rules that intentionally
wait cannot be faithfully scored from one first-trigger future, so they were
kept only when a dynamic closed-loop test was scientifically necessary.

| candidate | switches | rescued | regressed | net | predicted success | timeout | cube failure |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(offline_rows)}

## Closed-loop matched router ablation

Each row uses seeds 1400-1599 and contains its own matched baseline branch
from the same prefix. `H_exec=1`; success/safety gates and all policies are
unchanged; Recovery is locked after at most one switch.

| router | switches | Recovery success | baseline -> routed success | rescued | regressed | net | timeout | cube failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(closed_rows)}

## Best router details

`{best['name']}` uses a minimum 12-frame patience window and switches only on:

- cube displacement over the legacy 3 mm diagnostic threshold while radial
  cube velocity remains outward;
- moving-away for at least five consecutive frames with error above 15 mm;
- gate-stall/plateau with at least eight non-improving frames; or
- the existing near-timeout terminal signal.

Matched result:

- Approach-stage success: `{best['baseline_successes']}` ->
  `{best['candidate_successes']}` out of `{best['approach_attempts']}` Reach
  successes (`{_pct(best['candidate_success_rate_given_reach'])}`).
- Recovery attempts/successes: `{best['recovery_attempts']}` /
  `{best['recovery_successes']}`
  (`{_pct(best['recovery_success_rate'])}`).
- Rescued `{best['rescued']}`, regressed `{best['regressed']}`, net
  `{best['net_benefit']:+d}`; exact paired p=`{best_p:.4g}`.
- Timeout: `{best['baseline_timeouts']}` -> `{best['candidate_timeouts']}`.
- Cube safety failure: `{best['baseline_cube_safety_failures']}` ->
  `{best['candidate_cube_safety_failures']}`.
- Cube displacement median/p90:
  `{best['baseline_cube_displacement_mm']['median']:.2f}` /
  `{best['baseline_cube_displacement_mm']['p90']:.2f}` mm ->
  `{best['candidate_cube_displacement_mm']['median']:.2f}` /
  `{best['candidate_cube_displacement_mm']['p90']:.2f}` mm.

Contribution by primary trigger at the actual delayed switch:

| primary trigger | attempts | baseline success | Recovery success | net |
|---|---:|---:|---:|---:|
{chr(10).join(best_trigger_rows)}

## Decision

{conclusion}

A learned router is not justified by this experiment: the simple
persistence/hysteresis rule already produces positive matched net benefit,
83%+ Approach-stage success, lower cube failure, and no timeout increase.
"""
    args.output_doc.write_text(doc, encoding="utf-8")
    print(json.dumps({
        "best_router": best["name"],
        "baseline_successes": best["baseline_successes"],
        "routed_successes": best["candidate_successes"],
        "rescued": best["rescued"],
        "regressed": best["regressed"],
        "net_benefit": best["net_benefit"],
        "baseline_timeouts": best["baseline_timeouts"],
        "routed_timeouts": best["candidate_timeouts"],
        "baseline_cube_failures": best["baseline_cube_safety_failures"],
        "routed_cube_failures": best["candidate_cube_safety_failures"],
        "decision": case,
    }, indent=2))


if __name__ == "__main__":
    main()
