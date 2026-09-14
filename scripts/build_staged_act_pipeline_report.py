"""Build the machine-readable summary and report for the staged ACT experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "act_staged"
DOC = ROOT / "docs" / "staged_act_pipeline.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _failure_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in summary["episodes"]:
        if episode["approach_success"]:
            key = "success"
        elif not episode.get("approach_attempted", True):
            key = "not_attempted"
        else:
            key = episode.get("failure_reason") or "other"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _checkpoint_row(step: int, summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    return {
        "step": step,
        "success_count": aggregate["approach_success_count"],
        "episodes": aggregate["approach_attempted_count"],
        "mean_minimum_position_error_mm": aggregate["minimum_position_error_mean_mm"],
        "mean_final_position_error_mm": aggregate["final_position_error_mean_mm"],
        "mean_maximum_cube_displacement_mm": aggregate["maximum_cube_displacement_mean_mm"],
        "maximum_cube_displacement_mm": aggregate["maximum_cube_displacement_max_mm"],
        "early_contact_count": aggregate["early_contact_count"],
        "clipped_action_values": aggregate["clipped_action_values"],
        "outcomes": _failure_counts(summary),
    }


def _handoff_split(summary: dict[str, Any]) -> dict[str, Any]:
    attempted = [
        episode for episode in summary["episodes"]
        if episode.get("approach_attempted", False)
    ]
    result: dict[str, Any] = {}
    for label, selected in (
        ("success", [episode for episode in attempted if episode["approach_success"]]),
        ("failure", [episode for episode in attempted if not episode["approach_success"]]),
    ):
        result[label] = {
            "count": len(selected),
            "same_seed_arm_l2_mean_rad": (
                sum(item["handoff"]["same_seed_arm_l2_rad"] for item in selected)
                / len(selected) if selected else None
            ),
            "nearest_expert_state_l2_mean_rad": (
                sum(item["handoff"]["nearest_expert_state_l2_rad"] for item in selected)
                / len(selected) if selected else None
            ),
        }
    return result


def main() -> None:
    reach_manifest_path = ROOT / "outputs/act_reach_only/dataset/manifest.json"
    reach_rollout_path = ROOT / "outputs/act_reach_only/rollouts_step_002000/summary.json"
    approach_manifest_path = OUTPUT / "approach_only/dataset/manifest.json"
    expert_replay_path = OUTPUT / "approach_only/expert_replay_validation.json"
    approach_paths = {
        step: OUTPUT / f"approach_only/rollouts_step_{step:06d}/summary.json"
        for step in (500, 1000, 1500, 2000)
    }
    chain_paths = {
        step: OUTPUT / f"reach_approach_step_{step:06d}/summary.json"
        for step in (1500, 2000)
    }
    reach_manifest = _load(reach_manifest_path)
    reach_rollout = _load(reach_rollout_path)
    approach_manifest = _load(approach_manifest_path)
    expert_replay = _load(expert_replay_path)
    approaches = {step: _load(path) for step, path in approach_paths.items()}
    chains = {step: _load(path) for step, path in chain_paths.items()}
    selected_step = 2000
    selected_approach = approaches[selected_step]
    selected_chain = chains[selected_step]
    reach_aggregate = reach_rollout["aggregate"]
    approach_aggregate = selected_approach["aggregate"]
    chain_aggregate = selected_chain["aggregate"]

    summary = {
        "scope": {
            "completed": ["freeze_reach", "approach_only", "reach_to_approach"],
            "not_started": ["grasp_preload_act", "lift_policy"],
            "scripted_lift_preserved": True,
        },
        "common_interface": {
            "observation_state_dim": 27,
            "action_dim": 27,
            "images": ["observation.images.front", "observation.images.wrist"],
            "image_shape_hwc": [240, 320, 3],
            "fps": 30,
            "chunk_size": 20,
            "n_action_steps": 20,
            "execution_horizon": 1,
            "expert_action_during_rollout": False,
        },
        "training_seeds": approach_manifest["episodes_detail"] and [
            item["seed"] for item in approach_manifest["episodes_detail"]
        ],
        "policy_1_reach": {
            "checkpoint": reach_rollout["checkpoint"],
            "dataset_frames": reach_manifest["frames"],
            "dataset_real_frames": reach_manifest["real_reach_frames"],
            "dataset_terminal_hold_frames": reach_manifest["synthetic_terminal_hold_frames"],
            "gate": {"position_mm": 12, "consecutive_frames": 5},
            "single_stage_success_count": reach_aggregate["reach_success_count"],
            "single_stage_episodes": 20,
            "rollout_summary": _relative(reach_rollout_path),
        },
        "policy_2_approach": {
            "selected_checkpoint": selected_approach["approach_checkpoint"],
            "selection_reason": (
                "Best chained closed-loop result (16/19 conditional); step 1500 had the "
                "best nominal-start count (12/20) but only 11/19 after real handoff."
            ),
            "dataset_frames": approach_manifest["frames"],
            "dataset_real_frames": approach_manifest["real_approach_frames"],
            "dataset_terminal_hold_frames": approach_manifest["synthetic_terminal_hold_frames"],
            "wuji_target_constant": approach_manifest["wuji_target_constant_during_approach"],
            "gate": selected_approach["gates"]["approach"],
            "expert_target_replay_validation": expert_replay["aggregate"],
            "closed_loop_by_checkpoint": [
                _checkpoint_row(step, approaches[step]) for step in approaches
            ],
            "selected_single_stage_success_count": approach_aggregate["approach_success_count"],
            "selected_single_stage_episodes": approach_aggregate["approach_attempted_count"],
            "selected_rollout_summary": _relative(approach_paths[selected_step]),
        },
        "reach_to_approach": {
            "selected_approach_step": selected_step,
            "reach_success_count": chain_aggregate["reach_success_count"],
            "approach_attempted_count": chain_aggregate["approach_attempted_count"],
            "approach_conditional_success_count": chain_aggregate["approach_success_count"],
            "joint_success_count": chain_aggregate["joint_reach_approach_success_count"],
            "total_episodes": chain_aggregate["episodes"],
            "failure_counts": _failure_counts(selected_chain),
            "handoff_distribution": selected_chain["handoff_distribution"],
            "handoff_success_failure_split": _handoff_split(selected_chain),
            "rollout_summary": _relative(chain_paths[selected_step]),
            "comparison_step_1500": {
                "conditional_success_count": chains[1500]["aggregate"]["approach_success_count"],
                "approach_attempted_count": chains[1500]["aggregate"]["approach_attempted_count"],
                "rollout_summary": _relative(chain_paths[1500]),
            },
        },
        "answers": {
            "approach_only_safe": (
                "Partially. The selected step-2000 policy succeeds 11/20 from exact expert "
                "pregrasp, but 6/20 cross the 25 mm cube-displacement safety gate."
            ),
            "real_reach_handoff": (
                "Yes for most seeds: Reach succeeds 19/20 and Approach succeeds on 16/19 "
                "real Reach endpoints, for 16/20 joint success."
            ),
            "remaining_failure_diagnosis": (
                "Not a handoff distribution mismatch. Chained performance improves over the "
                "exact expert-start evaluation, and failed handoffs are not farther from the "
                "expert initial-state set than successful handoffs. Remaining failures are "
                "Approach terminal/contact stabilization failures."
            ),
            "advance_to_grasp_training": False,
            "next_step": (
                "Add a small targeted set of Approach correction/stop demonstrations around "
                "seeds 2, 6, and 16 and around perturbed pregrasp endpoints, then rerun Stage 2/3."
            ),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    rows = []
    for item in summary["policy_2_approach"]["closed_loop_by_checkpoint"]:
        outcomes = item["outcomes"]
        rows.append(
            f"| {item['step']} | {item['success_count']}/20 | "
            f"{item['mean_minimum_position_error_mm']:.2f} | "
            f"{item['mean_final_position_error_mm']:.2f} | "
            f"{item['maximum_cube_displacement_mm']:.2f} | "
            f"{outcomes.get('cube_displacement', 0)} / {outcomes.get('timeout', 0)} | "
            f"{item['clipped_action_values']} |"
        )
    handoff = selected_chain["handoff_distribution"]
    split = summary["reach_to_approach"]["handoff_success_failure_split"]
    report = f"""# Staged ACT pipeline: Reach → Approach

## Scope and outcome

This run completed Stage 1–3 only: the best Reach-only ACT was frozen, a new
Approach-only ACT was trained from scratch, and both policies were composed with
explicit gates. Grasp/Preload ACT was **not started** and Lift remains scripted.

The main result is **19/20 Reach**, **16/19 conditional Approach**, and **16/20
joint Reach→Approach success** on the 20 training seeds. The real closed-loop
Reach endpoint is therefore usable as the Approach input; it is not the present
bottleneck. Approach is not yet robust enough to start Grasp/Preload training,
because three chained seeds still fail at the terminal approach/contact region.

## Shared policy interface

- Observation: front + wrist RGB at 240×320 and 27-D actual qpos
- Action: 27-D absolute controller target
- ACT: ImageNet ResNet-18, chunk size 20, `n_action_steps=20`
- Execution: `H_exec=1`; a new prediction is made from the current observation
- Training/evaluation seeds: `{summary['training_seeds']}`
- No expert action is read during any reported ACT rollout

The staged API is `ReachPolicy` / `ApproachPolicy`, each exposing `reset()`,
`select_action()`, `is_success()`, and `is_timeout()`. The outer evaluator owns
the phase switch; phase is not hidden inside a monolithic policy.

## Policy 1: frozen Reach

- Checkpoint: `{reach_rollout['checkpoint']}`
- Dataset: 20 episodes, {reach_manifest['frames']} frames
  ({reach_manifest['real_reach_frames']} real Reach +
  {reach_manifest['synthetic_terminal_hold_frames']} terminal hold)
- Gate: pregrasp position error ≤12 mm for five consecutive frames
- Closed-loop result: {reach_aggregate['reach_success_count']}/20; all 20 enter
  the 12 mm basin; no early cube contact and no action clipping
- Rollout: `{_relative(reach_rollout_path)}`

## Policy 2: Approach-only

The dataset starts at each demonstration's exact stable pregrasp state and ends
at the grasp-start state. It contains 20 episodes and {approach_manifest['frames']}
frames: {approach_manifest['real_approach_frames']} real Approach frames plus
{approach_manifest['synthetic_terminal_hold_frames']} terminal hold frames.
The Wuji 20-D target stays constant; no grasp-close action is present.

The scripted expert-target replay reaches the same gate on 20/20 seeds, with
mean minimum error {expert_replay['aggregate']['minimum_error_mean_mm']:.2f} mm
and maximum cube displacement
{expert_replay['aggregate']['maximum_cube_displacement_max_mm']:.3f} mm. Thus the
dataset endpoint and gate are dynamically reachable.

| step | Approach success | mean min error (mm) | mean final error (mm) | max cube move (mm) | cube-gate / timeout failures | clipped values |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Step 1500 has the best isolated nominal-start count (12/20). Step 2000 is kept
as Policy 2 for the composed pipeline because it is substantially better after
the real Reach handoff (16/19 versus 11/19 at step 1500) and has much lower
nominal-start mean minimum error (4.59 mm). Its isolated result is still only
11/20: six seeds cross the 25 mm cube-displacement safety gate and three time
out. Therefore the answer to “safe Approach-only?” is **partially, not robustly**.

Selected checkpoint: `{selected_approach['approach_checkpoint']}`

## Stage 3: real Reach → Approach handoff

No expert state or action is injected at the transition. The frame that satisfies
the Reach gate is passed directly to Approach ACT.

- Reach: {chain_aggregate['reach_success_count']}/20
- Approach given Reach: {chain_aggregate['approach_success_count']}/{chain_aggregate['approach_attempted_count']}
  ({chain_aggregate['approach_conditional_success_rate'] * 100:.1f}%)
- Joint Reach→Approach: {chain_aggregate['joint_reach_approach_success_count']}/20
- Remaining failures: seed 8 fails Reach; seed 2 crosses the Approach cube-motion
  gate; seeds 6 and 16 time out near the terminal/contact region
- Mean same-seed arm-state handoff distance: {handoff['same_seed_arm_l2_mean_rad']:.5f} rad
- Mean nearest expert 27-D start distance: {handoff['nearest_expert_state_l2_mean_rad']:.5f} rad
- Mean cube-position handoff mismatch: {handoff['cube_position_error_mean_mm']:.3f} mm
- Successful handoffs' mean same-seed arm distance: {split['success']['same_seed_arm_l2_mean_rad']:.5f} rad
- Failed handoffs' mean same-seed arm distance: {split['failure']['same_seed_arm_l2_mean_rad']:.5f} rad
- Rollout: `{_relative(chain_paths[selected_step])}`

This is **not evidence of a harmful handoff distribution mismatch**. The chained
policy improves from 11/20 at exact expert starts to 16/19 conditional success,
and failed handoffs are not farther from the same-seed expert start than successful
handoffs (0.00739 versus 0.00984 rad arm L2). The remaining failure mode belongs
to Policy 2's terminal/contact stabilization: it can enter the grasp-pose basin
but sometimes fails to stop before pushing the cube or cannot hold the full gate.

## Decision

Do not begin Grasp/Preload ACT yet. Keep the staged architecture and collect a
small, targeted Approach correction/stop set around failed chained seeds 2, 6,
and 16, plus modest perturbed-pregrasp endpoints produced by Reach ACT. This is
not a request for more generic full-task demonstrations: the missing supervision
is specifically the last Approach frames and recovery/stop behavior.

Machine-readable summary: `{_relative(OUTPUT / 'summary.json')}`

## Verification

- Native LeRobotDataset v3.0 export round-trip: state/action exact
- DataLoader smoke: state `[8,27]`, action `[8,20,27]`, both images
  `[8,3,240,320]`, padding mask `[8,20]`
- Scripted Approach endpoint replay: 20/20
- ACT checkpoints: 500, 1000, 1500, and 2000 steps
- Syntax check passed for the dataset builder, staged controller, replay, and
  evaluator; native exporter unit test passed
"""
    DOC.write_text(report, encoding="utf-8")
    print(DOC)
    print(OUTPUT / "summary.json")


if __name__ == "__main__":
    main()
