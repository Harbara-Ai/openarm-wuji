"""Combine fixed-seed ACT rollout metrics across training checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LOSS_BY_STEP = {
    100: 3.619,
    200: 3.171,
    300: 2.505,
    400: 2.268,
    500: 2.111,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=Path,
        default=root / "outputs/act_e2e_smoke",
    )
    parser.add_argument("--steps", default="100,200,300,400,500")
    parser.add_argument("--seeds", default="0,7,11")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    successful_dir = args.experiment / "coordinated_demos/raw/successful"
    rows = []

    for step in steps:
        rollout_dir = args.experiment / f"rollouts/step_{step:06d}"
        rollout_summary = json.loads(
            (rollout_dir / "summary.json").read_text(encoding="utf-8")
        )
        episodes = {int(item["seed"]): item for item in rollout_summary["episodes"]}
        step_rows = []
        for seed in seeds:
            source_matches = sorted(successful_dir.glob(f"*seed_{seed:06d}.npz"))
            if len(source_matches) != 1:
                raise RuntimeError(f"expected one expert episode for seed {seed}")
            with np.load(source_matches[0], allow_pickle=False) as expert, np.load(
                rollout_dir / f"rollout_seed_{seed:06d}.npz",
                allow_pickle=False,
            ) as rollout:
                predicted = rollout["predicted_action"]
                expected = expert["action"]
                first_error = np.abs(predicted[0] - expected[0])
                compare_frames = min(10, len(predicted), len(expected))
                first_ten_error = np.abs(
                    predicted[:compare_frames] - expected[:compare_frames]
                )
                row = {
                    **episodes[seed],
                    "first_action_mae_rad": float(first_error.mean()),
                    "first_action_arm_mae_rad": float(first_error[:7].mean()),
                    "first_action_hand_mae_rad": float(first_error[7:].mean()),
                    "first_10_action_mae_rad": float(first_ten_error.mean()),
                    "first_predicted_hand_target_mean_rad": float(
                        predicted[0, 7:].mean()
                    ),
                    "first_expert_hand_target_mean_rad": float(
                        expected[0, 7:].mean()
                    ),
                }
                step_rows.append(row)
        rows.append({
            "step": step,
            "training_loss": LOSS_BY_STEP.get(step),
            "successful_rollouts": sum(item["task_success"] for item in step_rows),
            "reach_successes": sum(item["reach_success"] for item in step_rows),
            "best_minimum_pregrasp_error_m": min(
                item["minimum_pregrasp_error_m"] for item in step_rows
            ),
            "mean_cube_displacement_m": float(np.mean([
                item["cube_displacement_m"] for item in step_rows
            ])),
            "maximum_simultaneous_contacts": max(
                item["max_simultaneous_contacts"] for item in step_rows
            ),
            "longest_two_contact_frames": max(
                item["longest_two_contact_frames"] for item in step_rows
            ),
            "mean_first_action_mae_rad": float(np.mean([
                item["first_action_mae_rad"] for item in step_rows
            ])),
            "mean_first_action_hand_mae_rad": float(np.mean([
                item["first_action_hand_mae_rad"] for item in step_rows
            ])),
            "mean_max_abs_action_clip_rad": float(np.mean([
                item["max_abs_action_clip_rad"] for item in step_rows
            ])),
            "episodes": step_rows,
        })

    report = {
        "fixed_training_seeds": seeds,
        "rollout_frames_per_seed": 160,
        "unchanged_contract": {
            "dataset_frames": 2804,
            "state_dim": 27,
            "action_dim": 27,
            "front_wrist_shape_chw": [3, 240, 320],
            "action_semantics": "absolute MuJoCo position-controller target",
        },
        "checkpoints": rows,
    }
    destination = args.output or args.experiment / "rollouts/checkpoint_comparison.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
