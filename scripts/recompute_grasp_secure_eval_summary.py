"""Recompute GraspSecure sub-outcomes from saved rollout trajectories.

This never reruns policy inference or MuJoCo.  It is useful when a diagnostic
definition changes but the formal grasp_preload_success gate and control
trajectory must remain exactly the matched rollout that was originally run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_grasp_secure_act import _plain, _summary


def _longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _find_rollout(directory: Path, seed: int) -> Path:
    matches = sorted(directory.glob(f"rollout_*_seed_{seed:06d}.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one rollout for seed {seed}, found {matches}"
        )
    return matches[0]


def _rewrite_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {
            key: loaded[key].copy()
            for key in loaded.files
            if key != "metrics_json"
        }
    arrays["metrics_json"] = np.asarray(json.dumps(_plain(metrics)))
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--minimum-fingers", type=int, default=2)
    parser.add_argument("--settle-window", type=int, default=8)
    args = parser.parse_args()
    old = json.loads(args.summary.read_text(encoding="utf-8"))
    directory = args.summary.parent
    gate = float(old["preload_gate_rad"])
    corrected: list[dict[str, Any]] = []
    for item in old["episodes_detail"]:
        if not item.get("grasp_attempted", True):
            corrected.append(item)
            continue
        rollout = _find_rollout(directory, int(item["seed"]))
        with np.load(rollout, allow_pickle=False) as loaded:
            active = np.asarray(loaded["active_finger_mask"], dtype=bool)
            preload = np.asarray(loaded["hand_preload_l2_rad"], dtype=float)
        multi = np.count_nonzero(active, axis=1) >= args.minimum_fingers
        grasp_success = _longest_run(multi) >= args.settle_window
        preload_success = False
        for end in range(args.settle_window, len(multi) + 1):
            start = end - args.settle_window
            if np.all(multi[start:end]) and float(np.mean(
                preload[start:end]
            )) >= gate:
                preload_success = True
                break
        success = bool(item["grasp_preload_success"])
        if success:
            failure_stage = None
        elif not np.any(multi):
            failure_stage = "contact_acquisition"
        elif not grasp_success:
            failure_stage = "grasp_formation"
        elif not preload_success:
            failure_stage = "preload_formation"
        else:
            failure_stage = "terminal_hold"
        item["grasp_success"] = bool(grasp_success)
        item["preload_success"] = bool(preload_success)
        item["failure_stage"] = failure_stage
        _rewrite_metrics(rollout, item)
        corrected.append(item)

    rebuilt = _summary(
        corrected,
        checkpoint=Path(old["checkpoint"]),
        mode=str(old["mode"]),
        preload_gate=gate,
    )
    for key in (
        "closed_loop_observation_only",
        "expert_state_or_action_injected",
        "frozen_upstream",
    ):
        if key in old:
            rebuilt[key] = old[key]
    rebuilt["diagnostic_recomputed_from_saved_trajectory"] = True
    rebuilt["formal_success_and_control_trajectory_unchanged"] = True
    args.summary.write_text(
        json.dumps(_plain(rebuilt), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
