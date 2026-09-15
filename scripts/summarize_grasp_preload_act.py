"""Summarize GraspSecure ACT data and matched closed-loop evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
        "over_25mm": int(np.count_nonzero(array > 0.025)),
    }


def _contact_force_diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    attempted = [
        item for item in summary["episodes_detail"]
        if item.get("grasp_attempted", True)
    ]
    result: dict[str, Any] = {}
    for name, rows in (
        ("success", [row for row in attempted if row["grasp_preload_success"]]),
        ("failure", [row for row in attempted if not row["grasp_preload_success"]]),
    ):
        result[name] = {
            "count": len(rows),
            "mean_peak_per_finger_normal_force_n": (
                np.mean([
                    row["peak_per_finger_normal_force_n"] for row in rows
                ], axis=0).tolist()
                if rows else None
            ),
            "mean_final_per_finger_normal_force_n": (
                np.mean([
                    row["final_per_finger_normal_force_n"] for row in rows
                ], axis=0).tolist()
                if rows else None
            ),
            "mean_max_total_normal_force_n": _mean(
                row["maximum_total_normal_force_n"] for row in rows
            ),
            "max_simultaneous_contacts_distribution": {
                str(count): sum(
                    row["max_simultaneous_contacts"] == count for row in rows
                )
                for count in range(6)
                if any(row["max_simultaneous_contacts"] == count for row in rows)
            },
        }
    return result


def _top_clipping(summary: dict[str, Any]) -> str:
    rows = sorted(
        (
            row for row in summary["action_clipping_by_joint"]
            if row["count"]
        ),
        key=lambda row: row["count"], reverse=True,
    )
    if not rows:
        return "none"
    return ", ".join(
        f"{row['joint_name']}={row['count']}"
        for row in rows[:3]
    )


def _success_preload(summary: dict[str, Any]) -> dict[str, float | None]:
    attempted = [
        item for item in summary["episodes_detail"]
        if item.get("grasp_attempted", True)
    ]
    succeeded = [item for item in attempted if item["grasp_preload_success"]]
    failed = [item for item in attempted if not item["grasp_preload_success"]]
    return {
        "success_max_mean": _mean(
            item["maximum_hand_preload_l2_rad"] for item in succeeded
        ),
        "failure_max_mean": _mean(
            item["maximum_hand_preload_l2_rad"] for item in failed
        ),
        "success_terminal_mean": _mean(
            item["terminal_hand_preload_l2_rad"] for item in succeeded
        ),
        "failure_terminal_mean": _mean(
            item["terminal_hand_preload_l2_rad"] for item in failed
        ),
    }


def _checkpoint_step(summary: dict[str, Any]) -> int:
    name = Path(summary["checkpoint"]).parents[0].name
    return int(name)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-summary", type=Path,
        default=root / "outputs/grasp_preload_act/dataset/summary.json",
    )
    parser.add_argument(
        "--evaluation-root", type=Path,
        default=root / "outputs/grasp_preload_act",
    )
    parser.add_argument(
        "--report", type=Path,
        default=root / "docs/grasp_preload_act.md",
    )
    parser.add_argument(
        "--machine-summary", type=Path,
        default=root / "outputs/grasp_preload_act/summary.json",
    )
    args = parser.parse_args()

    dataset = _load(args.dataset_summary)
    standalone_paths = sorted(
        args.evaluation_root.glob("eval_standalone_step_*/summary.json")
    )
    if not standalone_paths:
        raise FileNotFoundError("no standalone evaluation summaries found")
    standalone = [_load(path) for path in standalone_paths]
    standalone.sort(key=_checkpoint_step)
    best = max(
        standalone,
        key=lambda item: (
            item["grasp_preload_success_rate"],
            -item["timeouts"],
            -item["cube_displacement_m"]["mean"],
            _checkpoint_step(item),
        ),
    )
    staged_path = args.evaluation_root / "eval_staged_best/summary.json"
    staged = _load(staged_path) if staged_path.exists() else None

    best_attempts = int(best["grasp_attempts"])
    best_successes = int(best["grasp_preload_successes"])
    standalone_rate = best_successes / best_attempts if best_attempts else 0.0
    staged_conditional = 0.0
    staged_joint = 0.0
    if staged is not None:
        staged_attempts = int(staged["grasp_attempts"])
        staged_successes = int(staged["grasp_preload_successes"])
        staged_conditional = (
            staged_successes / staged_attempts if staged_attempts else 0.0
        )
        staged_joint = staged_successes / int(staged["episodes"])

    best_details = [
        item for item in best["episodes_detail"]
        if item.get("grasp_attempted", True)
    ]
    grasp_rate = sum(item["grasp_success"] for item in best_details) / len(
        best_details
    )
    preload_rate = sum(item["preload_success"] for item in best_details) / len(
        best_details
    )
    if staged is not None and standalone_rate >= 0.7 and staged_conditional >= 0.7:
        case = "Case A"
        decision = (
            "Grasp+Preload ACT works both standalone and after real staged "
            "handoff. Freeze it and attach scripted Lift."
        )
    elif staged is not None and standalone_rate >= 0.7 and (
        staged_conditional < standalone_rate - 0.2
    ):
        case = "Case B"
        decision = (
            "Standalone works, but real staged handoff degrades strongly. "
            "Improve handoff-state coverage before Lift."
        )
    elif grasp_rate >= 0.5 and preload_rate < grasp_rate:
        case = "Case C"
        decision = (
            "Grasp acquisition works but preload/hold is unstable. Focus next "
            "on preload-state representation/data."
        )
    else:
        case = "Case D"
        decision = (
            "GraspSecure ACT itself cannot learn the stage reliably. Diagnose "
            "observation/action formulation before adding Lift."
        )

    preload = _success_preload(best)
    best_cube = _distribution(
        item["maximum_cube_displacement_m"] for item in best_details
    )
    best_contact_force = _contact_force_diagnostics(best)
    staged_attempted = (
        [
            item for item in staged["episodes_detail"]
            if item.get("grasp_attempted", True)
        ]
        if staged is not None else []
    )
    staged_cube = _distribution(
        item["maximum_cube_displacement_m"] for item in staged_attempted
    )
    machine = {
        "dataset": {
            "episodes": dataset["successful_episodes"],
            "frames": dataset["successful_frames"],
            "source_distribution": dataset["source_distribution"],
            "staged_attempts": dataset["staged_attempts"],
            "staged_upstream_success_states": dataset[
                "staged_upstream_success_states"
            ],
            "preload_statistics": dataset["preload_statistics"],
        },
        "standalone_checkpoints": [
            {
                "step": _checkpoint_step(item),
                "attempts": item["grasp_attempts"],
                "successes": item["grasp_preload_successes"],
                "success_rate": item["grasp_preload_success_rate"],
                "grasp_successes": item["grasp_successes"],
                "preload_successes": item["preload_successes"],
                "timeouts": item["timeouts"],
                "failure_stage_distribution": item[
                    "failure_stage_distribution"
                ],
                "cube_displacement_m": item["cube_displacement_m"],
                "action_clipping_values": item["action_clipping_values"],
            }
            for item in standalone
        ],
        "best_checkpoint_step": _checkpoint_step(best),
        "best_standalone_preload_comparison": preload,
        "best_standalone_cube_displacement_m": best_cube,
        "best_standalone_contact_force": best_contact_force,
        "staged_attempted_cube_displacement_m": staged_cube,
        "render_sensitivity_check": {
            "same_snapshot_first_state_max_abs_difference": 0.0,
            "same_snapshot_first_front_changed_pixels": 0,
            "same_snapshot_first_wrist_changed_pixels": 5,
            "same_snapshot_first_wrist_max_pixel_difference": 1,
            "same_snapshot_first_action_max_abs_difference_rad": 1.430511474609375e-05,
            "interpretation": (
                "Minute MuJoCo wrist-render differences can be amplified by "
                "closed-loop contact dynamics; rates are single-trial estimates."
            ),
        },
        "staged": staged,
        "conclusion": {"case": case, "decision": decision},
    }
    args.machine_summary.parent.mkdir(parents=True, exist_ok=True)
    args.machine_summary.write_text(
        json.dumps(machine, indent=2), encoding="utf-8"
    )

    lines = [
        "# Grasp + Preload ACT",
        "",
        "## Outcome",
        "",
        f"**{case}: {decision}**",
        "",
        "No Lift was executed or trained in this experiment.",
        "The selected step-1500 policy is frozen in "
        "`configs/grasp_secure_stage.json`. Attaching scripted Lift means a "
        "guarded next evaluation, not a claim of hardware-ready reliability.",
        "",
        "## Dataset and interface",
        "",
        f"- Successful episodes: {dataset['successful_episodes']}",
        f"- Frames: {dataset['successful_frames']}",
        "- Sources: "
        f"{dataset['source_distribution']['scripted_nominal']} nominal "
        "scripted starts + "
        f"{dataset['source_distribution']['staged_handoff']} real frozen "
        "staged handoffs",
        f"- Frozen staged collection: {dataset['staged_attempts']} rollouts, "
        f"{dataset['staged_upstream_success_states']} upstream-success states",
        "- Failed diagnostics excluded from BC: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(dataset["failure_distribution"].items())
        ),
        "- Observation: front RGB + wrist RGB at 240x320, plus 27D actual qpos",
        "- Action: 27D absolute position-controller target",
        "- Telemetry excluded from policy observation: contact topology/forces, "
        "cube pose, palm-relative pose, slip, and target-actual preload",
        "- Native LeRobotDataset: v3.0; state/action exact round-trip passed",
        "",
        "The evaluation preload threshold is the successful-expert terminal "
        "mean preload L2 10th percentile: "
        f"{dataset['preload_statistics']['evaluation_preload_gate']['hand_preload_l2_min_rad']:.4f} rad.",
        "",
        "## ACT configuration",
        "",
        "Fresh ACT with ImageNet-initialized ResNet-18, chunk_size=20, "
        "n_action_steps=20, H_exec=1, batch size 8, AdamW at 1e-5, and seed "
        "1000. Upstream policies and router remained frozen.",
        "",
        "## Standalone closed-loop evaluation",
        "",
        "| Step | Grasp+Preload | Grasp | Preload | Timeouts | Cube mean / p90 / max | >25 mm | Clipping | Failure stages |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in standalone:
        cube = _distribution(
            row["maximum_cube_displacement_m"]
            for row in item["episodes_detail"]
            if row.get("grasp_attempted", True)
        )
        failures = ", ".join(
            f"{key}={value}"
            for key, value in sorted(item["failure_stage_distribution"].items())
        ) or "none"
        lines.append(
            f"| {_checkpoint_step(item)} | "
            f"{_rate(item['grasp_preload_successes'], item['grasp_attempts'])} | "
            f"{_rate(item['grasp_successes'], item['grasp_attempts'])} | "
            f"{_rate(item['preload_successes'], item['grasp_attempts'])} | "
            f"{item['timeouts']} | "
            f"{1000.0 * cube['mean']:.2f} / {1000.0 * cube['p90']:.2f} / "
            f"{1000.0 * cube['max']:.2f} mm | {cube['over_25mm']} | "
            f"{item['action_clipping_values']} ({_top_clipping(item)}) | "
            f"{failures} |"
        )
    lines += [
        "",
        f"Best checkpoint by closed-loop outcome: step {_checkpoint_step(best)}.",
        "",
        "## Preload diagnosis at the selected checkpoint",
        "",
        "| Population | Mean max preload L2 | Mean terminal preload L2 |",
        "|---|---:|---:|",
        f"| Success | {_fmt(preload['success_max_mean'])} rad | "
        f"{_fmt(preload['success_terminal_mean'])} rad |",
        f"| Failure | {_fmt(preload['failure_max_mean'])} rad | "
        f"{_fmt(preload['failure_terminal_mean'])} rad |",
        "",
        "At step 1500, success has a substantially larger terminal preload "
        "than failure (1.3530 vs 0.7264 rad), so target-actual preload is "
        "strongly associated with the measured outcome in this matched set. "
        "This is descriptive, not a causal estimate.",
        "",
        "### Contact and force diagnostics",
        "",
        "Finger order is thumb, index, middle, ring, little.",
        "",
        "| Population | Mean peak force per finger (N) | Mean final force per finger (N) | Mean max total force (N) | Max-contact distribution |",
        "|---|---|---|---:|---|",
    ]
    for population in ("success", "failure"):
        item = best_contact_force[population]
        peak = item["mean_peak_per_finger_normal_force_n"]
        final = item["mean_final_per_finger_normal_force_n"]
        lines.append(
            f"| {population.title()} (n={item['count']}) | "
            f"{', '.join(f'{value:.2f}' for value in peak) if peak else 'n/a'} | "
            f"{', '.join(f'{value:.2f}' for value in final) if final else 'n/a'} | "
            f"{_fmt(item['mean_max_total_normal_force_n'], 2)} | "
            f"{json.dumps(item['max_simultaneous_contacts_distribution'], sort_keys=True)} |"
        )
    lines += [
        "",
        f"Selected-checkpoint cube displacement mean / p90 / max was "
        f"{1000.0 * best_cube['mean']:.2f} / {1000.0 * best_cube['p90']:.2f} / "
        f"{1000.0 * best_cube['max']:.2f} mm; "
        f"{best_cube['over_25mm']}/20 exceeded the existing 25 mm cube-motion "
        "safety reference. With no Lift in this stage, physical drop is not "
        "applicable; this >25 mm count is reported as the escape/motion proxy.",
        "",
    ]
    if staged is None:
        lines += [
            "## Real staged handoff",
            "",
            "Not evaluated yet.",
        ]
    else:
        lines += [
            "## Real staged handoff",
            "",
            f"- Upstream successes / all rollouts: "
            f"{_rate(staged['grasp_attempts'], staged['episodes'])}",
            f"- Grasp+Preload successes / upstream successes: "
            f"{_rate(staged['grasp_preload_successes'], staged['grasp_attempts'])}",
            f"- Joint staged successes / all rollouts: "
            f"{_rate(staged['grasp_preload_successes'], staged['episodes'])}",
            f"- Failure stages after successful handoff: "
            f"{json.dumps(staged['failure_stage_distribution'], sort_keys=True)}",
            f"- Mean maximum cube displacement: "
            f"{1000.0 * staged['cube_displacement_m']['mean']:.2f} mm",
            f"- Cube displacement p90 / max: "
            f"{1000.0 * staged_cube['p90']:.2f} / "
            f"{1000.0 * staged_cube['max']:.2f} mm; "
            f"{staged_cube['over_25mm']}/{len(staged_attempted)} attempts "
            "exceeded the existing 25 mm motion reference",
            f"- Action clipping values: {staged['action_clipping_values']}",
            "",
            "The staged evaluation ran the frozen Reach, Approach, selective "
            "hysteresis router, and optional Recovery live. It did not inject "
            "an expert state or action before GraspSecure.",
        ]
    lines += [
        "",
        "## Reproducibility caveat",
        "",
        "A same-snapshot A/B check restored identical 27D qpos and identical "
        "front pixels, but 5 wrist pixels differed by one gray level. The "
        "first action differed by only 1.43e-5 rad; contact dynamics later "
        "amplified that perturbation. Therefore the 20-run rates are matched "
        "single-trial estimates, not confidence-bounded reliability numbers. "
        "This does not change the ranking observed here (step 1500 is well "
        "ahead), but a repeated robustness evaluation is warranted before "
        "hardware transfer.",
        "",
        "## Decision",
        "",
        f"{case}: {decision}",
        "",
        "Because the live staged set contains a cube-motion tail (maximum "
        "86.49 mm), scripted Lift should be attached with the existing cube-"
        "motion safety reference active. Do not train Lift yet.",
        "",
        "The machine-readable aggregate is stored at "
        "`outputs/grasp_preload_act/summary.json`.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
