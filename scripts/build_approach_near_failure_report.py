"""Summarize targeted Approach mining and correction demonstrations."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/approach_correction_demos"
DOC = ROOT / "docs/approach_near_failure_mining.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float], *, scale: float = 1.0) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float) * scale
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _direction(vector: list[float]) -> str:
    array = np.asarray(vector, dtype=float)
    axis = int(np.argmax(np.abs(array)))
    return f"{'+' if array[axis] >= 0 else '-'}{'XYZ'[axis]}"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    mining_path = OUTPUT / "mining/summary.json"
    selection_path = OUTPUT / "selection.json"
    correction_path = OUTPUT / "correction_results.json"
    manifest_path = OUTPUT / "manifest.json"
    mining = _load(mining_path)
    selection = _load(selection_path)
    correction = _load(correction_path)
    validation = _load(OUTPUT / "validation.json")
    alignment = validation["native_alignment"]
    selected = selection["selected"]
    successful = [
        item for item in correction["results"] if item["correction_success"]
    ]
    approach_failures = [
        item for item in mining["rollouts"]
        if item["approach_attempted"] and not item["approach_success"]
    ]
    failure_modes = Counter(item["outcome"] for item in approach_failures)
    directions = Counter(
        _direction(item["initial_terminal_error_vector_m"])
        for item in successful
    )
    trigger_sources = Counter(item["trigger_type"] for item in selected)
    selected_outcomes = Counter(item["source_outcome"] for item in selected)
    successful_outcomes = Counter(item["source_outcome"] for item in successful)
    octants = {
        tuple(np.sign(np.asarray(
            item["initial_terminal_error_vector_m"], dtype=float
        )).astype(int).tolist())
        for item in successful
    }
    analysis = {
        "total_rollouts": mining["completed_rollouts"],
        "reach_successes": mining["reach_successes"],
        "approach_attempts": mining["approach_attempts"],
        "approach_successes": mining["approach_successes"],
        "approach_failures": mining["approach_failures"],
        "near_failure_triggers_total": mining["candidate_snapshots"],
        "rollouts_with_trigger": mining["rollouts_with_trigger"],
        "unique_snapshots_after_dedup": len(selected),
        "successful_correction_demos": len(successful),
        "failed_corrections": len(selected) - len(successful),
        "correction_failure_distribution": correction[
            "correction_failure_distribution"
        ],
        "trigger_type_distribution_all": mining["trigger_type_distribution"],
        "trigger_type_distribution_selected": dict(trigger_sources),
        "selected_source_outcome_distribution": dict(selected_outcomes),
        "successful_source_outcome_distribution": dict(successful_outcomes),
        "approach_failure_mode_distribution": dict(failure_modes),
        "most_common_failure_mode": (
            failure_modes.most_common(1)[0][0] if failure_modes else None
        ),
        "correction_direction_distribution": dict(directions),
        "most_common_correction_direction": (
            directions.most_common(1)[0][0] if directions else None
        ),
        "covered_dominant_error_directions": sorted(directions),
        "covered_error_octants": [list(item) for item in sorted(octants)],
        "selected_terminal_error_mm": _stats([
            item["terminal_error_m"] for item in selected
        ], scale=1000),
        "selected_cube_displacement_mm": _stats([
            item["cube_displacement_m"] for item in selected
        ], scale=1000),
        "successful_recovery_length_frames": _stats([
            item["recovery_length"] for item in successful
        ]),
        "successful_minimum_terminal_error_mm": _stats([
            item["minimum_terminal_error_m"] for item in successful
        ], scale=1000),
        "successful_maximum_additional_cube_displacement_mm": _stats([
            item["maximum_additional_cube_displacement_m"]
            for item in successful
        ], scale=1000),
        "restore_checks_all_passed": correction["restore_checks_all_passed"],
        "native_dataset": correction["native_dataset"],
        "native_alignment_validation": alignment,
        "act_chunk20_dataloader": validation["act_chunk20_dataloader"],
        "coverage_has_multiple_error_directions": len(directions) >= 3,
        "coverage_has_multiple_source_failure_modes": len(successful_outcomes) >= 2,
        "recommended_training_mix": {
            "original_approach_probability": 0.70,
            "correction_probability": 0.30,
            "sampling_unit": "episode/source-balanced windows, not raw frame concatenation",
            "correction_stratification": "balance trigger_type and source_outcome",
            "keep_action_semantics": "27D absolute controller target",
            "initial_ablation": [
                "original-only frozen baseline",
                "70% original + 30% corrections from scratch",
            ],
            "do_not_train_automatically": True,
        },
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    terminal = analysis["selected_terminal_error_mm"]
    cube = analysis["selected_cube_displacement_mm"]
    recovery = analysis["successful_recovery_length_frames"]
    extra_cube = analysis[
        "successful_maximum_additional_cube_displacement_mm"
    ]
    report = f"""# Approach near-failure targeted data mining

## 结论

冻结的 Reach ACT 与 Approach ACT 在新 seeds 上完成了
{analysis['total_rollouts']} 条纯 closed-loop rollout。Reach 成功
{analysis['reach_successes']}/{analysis['total_rollouts']}；在
{analysis['approach_attempts']} 次真实 Approach 中成功
{analysis['approach_successes']}，失败 {analysis['approach_failures']}。

检测器产生 {analysis['near_failure_triggers_total']} 个候选 snapshot，覆盖
{analysis['rollouts_with_trigger']} 条 rollout。按 terminal error 向量/幅值、
7D arm configuration、cube displacement 和 failure mode 去重后保留
{analysis['unique_snapshots_after_dedup']} 个；scripted Approach expert 从中成功
恢复 {analysis['successful_correction_demos']} 条。失败 correction 不进入数据集。

本轮没有修改或重训 Reach/Approach policy，也没有进入 Grasp、Preload、Lift、
RL 或 SmolVLA。

## Mining 设置

- Seeds: {mining['seed_start']}–{mining['seed_end_inclusive']}
- 流程：reset → frozen Reach ACT → Reach gate → frozen Approach ACT
- ACT rollout 不读取 expert action
- Near-failure trigger：terminal plateau、进入后远离、cube 位移 >3 mm、
  terminal-region near-timeout、目标附近 gate stall
- Snapshot 保存 `mjSTATE_INTEGRATION`、qpos/qvel/act/ctrl、solver warm-start、
  外力/mocap、controller bookkeeping、task reference、双相机与 27D observation
- 恢复检查：physics/state/ctrl 精确；离屏重渲染保持亚灰度级一致

Trigger 总分布：`{analysis['trigger_type_distribution_all']}`

Approach failure 分布：`{analysis['approach_failure_mode_distribution']}`。
最常见 failure mode 是 `{analysis['most_common_failure_mode']}`。

## 去重后 snapshot

- 数量：{analysis['unique_snapshots_after_dedup']}
- 来源 outcome：`{analysis['selected_source_outcome_distribution']}`
- Trigger：`{analysis['trigger_type_distribution_selected']}`
- Terminal error：median {terminal['median']:.2f} mm，p10–p90
  {terminal['p10']:.2f}–{terminal['p90']:.2f} mm，range
  {terminal['min']:.2f}–{terminal['max']:.2f} mm
- Cube displacement：median {cube['median']:.2f} mm，p10–p90
  {cube['p10']:.2f}–{cube['p90']:.2f} mm，range
  {cube['min']:.2f}–{cube['max']:.2f} mm

## Scripted correction

Expert 从完整恢复的 near-failure state 重新对准当前 cube 的 grasp-start，
Wuji 保持 snapshot 中的 open/pre-shape target，并在 Approach gate 后额外 hold
8 帧。动作仍是 27D absolute position-controller target。

- 成功：{analysis['successful_correction_demos']}/{analysis['unique_snapshots_after_dedup']}
- 失败 correction：`{analysis['correction_failure_distribution']}`（只保留诊断）
- Recovery length：median {recovery['median']:.1f} frames，p10–p90
  {recovery['p10']:.1f}–{recovery['p90']:.1f}
- Correction 造成的额外 cube 位移：median {extra_cube['median']:.3f} mm，
  p90 {extra_cube['p90']:.3f} mm
- 主导 correction direction：`{analysis['correction_direction_distribution']}`
- 最常见 correction direction：`{analysis['most_common_correction_direction']}`
- 覆盖的 dominant directions：`{analysis['covered_dominant_error_directions']}`
- 覆盖 error octants：{len(analysis['covered_error_octants'])}

## 数据集

- Raw correction trajectories：`{_relative(OUTPUT / 'raw')}`
- Native LeRobotDataset v3.0：`{_relative(OUTPUT / 'lerobot_dataset')}`
- Episodes / frames：{correction['native_dataset']['episodes']} /
  {correction['native_dataset']['frames']}
- Observation：front+wrist RGB 240×320，27D actual qpos
- Action：27D absolute controller target
- State/action exact round-trip：
  `{correction['native_dataset']['exact_round_trip']}`
- Episode boundary / timestamp / image alignment：`{alignment['passed']}`
- ACT chunk20 DataLoader：
  state `{validation['act_chunk20_dataloader']['observation.state']}`，
  action `{validation['act_chunk20_dataloader']['action']}`，pad
  `{validation['act_chunk20_dataloader']['action_is_pad']}`
- Manifest：`{_relative(manifest_path)}`
- Validation：`{_relative(OUTPUT / 'validation.json')}`

## 下一轮混合建议（本轮不执行）

第一组 matched experiment 建议从头训练，不 resume：每个 batch/window 以
**70% 原 Approach-only + 30% correction** 采样。不要直接按 raw frame 拼接，
因为 correction episode 较短；先按 source episode 平衡，再在 correction 内按
`trigger_type` 和 `source_outcome` 分层。保持 ImageNet ResNet-18、chunk20、
H_exec=1、27D absolute action 和现有 gate 不变，并与 original-only baseline
使用相同 seeds 对照。若安全位移改善但 nominal Approach 退化，再把 correction
比例降到 20%，不要同时改模型或 gate。

Machine-readable analysis：`{_relative(OUTPUT / 'analysis_summary.json')}`
"""
    DOC.write_text(report, encoding="utf-8")
    print(DOC)
    print(OUTPUT / "analysis_summary.json")


if __name__ == "__main__":
    main()
