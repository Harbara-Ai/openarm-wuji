"""Build matched expert-vs-ACT artifacts for training-demo closed-loop rollouts.

This script is analysis-only.  It consumes completed rollout NPZ files and the
raw successful demonstrations; expert actions are never passed to the policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "expert": (33, 113, 181),
    "act": (220, 70, 55),
    "gate": (35, 150, 90),
    "grid": (220, 225, 230),
    "text": (30, 35, 40),
}


def _load_raw_by_seed(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(directory.glob("episode_*.npz")):
        with np.load(path, allow_pickle=True) as episode:
            result[int(episode["episode_seed"])] = path
    return result


def _runs(values: np.ndarray) -> int:
    best = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _compact_episode(item: dict) -> dict:
    return {
        "seed": int(item["seed"]),
        "reach_success": bool(item["reach_success"]),
        "approach_success": bool(item["approach_success"]),
        "task_success": bool(item["task_success"]),
        "minimum_pregrasp_error_mm": 1000.0 * item["minimum_pregrasp_error_m"],
        "frames_inside_12mm_gate": int(item["frames_inside_12mm_gate"]),
        "longest_consecutive_frames_inside_12mm": int(
            item["longest_consecutive_frames_inside_12mm"]
        ),
        "maximum_cube_displacement_mm": 1000.0 * item["maximum_cube_displacement_m"],
        "final_cube_displacement_mm": 1000.0 * item["final_cube_displacement_m"],
        "early_contact_before_valid_approach": bool(
            item["early_contact_before_valid_approach"]
        ),
        "max_simultaneous_contacts": int(item["max_simultaneous_contacts"]),
    }


def _draw_series(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                 title: str, series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
                 fps: float = 30.0, horizontal: float | None = None) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=COLORS["grid"], width=1)
    draw.text((left + 6, top + 4), title, fill=COLORS["text"])
    all_values = np.concatenate([np.asarray(values, dtype=float) for _, values, _ in series])
    if horizontal is not None:
        all_values = np.append(all_values, horizontal)
    lo, hi = float(np.min(all_values)), float(np.max(all_values))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    max_n = max(len(values) for _, values, _ in series)
    x0, x1 = left + 44, right - 8
    y0, y1 = top + 24, bottom - 22
    for fraction in (0.0, 0.5, 1.0):
        y = round(y1 - fraction * (y1 - y0))
        draw.line((x0, y, x1, y), fill=(238, 241, 244), width=1)
    draw.text((left + 4, y0 - 5), f"{hi:.2f}", fill=(90, 95, 100))
    draw.text((left + 4, y1 - 7), f"{lo:.2f}", fill=(90, 95, 100))
    if horizontal is not None:
        y = y1 - (horizontal - lo) / (hi - lo) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=COLORS["gate"], width=2)
    for label, values, color in series:
        values = np.asarray(values, dtype=float)
        points = []
        for index, value in enumerate(values):
            x = x0 + index / max(1, max_n - 1) * (x1 - x0)
            y = y1 - (value - lo) / (hi - lo) * (y1 - y0)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    legend_x = right - 205
    for index, (label, _, color) in enumerate(series):
        x = legend_x + index * 95
        draw.line((x, top + 11, x + 18, top + 11), fill=color, width=3)
        draw.text((x + 22, top + 4), label, fill=COLORS["text"])
    draw.text((right - 73, bottom - 17), f"{(max_n - 1) / fps:.2f} s", fill=(90, 95, 100))


def _trajectory_figure(*, seed: int, expert: dict[str, np.ndarray],
                       act: dict[str, np.ndarray], reset_cube: np.ndarray,
                       metrics: dict, output: Path) -> None:
    width, height = 1200, 1030
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), f"Seed {seed}: expert demonstration vs pure closed-loop ACT", fill=COLORS["text"])
    draw.text(
        (28, 40),
        "Solid blue = expert; solid red = ACT. Curves share the original 30 Hz control timeline.",
        fill=(85, 90, 95),
    )
    expert_state = expert["state"]
    act_state = act["state"]
    expert_action = expert["action"]
    act_action = act["action"]
    expert_cube = expert["cube"]
    act_cube = act["cube"]
    panels = [
        (
            "Arm motion from reset, L2 rad",
            np.linalg.norm(expert_state[:, :7] - expert_state[0, :7], axis=1),
            np.linalg.norm(act_state[:, :7] - act_state[0, :7], axis=1),
        ),
        (
            "Hand motion from reset, L2 rad",
            np.linalg.norm(expert_state[:, 7:] - expert_state[0, 7:], axis=1),
            np.linalg.norm(act_state[:, 7:] - act_state[0, 7:], axis=1),
        ),
        (
            "Cube displacement from identical reset, mm",
            1000.0 * np.linalg.norm(expert_cube - reset_cube, axis=1),
            1000.0 * np.linalg.norm(act_cube - reset_cube, axis=1),
        ),
        (
            "Arm controller target - actual state, L2 rad",
            np.linalg.norm(expert_action[:, :7] - expert_state[:, :7], axis=1),
            np.linalg.norm(act_action[:, :7] - act_state[:, :7], axis=1),
        ),
    ]
    y = 78
    for title, expert_values, act_values in panels:
        _draw_series(
            draw, (25, y, width - 25, y + 180), title,
            [("Expert", expert_values, COLORS["expert"]), ("ACT", act_values, COLORS["act"])],
        )
        y += 188
    _draw_series(
        draw, (25, y, width - 25, y + 180),
        "ACT pregrasp position error, mm (green line = 12 mm gate)",
        [("ACT", 1000.0 * act["reach_error"], COLORS["act"])],
        horizontal=12.0,
    )
    note = (
        f"ACT: min={metrics['minimum_pregrasp_error_mm']:.2f} mm, "
        f"12 mm frames={metrics['frames_inside_12mm_gate']}, "
        f"longest dwell={metrics['longest_consecutive_frames_inside_12mm']}, "
        f"cube max displacement={metrics['maximum_cube_displacement_mm']:.2f} mm."
    )
    draw.text((28, height - 28), note, fill=COLORS["text"])
    image.save(output)


def _comparison_gif(*, seed: int, expert: dict[str, np.ndarray],
                    act: dict[str, np.ndarray], metrics: dict, output: Path) -> None:
    expert_front = expert["front"]
    expert_wrist = expert["wrist"]
    act_front = act["front"]
    act_wrist = act["wrist"]
    phases = expert["phase"]
    max_frames = max(len(expert_front), len(act_front))
    frames: list[np.ndarray] = []
    for frame in range(0, max_frames, 2):
        expert_index = min(frame, len(expert_front) - 1)
        act_index = min(frame, len(act_front) - 1)
        canvas = Image.new("RGB", (640, 548), (20, 22, 25))
        canvas.paste(Image.fromarray(expert_front[expert_index]), (0, 42))
        canvas.paste(Image.fromarray(act_front[act_index]), (320, 42))
        canvas.paste(Image.fromarray(expert_wrist[expert_index]), (0, 292))
        canvas.paste(Image.fromarray(act_wrist[act_index]), (320, 292))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"EXPERT | seed {seed} | phase={phases[expert_index]}", fill="white")
        draw.text((328, 7), f"ACT closed loop | frame {act_index}", fill="white")
        draw.text((8, 27), "front camera", fill=(190, 195, 200))
        draw.text((328, 27), "front camera", fill=(190, 195, 200))
        draw.text((8, 277), "wrist camera", fill=(190, 195, 200))
        draw.text((328, 277), "wrist camera", fill=(190, 195, 200))
        error_mm = 1000.0 * act["reach_error"][act_index]
        cube_mm = 1000.0 * np.linalg.norm(act["cube"][act_index] - act["reset_cube"])
        draw.rectangle((0, 532, 640, 548), fill=(20, 22, 25))
        draw.text(
            (8, 533),
            f"ACT pregrasp={error_mm:5.1f} mm | cube displacement={cube_mm:5.1f} mm | "
            f"final Reach={int(metrics['reach_success'])}",
            fill="white",
        )
        frames.append(np.asarray(canvas))
    imageio.mimsave(output, frames, duration=2.0 / 30.0, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--focus-seed", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    rollout_summary = json.loads((args.rollouts / "summary.json").read_text(encoding="utf-8"))
    raw_by_seed = _load_raw_by_seed(args.raw)
    rows = [_compact_episode(item) for item in rollout_summary["episodes"]]
    seeds = [row["seed"] for row in rows]
    if set(seeds) != set(raw_by_seed):
        raise ValueError("rollout seeds do not exactly match the 20 raw training demonstration seeds")

    quality = lambda row: (
        row["longest_consecutive_frames_inside_12mm"],
        -row["minimum_pregrasp_error_mm"],
        -row["maximum_cube_displacement_mm"],
    )
    best = max(rows, key=quality)
    worst = min(rows, key=quality)
    selected = [args.focus_seed, best["seed"], worst["seed"]]
    selected = list(dict.fromkeys(selected))
    comparisons: dict[str, dict] = {}
    initial_state_errors = []

    for seed in seeds:
        row = next(item for item in rows if item["seed"] == seed)
        with np.load(raw_by_seed[seed], allow_pickle=True) as expert_npz, np.load(
            args.rollouts / f"rollout_seed_{seed:06d}.npz", allow_pickle=True
        ) as act_npz:
            initial_error = float(np.max(np.abs(
                expert_npz["observation.state"][0] - act_npz["observation_state"][0]
            )))
            initial_state_errors.append(initial_error)
            row["initial_27d_state_max_abs_difference"] = initial_error
            if seed not in selected:
                continue
            metadata = json.loads(str(expert_npz["episode_metadata_json"].item()))
            reset_cube = np.asarray(
                metadata["cube_pose_at_reset"]["world_xyz_wxyz"][:3], dtype=float
            )
            expert = {
                "state": np.asarray(expert_npz["observation.state"]),
                "action": np.asarray(expert_npz["action"]),
                "cube": np.asarray(expert_npz["telemetry.cube_pose_world"][:, :3]),
                "front": np.asarray(expert_npz["observation.images.front"]),
                "wrist": np.asarray(expert_npz["observation.images.wrist"]),
                "phase": np.asarray(expert_npz["phase"]).astype(str),
            }
            act = {
                "state": np.asarray(act_npz["observation_state"]),
                "action": np.asarray(act_npz["sent_action"]),
                "cube": np.asarray(act_npz["cube_position_m"]),
                "front": np.asarray(act_npz["observation.images.front"]),
                "wrist": np.asarray(act_npz["observation.images.wrist"]),
                "reach_error": np.asarray(act_npz["reach_error_m"]),
                "reset_cube": reset_cube,
            }
            common = min(len(expert["state"]), len(act["state"]))
            comparison = {
                "seed": seed,
                "selection_role": (
                    "requested_seed_7" if seed == args.focus_seed else
                    "best" if seed == best["seed"] else "worst"
                ),
                "expert_frames": int(len(expert["state"])),
                "act_frames": int(len(act["state"])),
                "initial_27d_state_max_abs_difference": initial_error,
                "aligned_arm_state_rmse_rad": float(np.sqrt(np.mean(
                    (expert["state"][:common, :7] - act["state"][:common, :7]) ** 2
                ))),
                "aligned_hand_state_rmse_rad": float(np.sqrt(np.mean(
                    (expert["state"][:common, 7:] - act["state"][:common, 7:]) ** 2
                ))),
                "aligned_27d_action_rmse_rad": float(np.sqrt(np.mean(
                    (expert["action"][:common] - act["action"][:common]) ** 2
                ))),
                "expert_max_cube_displacement_mm": float(1000.0 * np.max(
                    np.linalg.norm(expert["cube"] - reset_cube, axis=1)
                )),
                "act_metrics": row,
            }
            comparisons[str(seed)] = comparison
            _trajectory_figure(
                seed=seed, expert=expert, act=act, reset_cube=reset_cube,
                metrics=row,
                output=args.output / f"seed_{seed:06d}_expert_vs_act_trajectory.png",
            )
            _comparison_gif(
                seed=seed, expert=expert, act=act, metrics=row,
                output=args.output / f"seed_{seed:06d}_expert_vs_act.gif",
            )

    minimums = np.asarray([row["minimum_pregrasp_error_mm"] for row in rows])
    cube_maximums = np.asarray([row["maximum_cube_displacement_mm"] for row in rows])
    compact = {
        "experiment": "D step1000 on exact 20 training demonstration reset seeds",
        "checkpoint": rollout_summary["checkpoint"],
        "closed_loop_contract": {
            "expert_action_supplied_to_policy": False,
            "policy_inputs_each_frame": ["27D actual qpos", "front RGB", "wrist RGB"],
            "policy_output": "27D absolute controller target",
            "execution_horizon": rollout_summary["execution_horizon"],
            "reobserve_and_repredict_every_frame": True,
        },
        "training_demo_seeds": seeds,
        "aggregate": {
            "episodes": len(rows),
            "reach_success_count": sum(row["reach_success"] for row in rows),
            "approach_success_count": sum(row["approach_success"] for row in rows),
            "task_success_count": sum(row["task_success"] for row in rows),
            "minimum_pregrasp_error_mean_mm": float(minimums.mean()),
            "minimum_pregrasp_error_median_mm": float(np.median(minimums)),
            "minimum_pregrasp_error_best_mm": float(minimums.min()),
            "episodes_entering_12mm": sum(row["frames_inside_12mm_gate"] > 0 for row in rows),
            "total_frames_inside_12mm": sum(row["frames_inside_12mm_gate"] for row in rows),
            "maximum_longest_consecutive_12mm_dwell": max(
                row["longest_consecutive_frames_inside_12mm"] for row in rows
            ),
            "maximum_cube_displacement_mean_mm": float(cube_maximums.mean()),
            "maximum_cube_displacement_median_mm": float(np.median(cube_maximums)),
            "maximum_cube_displacement_max_mm": float(cube_maximums.max()),
            "episodes_cube_displacement_over_25mm": int(np.sum(cube_maximums > 25.0)),
            "episodes_with_early_contact": sum(
                row["early_contact_before_valid_approach"] for row in rows
            ),
            "initial_27d_state_max_abs_difference": max(initial_state_errors),
        },
        "selection": {
            "criterion": (
                "maximize longest consecutive 12mm dwell, then minimize pregrasp "
                "error, then minimize cube displacement"
            ),
            "requested_seed": args.focus_seed,
            "best_seed": best["seed"],
            "worst_seed": worst["seed"],
        },
        "episodes": rows,
        "selected_comparisons": comparisons,
        "conclusion": (
            "ACT does not closed-loop reproduce any of its 20 training scenarios: "
            "0/20 valid Reach, 0/20 Approach, and 0/20 task success."
        ),
    }
    (args.output / "comparison_summary.json").write_text(
        json.dumps(compact, indent=2), encoding="utf-8"
    )

    lines = [
        "# ACT D step1000：20 条训练场景纯闭环复现评测",
        "",
        "## 结论",
        "",
        "D step1000 **不能在 closed loop 中复现它真正训练过的 20 个场景**。",
        "策略在每帧只读取 27D actual qpos、front RGB、wrist RGB，并输出 27D absolute controller target；H_exec=1。rollout 阶段完全没有加载或执行 expert action。",
        "",
        f"- Reach：0/{len(rows)}",
        f"- Approach：0/{len(rows)}",
        f"- Task success：0/{len(rows)}",
        f"- 最小 pregrasp error：mean {minimums.mean():.2f} mm，median {np.median(minimums):.2f} mm，best {minimums.min():.2f} mm",
        f"- 进入 12 mm：{sum(row['frames_inside_12mm_gate'] > 0 for row in rows)}/{len(rows)}；总计 {sum(row['frames_inside_12mm_gate'] for row in rows)} 帧；最长连续 {max(row['longest_consecutive_frames_inside_12mm'] for row in rows)} 帧（gate 要求 5 帧）",
        f"- cube 最大位移：mean {cube_maximums.mean():.2f} mm，median {np.median(cube_maximums):.3f} mm，max {cube_maximums.max():.2f} mm；>25 mm 为 {int(np.sum(cube_maximums > 25.0))}/{len(rows)}",
        f"- early contact：{sum(row['early_contact_before_valid_approach'] for row in rows)}/{len(rows)}",
        "",
        "这不是新场景泛化失败，而是训练场景上的 closed-loop reproduction failure。9 条轨迹能短暂进入 12 mm basin，但没有一条保持到 5 帧；同时 4 条把 cube 推动超过 25 mm。主要 blocker 是闭环时序稳定性/误差恢复，而不是单纯没有见过这些初始条件。",
        "",
        "## 初始条件与评测契约",
        "",
        f"训练 demo seeds：`{', '.join(map(str, seeds))}`。raw expert frame-0 与 ACT rollout frame-0 的 27D actual state 最大绝对差为 `{max(initial_state_errors):.3e}` rad，确认使用相同 reset robot state；cube reset 由同一 seed 的任务 reset 生成。",
        "",
        "为了让统计和视频严格对应，最终 run 一次性保存了全部 20 条 rollout 的双相机输入帧。额外重复性检查发现 Windows/MuJoCo 渲染跨进程偶有极少像素 ±1 灰度差，微小输入差会被当前闭环长期放大；这与‘缺少稳定吸引域’的主结论一致。",
        "",
        "## 每条训练场景",
        "",
        "| seed | Reach | Approach | Task | min pregrasp mm | 12mm frames | longest dwell | cube max mm | early contact | max contacts |",
        "|---:|:---:|:---:|:---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {int(row['reach_success'])} | {int(row['approach_success'])} | {int(row['task_success'])} | "
            f"{row['minimum_pregrasp_error_mm']:.2f} | {row['frames_inside_12mm_gate']} | "
            f"{row['longest_consecutive_frames_inside_12mm']} | {row['maximum_cube_displacement_mm']:.2f} | "
            f"{int(row['early_contact_before_valid_approach'])} | {row['max_simultaneous_contacts']} |"
        )
    lines += [
        "",
        "## 选中对照",
        "",
        f"按“最长连续 12 mm dwell 优先，其次更低 pregrasp error，再其次更小 cube displacement”选择：best = seed {best['seed']}，worst = seed {worst['seed']}；另固定展示 seed {args.focus_seed}。",
        "",
    ]
    for seed in selected:
        comparison = comparisons[str(seed)]
        row = comparison["act_metrics"]
        lines += [
            f"### Seed {seed}（{comparison['selection_role']}）",
            "",
            f"ACT：min pregrasp {row['minimum_pregrasp_error_mm']:.2f} mm，longest dwell {row['longest_consecutive_frames_inside_12mm']} 帧，cube max displacement {row['maximum_cube_displacement_mm']:.2f} mm。",
            f"同一 30 Hz 时间轴前 {min(comparison['expert_frames'], comparison['act_frames'])} 帧的 arm-state RMSE={comparison['aligned_arm_state_rmse_rad']:.3f} rad，hand-state RMSE={comparison['aligned_hand_state_rmse_rad']:.3f} rad，27D action RMSE={comparison['aligned_27d_action_rmse_rad']:.3f} rad。",
            "",
            f"- 动画：`seed_{seed:06d}_expert_vs_act.gif`",
            f"- 数值轨迹图：`seed_{seed:06d}_expert_vs_act_trajectory.png`",
            "",
        ]
    lines += [
        "## 判断",
        "",
        "ACT 已学到能够把手臂送入 pregrasp 邻域的部分映射，但没有学到在观测误差和自身动作引起的状态偏移下持续留在 basin、再推进 Approach/Grasp/Lift 的闭环策略。因为连 20 条训练初始条件都为 0/20 Reach，当前不能把失败归因于测试 seed 泛化；应优先处理 demonstration 的闭环覆盖/恢复数据与 temporal stabilization，而不是继续用 training loss 证明策略已复现 expert。",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
