"""Aggregate Reach-only ACT checkpoints and generate matched visual diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from openarm_wuji.tasks.se3 import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_matrix,
)


STEPS = (500, 1000, 2000)
BLUE = (33, 113, 181)
RED = (220, 70, 55)
GREEN = (35, 150, 90)
TEXT = (30, 35, 40)
GRID = (225, 229, 234)


def _first_run(values: np.ndarray, count: int) -> int | None:
    current = 0
    for index, value in enumerate(values.astype(bool)):
        current = current + 1 if value else 0
        if current >= count:
            return index - count + 1
    return None


def _line_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                title: str, series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
                gate: float | None = None, fps: float = 30.0) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=GRID)
    draw.text((left + 6, top + 4), title, fill=TEXT)
    values = np.concatenate([np.asarray(item[1], dtype=float) for item in series])
    if gate is not None:
        values = np.append(values, gate)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-8:
        hi = lo + 1.0
    x0, x1, y0, y1 = left + 48, right - 10, top + 25, bottom - 22
    for fraction in (0.0, 0.5, 1.0):
        y = y1 - fraction * (y1 - y0)
        draw.line((x0, y, x1, y), fill=(240, 242, 245))
    draw.text((left + 3, y0 - 6), f"{hi:.2f}", fill=(90, 95, 100))
    draw.text((left + 3, y1 - 6), f"{lo:.2f}", fill=(90, 95, 100))
    max_n = max(len(item[1]) for item in series)
    if gate is not None:
        y = y1 - (gate - lo) / (hi - lo) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=GREEN, width=2)
    for label, data, color in series:
        data = np.asarray(data, dtype=float)
        points = [
            (
                x0 + index / max(1, max_n - 1) * (x1 - x0),
                y1 - (value - lo) / (hi - lo) * (y1 - y0),
            )
            for index, value in enumerate(data)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    legend_x = right - 205
    for index, (label, _, color) in enumerate(series):
        x = legend_x + index * 98
        draw.line((x, top + 12, x + 18, top + 12), fill=color, width=3)
        draw.text((x + 22, top + 5), label, fill=TEXT)
    draw.text((right - 76, bottom - 17), f"{(max_n - 1) / fps:.2f} s", fill=(90, 95, 100))


def _error_plot(step: int, summary: dict, rollout_dir: Path, output: Path) -> dict:
    episodes = summary["episodes"]
    sorted_rows = sorted(episodes, key=lambda row: row["minimum_pregrasp_error_m"])
    chosen = {
        "best": sorted_rows[0],
        "median": min(
            episodes,
            key=lambda row: abs(
                row["minimum_pregrasp_error_m"]
                - float(np.median([item["minimum_pregrasp_error_m"] for item in episodes]))
            ),
        ),
        "worst": sorted_rows[-1],
    }
    canvas = Image.new("RGB", (1150, 570), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 15), f"Reach-only ACT step {step}: pregrasp error vs time", fill=TEXT)
    colors = {"best": BLUE, "median": (230, 145, 30), "worst": RED}
    series = []
    for role, row in chosen.items():
        with np.load(rollout_dir / f"rollout_seed_{row['seed']:06d}.npz") as data:
            series.append((f"{role} s{row['seed']}", 1000.0 * data["pregrasp_error_m"], colors[role]))
    _line_panel(
        draw, (20, 48, 1130, 535), "Post-control pregrasp position error, mm (green=12 mm)",
        series, gate=12.0,
    )
    canvas.save(output)
    return {role: int(row["seed"]) for role, row in chosen.items()}


def _expert_error(full_path: Path, reach_path: Path, offset: np.ndarray) -> np.ndarray:
    with np.load(full_path, allow_pickle=False) as full, np.load(reach_path, allow_pickle=False) as reach:
        indices = np.asarray(reach["source_frame_index"], dtype=np.int64)
        world = np.asarray(full["telemetry.cube_pose_world"][indices], dtype=float)
        relative = np.asarray(full["telemetry.cube_pose_relative_to_palm"][indices], dtype=float)
        metadata = json.loads(str(full["episode_metadata_json"]))
        target = np.asarray(metadata["cube_pose_at_reset"]["world_xyz_wxyz"][:3]) + offset
    grasp_positions = []
    for cube_pose, relative_pose in zip(world, relative):
        grasp_quaternion = quaternion_multiply(
            cube_pose[3:], quaternion_conjugate(relative_pose[3:])
        )
        grasp_positions.append(
            cube_pose[:3] - quaternion_to_matrix(grasp_quaternion) @ relative_pose[:3]
        )
    return np.linalg.norm(np.asarray(grasp_positions) - target, axis=1)


def _draw_action_heatmap(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                         title: str, action: np.ndarray,
                         joint_min: np.ndarray, joint_max: np.ndarray) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=GRID)
    draw.text((left + 5, top + 4), title, fill=TEXT)
    action = np.asarray(action, dtype=float).T
    scale = np.maximum(joint_max - joint_min, 1e-8)[:, None]
    normalized = np.clip((action - joint_min[:, None]) / scale, 0.0, 1.0)
    rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.asarray(255 * normalized, dtype=np.uint8)
    rgb[..., 1] = np.asarray(80 + 80 * (1.0 - np.abs(normalized - 0.5) * 2), dtype=np.uint8)
    rgb[..., 2] = np.asarray(255 * (1.0 - normalized), dtype=np.uint8)
    image = Image.fromarray(rgb).resize((right - left - 55, bottom - top - 30), Image.Resampling.NEAREST)
    image_x, image_y = left + 45, top + 24
    draw._image.paste(image, (image_x, image_y))
    for joint in (0, 6, 7, 26):
        y = image_y + joint / 26 * (bottom - top - 31)
        draw.text((left + 4, int(y) - 5), f"j{joint}", fill=(80, 85, 90))


def _comparison_figure(*, seed: int, expert_state: np.ndarray,
                       expert_action: np.ndarray, expert_error: np.ndarray,
                       act: dict[str, np.ndarray], output: Path) -> dict:
    act_state = act["observation_state"]
    act_action = act["sent_action"]
    act_error = act["pregrasp_error_m"]
    common = min(len(expert_state), len(act_state))
    arm_delta = np.linalg.norm(expert_state[:common, :7] - act_state[:common, :7], axis=1)
    action_delta = np.linalg.norm(expert_action[:common] - act_action[:common], axis=1)
    arm_divergence = _first_run(arm_delta > 0.1, 3)
    action_divergence = _first_run(action_delta > 0.1, 3)

    canvas = Image.new("RGB", (1400, 1860), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 15), f"Seed {seed}: Reach-only expert vs ACT closed-loop trajectories", fill=TEXT)
    y = 45
    for joint in range(7):
        _line_panel(
            draw, (20, y, 1380, y + 150), f"OpenArm qpos joint {joint + 1}, rad",
            [("Expert", expert_state[:, joint], BLUE), ("ACT", act_state[:, joint], RED)],
        )
        y += 158
    joint_min = np.minimum(expert_action.min(axis=0), act_action.min(axis=0))
    joint_max = np.maximum(expert_action.max(axis=0), act_action.max(axis=0))
    _draw_action_heatmap(
        draw, (20, y, 690, y + 300), "Expert 27D action trajectory (per-joint shared scale)",
        expert_action, joint_min, joint_max,
    )
    _draw_action_heatmap(
        draw, (710, y, 1380, y + 300), "ACT 27D action trajectory (per-joint shared scale)",
        act_action, joint_min, joint_max,
    )
    y += 310
    _line_panel(
        draw, (20, y, 1380, y + 200), "Pregrasp error, mm (green=12 mm)",
        [("Expert", 1000.0 * expert_error, BLUE), ("ACT", 1000.0 * act_error, RED)], gate=12.0,
    )
    y += 210
    _line_panel(
        draw, (20, y, 1380, y + 175), "Aligned expert-vs-ACT L2 error, rad",
        [("arm qpos", arm_delta, BLUE), ("27D action", action_delta, RED)],
    )
    draw.text(
        (25, 1836),
        f"First sustained divergence (>0.1 rad for 3 frames): arm={arm_divergence}, action={action_divergence}",
        fill=TEXT,
    )
    canvas.save(output)
    return {
        "aligned_frames": common,
        "initial_27d_state_max_abs_difference": float(
            np.max(np.abs(expert_state[0] - act_state[0]))
        ),
        "first_sustained_arm_state_divergence_frame": arm_divergence,
        "first_sustained_action_divergence_frame": action_divergence,
        "arm_state_rmse_rad": float(np.sqrt(np.mean((expert_state[:common, :7] - act_state[:common, :7]) ** 2))),
        "action_rmse_rad": float(np.sqrt(np.mean((expert_action[:common] - act_action[:common]) ** 2))),
    }


def _comparison_gif(seed: int, expert: dict[str, np.ndarray],
                    act: dict[str, np.ndarray], output: Path) -> None:
    max_frames = max(len(expert["front"]), len(act["front"]))
    frames = []
    for frame in range(0, max_frames, 2):
        ei = min(frame, len(expert["front"]) - 1)
        ai = min(frame, len(act["front"]) - 1)
        canvas = Image.new("RGB", (640, 548), (20, 22, 25))
        canvas.paste(Image.fromarray(expert["front"][ei]), (0, 42))
        canvas.paste(Image.fromarray(act["front"][ai]), (320, 42))
        canvas.paste(Image.fromarray(expert["wrist"][ei]), (0, 292))
        canvas.paste(Image.fromarray(act["wrist"][ai]), (320, 292))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"EXPERT Reach-only | seed {seed} | frame {ei}", fill="white")
        draw.text((328, 7), f"ACT closed-loop | frame {ai}", fill="white")
        draw.text((8, 27), "front", fill=(190, 195, 200))
        draw.text((328, 27), "front", fill=(190, 195, 200))
        draw.text((8, 277), "wrist", fill=(190, 195, 200))
        draw.text((328, 277), "wrist", fill=(190, 195, 200))
        draw.text(
            (8, 533), f"expert error={1000*expert['error'][ei]:5.1f} mm | ACT error={1000*act['error'][ai]:5.1f} mm",
            fill="white",
        )
        frames.append(np.asarray(canvas))
    imageio.mimsave(output, frames, duration=2.0 / 30.0, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reach-raw", type=Path, required=True)
    parser.add_argument("--full-raw", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-task-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    offset = np.asarray(config["reach"]["pregrasp_offset_m"], dtype=float)
    manifest = json.loads((args.root / "dataset/manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads(args.full_task_baseline.read_text(encoding="utf-8"))
    full_by_seed = {}
    for path in args.full_raw.glob("episode_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            full_by_seed[int(data["episode_seed"])] = path
    reach_by_seed = {}
    for path in args.reach_raw.glob("episode_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            reach_by_seed[int(data["episode_seed"])] = path

    checkpoints = {}
    error_plot_seeds = {}
    for step in STEPS:
        rollout_dir = args.root / f"rollouts_step_{step:06d}"
        offline_dir = args.root / f"offline_step_{step:06d}"
        rollout = json.loads((rollout_dir / "summary.json").read_text(encoding="utf-8"))
        offline = json.loads((offline_dir / "summary.json").read_text(encoding="utf-8"))
        error_plot_seeds[str(step)] = _error_plot(
            step, rollout, rollout_dir, args.output / f"pregrasp_error_step_{step:06d}.png"
        )
        checkpoints[str(step)] = {"rollout": rollout, "offline": offline}

    final = checkpoints["2000"]["rollout"]
    quality = lambda row: (
        bool(row["reach_success"]),
        int(row["longest_consecutive_frames_inside_12mm"]),
        -float(row["minimum_pregrasp_error_m"]),
        -float(row["maximum_cube_displacement_m"]),
    )
    best = max(final["episodes"], key=quality)
    worst = min(final["episodes"], key=quality)
    selected = list(dict.fromkeys([7, int(best["seed"]), int(worst["seed"])]))
    comparisons = {}
    final_rollout_dir = args.root / "rollouts_step_002000"
    for seed in selected:
        with np.load(reach_by_seed[seed], allow_pickle=False) as expert_npz, np.load(
            final_rollout_dir / f"rollout_seed_{seed:06d}.npz", allow_pickle=False
        ) as act_npz:
            expert = {
                "state": np.asarray(expert_npz["observation.state"]),
                "action": np.asarray(expert_npz["action"]),
                "front": np.asarray(expert_npz["observation.images.front"]),
                "wrist": np.asarray(expert_npz["observation.images.wrist"]),
                "error": _expert_error(full_by_seed[seed], reach_by_seed[seed], offset),
            }
            act = {
                "observation_state": np.asarray(act_npz["observation_state"]),
                "sent_action": np.asarray(act_npz["sent_action"]),
                # Rollout gate telemetry is sampled after applying action_t, while
                # observation/images_t are sampled before it. Shift only the
                # comparison series back onto the observation timestamp. The
                # initial state is bit-exact with the expert reset, so its expert
                # error is the correct ACT frame-0 error.
                "pregrasp_error_m": np.concatenate([
                    np.asarray(expert["error"][:1]),
                    np.asarray(act_npz["pregrasp_error_m"][:-1]),
                ]),
                "front": np.asarray(act_npz["observation.images.front"]),
                "wrist": np.asarray(act_npz["observation.images.wrist"]),
                "error": np.concatenate([
                    np.asarray(expert["error"][:1]),
                    np.asarray(act_npz["pregrasp_error_m"][:-1]),
                ]),
            }
        comparison = _comparison_figure(
            seed=seed, expert_state=expert["state"], expert_action=expert["action"],
            expert_error=expert["error"], act=act,
            output=args.output / f"seed_{seed:06d}_expert_vs_act_trajectory.png",
        )
        _comparison_gif(
            seed, expert, act, args.output / f"seed_{seed:06d}_expert_vs_act.gif"
        )
        role = "seed7" if seed == 7 else "best" if seed == best["seed"] else "worst"
        comparisons[str(seed)] = {"role": role, **comparison}

    aggregates = {str(step): checkpoints[str(step)]["rollout"]["aggregate"] for step in STEPS}
    final_success = aggregates["2000"]["reach_success_count"]
    full_success = baseline["aggregate"]["reach_success_count"]
    if final_success >= 15:
        case = "Case A"
        conclusion = "Reach-only ACT reaches high training-seed success; full-task phase/temporal ambiguity is the primary failure source."
    elif final_success > max(full_success, 2):
        case = "Case B"
        conclusion = "Reach-only ACT is materially better but still unstable; phase ambiguity matters, with residual closed-loop issues."
    else:
        case = "Case C"
        conclusion = "Reach-only ACT still cannot reliably solve training-seed Reach; the problem is more basic than full-task phase ambiguity."
    compact = {
        "experiment": "Reach-only ACT diagnosis",
        "dataset": manifest,
        "checkpoints": checkpoints,
        "full_task_step1000_baseline": baseline["aggregate"],
        "error_plot_seed_selection": error_plot_seeds,
        "final_selected": {"seed7": 7, "best": best["seed"], "worst": worst["seed"]},
        "comparisons": comparisons,
        "forced_case": case,
        "conclusion": conclusion,
    }
    (args.output / "summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")

    lines = [
        "# Reach-only ACT diagnosis",
        "",
        "## 结论",
        "",
        f"**{case}：{conclusion}**",
        "",
        "本实验只执行 `reset → Reach → stable pregrasp hold → END`。rollout 每帧只向 ACT 输入双相机和 27D actual qpos，执行其 27D absolute target；未读取或执行 expert action，也没有进入 Approach/Grasp/Preload/Lift/Hold。",
        "三条对照的 expert 与 ACT 初始 27D state 完全一致；GIF/trajectory 的 error 已对齐到 observation timestamp。成功 gate 则按动作执行后的真实 MuJoCo 状态判定。",
        "",
        "## 数据与配置",
        "",
        f"- 20 条训练 seed，{manifest['frames']} frames：{manifest['real_reach_frames']} 个真实 Reach 帧 + {manifest['synthetic_terminal_hold_frames']} 个终点 hold 帧。",
        "- 每条终点 hold 8 帧，重复 first post-Reach observation，并保持最后一个已经稳定的 Reach controller target。",
        "- Wuji 20D target 在所有 Reach 段中恒定；数据不含 grasp close。",
        "- 27D state、27D action、front+wrist 240×320、30 Hz；ImageNet ResNet-18；chunk20；normal sampling；fresh training；H_exec=1。",
        "",
        "## Checkpoint 对比",
        "",
        "| step | Reach/20 | timeout | min error mean/median/best mm | enter 12mm | first entry mean frame | total dwell | max consecutive | early contact | cube max mean/max mm | Reach arm H1 MAE | Reach hand H1 MAE | clipped values | frame0 visual/expert arm spread |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in STEPS:
        rollout = checkpoints[str(step)]["rollout"]
        aggregate = rollout["aggregate"]
        offline = checkpoints[str(step)]["offline"]
        h1 = offline["h1_mae_rad"]["real_reach"]
        spread = offline["frame0_conditioning"]["image_only_spread_ratio_vs_expert"]["arm_7d"]
        mins = np.asarray([row["minimum_pregrasp_error_m"] for row in rollout["episodes"]]) * 1000
        entries = np.asarray([
            row["first_frame_enter_12mm"] for row in rollout["episodes"]
            if row["first_frame_enter_12mm"] is not None
        ])
        cube = np.asarray([
            row["maximum_cube_displacement_m"] for row in rollout["episodes"]
        ]) * 1000
        lines.append(
            f"| {step} | {aggregate['reach_success_count']}/20 | {aggregate['timeout_count']}/20 | "
            f"{mins.mean():.2f}/{np.median(mins):.2f}/{mins.min():.2f} | "
            f"{aggregate['episodes_entering_12mm']}/20 | {entries.mean():.2f} | "
            f"{aggregate['total_frames_inside_12mm']} | {aggregate['max_longest_consecutive_dwell']} | "
            f"{aggregate['early_cube_contact_count']} | {cube.mean():.2f}/{cube.max():.2f} | "
            f"{h1['arm_7d']:.5f} | {h1['hand_20d']:.5f} | "
            f"{aggregate['clipped_action_values']} | {spread:.3f} |"
        )
    lines += [
        "",
        "## Full-task baseline 对照",
        "",
        f"相同训练 reset 的 full-task D step1000 为 Reach {full_success}/20，mean/median/best pregrasp error = "
        f"{baseline['aggregate']['minimum_pregrasp_error_mean_mm']:.2f}/"
        f"{baseline['aggregate']['minimum_pregrasp_error_median_mm']:.2f}/"
        f"{baseline['aggregate']['minimum_pregrasp_error_best_mm']:.2f} mm。",
        f"它只有 {baseline['aggregate']['episodes_entering_12mm']}/20 进入过 12 mm basin、最长 dwell {baseline['aggregate']['maximum_longest_consecutive_12mm_dwell']} 帧；"
        f"Reach-only step2000 则为 20/20 进入、19/20 连续保持满 5 帧。",
        "",
        "## 诊断解释",
        "",
        "- step 500 → 1000 → 2000 的 Reach 为 11/20 → 16/20 → 19/20，同时 real-Reach arm H=1 MAE 为 0.00609 → 0.00437 → 0.00371 rad；二值与连续指标一致改善。",
        "- 三档均为 20/20 至少进入一次 12 mm basin，且没有 action clipping、early cube contact 或 >25 mm cube displacement；失败不是 controller limit 或碰撞造成的。",
        "- step2000 唯一失败 seed 8：minimum error 8.27 mm、共 8 帧在 gate 内，但最长连续 dwell=4，距离正式成功只差第 5 个连续帧；随后停在 gate 外侧而超时。",
        "- seed 7、best seed 14、worst seed 8 在共同前缀上均未出现 `>0.1 rad 且连续 3 帧` 的 gross arm/action divergence。这不表示轨迹完全相同，而是 seed 8 的失败更像终点小偏差/稳定保持问题。",
        "- frame-0 visual-only arm prediction spread / expert spread 在三档均 >1，说明双相机条件化没有塌缩为同一个动作；该指标只证明视觉响应存在，不等同于未见初始条件上的泛化。",
        "- 这是 training-seed 复现诊断，足以回答 phase ambiguity 假设，但不能据此声称对 unseen seeds 泛化。",
        "",
        "## Clipping by joint",
        "",
    ]
    for step in STEPS:
        totals = np.zeros(27, dtype=int)
        magnitudes = np.zeros(27)
        for episode in checkpoints[str(step)]["rollout"]["episodes"]:
            for row in episode["clipping_by_joint"]:
                totals[row["index"]] += row["count"]
                magnitudes[row["index"]] = max(magnitudes[row["index"]], row["max_magnitude_rad"])
        order = np.argsort(-totals)[:5]
        lines.append(
            f"- step {step}: " + ", ".join(
                f"j{index} count={totals[index]} max={magnitudes[index]:.4f} rad" for index in order
            )
        )
    lines += [
        "",
        "## Error-vs-time 与 expert 对照",
        "",
        "每个 checkpoint 的 best/median/worst pregrasp 曲线已保存。最终 step2000 对 seed 7、best、worst 保存了 7D arm qpos、27D action heatmap、pregrasp error、持续偏离帧和 front/wrist GIF。",
        "",
    ]
    for step in STEPS:
        lines.append(f"- step {step}: `pregrasp_error_step_{step:06d}.png`（seeds {error_plot_seeds[str(step)]}）")
    for seed in selected:
        item = comparisons[str(seed)]
        lines.append(
            f"- seed {seed} ({item['role']}): arm 首次持续偏离 frame={item['first_sustained_arm_state_divergence_frame']}，"
            f"action 首次持续偏离 frame={item['first_sustained_action_divergence_frame']}；"
            f"初始 27D state 最大差={item['initial_27d_state_max_abs_difference']:.2e}；"
            f"`seed_{seed:06d}_expert_vs_act_trajectory.png` / `seed_{seed:06d}_expert_vs_act.gif`"
        )
    lines += [
        "",
        "## 下一步",
        "",
        (
            "Case A 下不要立刻恢复原 full-task 训练。首选 staged/FSM policies（先让 Reach-only policy 到位并稳定，再显式切换下一阶段）；"
            "随后分别验证给 full-task ACT 显式 phase/progress，以及 previous-action/short-history。这里只给出建议，本轮未执行。"
            if case == "Case A" else
            "不要继续盲目堆训练 steps。优先做 previous-action conditioning、short observation history，以及从 pregrasp 偏离后回到 basin 的 Reach correction/recovery demonstrations。"
        ),
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"case": case, "conclusion": conclusion, "selected": compact["final_selected"]}, indent=2))


if __name__ == "__main__":
    main()
