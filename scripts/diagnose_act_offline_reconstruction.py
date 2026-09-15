"""Measure ACT's inference-only full-chunk reconstruction on expert observations.

This diagnostic deliberately removes ``action`` from the policy input.  ACT uses
the ground-truth action chunk to construct its VAE latent during training, so
passing it here would measure teacher-forced reconstruction rather than the
deterministic inference path used by the MuJoCo rollout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PHASE_ORDER = ["Reach", "Approach", "Grasp", "Preload", "Lift", "Hold"]
PHASE_MAP = {
    "reach": "Reach",
    "approach": "Approach",
    "grasp_close": "Grasp",
    "freeze_synergy": "Grasp",
    "grasp_settle": "Grasp",
    "preload": "Preload",
    "preload_settle": "Preload",
    "lift": "Lift",
    "lift_s_curve": "Lift",
    "hold": "Hold",
}
REPORT_HORIZONS = [1, 5, 10, 20, 50, 100]
TASK = "Coordinate OpenArm and Wuji to grasp the cube, lift it, and hold it unsupported."


def _checkpoint_label(checkpoint: Path) -> str:
    """Return the numbered checkpoint directory for artifact labels."""
    if checkpoint.name == "pretrained_model":
        return checkpoint.parent.name
    return checkpoint.name


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _masked_mae(error: np.ndarray, valid: np.ndarray, joint_slice: slice) -> float:
    selected = error[..., joint_slice]
    expanded = np.broadcast_to(valid[..., None], selected.shape)
    return _mean(selected[expanded])


def _evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or len(indices) <= count:
        return indices
    positions = np.rint(np.linspace(0, len(indices) - 1, count)).astype(np.int64)
    return indices[np.unique(positions)]


def _load_phase_metadata(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    phases: list[str] = []
    raw_phases: list[str] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    episode_lengths: list[int] = []
    raw_actions: list[np.ndarray] = []
    raw_states: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []

    for row in manifest["source_episodes"]:
        episode_index = int(row["native_episode_index"])
        source = Path(row["source"])
        if not source.is_absolute():
            source = root / source
        with np.load(source, allow_pickle=False) as episode:
            local_raw_phases = [str(value) for value in episode["phase"]]
            unknown = sorted(set(local_raw_phases) - set(PHASE_MAP))
            if unknown:
                raise ValueError(f"unmapped phases in {source}: {unknown}")
            local_phases = [PHASE_MAP[value] for value in local_raw_phases]
            actions = np.asarray(episode["action"], dtype=np.float32)
            states = np.asarray(episode["observation.state"], dtype=np.float32)
            length = len(actions)
            if length != int(row["frames"]) or len(states) != length:
                raise ValueError(f"manifest length mismatch for {source}")
            if actions.shape[1:] != (27,) or states.shape[1:] != (27,):
                raise ValueError(f"expected 27-D state/action in {source}")

            phases.extend(local_phases)
            raw_phases.extend(local_raw_phases)
            episode_indices.extend([episode_index] * length)
            frame_indices.extend(range(length))
            episode_lengths.extend([length] * length)
            raw_actions.append(actions)
            raw_states.append(states)
            sources.append({
                "native_episode_index": episode_index,
                "seed": int(row["seed"]),
                "frames": length,
                "source": source.as_posix(),
            })

    return {
        "phase": np.asarray(phases),
        "raw_phase": np.asarray(raw_phases),
        "episode_index": np.asarray(episode_indices, dtype=np.int64),
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "episode_length": np.asarray(episode_lengths, dtype=np.int64),
        "action": np.concatenate(raw_actions),
        "state": np.concatenate(raw_states),
        "sources": sources,
    }


def _draw_line_chart(
    path: Path,
    *,
    x: np.ndarray,
    series: dict[str, np.ndarray],
    title: str,
    y_label: str,
    marker_x: list[int] | None = None,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1200, 720
    left, top, right, bottom = 105, 70, 40, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    finite = np.concatenate([
        values[np.isfinite(values)] for values in series.values()
        if np.any(np.isfinite(values))
    ])
    y_max = max(float(np.max(finite)) * 1.08, 1e-6)
    x_min, x_max = float(np.min(x)), float(np.max(x))

    draw.text((left, 20), title, fill="black", font=font)
    for tick in range(6):
        value = y_max * tick / 5
        py = top + plot_h - plot_h * tick / 5
        draw.line((left, py, left + plot_w, py), fill="#e5e7eb", width=1)
        draw.text((12, py - 7), f"{value:.3f}", fill="#374151", font=font)
    for value in [1, 20, 40, 60, 80, 100]:
        if value < x_min or value > x_max:
            continue
        px = left + (value - x_min) / max(x_max - x_min, 1) * plot_w
        draw.line((px, top, px, top + plot_h), fill="#f3f4f6", width=1)
        draw.text((px - 10, top + plot_h + 10), str(value), fill="#374151", font=font)
    if marker_x:
        for value in marker_x:
            if value < x_min or value > x_max:
                continue
            px = left + (value - x_min) / max(x_max - x_min, 1) * plot_w
            draw.line((px, top, px, top + plot_h), fill="#9ca3af", width=1)

    for color, (name, values) in zip(colors, series.items()):
        points = []
        for x_value, y_value in zip(x, values):
            if not np.isfinite(y_value):
                continue
            px = left + (float(x_value) - x_min) / max(x_max - x_min, 1) * plot_w
            py = top + plot_h - float(y_value) / y_max * plot_h
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for horizon in marker_x or []:
            index = horizon - 1
            if 0 <= index < len(values) and np.isfinite(values[index]):
                px = left + (horizon - x_min) / max(x_max - x_min, 1) * plot_w
                py = top + plot_h - float(values[index]) / y_max * plot_h
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)

    legend_x = left + 20
    for color, name in zip(colors, series):
        draw.line((legend_x, top + 18, legend_x + 28, top + 18), fill=color, width=3)
        draw.text((legend_x + 35, top + 11), name, fill="#111827", font=font)
        legend_x += 180
    draw.text((left + plot_w / 2 - 50, height - 35), "Action horizon", fill="black", font=font)
    draw.text((12, 45), y_label, fill="black", font=font)
    image.save(path)


def _draw_phase_bars(
    path: Path,
    phase_metrics: dict[str, dict[str, Any]],
    *,
    checkpoint_label: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1200, 720
    left, top, right, bottom = 100, 70, 40, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    names = ["overall", "arm", "hand"]
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    values = np.asarray([
        [phase_metrics[phase]["horizons"]["1"][name + "_mae_rad"] for name in names]
        for phase in PHASE_ORDER
    ], dtype=float)
    finite = values[np.isfinite(values)]
    y_max = max(float(np.max(finite)) * 1.12, 1e-6)
    draw.text(
        (left, 20),
        f"ACT checkpoint {checkpoint_label}: inference-only first-action MAE by expert phase",
        fill="black",
        font=font,
    )
    for tick in range(6):
        value = y_max * tick / 5
        py = top + plot_h - plot_h * tick / 5
        draw.line((left, py, left + plot_w, py), fill="#e5e7eb", width=1)
        draw.text((12, py - 7), f"{value:.3f}", fill="#374151", font=font)
    group_w = plot_w / len(PHASE_ORDER)
    bar_w = group_w * 0.2
    for phase_index, phase in enumerate(PHASE_ORDER):
        center = left + group_w * (phase_index + 0.5)
        for metric_index, color in enumerate(colors):
            value = values[phase_index, metric_index]
            if not np.isfinite(value):
                continue
            x0 = center + (metric_index - 1.5) * bar_w
            x1 = x0 + bar_w * 0.85
            y0 = top + plot_h - value / y_max * plot_h
            draw.rectangle((x0, y0, x1, top + plot_h), fill=color)
        draw.text((center - 24, top + plot_h + 12), phase, fill="#111827", font=font)
    legend_x = left + 250
    for name, color in zip(names, colors):
        draw.rectangle((legend_x, top + 12, legend_x + 18, top + 28), fill=color)
        draw.text((legend_x + 25, top + 13), name, fill="#111827", font=font)
        legend_x += 170
    draw.text((12, 45), "MAE (rad)", fill="black", font=font)
    image.save(path)


def _metrics_at_horizon(error: np.ndarray, valid: np.ndarray, horizon: int) -> dict[str, Any]:
    index = horizon - 1
    is_valid = valid[:, index]
    return {
        "valid_observations": int(np.count_nonzero(is_valid)),
        "overall_mae_rad": _json_number(_masked_mae(error[:, index:index + 1], is_valid[:, None], slice(None))),
        "arm_mae_rad": _json_number(_masked_mae(error[:, index:index + 1], is_valid[:, None], slice(0, 7))),
        "hand_mae_rad": _json_number(_masked_mae(error[:, index:index + 1], is_valid[:, None], slice(7, 27))),
    }


def _make_summary(
    *,
    error: np.ndarray,
    valid: np.ndarray,
    phases: np.ndarray,
    selected_indices: np.ndarray,
    metadata: dict[str, Any],
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    samples_per_phase: int,
    padding_mask_matches_manual: bool,
    max_padding_action_error: float,
    max_state_roundtrip_error: float,
) -> dict[str, Any]:
    horizons = np.arange(1, error.shape[1] + 1)
    report_horizons = [
        horizon for horizon in REPORT_HORIZONS if horizon <= error.shape[1]
    ]
    overall_curve = []
    arm_curve = []
    hand_curve = []
    valid_counts = []
    for horizon in horizons:
        row = _metrics_at_horizon(error, valid, int(horizon))
        overall_curve.append(row["overall_mae_rad"])
        arm_curve.append(row["arm_mae_rad"])
        hand_curve.append(row["hand_mae_rad"])
        valid_counts.append(row["valid_observations"])

    phasewise: dict[str, dict[str, Any]] = {}
    for phase in PHASE_ORDER:
        mask = phases == phase
        phase_error = error[mask]
        phase_valid = valid[mask]
        phasewise[phase] = {
            "sampled_observations": int(np.count_nonzero(mask)),
            "all_valid_horizons": {
                "valid_action_steps": int(np.count_nonzero(phase_valid)),
                "overall_mae_rad": _json_number(_masked_mae(phase_error, phase_valid, slice(None))),
                "arm_mae_rad": _json_number(_masked_mae(phase_error, phase_valid, slice(0, 7))),
                "hand_mae_rad": _json_number(_masked_mae(phase_error, phase_valid, slice(7, 27))),
            },
            "horizons": {
                str(horizon): _metrics_at_horizon(phase_error, phase_valid, horizon)
                for horizon in report_horizons
            },
        }

    return {
        "schema_version": 1,
        "diagnostic": "ACT inference-only offline full-chunk reconstruction",
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_label": _checkpoint_label(checkpoint),
        "dataset_root": dataset_root.resolve().as_posix(),
        "repo_id": repo_id,
        "state_dim": 27,
        "action_dim": 27,
        "arm_slice": [0, 7],
        "hand_slice": [7, 27],
        "chunk_size": int(error.shape[1]),
        "inference_semantics": {
            "ground_truth_action_passed_to_policy": False,
            "vae_inference_latent": "deterministic zero latent used by ACT when action is absent",
            "prediction_space": "unnormalized absolute controller target (radians)",
            "current_phase_assignment": "phase of expert observation o_t",
            "target": (
                f"[a_t, ..., a_t+{error.shape[1] - 1}] within the same episode"
            ),
            "padding": "repeat-last values may be stored by LeRobot but are excluded from every MAE",
        },
        "sampling": {
            "strategy": "all frames" if samples_per_phase <= 0 else "deterministic evenly spaced per phase",
            "samples_per_phase_limit": samples_per_phase if samples_per_phase > 0 else None,
            "dataset_frames": int(len(metadata["phase"])),
            "sampled_observations": int(len(selected_indices)),
            "selected_global_indices": selected_indices.tolist(),
            "phase_counts": {
                phase: int(np.count_nonzero(phases == phase)) for phase in PHASE_ORDER
            },
        },
        "boundary_and_roundtrip_checks": {
            "padding_mask_matches_manual_episode_boundary": padding_mask_matches_manual,
            "max_abs_dataset_vs_raw_padded_action_rad": max_padding_action_error,
            "max_abs_dataset_vs_raw_state_rad": max_state_roundtrip_error,
            "cross_episode_targets_in_mae": False,
        },
        "reported_horizons": {
            str(horizon): _metrics_at_horizon(error, valid, horizon)
            for horizon in report_horizons
        },
        "mae_vs_horizon": {
            "horizon": horizons.tolist(),
            "valid_observations": valid_counts,
            "overall_mae_rad": overall_curve,
            "arm_mae_rad": arm_curve,
            "hand_mae_rad": hand_curve,
        },
        "phasewise": phasewise,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    report_horizons = [
        horizon for horizon in REPORT_HORIZONS
        if str(horizon) in summary["reported_horizons"]
    ]
    lines = [
        "# ACT offline full-chunk reconstruction",
        "",
        "This is inference-only evaluation on real expert observations. Ground-truth actions are not passed to ACT's VAE encoder, and padded targets are excluded.",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Sampled observations: {summary['sampling']['sampled_observations']} / {summary['sampling']['dataset_frames']}",
        f"- Sampling: {summary['sampling']['strategy']}",
        "",
        "## MAE by action horizon",
        "",
        "| horizon | valid observations | overall (rad) | arm 7D (rad) | hand 20D (rad) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in report_horizons:
        row = summary["reported_horizons"][str(horizon)]
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.6f}"
        lines.append(
            f"| {horizon} | {row['valid_observations']} | {fmt(row['overall_mae_rad'])} | "
            f"{fmt(row['arm_mae_rad'])} | {fmt(row['hand_mae_rad'])} |"
        )
    lines.extend([
        "",
        "## Reach observations: MAE by action horizon",
        "",
        "| horizon | valid observations | overall (rad) | arm 7D (rad) | hand 20D (rad) |",
        "|---:|---:|---:|---:|---:|",
    ])
    for horizon in report_horizons:
        row = summary["phasewise"]["Reach"]["horizons"][str(horizon)]
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.6f}"
        lines.append(
            f"| {horizon} | {row['valid_observations']} | {fmt(row['overall_mae_rad'])} | "
            f"{fmt(row['arm_mae_rad'])} | {fmt(row['hand_mae_rad'])} |"
        )
    lines.extend([
        "",
        "## Phase-wise first-action MAE",
        "",
        "| phase | observations | overall (rad) | arm 7D (rad) | hand 20D (rad) |",
        "|---|---:|---:|---:|---:|",
    ])
    for phase in PHASE_ORDER:
        phase_row = summary["phasewise"][phase]
        row = phase_row["horizons"]["1"]
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.6f}"
        lines.append(
            f"| {phase} | {phase_row['sampled_observations']} | {fmt(row['overall_mae_rad'])} | "
            f"{fmt(row['arm_mae_rad'])} | {fmt(row['hand_mae_rad'])} |"
        )
    reach = summary["phasewise"]["Reach"]["horizons"]["1"]
    lines.extend([
        "",
        "## Direct answers",
        "",
        f"- First action overall MAE: {summary['reported_horizons']['1']['overall_mae_rad']:.6f} rad.",
        f"- Reach first-action arm MAE: {reach['arm_mae_rad']:.6f} rad; hand MAE: {reach['hand_mae_rad']:.6f} rad.",
        (
            f"- Horizon-{summary['chunk_size']} statistics only include observations "
            f"with {summary['chunk_size']} real future actions in the same episode; "
            "later phases therefore legitimately have fewer samples."
        ),
        "- This diagnostic alone measures on-manifold fit. It does not establish closed-loop robustness or input parity.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "outputs/act_e2e_smoke/act_train/checkpoints/000500/pretrained_model",
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=root / "outputs/act_e2e_smoke/lerobot_dataset",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "outputs/act_e2e_smoke/lerobot_dataset/conversion_manifest.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/act_e2e_smoke/diagnosis/offline_chunk",
    )
    parser.add_argument(
        "--cache", type=Path,
        default=root.parent / ".hf_act_offline_chunk",
        help="Short local cache path (Windows Hugging Face lock names can be long).",
    )
    parser.add_argument(
        "--samples-per-phase", type=int, default=48,
        help="Evenly sample this many observations per phase; <=0 evaluates all frames.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")

    import datasets
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from torch.utils.data import DataLoader, Subset

    datasets.config.HF_DATASETS_CACHE = str(cache_root / "datasets")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    repo_id = str(manifest["repo_id"])
    checkpoint_config = json.loads(
        (args.checkpoint / "config.json").read_text(encoding="utf-8")
    )
    chunk_size = int(checkpoint_config["chunk_size"])
    if chunk_size < 1:
        raise ValueError("checkpoint chunk_size must be positive")
    fps = int(manifest["fps"])
    metadata = _load_phase_metadata(root, manifest)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=args.dataset,
        delta_timestamps={"action": [step / fps for step in range(chunk_size)]},
        video_backend="pyav",
        return_uint8=True,
    )
    if len(dataset) != len(metadata["phase"]):
        raise ValueError("native dataset and raw phase metadata have different lengths")
    native_episode = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    native_frame = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
    native_state = np.stack(dataset.hf_dataset["observation.state"]).astype(np.float32)
    if not np.array_equal(native_episode, metadata["episode_index"]):
        raise ValueError("native episode indices do not match source manifest order")
    if not np.array_equal(native_frame, metadata["frame_index"]):
        raise ValueError("native frame indices do not match source episode frames")
    max_state_roundtrip_error = float(np.max(np.abs(native_state - metadata["state"])))

    selected_parts = []
    for phase in PHASE_ORDER:
        candidates = np.flatnonzero(metadata["phase"] == phase)
        selected_parts.append(_evenly_spaced(candidates, args.samples_per_phase))
    selected_indices = np.sort(np.concatenate(selected_parts)).astype(np.int64)
    selected_phases = metadata["phase"][selected_indices]
    loader = DataLoader(
        Subset(dataset, selected_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(torch.device("cpu"))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    all_indices: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    padding_mask_matches_manual = True
    max_padding_action_error = 0.0

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            global_indices = batch["index"].cpu().numpy().astype(np.int64)
            target = batch["action"].cpu().numpy().astype(np.float32)
            dataset_pad = batch["action_is_pad"].cpu().numpy().astype(bool)
            model_batch = {
                "observation.state": batch["observation.state"],
                "observation.images.front": batch["observation.images.front"],
                "observation.images.wrist": batch["observation.images.wrist"],
                "task": list(batch.get("task", [TASK] * len(global_indices))),
            }
            for camera_key in ["observation.images.front", "observation.images.wrist"]:
                if model_batch[camera_key].dtype == torch.uint8:
                    model_batch[camera_key] = model_batch[camera_key].to(torch.float32) / 255.0
            processed = preprocessor(model_batch)
            normalized_prediction = policy.predict_action_chunk(processed)
            prediction = postprocessor(normalized_prediction).detach().cpu().numpy().astype(np.float32)
            if prediction.shape != target.shape or prediction.shape[1:] != (chunk_size, 27):
                raise RuntimeError(
                    f"unexpected prediction/target shapes: {prediction.shape}, {target.shape}"
                )

            manual_pad = np.zeros_like(dataset_pad)
            manual_target = np.empty_like(target)
            for local_index, global_index in enumerate(global_indices):
                episode_frame = int(metadata["frame_index"][global_index])
                episode_length = int(metadata["episode_length"][global_index])
                episode_start = global_index - episode_frame
                for horizon in range(chunk_size):
                    future_frame = episode_frame + horizon
                    manual_pad[local_index, horizon] = future_frame >= episode_length
                    clamped_frame = min(future_frame, episode_length - 1)
                    manual_target[local_index, horizon] = metadata["action"][episode_start + clamped_frame]
            padding_mask_matches_manual &= bool(np.array_equal(dataset_pad, manual_pad))
            max_padding_action_error = max(
                max_padding_action_error,
                float(np.max(np.abs(target - manual_target))),
            )
            all_indices.append(global_indices)
            all_predictions.append(prediction)
            all_targets.append(target)
            all_valid.append(~dataset_pad)
            print(
                f"batch {batch_number}/{len(loader)}: {len(global_indices)} observations",
                flush=True,
            )

    observed_indices = np.concatenate(all_indices)
    if not np.array_equal(observed_indices, selected_indices):
        raise RuntimeError("DataLoader changed the selected observation order")
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    valid = np.concatenate(all_valid)
    error = np.abs(predictions - targets)
    summary = _make_summary(
        error=error,
        valid=valid,
        phases=selected_phases,
        selected_indices=selected_indices,
        metadata=metadata,
        checkpoint=args.checkpoint,
        dataset_root=args.dataset,
        repo_id=repo_id,
        samples_per_phase=args.samples_per_phase,
        padding_mask_matches_manual=padding_mask_matches_manual,
        max_padding_action_error=max_padding_action_error,
        max_state_roundtrip_error=max_state_roundtrip_error,
    )
    if not padding_mask_matches_manual or max_padding_action_error > 1e-6:
        raise RuntimeError("LeRobot action chunk padding differs from the raw episode boundary")

    (args.output / "offline_chunk_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "offline_chunk_reconstruction.npz",
        global_index=selected_indices,
        episode_index=metadata["episode_index"][selected_indices],
        frame_index=metadata["frame_index"][selected_indices],
        phase=selected_phases,
        predicted_action_chunk=predictions,
        expert_action_chunk=targets,
        valid_action_mask=valid,
        absolute_error=error,
    )
    curve = summary["mae_vs_horizon"]
    horizon_array = np.asarray(curve["horizon"])
    checkpoint_label = _checkpoint_label(args.checkpoint)
    _draw_line_chart(
        args.output / "offline_chunk_mae_vs_horizon.png",
        x=horizon_array,
        series={
            "overall 27D": np.asarray(curve["overall_mae_rad"], dtype=float),
            "arm 7D": np.asarray(curve["arm_mae_rad"], dtype=float),
            "hand 20D": np.asarray(curve["hand_mae_rad"], dtype=float),
        },
        title=f"ACT checkpoint {checkpoint_label}: inference-only full-chunk reconstruction",
        y_label="MAE (rad)",
        marker_x=[horizon for horizon in REPORT_HORIZONS if horizon <= chunk_size],
    )
    _draw_line_chart(
        args.output / "arm_vs_hand_mae.png",
        x=horizon_array,
        series={
            "arm 7D": np.asarray(curve["arm_mae_rad"], dtype=float),
            "hand 20D": np.asarray(curve["hand_mae_rad"], dtype=float),
        },
        title=f"ACT checkpoint {checkpoint_label}: arm versus hand action-chunk error",
        y_label="MAE (rad)",
        marker_x=REPORT_HORIZONS,
    )
    _draw_phase_bars(
        args.output / "phasewise_action_mae.png",
        summary["phasewise"],
        checkpoint_label=checkpoint_label,
    )
    _write_markdown(args.output / "README.md", summary)
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "sampled_observations": len(selected_indices),
        "reported_horizons": summary["reported_horizons"],
        "reach_first_action": summary["phasewise"]["Reach"]["horizons"]["1"],
        "padding_check": summary["boundary_and_roundtrip_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
