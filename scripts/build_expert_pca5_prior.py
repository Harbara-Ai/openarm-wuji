from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


FPS = 30.0
FINGERS = ("thumb", "finger2", "middle", "ring", "little")


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / max(norm, 1e-12)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _load_analysis_helpers():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from analyze_wuji_cube_teleop import analyze_episode, joint_names, load_episode
    return analyze_episode, joint_names, load_episode


def _joint_mapping(root: Path, dataset: Path) -> dict:
    import mujoco

    synergy = json.loads((root / "configs/wuji_hand_left_synergies.json").read_text(
        encoding="utf-8"
    ))
    assumed = list(synergy["joint_names"])
    model_path = root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb"
    model = mujoco.MjModel.from_binary_path(str(model_path))
    rows = []
    for index, name in enumerate(assumed):
        full_name = f"wuji_{name}"
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, full_name
        )
        actual = None if joint_id < 0 else mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
        )
        rows.append({
            "expert_index_global_54d": 14 + index,
            "expert_index_left_hand_20d": index,
            "expert_assumed_joint_name": name,
            "mujoco_joint_name": actual,
            "mujoco_joint_id": int(joint_id),
            "mapped_index": index,
            "sign": 1,
            "unit": "rad",
            "exact_name_match": actual == full_name,
        })
    return {
        "source": "dataset info.json names=null; official Wuji flat-array convention",
        "model_path": model_path.as_posix(),
        "left_hand_dataset_slice": "observation.state/action[14:34]",
        "muJoCo_hand_order_source": "configs/wuji_hand_left_synergies.json",
        "mapping_proven": bool(all(row["exact_name_match"] for row in rows)),
        "recorder_reorder_uncertainty": (
            "Parquet does not embed per-channel names. The mapping is exact against "
            "the current MuJoCo model and official Wuji order, but the external "
            "dataset recorder's internal channel construction cannot be proven from "
            "the file alone."
        ),
        "rows": rows,
    }


def _phase_rows(root: Path, dataset: Path) -> tuple[list[dict], list[dict]]:
    analyze_episode, _, _ = _load_analysis_helpers()
    data_dir = dataset / "teleop/data/chunk-000"
    rows: list[dict] = []
    raws: list[dict] = []
    for episode_index in range(60, 90):
        row, raw = analyze_episode(
            data_dir / f"episode_{episode_index:06d}.parquet",
            episode_index,
            "left",
        )
        start = 0
        close_start = int(row["inferred_onset_frame"])
        established = int(row["stable_start_frame"])
        end = int(row["stable_end_frame_exclusive"])
        end = min(end, int(row["trajectory_length_frames"]))
        if not (0 <= start < close_start <= established < end):
            raise RuntimeError(f"invalid phase segmentation for episode {episode_index}")
        row["phase"] = {
            "episode_id": episode_index,
            "start_frame": start,
            "close_start_frame": close_start,
            "grasp_established_frame": established,
            "end_frame_exclusive": end,
            "duration_s": (end - start) / FPS,
            "segmentation_source": "Parquet action activity heuristic",
            "video_role": "optional interval QA only; no joint values used",
        }
        rows.append(row)
        raws.append(raw)
    return rows, raws


def _pca(phase_arrays: list[np.ndarray], components: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    pooled = np.concatenate(phase_arrays, axis=0)
    mean = np.mean(pooled, axis=0)
    centered = pooled - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:components].T.copy()
    # Make signs reproducible and useful for reading: largest-magnitude loading
    # in each PC is positive. PCA signs are otherwise mathematically arbitrary.
    for index in range(components):
        anchor = int(np.argmax(np.abs(basis[:, index])))
        if basis[anchor, index] < 0.0:
            basis[:, index] *= -1.0
    variance = singular ** 2 / max(len(pooled) - 1, 1)
    explained = variance / max(float(np.sum(variance)), 1e-12)
    cumulative = np.cumsum(explained)
    return pooled, mean, basis, [
        {
            "pc": index + 1,
            "explained_variance_ratio": float(explained[index]),
            "cumulative_explained_variance_ratio": float(cumulative[index]),
            "vector": basis[:, index].tolist(),
            "by_finger": {
                finger: basis[4 * finger_index:4 * (finger_index + 1), index].tolist()
                for finger_index, finger in enumerate(FINGERS)
            },
        }
        for index in range(components)
    ]


def _reconstruction_metrics(
    rows: list[dict], raws: list[dict], mean: np.ndarray, basis: np.ndarray,
) -> tuple[dict, list[dict], list[np.ndarray]]:
    all_errors = []
    trajectory_metrics = []
    reconstructions = []
    for row, raw in zip(rows, raws, strict=True):
        start = row["phase"]["start_frame"]
        end = row["phase"]["end_frame_exclusive"]
        q = raw["q_hand"][start:end]
        latent = (q - mean) @ basis
        recon = mean + latent @ basis.T
        error = recon - q
        all_errors.append(error)
        reconstructions.append(recon)
        trajectory_metrics.append({
            "episode_id": row["episode_index"],
            "frames": int(len(q)),
            "rmse_rad": float(np.sqrt(np.mean(error ** 2))),
            "max_abs_error_rad": float(np.max(np.abs(error))),
            "thumb_rmse_rad": float(np.sqrt(np.mean(error[:, :4] ** 2))),
            "other_finger_rmse_rad": float(np.sqrt(np.mean(error[:, 4:] ** 2))),
            "max_abs_error_joint_index": int(np.unravel_index(
                np.argmax(np.abs(error)), error.shape
            )[1]),
        })
    errors = np.concatenate(all_errors, axis=0)
    per_joint = np.sqrt(np.mean(errors ** 2, axis=0))
    return {
        "overall_rmse_rad": float(np.sqrt(np.mean(errors ** 2))),
        "max_absolute_error_rad": float(np.max(np.abs(errors))),
        "thumb_rmse_rad": float(np.sqrt(np.mean(errors[:, :4] ** 2))),
        "other_finger_rmse_rad": float(np.sqrt(np.mean(errors[:, 4:] ** 2))),
        "per_joint_rmse_rad": per_joint.tolist(),
        "trajectory_count": len(trajectory_metrics),
        "trajectory_rmse_mean_rad": float(np.mean([
            item["rmse_rad"] for item in trajectory_metrics
        ])),
        "trajectory_rmse_median_rad": float(np.median([
            item["rmse_rad"] for item in trajectory_metrics
        ])),
        "trajectory_rmse_max_rad": float(np.max([
            item["rmse_rad"] for item in trajectory_metrics
        ])),
    }, trajectory_metrics, reconstructions


def _select_representatives(rows: list[dict], trajectory_metrics: list[dict]) -> list[dict]:
    by_episode = {item["episode_id"]: item for item in trajectory_metrics}
    ordered = sorted(trajectory_metrics, key=lambda item: item["rmse_rad"])
    ids: list[int] = []
    ids.append(ordered[0]["episode_id"])
    ids.append(ordered[len(ordered) // 2]["episode_id"])
    ids.append(ordered[-1]["episode_id"])
    thumb_motion = sorted(
        rows,
        key=lambda row: float(np.linalg.norm(np.asarray(row["delta_action_rad"][:4]))),
        reverse=True,
    )
    ids.append(thumb_motion[0]["episode_index"])
    staged = sorted(
        rows,
        key=lambda row: max(
            value for value in row["finger_onset_offset_s"].values()
            if value is not None
        ) - min(
            value for value in row["finger_onset_offset_s"].values()
            if value is not None
        ),
        reverse=True,
    )
    ids.append(staged[0]["episode_index"])
    result = []
    for label, episode_id in zip(
        ("low_reconstruction_error", "median_reconstruction_error",
         "high_reconstruction_error", "largest_thumb_motion", "largest_staging_spread"),
        ids,
        strict=True,
    ):
        if episode_id in {item["episode_id"] for item in result}:
            continue
        result.append({"label": label, "episode_id": episode_id,
                       "reconstruction": by_episode[episode_id]})
    return result


def _write_plots(
    output_dir: Path, rows: list[dict], raws: list[dict], reconstructions: list[np.ndarray],
    representatives: list[dict], joint_names: tuple[str, ...],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_plots_pillow(output_dir, rows, raws, reconstructions, representatives)
        return
    raw_by_episode = {row["episode_index"]: (row, raw) for row, raw in zip(rows, raws, strict=True)}
    recon_by_episode = {
        row["episode_index"]: reconstruction
        for row, reconstruction in zip(rows, reconstructions, strict=True)
    }
    for item in representatives:
        row, raw = raw_by_episode[item["episode_id"]]
        start = row["phase"]["start_frame"]
        end = row["phase"]["end_frame_exclusive"]
        q = raw["q_hand"][start:end]
        recon = recon_by_episode[item["episode_id"]]
        fig, ax = plt.subplots(figsize=(13, 6))
        time = np.arange(len(q)) / FPS
        for joint in range(20):
            ax.plot(time, q[:, joint], linewidth=1.0, alpha=0.35)
            ax.plot(time, recon[:, joint], linewidth=1.0, linestyle="--", alpha=0.75)
        ax.set_title(f"Episode {item['episode_id']} — raw q_hand vs PCA5 reconstruction")
        ax.set_xlabel("Acquisition time (s)")
        ax.set_ylabel("Joint position (rad)")
        ax.grid(alpha=0.2)
        for boundary in (row["phase"]["close_start_frame"], row["phase"]["grasp_established_frame"]):
            ax.axvline((boundary - start) / FPS, color="black", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_dir / f"episode_{item['episode_id']:06d}_raw_vs_pca5.png", dpi=140)
        plt.close(fig)


def _write_plots_pillow(
    output_dir: Path, rows: list[dict], raws: list[dict],
    reconstructions: list[np.ndarray], representatives: list[dict],
) -> None:
    """Small dependency-free fallback so reconstruction plots are always emitted."""
    from PIL import Image, ImageDraw

    raw_by_episode = {row["episode_index"]: (row, raw) for row, raw in zip(rows, raws, strict=True)}
    recon_by_episode = {
        row["episode_index"]: reconstruction
        for row, reconstruction in zip(rows, reconstructions, strict=True)
    }
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44),
        (214, 39, 40), (148, 103, 189),
    ]
    for item in representatives:
        row, raw = raw_by_episode[item["episode_id"]]
        start = row["phase"]["start_frame"]
        end = row["phase"]["end_frame_exclusive"]
        q = raw["q_hand"][start:end]
        recon = recon_by_episode[item["episode_id"]]
        width, height = 1300, 600
        left, right, top, bottom = 70, 30, 35, 60
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        values = np.concatenate([q, recon], axis=0)
        ymin = float(np.min(values))
        ymax = float(np.max(values))
        pad = max((ymax - ymin) * 0.06, 0.05)
        ymin -= pad
        ymax += pad

        def point(frame: int, value: float) -> tuple[int, int]:
            x = left + int((width - left - right) * frame / max(len(q) - 1, 1))
            y = top + int((height - top - bottom) * (ymax - value) / (ymax - ymin))
            return x, y

        draw.rectangle((left, top, width - right, height - bottom), outline=(80, 80, 80))
        draw.text((left, 8), f"Episode {item['episode_id']} raw q_hand (solid) vs PCA5 (dashed)", fill=(20, 20, 20))
        draw.text((left, height - 30), "Acquisition frames", fill=(20, 20, 20))
        draw.text((8, top), f"{ymax:.2f} rad", fill=(20, 20, 20))
        draw.text((8, height - bottom - 12), f"{ymin:.2f} rad", fill=(20, 20, 20))
        for joint in range(20):
            color = palette[joint // 4]
            raw_points = [point(index, float(q[index, joint])) for index in range(len(q))]
            recon_points = [point(index, float(recon[index, joint])) for index in range(len(q))]
            draw.line(raw_points, fill=color, width=1)
            for index in range(0, len(recon_points) - 1, 4):
                draw.line(recon_points[index:index + 3], fill=color, width=1)
        for boundary in (row["phase"]["close_start_frame"], row["phase"]["grasp_established_frame"]):
            x = left + int((width - left - right) * (boundary - start) / max(len(q) - 1, 1))
            draw.line((x, top, x, height - bottom), fill=(30, 30, 30), width=1)
        image.save(output_dir / f"episode_{item['episode_id']:06d}_raw_vs_pca5.png")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a left-Wuji expert PCA5 absolute posture prior")
    parser.add_argument("--dataset", type=Path, default=root / "data/external/wuji-pick-and-place")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/expert_pca5")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyze_episode, official_joint_names, _ = _load_analysis_helpers()

    mapping = _joint_mapping(root, args.dataset)
    rows, raws = _phase_rows(root, args.dataset)
    phase_arrays = [
        raw["q_hand"][row["phase"]["start_frame"]:row["phase"]["end_frame_exclusive"]]
        for row, raw in zip(rows, raws, strict=True)
    ]
    pooled, mean, basis, pcs = _pca(phase_arrays, components=5)
    metrics, trajectory_metrics, reconstructions = _reconstruction_metrics(
        rows, raws, mean, basis
    )
    latent_all = np.concatenate([
        (array - mean) @ basis for array in phase_arrays
    ], axis=0)
    latent_min = np.min(latent_all, axis=0)
    latent_max = np.max(latent_all, axis=0)
    latent_center = 0.5 * (latent_min + latent_max)
    latent_half_range = np.maximum(0.5 * (latent_max - latent_min), 1e-8)
    representatives = _select_representatives(rows, trajectory_metrics)
    names = official_joint_names("left")
    phase_metadata = {
        "dataset": "yeeeiii111/wuji-pick-and-place",
        "dataset_revision": json.loads((args.dataset / "download_manifest.json").read_text(
            encoding="utf-8"
        ))["revision"],
        "fps": FPS,
        "phase_definition": {
            "start": "frame 0 (pregrasp context)",
            "close_start": "first sustained 12% action-excursion crossing, floor 0.08 rad",
            "grasp_established": "inferred close + 3 frames; start of one-second early-hold window",
            "end": "one-second early-hold window end",
            "contact_source": "none; dataset has no tactile/contact field",
        },
        "episodes": [row["phase"] for row in rows],
    }
    scaling = {
        "representation": "expert_pca5_absolute",
        "formula": "q_target = clip(mu + B @ (latent_center + latent_half_range * action), joint_limits)",
        "policy_action_range": [-1.0, 1.0],
        "latent_min": latent_min.tolist(),
        "latent_max": latent_max.tolist(),
        "latent_center": latent_center.tolist(),
        "latent_half_range": latent_half_range.tolist(),
        "latent_percentiles": {
            str(percentile): np.percentile(latent_all, percentile, axis=0).tolist()
            for percentile in (1, 5, 50, 95, 99)
        },
        "rate_limit": "environment joint-rate limit; no target teleport",
    }
    report = {
        "representation": "expert_pca5_absolute",
        "dataset": "yeeeiii111/wuji-pick-and-place",
        "dataset_revision": phase_metadata["dataset_revision"],
        "source_episode_ids": [row["episode_index"] for row in rows],
        "joint_mapping": mapping,
        "phase_metadata": phase_metadata,
        "mean_vector_rad": mean.tolist(),
        "basis_shape": list(basis.shape),
        "principal_components": pcs,
        "explained_variance_ratio": [pc["explained_variance_ratio"] for pc in pcs],
        "cumulative_explained_variance_ratio": [pc["cumulative_explained_variance_ratio"] for pc in pcs],
        "reconstruction_metrics": metrics,
        "trajectory_reconstruction_metrics": trajectory_metrics,
        "latent_scaling": scaling,
        "representatives": representatives,
        "constraints": {
            "uses_left_cube_episodes_only": True,
            "uses_right_hand_data": False,
            "modifies_reward_v2": False,
            "modifies_observation": False,
            "modifies_reset_or_success": False,
            "includes_20d_residual": False,
            "uses_manual_synergy_direction": False,
        },
    }
    np.save(args.output_dir / "basis.npy", basis.astype(np.float64))
    np.save(args.output_dir / "mean.npy", mean.astype(np.float64))
    (args.output_dir / "joint_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (args.output_dir / "phase_metadata.json").write_text(json.dumps(phase_metadata, indent=2), encoding="utf-8")
    (args.output_dir / "latent_scaling.json").write_text(json.dumps(scaling, indent=2), encoding="utf-8")
    (args.output_dir / "reconstruction_metrics.json").write_text(json.dumps({
        "overall": metrics, "trajectory": trajectory_metrics, "representatives": representatives,
    }, indent=2), encoding="utf-8")
    (args.output_dir / "expert_pca5_prior.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_plots(args.output_dir, rows, raws, reconstructions, representatives, names)
    print(json.dumps({
        "output_dir": args.output_dir.as_posix(),
        "mapping_proven": mapping["mapping_proven"],
        "phase_frames": int(len(pooled)),
        "explained_variance_ratio": report["explained_variance_ratio"],
        "reconstruction_metrics": metrics,
        "representatives": representatives,
    }, indent=2))


if __name__ == "__main__":
    main()
