from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FPS = 30.0
FINGERS = ("thumb", "finger2", "middle", "ring", "little")
FINGER_NUMBERS = (1, 2, 3, 4, 5)
HAND_SLICES = {"left": slice(14, 34), "right": slice(34, 54)}
# Exactly what the current structured5 environment computes from the checked-in
# open/close poses: sign(close_pose - open_pose), grouped as five by four.
CURRENT_SCRIPTED_DIRECTION = np.asarray(
    [[1, 1, 1, 1]] + [[1, 0, 1, 1]] * 4, dtype=float
)


def joint_names(side: str) -> tuple[str, ...]:
    return tuple(
        f"{side}_finger{finger}_joint{joint}"
        for finger in FINGER_NUMBERS
        for joint in range(1, 5)
    )


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    if values.ndim == 1:
        padded = np.pad(values, (window // 2,), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:len(values)]
    return np.column_stack([
        _smooth(values[:, index], window) for index in range(values.shape[1])
    ])


def _first_sustained(
    values: np.ndarray,
    threshold: float,
    *,
    start: int,
    stop: int,
    frames: int = 3,
) -> int | None:
    above = values >= threshold
    stop = min(stop, len(values))
    for index in range(max(0, start), max(0, stop - frames + 1)):
        if bool(np.all(above[index:index + frames])):
            return index
    return None


def _unit(values: np.ndarray) -> np.ndarray:
    return values / max(float(np.linalg.norm(values)), 1e-12)


def _vector_stats(values: np.ndarray) -> dict:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def _best_command_state_lag(action: np.ndarray, state: np.ndarray) -> int:
    errors = []
    for lag in range(11):
        predicted = action[:-lag or None]
        measured = state[lag:]
        errors.append(float(np.median(np.abs(measured - predicted))))
    return int(np.argmin(errors))


def load_episode(path: Path, side: str) -> dict:
    table = pq.read_table(path, columns=[
        "observation.state", "action", "timestamp", "frame_index",
        "episode_index", "task_index",
    ])
    state54 = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    action54 = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    hand_slice = HAND_SLICES[side]
    return {
        "q_hand": state54[:, hand_slice],
        "a_hand": action54[:, hand_slice],
        "timestamp": np.asarray(table["timestamp"], dtype=np.float64),
        "frame_index": np.asarray(table["frame_index"], dtype=np.int64),
        "episode_index": int(table["episode_index"][0].as_py()),
        "task_index": int(table["task_index"][0].as_py()),
    }


def analyze_episode(path: Path, episode_index: int, side: str) -> tuple[dict, dict]:
    raw = load_episode(path, side)
    qhand = raw["q_hand"]
    ahand = raw["a_hand"]
    command = _smooth(ahand)
    baseline_frames = min(15, max(5, len(command) // 10))
    action_start = np.median(command[:baseline_frames], axis=0)
    state_start = np.median(qhand[:baseline_frames], axis=0)

    finger_displacement = np.column_stack([
        np.linalg.norm(
            command[:, 4 * index:4 * (index + 1)]
            - action_start[4 * index:4 * (index + 1)], axis=1,
        )
        for index in range(5)
    ])
    displacement = np.linalg.norm(command - action_start, axis=1)
    peak_value = float(np.percentile(displacement, 99))
    peak_index = int(np.argmax(displacement))
    onset = _first_sustained(
        displacement, max(0.08, 0.12 * peak_value),
        start=baseline_frames, stop=peak_index + 1,
    )
    if onset is None:
        onset = baseline_frames
    close = _first_sustained(
        displacement, 0.85 * peak_value,
        start=onset, stop=peak_index + 1,
    )
    if close is None:
        close = peak_index

    stable_start = min(close + 3, len(command) - 1)
    stable_end = min(len(command), stable_start + 30)
    if stable_end - stable_start < 5:
        stable_start = max(onset, peak_index - 5)
        stable_end = min(len(command), peak_index + 6)
    action_grasp = np.median(command[stable_start:stable_end], axis=0)
    state_grasp = np.median(qhand[stable_start:stable_end], axis=0)
    delta_action = action_grasp - action_start
    delta_q = state_grasp - state_start

    finger_onsets_frames: dict[str, int | None] = {}
    finger_onsets_seconds: dict[str, float | None] = {}
    for index, finger in enumerate(FINGERS):
        values = finger_displacement[:, index]
        threshold = max(0.03, 0.15 * float(np.percentile(values, 99)))
        finger_onset = _first_sustained(
            values, threshold, start=baseline_frames, stop=peak_index + 1
        )
        finger_onsets_frames[finger] = finger_onset
        finger_onsets_seconds[finger] = (
            None if finger_onset is None else (finger_onset - onset) / FPS
        )

    row = {
        "episode_index": episode_index,
        "task_index": raw["task_index"],
        "side": side,
        "source_parquet": path.as_posix(),
        "source_parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trajectory_length_frames": len(qhand),
        "duration_s": float(raw["timestamp"][-1] - raw["timestamp"][0] + 1.0 / FPS),
        "inferred_onset_frame": onset,
        "inferred_close_frame": close,
        "stable_start_frame": stable_start,
        "stable_end_frame_exclusive": stable_end,
        "closing_phase_length_frames": close - onset + 1,
        "close_duration_s": (close - onset) / FPS,
        "finger_onset_frame": finger_onsets_frames,
        "finger_onset_offset_s": finger_onsets_seconds,
        "state_start_rad": state_start.tolist(),
        "state_grasp_rad": state_grasp.tolist(),
        "delta_q_rad": delta_q.tolist(),
        "state_closing_unit_direction": _unit(delta_q).tolist(),
        "action_start_rad": action_start.tolist(),
        "action_grasp_rad": action_grasp.tolist(),
        "delta_action_rad": delta_action.tolist(),
        "action_closing_unit_direction": _unit(delta_action).tolist(),
        "action_mean_rad": np.mean(ahand, axis=0).tolist(),
        "action_std_rad": np.std(ahand, axis=0).tolist(),
        "action_min_rad": np.min(ahand, axis=0).tolist(),
        "action_max_rad": np.max(ahand, axis=0).tolist(),
        "state_mean_rad": np.mean(qhand, axis=0).tolist(),
        "state_std_rad": np.std(qhand, axis=0).tolist(),
        "state_min_rad": np.min(qhand, axis=0).tolist(),
        "state_max_rad": np.max(qhand, axis=0).tolist(),
        "closing_action_mean_rad": np.mean(ahand[onset:close + 1], axis=0).tolist(),
        "closing_action_std_rad": np.std(ahand[onset:close + 1], axis=0).tolist(),
        "best_state_lag_frames": _best_command_state_lag(ahand, qhand),
        "stable_state_action_mae_rad": float(np.mean(np.abs(state_grasp - action_grasp))),
    }
    return row, raw


def _pca(values: np.ndarray) -> dict:
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variances = singular ** 2 / max(len(values) - 1, 1)
    ratio = variances / max(float(np.sum(variances)), 1e-12)
    mean = np.mean(values, axis=0)
    for index in range(len(components)):
        projection = float(np.dot(components[index], mean))
        anchor = int(np.argmax(np.abs(components[index])))
        if projection < -1e-12 or (
            abs(projection) <= 1e-12 and components[index, anchor] < 0
        ):
            components[index] *= -1
    cumulative = np.cumsum(ratio)
    return {
        "method": "centered PCA via SVD on per-episode 20D delta",
        "samples": len(values),
        "explained_variance_ratio": ratio.tolist(),
        "cumulative_explained_variance_ratio": cumulative.tolist(),
        "dimensions_for_percent_variance": {
            str(percent): int(np.searchsorted(cumulative, percent / 100.0) + 1)
            for percent in (80, 90, 95)
        },
        "components": components.tolist(),
        "mean_vector_rad": mean.tolist(),
    }


def _uncentered_direction_svd(values: np.ndarray) -> dict:
    _, singular, components = np.linalg.svd(values, full_matrices=False)
    energy = singular ** 2 / max(float(np.sum(singular ** 2)), 1e-12)
    if float(np.dot(components[0], np.mean(values, axis=0))) < 0:
        components[0] *= -1
    return {
        "energy_ratio": energy.tolist(),
        "components": components.tolist(),
        "first_component_energy": float(energy[0]),
        "first_three_component_energy": float(np.sum(energy[:3])),
    }


def _finger_coupling(values: np.ndarray) -> dict:
    result = {}
    for finger_index, finger in enumerate(FINGERS):
        chunk = values[:, 4 * finger_index:4 * (finger_index + 1)]
        valid = np.linalg.norm(chunk, axis=1) > 0.03
        used = chunk[valid]
        if not len(used):
            result[finger] = {"valid_episodes": 0}
            continue
        directions = used / np.linalg.norm(used, axis=1, keepdims=True)
        _, singular, components = np.linalg.svd(used, full_matrices=False)
        energy = singular ** 2 / np.sum(singular ** 2)
        principal = components[0]
        if float(np.dot(principal, np.median(used, axis=0))) < 0:
            principal *= -1
        cosine = directions @ _unit(np.median(used, axis=0))
        result[finger] = {
            "valid_episodes": int(np.sum(valid)),
            "first_uncentered_svd_energy": float(energy[0]),
            "median_cosine_to_median_direction": float(np.median(cosine)),
            "q25_cosine_to_median_direction": float(np.percentile(cosine, 25)),
            "principal_ratio": principal.tolist(),
            "positive_fraction_over_0p02_rad": np.mean(used > 0.02, axis=0).tolist(),
            "negative_fraction_below_minus_0p02_rad": np.mean(used < -0.02, axis=0).tolist(),
        }
    return result


def _timing_summary(rows: list[dict]) -> dict:
    offsets = {finger: [] for finger in FINGERS}
    cofirst_counts = {finger: 0 for finger in FINGERS}
    strict_first_counts = {finger: 0 for finger in FINGERS}
    spreads = []
    for row in rows:
        valid = {
            finger: value for finger, value in row["finger_onset_offset_s"].items()
            if value is not None
        }
        for finger, value in valid.items():
            offsets[finger].append(value)
        if valid:
            first_value = min(valid.values())
            strict_first_counts[min(valid, key=valid.get)] += 1
            for finger, value in valid.items():
                if value <= first_value + 1.0 / FPS + 1e-9:
                    cofirst_counts[finger] += 1
            spreads.append(max(valid.values()) - min(valid.values()))
    return {
        "finger_onset_offset_s": {
            finger: {
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "std": float(np.std(values)) if values else None,
            }
            for finger, values in offsets.items()
        },
        "strict_first_counts": strict_first_counts,
        "cofirst_within_one_frame_counts": cofirst_counts,
        "onset_spread_s": {
            "mean": float(np.mean(spreads)),
            "median": float(np.median(spreads)),
            "std": float(np.std(spreads)),
            "fraction_le_0p20_s": float(np.mean(np.asarray(spreads) <= 0.20)),
            "fraction_gt_0p30_s": float(np.mean(np.asarray(spreads) > 0.30)),
        },
    }


def _scripted_comparison(expert_delta: np.ndarray) -> dict:
    expert = np.median(expert_delta, axis=0).reshape(5, 4)
    expert_direction = np.vstack([_unit(row) for row in expert])
    scripted_direction = np.vstack([_unit(row) for row in CURRENT_SCRIPTED_DIRECTION])
    comparisons = []
    sign_mismatches = []
    unsupported_zeros = []
    for finger_index, finger in enumerate(FINGERS):
        comparisons.append({
            "finger": finger,
            "expert_median_delta_action_rad": expert[finger_index].tolist(),
            "expert_normalized_ratio": expert_direction[finger_index].tolist(),
            "current_scripted_normalized_ratio": scripted_direction[finger_index].tolist(),
            "cosine": float(np.dot(
                expert_direction[finger_index], scripted_direction[finger_index]
            )),
            "normalized_l2_error": float(np.linalg.norm(
                expert_direction[finger_index] - scripted_direction[finger_index]
            )),
        })
        for joint in range(4):
            expert_value = expert[finger_index, joint]
            scripted_value = CURRENT_SCRIPTED_DIRECTION[finger_index, joint]
            label = f"{finger}.joint{joint + 1}"
            if abs(expert_value) <= 0.02:
                continue
            if scripted_value == 0:
                unsupported_zeros.append(label)
            elif np.sign(expert_value) != np.sign(scripted_value):
                sign_mismatches.append(label)
    return {
        "current_implementation": "sign(close_pose-open_pose), per finger",
        "current_scripted_raw": CURRENT_SCRIPTED_DIRECTION.tolist(),
        "per_finger": comparisons,
        "opposite_sign_joints": sign_mismatches,
        "scripted_zero_but_expert_moves_joints": unsupported_zeros,
        "largest_coupling_difference_first": [
            item["finger"] for item in sorted(
                comparisons, key=lambda item: item["normalized_l2_error"], reverse=True
            )
        ],
    }


def _aggregate(rows: list[dict]) -> dict:
    arrays = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in (
            "state_start_rad", "state_grasp_rad", "delta_q_rad",
            "action_start_rad", "action_grasp_rad", "delta_action_rad",
            "action_mean_rad", "action_std_rad", "action_min_rad", "action_max_rad",
            "state_mean_rad", "state_std_rad", "state_min_rad", "state_max_rad",
        )
    }
    return {
        "episodes": len(rows),
        "trajectory_length_frames": {
            "mean": float(np.mean([row["trajectory_length_frames"] for row in rows])),
            "median": float(np.median([row["trajectory_length_frames"] for row in rows])),
            "std": float(np.std([row["trajectory_length_frames"] for row in rows])),
            "min": int(min(row["trajectory_length_frames"] for row in rows)),
            "max": int(max(row["trajectory_length_frames"] for row in rows)),
        },
        "per_joint_vector_statistics": {
            key: _vector_stats(value) for key, value in arrays.items()
        },
        "per_joint_global_extrema": {
            "action_min_rad": np.min(arrays["action_min_rad"], axis=0).tolist(),
            "action_max_rad": np.max(arrays["action_max_rad"], axis=0).tolist(),
            "state_min_rad": np.min(arrays["state_min_rad"], axis=0).tolist(),
            "state_max_rad": np.max(arrays["state_max_rad"], axis=0).tolist(),
        },
        "per_finger_delta_q_rad": {
            key: np.asarray(value).reshape(5, 4).tolist()
            for key, value in _vector_stats(arrays["delta_q_rad"]).items()
        },
        "per_finger_delta_action_rad": {
            key: np.asarray(value).reshape(5, 4).tolist()
            for key, value in _vector_stats(arrays["delta_action_rad"]).items()
        },
        "state_delta_q_pca": _pca(arrays["delta_q_rad"]),
        "action_delta_pca": _pca(arrays["delta_action_rad"]),
        "state_delta_q_uncentered_svd": _uncentered_direction_svd(arrays["delta_q_rad"]),
        "action_delta_uncentered_svd": _uncentered_direction_svd(arrays["delta_action_rad"]),
        "state_per_finger_coupling": _finger_coupling(arrays["delta_q_rad"]),
        "action_per_finger_coupling": _finger_coupling(arrays["delta_action_rad"]),
        "timing": _timing_summary(rows),
        "scripted_comparison_using_action_delta": _scripted_comparison(
            arrays["delta_action_rad"]
        ),
        "median_best_state_lag_frames": float(np.median([
            row["best_state_lag_frames"] for row in rows
        ])),
        "median_stable_state_action_mae_rad": float(np.median([
            row["stable_state_action_mae_rad"] for row in rows
        ])),
    }


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)


def _k_medoids(rows: list[dict], count: int) -> list[tuple[int, float]]:
    delta = np.asarray([row["delta_action_rad"] for row in rows], dtype=float)
    direction = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
    norm = np.linalg.norm(delta, axis=1)
    duration = np.asarray([row["close_duration_s"] for row in rows])
    features = np.column_stack([
        direction,
        0.25 * (norm - np.median(norm)) / max(float(np.std(norm)), 1e-12),
        0.25 * (duration - np.median(duration)) / max(float(np.std(duration)), 1e-12),
    ])
    distances = _pairwise_distances(features)
    medoids = [int(np.argmin(np.sum(distances, axis=1)))]
    while len(medoids) < count:
        medoids.append(int(np.argmax(np.min(distances[:, medoids], axis=1))))
    for _ in range(50):
        labels = np.argmin(distances[:, medoids], axis=1)
        updated = []
        for cluster in range(count):
            members = np.flatnonzero(labels == cluster)
            updated.append(int(members[np.argmin(
                np.sum(distances[np.ix_(members, members)], axis=1)
            )]))
        if updated == medoids:
            break
        medoids = updated
    labels = np.argmin(distances[:, medoids], axis=1)
    return [
        (index, float(np.mean(distances[labels == cluster, index])))
        for cluster, index in enumerate(medoids)
    ]


def _write_all_trajectories(
    raws: list[dict], rows: list[dict], tasks: dict[int, str], path: Path
) -> None:
    columns: dict[str, list] = {
        "timestamp": [], "frame_index": [], "episode_index": [], "task_index": [],
        "task": [], "side": [], "q_hand": [], "a_hand": [],
    }
    for raw, row in zip(raws, rows, strict=True):
        length = len(raw["timestamp"])
        columns["timestamp"].extend(raw["timestamp"].tolist())
        columns["frame_index"].extend(raw["frame_index"].tolist())
        columns["episode_index"].extend([row["episode_index"]] * length)
        columns["task_index"].extend([row["task_index"]] * length)
        columns["task"].extend([tasks[row["task_index"]]] * length)
        columns["side"].extend([row["side"]] * length)
        columns["q_hand"].extend(raw["q_hand"].astype(np.float32).tolist())
        columns["a_hand"].extend(raw["a_hand"].astype(np.float32).tolist())
    table = pa.table({
        "timestamp": pa.array(columns["timestamp"], type=pa.float32()),
        "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
        "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
        "task_index": pa.array(columns["task_index"], type=pa.int64()),
        "task": pa.array(columns["task"], type=pa.string()),
        "side": pa.array(columns["side"], type=pa.string()),
        "q_hand": pa.array(columns["q_hand"], type=pa.list_(pa.float32(), 20)),
        "a_hand": pa.array(columns["a_hand"], type=pa.list_(pa.float32(), 20)),
    })
    pq.write_table(table, path, compression="zstd")


def _validate_all_trajectories(path: Path, raws: list[dict]) -> dict:
    table = pq.read_table(path)
    actual_q = np.asarray(table["q_hand"].to_pylist(), dtype=np.float32)
    actual_a = np.asarray(table["a_hand"].to_pylist(), dtype=np.float32)
    actual_t = np.asarray(table["timestamp"], dtype=np.float32)
    expected_q = np.concatenate([raw["q_hand"].astype(np.float32) for raw in raws])
    expected_a = np.concatenate([raw["a_hand"].astype(np.float32) for raw in raws])
    expected_t = np.concatenate([
        raw["timestamp"].astype(np.float32) for raw in raws
    ])
    return {
        "rows": table.num_rows,
        "q_hand_shape": list(actual_q.shape),
        "a_hand_shape": list(actual_a.shape),
        "q_hand_bit_exact_to_source": bool(np.array_equal(actual_q, expected_q)),
        "a_hand_bit_exact_to_source": bool(np.array_equal(actual_a, expected_a)),
        "timestamp_bit_exact_to_source": bool(np.array_equal(actual_t, expected_t)),
    }


def _write_episode_metrics(rows: list[dict], tasks: dict[int, str], path: Path) -> None:
    vector_fields = (
        "state_start_rad", "state_grasp_rad", "delta_q_rad",
        "action_start_rad", "action_grasp_rad", "delta_action_rad",
        "action_mean_rad", "action_std_rad", "action_min_rad", "action_max_rad",
    )
    columns: dict[str, pa.Array] = {
        "episode_index": pa.array([row["episode_index"] for row in rows], pa.int64()),
        "task_index": pa.array([row["task_index"] for row in rows], pa.int64()),
        "task": pa.array([tasks[row["task_index"]] for row in rows], pa.string()),
        "side": pa.array([row["side"] for row in rows], pa.string()),
        "trajectory_length_frames": pa.array([
            row["trajectory_length_frames"] for row in rows
        ], pa.int64()),
        "inferred_onset_frame": pa.array([
            row["inferred_onset_frame"] for row in rows
        ], pa.int64()),
        "inferred_close_frame": pa.array([
            row["inferred_close_frame"] for row in rows
        ], pa.int64()),
    }
    columns.update({
        field: pa.array([row[field] for row in rows], pa.list_(pa.float64(), 20))
        for field in vector_fields
    })
    pq.write_table(pa.table(columns), path, compression="zstd")


def _normalized_finger_rows(values: list[float]) -> list[list[float]]:
    rows = np.asarray(values, dtype=float).reshape(5, 4)
    return [_unit(row).tolist() for row in rows]


def _export_selected(
    rows: list[dict], raws: list[dict], tasks: dict[int, str], output_dir: Path,
    revision: str,
) -> list[dict]:
    selected: list[tuple[int, float]] = []
    for side, count in (("left", 3), ("right", 2)):
        indices = [index for index, row in enumerate(rows) if row["side"] == side]
        local = _k_medoids([rows[index] for index in indices], count)
        selected.extend((indices[index], score) for index, score in local)
    export_dir = output_dir / "selected_expert_trajectories"
    export_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for index, score in sorted(selected, key=lambda item: rows[item[0]]["episode_index"]):
        row = rows[index]
        raw = raws[index]
        path = export_dir / f"episode_{row['episode_index']:06d}_{row['side']}_expert.npz"
        np.savez_compressed(
            path,
            q_hand=raw["q_hand"].astype(np.float32),
            a_hand=raw["a_hand"].astype(np.float32),
            timestamp=raw["timestamp"].astype(np.float32),
            frame_index=raw["frame_index"],
            episode_index=np.int64(row["episode_index"]),
            task_index=np.int64(row["task_index"]),
            task=np.asarray(tasks[row["task_index"]]),
            side=np.asarray(row["side"]),
            joint_names=np.asarray(joint_names(row["side"])),
            inferred_onset_frame=np.int64(row["inferred_onset_frame"]),
            inferred_close_frame=np.int64(row["inferred_close_frame"]),
            stable_start_frame=np.int64(row["stable_start_frame"]),
            stable_end_frame_exclusive=np.int64(row["stable_end_frame_exclusive"]),
            dataset_revision=np.asarray(revision),
        )
        with np.load(path) as check:
            q_exact = bool(np.array_equal(
                check["q_hand"], raw["q_hand"].astype(np.float32)
            ))
            a_exact = bool(np.array_equal(
                check["a_hand"], raw["a_hand"].astype(np.float32)
            ))
            t_exact = bool(np.array_equal(
                check["timestamp"], raw["timestamp"].astype(np.float32)
            ))
        exported.append({
            "episode_index": row["episode_index"],
            "side": row["side"],
            "task_index": row["task_index"],
            "task": tasks[row["task_index"]],
            "trajectory_length_frames": row["trajectory_length_frames"],
            "typicality_cluster_mean_distance": score,
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "q_hand_bit_exact_to_source": q_exact,
            "a_hand_bit_exact_to_source": a_exact,
            "timestamp_bit_exact_to_source": t_exact,
        })
    return exported


def _read_tasks(path: Path) -> dict[int, str]:
    tasks = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        tasks[int(row["task_index"])] = row["task"]
    return tasks


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze raw LeRobot Wuji cube state/action Parquet trajectories"
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=root / "data/external/wuji-pick-and-place",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs/wuji_teleop_analysis",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.dataset / "download_manifest.json").read_text(
        encoding="utf-8"
    ))
    revision = manifest["revision"]
    tasks = _read_tasks(args.dataset / "teleop/meta/tasks.jsonl")
    data_dir = args.dataset / "teleop/data/chunk-000"
    rows: list[dict] = []
    raws: list[dict] = []
    for episode_index in range(60, 120):
        side = "left" if episode_index < 90 else "right"
        row, raw = analyze_episode(
            data_dir / f"episode_{episode_index:06d}.parquet", episode_index, side
        )
        rows.append(row)
        raws.append(raw)

    all_trajectories = args.output_dir / "cube_60_hand_trajectories.parquet"
    _write_all_trajectories(raws, rows, tasks, all_trajectories)
    all_export_validation = _validate_all_trajectories(all_trajectories, raws)
    episode_metrics = args.output_dir / "cube_episode_metrics.parquet"
    _write_episode_metrics(rows, tasks, episode_metrics)
    selected = _export_selected(rows, raws, tasks, args.output_dir, revision)
    aggregates = {
        "all": _aggregate(rows),
        "left": _aggregate([row for row in rows if row["side"] == "left"]),
        "right": _aggregate([row for row in rows if row["side"] == "right"]),
    }
    report = {
        "dataset": "yeeeiii111/wuji-pick-and-place",
        "dataset_revision": revision,
        "analysis_source": "raw teleop/data/chunk-000 LeRobot Parquet columns",
        "fps": FPS,
        "cube_episode_ranges": {
            "left": [60, 89], "right": [90, 119], "all": [60, 119]
        },
        "cube_tasks": {"2": tasks[2], "3": tasks[3]},
        "state_action_layout_0_based": {
            "left_arm": {"slice": "[0:7]", "indices_inclusive": [0, 6]},
            "right_arm": {"slice": "[7:14]", "indices_inclusive": [7, 13]},
            "left_hand": {"slice": "[14:34]", "indices_inclusive": [14, 33]},
            "right_hand": {"slice": "[34:54]", "indices_inclusive": [34, 53]},
        },
        "joint_name_provenance": (
            "Dataset info.json stores names=null. Names below use the official Wuji "
            "firmware/ROS convention: finger1..5=thumb,index,middle,ring,pinky; "
            "joint1..4=proximal-to-distal; flat index=finger_id*4+joint_id."
        ),
        "hand_joint_names": {
            "left": joint_names("left"), "right": joint_names("right")
        },
        "phase_inference_caveat": (
            "No contact/tactile column exists. Onset and main-close boundaries are "
            "inferred only from a_hand displacement; video is optional interval QA "
            "and is never used to infer joint values or coordination."
        ),
        "raw_numeric_validation": {
            "source_parquet_files": 60,
            "total_frames": int(sum(row["trajectory_length_frames"] for row in rows)),
            "state_shape": [54],
            "action_shape": [54],
            "state_dtype": "float32",
            "action_dtype": "float32",
            "all_hand_only_trajectories_path": all_trajectories.as_posix(),
            "all_hand_only_trajectories_sha256": hashlib.sha256(
                all_trajectories.read_bytes()
            ).hexdigest(),
            "all_hand_only_export_validation": all_export_validation,
            "episode_metrics_path": episode_metrics.as_posix(),
            "episode_metrics_sha256": hashlib.sha256(
                episode_metrics.read_bytes()
            ).hexdigest(),
        },
        **aggregates,
        "selected_episode_ids": [item["episode_index"] for item in selected],
        "selected_expert_trajectories": selected,
        "episodes": rows,
    }
    report["action_prior_summary"] = {
        "dataset": report["dataset"],
        "dataset_revision": revision,
        "joint_order": {
            "left": joint_names("left"), "right": joint_names("right")
        },
        "state_delta_q": {
            side: {
                "expert_mean_closing_vector_rad": aggregates[side][
                    "per_joint_vector_statistics"
                ]["delta_q_rad"]["mean"],
                "expert_median_closing_vector_rad": aggregates[side][
                    "per_joint_vector_statistics"
                ]["delta_q_rad"]["median"],
                "pca_basis": aggregates[side]["state_delta_q_pca"]["components"],
                "explained_variance_ratio": aggregates[side][
                    "state_delta_q_pca"
                ]["explained_variance_ratio"],
            }
            for side in ("all", "left", "right")
        },
        "action_delta": {
            side: {
                "expert_mean_closing_vector_rad": aggregates[side][
                    "per_joint_vector_statistics"
                ]["delta_action_rad"]["mean"],
                "expert_median_closing_vector_rad": aggregates[side][
                    "per_joint_vector_statistics"
                ]["delta_action_rad"]["median"],
                "per_finger_normalized_median_closing_ratios": (
                    _normalized_finger_rows(aggregates[side][
                        "per_joint_vector_statistics"
                    ]["delta_action_rad"]["median"])
                ),
                "pca_basis": aggregates[side]["action_delta_pca"]["components"],
                "explained_variance_ratio": aggregates[side][
                    "action_delta_pca"
                ]["explained_variance_ratio"],
            }
            for side in ("all", "left", "right")
        },
        "selected_episode_ids": report["selected_episode_ids"],
    }
    output = args.output_dir / "cube_hand_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_output = args.output_dir / "action_prior_summary.json"
    summary_output.write_text(json.dumps(
        report["action_prior_summary"], indent=2
    ), encoding="utf-8")
    print(json.dumps({
        "report": output.as_posix(),
        "summary": summary_output.as_posix(),
        "all_trajectories": all_trajectories.as_posix(),
        "total_frames": report["raw_numeric_validation"]["total_frames"],
        "selected_episode_ids": report["selected_episode_ids"],
        "state_delta_q_pca_dimensions": report["all"]["state_delta_q_pca"][
            "dimensions_for_percent_variance"
        ],
        "action_delta_pca_dimensions": report["all"]["action_delta_pca"][
            "dimensions_for_percent_variance"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
