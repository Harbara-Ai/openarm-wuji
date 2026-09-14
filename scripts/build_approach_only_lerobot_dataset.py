"""Build the staged Approach-only LeRobotDataset from successful demos.

Each episode starts at the first real Approach observation and ends at the
first Grasp observation.  The latter is repeated as a short absorbing hold
while the final stable Approach controller target is held.  No grasp-closing
action is copied into the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.lerobot_native import _features


TASK = "Move the open Wuji hand from stable pregrasp to the cube grasp-start pose and hold."
FORMAT_NAME = "openarm_wuji_approach_only"


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _repeat_terminal(array: np.ndarray, index: int, count: int) -> np.ndarray:
    return np.repeat(np.asarray(array[index])[None], count, axis=0)


def _make_episode(source: Path, hold_frames: int) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as episode:
        phases = episode["phase"].astype(str)
        indices = np.flatnonzero(phases == "approach")
        if not len(indices):
            raise ValueError(f"episode has no Approach phase: {source}")
        expected = np.arange(indices[0], indices[-1] + 1)
        if not np.array_equal(indices, expected):
            raise ValueError(f"Approach frames are not contiguous: {source}")
        if indices[0] == 0 or phases[indices[0] - 1] != "reach":
            raise ValueError(f"Approach does not follow Reach: {source}")
        terminal_index = int(indices[-1] + 1)
        if terminal_index >= len(phases) or phases[terminal_index] != "grasp_close":
            raise ValueError(f"first post-Approach observation is not Grasp: {source}")

        actions = np.asarray(episode["action"][indices], dtype=np.float32)
        if len(actions) < 5 or np.max(np.abs(np.diff(actions[-5:], axis=0))) > 1e-7:
            raise ValueError(f"final Approach target is not stable for five frames: {source}")
        if np.max(np.ptp(actions[:, 7:], axis=0)) > 1e-7:
            raise ValueError(f"Wuji target changes during Approach: {source}")

        real_frames = len(indices)
        final_target = actions[-1]
        states = np.concatenate([
            np.asarray(episode["observation.state"][indices], dtype=np.float32),
            _repeat_terminal(episode["observation.state"], terminal_index, hold_frames),
        ]).astype(np.float32)
        all_actions = np.concatenate([
            actions,
            np.repeat(final_target[None], hold_frames, axis=0),
        ]).astype(np.float32)
        front = np.concatenate([
            np.asarray(episode["observation.images.front"][indices], dtype=np.uint8),
            _repeat_terminal(episode["observation.images.front"], terminal_index, hold_frames),
        ]).astype(np.uint8)
        wrist = np.concatenate([
            np.asarray(episode["observation.images.wrist"][indices], dtype=np.uint8),
            _repeat_terminal(episode["observation.images.wrist"], terminal_index, hold_frames),
        ]).astype(np.uint8)
        cube_world = np.concatenate([
            np.asarray(episode["telemetry.cube_pose_world"][indices], dtype=np.float64),
            _repeat_terminal(episode["telemetry.cube_pose_world"], terminal_index, hold_frames),
        ])
        cube_relative = np.concatenate([
            np.asarray(
                episode["telemetry.cube_pose_relative_to_palm"][indices], dtype=np.float64
            ),
            _repeat_terminal(
                episode["telemetry.cube_pose_relative_to_palm"], terminal_index, hold_frames
            ),
        ])
        contacts = np.concatenate([
            np.asarray(episode["telemetry.contact_count"][indices], dtype=np.int32),
            _repeat_terminal(episode["telemetry.contact_count"], terminal_index, hold_frames),
        ]).astype(np.int32)
        active = np.concatenate([
            np.asarray(episode["telemetry.active_finger_mask"][indices], dtype=bool),
            _repeat_terminal(
                episode["telemetry.active_finger_mask"], terminal_index, hold_frames
            ),
        ]).astype(bool)
        forces = np.concatenate([
            np.asarray(episode["telemetry.finger_normal_force_n"][indices], dtype=np.float64),
            _repeat_terminal(
                episode["telemetry.finger_normal_force_n"], terminal_index, hold_frames
            ),
        ])
        fps = float(episode["control_hz"])
        seed = int(episode["episode_seed"])
        initial_controller_target = np.asarray(
            episode["action"][indices[0] - 1], dtype=np.float32
        )
        result = {
            "format_name": np.asarray(FORMAT_NAME),
            "schema_version": np.asarray(1, dtype=np.int64),
            "task_description": np.asarray(TASK),
            "source": np.asarray(source.resolve().as_posix()),
            "source_episode_index": np.asarray(int(episode["episode_index"]), dtype=np.int64),
            "episode_seed": np.asarray(seed, dtype=np.int64),
            "control_hz": np.asarray(fps, dtype=np.float64),
            "joint_names": np.asarray([str(name) for name in episode["joint_names"]]),
            "observation.state": states,
            "action": all_actions,
            "observation.images.front": front,
            "observation.images.wrist": wrist,
            "telemetry.cube_pose_world": cube_world,
            "telemetry.cube_pose_relative_to_palm": cube_relative,
            "telemetry.contact_count": contacts,
            "telemetry.active_finger_mask": active,
            "telemetry.finger_normal_force_n": forces,
            "initial_controller_target": initial_controller_target,
            "timestamp": np.arange(len(all_actions), dtype=np.float64) / fps,
            "frame_index": np.arange(len(all_actions), dtype=np.int64),
            "phase": np.asarray(
                ["approach"] * real_frames + ["approach_hold"] * hold_frames
            ),
            "source_frame_index": np.concatenate([
                indices.astype(np.int64),
                np.full(hold_frames, terminal_index, dtype=np.int64),
            ]),
            "synthetic_terminal_hold": np.concatenate([
                np.zeros(real_frames, dtype=bool), np.ones(hold_frames, dtype=bool)
            ]),
        }
    result["metadata_json"] = np.asarray(json.dumps({
        "seed": seed,
        "source": source.resolve().as_posix(),
        "real_approach_frames": real_frames,
        "terminal_hold_frames": hold_frames,
        "initial_observation_source_frame": int(indices[0]),
        "initial_controller_target_source_frame": int(indices[0] - 1),
        "terminal_observation_source_frame": terminal_index,
        "terminal_action_source_frame": int(indices[-1]),
        "excluded_action_phases": [
            "reach", "grasp_close", "preload", "preload_settle", "lift", "hold"
        ],
    }))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/openarm-wuji-approach-only")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--hold-frames", type=int, default=8)
    args = parser.parse_args()
    if not 5 <= args.hold_frames <= 10:
        raise ValueError("--hold-frames must be between 5 and 10")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    sources = sorted(args.input.glob("episode_*.npz"))
    if len(sources) != 20:
        raise ValueError(f"expected 20 successful demos, found {len(sources)}")

    raw_dir = args.output / "raw"
    native_dir = args.output / "lerobot_dataset"
    raw_dir.mkdir(parents=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(args.cache_dir / "datasets")
    import datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets.config.HF_DATASETS_CACHE = str(args.cache_dir / "datasets")
    built = []
    rows = []
    for episode_index, source in enumerate(sources):
        item = _make_episode(source, args.hold_frames)
        seed = int(item["episode_seed"])
        item["episode_index"] = np.asarray(episode_index, dtype=np.int64)
        raw_path = raw_dir / f"episode_{episode_index:06d}_seed_{seed:06d}.npz"
        np.savez_compressed(raw_path, **item)
        built.append((raw_path, item))
        rows.append({
            "episode_index": episode_index,
            "seed": seed,
            "frames": int(len(item["action"])),
            "real_approach_frames": int(np.count_nonzero(item["phase"] == "approach")),
            "terminal_hold_frames": args.hold_frames,
            "source": source.resolve().as_posix(),
            "raw": raw_path.resolve().as_posix(),
        })

    first = built[0][1]
    fps = int(round(float(first["control_hz"])))
    image_shape = tuple(int(value) for value in first["observation.images.front"].shape[1:])
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=_features(image_shape),
        root=native_dir,
        robot_type="openarm_wuji",
        use_videos=False,
        video_backend="pyav",
        image_writer_threads=0,
    )
    expected_states = []
    expected_actions = []
    for _, item in built:
        for frame in range(len(item["action"])):
            dataset.add_frame({
                "observation.state": item["observation.state"][frame],
                "action": item["action"][frame],
                "observation.images.front": item["observation.images.front"][frame],
                "observation.images.wrist": item["observation.images.wrist"][frame],
                "task": TASK,
            })
        dataset.save_episode(parallel_encoding=False)
        expected_states.append(item["observation.state"])
        expected_actions.append(item["action"])
    dataset.finalize()

    loaded = LeRobotDataset(repo_id=args.repo_id, root=native_dir, video_backend="pyav")
    actual_states = np.stack([np.asarray(value) for value in loaded.hf_dataset["observation.state"]])
    actual_actions = np.stack([np.asarray(value) for value in loaded.hf_dataset["action"]])
    state_round_trip = np.array_equal(actual_states, np.concatenate(expected_states))
    action_round_trip = np.array_equal(actual_actions, np.concatenate(expected_actions))
    if not state_round_trip or not action_round_trip:
        raise RuntimeError("native dataset changed state/action values")
    manifest = {
        "format": "LeRobotDataset",
        "codebase_version": str(loaded.meta.info.codebase_version),
        "repo_id": args.repo_id,
        "root": native_dir.resolve(),
        "task": TASK,
        "episodes": int(loaded.num_episodes),
        "frames": int(len(loaded)),
        "fps": int(loaded.fps),
        "state_dim": 27,
        "action_dim": 27,
        "image_shape_hwc": image_shape,
        "chunk_target_shape": [20, 27],
        "hold_frames_per_episode": args.hold_frames,
        "real_approach_frames": int(sum(row["real_approach_frames"] for row in rows)),
        "synthetic_terminal_hold_frames": int(args.hold_frames * len(rows)),
        "excluded_action_phases": [
            "reach", "grasp_close", "preload", "preload_settle", "lift", "hold"
        ],
        "wuji_target_constant_during_approach": True,
        "exact_round_trip": {"state": state_round_trip, "action": action_round_trip},
        "episodes_detail": rows,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(_plain(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps(_plain(manifest), indent=2))


if __name__ == "__main__":
    main()
