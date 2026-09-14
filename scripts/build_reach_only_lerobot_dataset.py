"""Build a native LeRobotDataset containing only Reach plus terminal hold.

The real Reach prefix is copied from each successful coordinated demo.  A
small absorbing terminal segment repeats the first post-Reach observation
while continuing to command the final (already constant) Reach target.  No
Approach/Grasp/Preload/Lift/Hold action is copied into this dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.dataset.lerobot_native import _features


TASK = "Move OpenArm and the open Wuji hand to the cube pregrasp pose and hold it stable."
FORMAT_NAME = "openarm_wuji_reach_only"


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


def _make_episode(source: Path, hold_frames: int) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as episode:
        phases = episode["phase"].astype(str)
        reach_indices = np.flatnonzero(phases == "reach")
        if not len(reach_indices) or reach_indices[0] != 0:
            raise ValueError(f"Reach must start at frame zero: {source}")
        if not np.array_equal(reach_indices, np.arange(reach_indices[-1] + 1)):
            raise ValueError(f"Reach frames are not a contiguous prefix: {source}")
        terminal_index = int(reach_indices[-1] + 1)
        if terminal_index >= len(phases) or phases[terminal_index] != "approach":
            raise ValueError(f"first post-Reach observation is not Approach: {source}")

        real_frames = len(reach_indices)
        reach_actions = np.asarray(episode["action"][reach_indices], dtype=np.float32)
        if real_frames >= 5 and np.max(np.abs(np.diff(reach_actions[-5:], axis=0))) > 1e-7:
            raise ValueError(f"final Reach target is not stable for five frames: {source}")
        if np.max(np.ptp(reach_actions[:, 7:], axis=0)) > 1e-7:
            raise ValueError(f"Wuji target changes during Reach: {source}")

        terminal_state = np.asarray(
            episode["observation.state"][terminal_index], dtype=np.float32
        )
        terminal_front = np.asarray(
            episode["observation.images.front"][terminal_index], dtype=np.uint8
        )
        terminal_wrist = np.asarray(
            episode["observation.images.wrist"][terminal_index], dtype=np.uint8
        )
        final_target = reach_actions[-1]
        states = np.concatenate([
            np.asarray(episode["observation.state"][reach_indices], dtype=np.float32),
            np.repeat(terminal_state[None], hold_frames, axis=0),
        ])
        actions = np.concatenate([
            reach_actions,
            np.repeat(final_target[None], hold_frames, axis=0),
        ])
        front = np.concatenate([
            np.asarray(episode["observation.images.front"][reach_indices], dtype=np.uint8),
            np.repeat(terminal_front[None], hold_frames, axis=0),
        ])
        wrist = np.concatenate([
            np.asarray(episode["observation.images.wrist"][reach_indices], dtype=np.uint8),
            np.repeat(terminal_wrist[None], hold_frames, axis=0),
        ])
        cube_pose = np.concatenate([
            np.asarray(episode["telemetry.cube_pose_world"][reach_indices], dtype=np.float64),
            np.repeat(
                np.asarray(episode["telemetry.cube_pose_world"][terminal_index], dtype=np.float64)[None],
                hold_frames, axis=0,
            ),
        ])
        fps = float(episode["control_hz"])
        seed = int(episode["episode_seed"])
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
            "action": actions,
            "observation.images.front": front,
            "observation.images.wrist": wrist,
            "telemetry.cube_pose_world": cube_pose,
            "timestamp": np.arange(len(actions), dtype=np.float64) / fps,
            "frame_index": np.arange(len(actions), dtype=np.int64),
            "phase": np.asarray(["reach"] * real_frames + ["reach_hold"] * hold_frames),
            "source_frame_index": np.concatenate([
                reach_indices.astype(np.int64),
                np.full(hold_frames, terminal_index, dtype=np.int64),
            ]),
            "synthetic_terminal_hold": np.concatenate([
                np.zeros(real_frames, dtype=bool), np.ones(hold_frames, dtype=bool)
            ]),
        }
    result["metadata_json"] = np.asarray(json.dumps({
        "seed": seed,
        "source": source.resolve().as_posix(),
        "real_reach_frames": real_frames,
        "terminal_hold_frames": hold_frames,
        "terminal_observation_source_frame": terminal_index,
        "terminal_action_source_frame": int(reach_indices[-1]),
        "terminal_construction": (
            "repeat first post-Reach observation and final constant Reach controller target"
        ),
        "excluded_action_phases": ["approach", "grasp_close", "preload", "preload_settle", "lift", "hold"],
    }))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/openarm-wuji-reach-only")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--hold-frames", type=int, default=8)
    args = parser.parse_args()
    if args.hold_frames < 5 or args.hold_frames > 10:
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
    rows = []
    built = []
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
            "real_reach_frames": int(np.count_nonzero(item["phase"] == "reach")),
            "terminal_hold_frames": args.hold_frames,
            "source": source.resolve().as_posix(),
            "raw": raw_path.resolve().as_posix(),
        })

    first = built[0][1]
    fps = int(round(float(first["control_hz"])))
    image_shape = tuple(int(v) for v in first["observation.images.front"].shape[1:])
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
        states = item["observation.state"]
        actions = item["action"]
        front = item["observation.images.front"]
        wrist = item["observation.images.wrist"]
        for frame in range(len(actions)):
            dataset.add_frame({
                "observation.state": states[frame],
                "action": actions[frame],
                "observation.images.front": front[frame],
                "observation.images.wrist": wrist[frame],
                "task": TASK,
            })
        dataset.save_episode(parallel_encoding=False)
        expected_states.append(states)
        expected_actions.append(actions)
    dataset.finalize()

    loaded = LeRobotDataset(repo_id=args.repo_id, root=native_dir, video_backend="pyav")
    actual_states = np.stack([np.asarray(v) for v in loaded.hf_dataset["observation.state"]])
    actual_actions = np.stack([np.asarray(v) for v in loaded.hf_dataset["action"]])
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
        "real_reach_frames": int(sum(row["real_reach_frames"] for row in rows)),
        "synthetic_terminal_hold_frames": int(args.hold_frames * len(rows)),
        "excluded_action_phases": ["approach", "grasp_close", "preload", "preload_settle", "lift", "hold"],
        "wuji_target_constant_during_reach": True,
        "exact_round_trip": {"state": state_round_trip, "action": action_round_trip},
        "episodes_detail": rows,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(_plain(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps(_plain(manifest), indent=2))


if __name__ == "__main__":
    main()
