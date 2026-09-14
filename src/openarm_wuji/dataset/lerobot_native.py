"""Convert successful coordinated episodes into a native LeRobotDataset."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .coordinated_demo_recorder import (
    EMBODIMENT_DOF,
    JOINT_NAMES,
    CoordinatedDemoRecorder,
)


def _load_lerobot_api():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "native export requires the LeRobot dataset dependencies; install "
            "the project's vendor/lerobot[dataset] extra"
        ) from error
    return LeRobotDataset, DataLoader


def _features(image_shape: tuple[int, int, int]) -> dict[str, dict[str, Any]]:
    height, width, channels = image_shape
    if channels != 3:
        raise ValueError(f"RGB observations require 3 channels, got {channels}")
    image_feature = {
        "dtype": "image",
        "shape": (channels, height, width),
        "names": ["channels", "height", "width"],
    }
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (EMBODIMENT_DOF,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (EMBODIMENT_DOF,),
            "names": JOINT_NAMES,
        },
        "observation.images.front": dict(image_feature),
        "observation.images.wrist": dict(image_feature),
    }


def validate_native_lerobot_dataset(
    root: str | Path,
    *,
    repo_id: str,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate v3 episode boundaries and synchronous 30 Hz frame indexing."""
    destination = Path(root)
    cache_root = Path(cache_dir) if cache_dir is not None else (
        destination.parent / ".hf_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    LeRobotDataset, _ = _load_lerobot_api()
    import datasets

    datasets.config.HF_DATASETS_CACHE = str(cache_root / "datasets")
    loaded = LeRobotDataset(
        repo_id=repo_id, root=destination, video_backend="pyav"
    )
    hf = loaded.hf_dataset
    episode_index = np.asarray(hf["episode_index"], dtype=np.int64)
    frame_index = np.asarray(hf["frame_index"], dtype=np.int64)
    global_index = np.asarray(hf["index"], dtype=np.int64)
    timestamp = np.asarray(hf["timestamp"], dtype=np.float64)
    expected_front_shape = list(
        loaded.meta.features["observation.images.front"]["shape"]
    )
    expected_wrist_shape = list(
        loaded.meta.features["observation.images.wrist"]["shape"]
    )
    failures: list[str] = []
    episode_rows = []
    if not np.array_equal(global_index, np.arange(len(loaded))):
        failures.append("global index is not contiguous")
    expected_episode_ids = np.arange(loaded.num_episodes)
    if not np.array_equal(np.unique(episode_index), expected_episode_ids):
        failures.append("episode indices are not contiguous")
    for episode_id in expected_episode_ids:
        positions = np.flatnonzero(episode_index == episode_id)
        local_frames = frame_index[positions]
        local_timestamps = timestamp[positions]
        expected_frames = np.arange(len(positions))
        expected_timestamps = expected_frames / loaded.fps
        contiguous_span = bool(
            len(positions) > 0
            and np.array_equal(positions, np.arange(positions[0], positions[-1] + 1))
        )
        frames_match = bool(np.array_equal(local_frames, expected_frames))
        timestamps_match = bool(np.allclose(
            local_timestamps, expected_timestamps, atol=1e-6, rtol=0.0
        ))
        if not contiguous_span:
            failures.append(f"episode {episode_id} is split across the table")
        if not frames_match:
            failures.append(f"episode {episode_id} frame_index does not reset at zero")
        if not timestamps_match:
            failures.append(f"episode {episode_id} timestamps are not frame_index/fps")
        first = loaded[int(positions[0])]
        front_shape = list(first["observation.images.front"].shape)
        wrist_shape = list(first["observation.images.wrist"].shape)
        images_match = (
            front_shape == expected_front_shape
            and wrist_shape == expected_wrist_shape
        )
        if not images_match:
            failures.append(f"episode {episode_id} has an unexpected image shape")
        episode_rows.append({
            "episode_index": int(episode_id),
            "frames": int(len(positions)),
            "global_start": int(positions[0]),
            "global_end_inclusive": int(positions[-1]),
            "frame_index_contiguous": frames_match,
            "timestamp_aligned_30hz": timestamps_match,
            "front_wrist_shape_aligned": images_match,
        })
    return {
        "passed": not failures,
        "codebase_version": str(loaded.meta.info.codebase_version),
        "episodes": int(loaded.num_episodes),
        "frames": int(len(loaded)),
        "fps": int(loaded.fps),
        "global_index_contiguous": bool(np.array_equal(
            global_index, np.arange(len(loaded))
        )),
        "failures": failures,
        "episode_boundaries": episode_rows,
    }
def export_native_lerobot_dataset(
    paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    repo_id: str = "local/openarm-wuji-coordinated",
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write successful raw episodes through the installed LeRobot 0.6 API.

    Visual features deliberately use LeRobot's image storage rather than video
    storage. This keeps conversion and loading independent of platform FFmpeg /
    TorchCodec support. A later storage-only migration to MP4 does not change
    state/action semantics.
    """
    sources = sorted((Path(item) for item in paths), key=lambda item: item.name)
    if not sources:
        raise ValueError("no raw episodes were provided")

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"native LeRobot output already exists: {destination}"
        )
    cache_root = Path(cache_dir) if cache_dir is not None else (
        destination.parent / ".hf_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    LeRobotDataset, DataLoader = _load_lerobot_api()
    import datasets

    datasets.config.HF_DATASETS_CACHE = str(cache_root / "datasets")

    expected_states: list[np.ndarray] = []
    expected_actions: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    fps: int | None = None
    image_shape: tuple[int, int, int] | None = None
    dataset = None

    for source in sources:
        validation = CoordinatedDemoRecorder.validate(source)
        if not validation["demonstration_success"]:
            continue
        with np.load(source, allow_pickle=False) as episode:
            current_hz = float(episode["control_hz"])
            rounded_hz = int(round(current_hz))
            if not np.isclose(current_hz, rounded_hz):
                raise ValueError("native LeRobot export requires an integer fps")
            current_shape = tuple(
                int(value)
                for value in episode["observation.images.front"].shape[1:]
            )
            wrist_shape = tuple(
                int(value)
                for value in episode["observation.images.wrist"].shape[1:]
            )
            if current_shape != wrist_shape:
                raise ValueError("front and wrist image shapes must match")
            if dataset is None:
                fps = rounded_hz
                image_shape = current_shape
                dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=fps,
                    features=_features(image_shape),
                    root=destination,
                    robot_type="openarm_wuji",
                    use_videos=False,
                    video_backend="pyav",
                    image_writer_threads=0,
                )
            elif rounded_hz != fps or current_shape != image_shape:
                raise ValueError(
                    "all native episodes must share fps and camera resolution"
                )

            states = episode["observation.state"].astype(np.float32)
            actions = episode["action"].astype(np.float32)
            front = episode["observation.images.front"]
            wrist = episode["observation.images.wrist"]
            task = str(episode["task_description"])
            for frame_index in range(len(actions)):
                dataset.add_frame({
                    "observation.state": states[frame_index],
                    "action": actions[frame_index],
                    "observation.images.front": front[frame_index],
                    "observation.images.wrist": wrist[frame_index],
                    "task": task,
                })
            dataset.save_episode(parallel_encoding=False)
            expected_states.append(states)
            expected_actions.append(actions)
            source_rows.append({
                "native_episode_index": len(source_rows),
                "source": source.as_posix(),
                "source_episode_index": int(episode["episode_index"]),
                "seed": int(episode["episode_seed"]),
                "frames": len(actions),
                "task": task,
            })

    if dataset is None:
        raise ValueError("no successful demonstrations were provided")
    dataset.finalize()

    loaded = LeRobotDataset(
        repo_id=repo_id, root=destination, video_backend="pyav"
    )
    loader = DataLoader(loaded, batch_size=min(8, len(loaded)), shuffle=False,
                        num_workers=0)
    first_batch = next(iter(loader))
    expected_state = np.concatenate(expected_states)
    expected_action = np.concatenate(expected_actions)
    loaded_state = np.stack([
        value.numpy() for value in loaded.hf_dataset["observation.state"]
    ])
    loaded_action = np.stack([
        value.numpy() for value in loaded.hf_dataset["action"]
    ])
    if not np.array_equal(loaded_state, expected_state):
        raise RuntimeError("native LeRobot state round-trip changed values")
    if not np.array_equal(loaded_action, expected_action):
        raise RuntimeError("native LeRobot action round-trip changed values")

    front_shape = list(first_batch["observation.images.front"].shape)
    wrist_shape = list(first_batch["observation.images.wrist"].shape)
    manifest = {
        "format": "LeRobotDataset",
        "codebase_version": str(loaded.meta.info.codebase_version),
        "native_lerobot_dataset": True,
        "repo_id": repo_id,
        "cache_dir": str(cache_root),
        "robot_type": "openarm_wuji",
        "successful_episodes_only": True,
        "episodes": loaded.num_episodes,
        "frames": len(loaded),
        "fps": fps,
        "state_dim": EMBODIMENT_DOF,
        "action_dim": EMBODIMENT_DOF,
        "visual_storage": "LeRobot image features (PNG, use_videos=False)",
        "dataloader_smoke_test": {
            "passed": True,
            "batch_size": int(first_batch["action"].shape[0]),
            "state_shape": list(first_batch["observation.state"].shape),
            "action_shape": list(first_batch["action"].shape),
            "front_shape": front_shape,
            "wrist_shape": wrist_shape,
        },
        "exact_round_trip": {"state": True, "action": True},
        "source_episodes": source_rows,
    }
    manifest["alignment_validation"] = validate_native_lerobot_dataset(
        destination, repo_id=repo_id, cache_dir=cache_root
    )
    if not manifest["alignment_validation"]["passed"]:
        raise RuntimeError(
            "native LeRobot episode/timestamp alignment validation failed"
        )
    (destination / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
