"""Validate raw and native Approach correction dataset compatibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from openarm_wuji.dataset import validate_native_lerobot_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=root / "outputs/approach_correction_demos",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=root / ".hf_approach_corrections",
    )
    parser.add_argument(
        "--repo-id", default="local/openarm-wuji-approach-corrections"
    )
    args = parser.parse_args()
    paths = sorted((args.root / "raw").glob("episode_*.npz"))
    if not paths:
        raise ValueError("no raw correction episodes")
    total_frames = 0
    triggers: set[str] = set()
    maximum_hand_change = 0.0
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            frames = len(episode["action"])
            total_frames += frames
            if episode["observation.state"].shape != (frames, 27):
                raise ValueError(f"invalid state shape: {path}")
            if episode["action"].shape != (frames, 27):
                raise ValueError(f"invalid action shape: {path}")
            for key in (
                "observation.images.front", "observation.images.wrist"
            ):
                if episode[key].shape != (frames, 240, 320, 3):
                    raise ValueError(f"invalid {key} shape: {path}")
            if not np.array_equal(episode["frame_index"], np.arange(frames)):
                raise ValueError(f"non-contiguous frame index: {path}")
            if not np.allclose(
                episode["timestamp"], np.arange(frames) / 30,
                rtol=0.0, atol=1e-12,
            ):
                raise ValueError(f"timestamp mismatch: {path}")
            if not bool(episode["correction_success"]):
                raise ValueError(f"failed correction included: {path}")
            forbidden = {"grasp", "preload", "lift", "hold"}
            if forbidden.intersection(episode["phase"].astype(str)):
                raise ValueError(f"forbidden phase in correction data: {path}")
            maximum_hand_change = max(
                maximum_hand_change,
                float(np.max(np.ptp(episode["action"][:, 7:], axis=0))),
            )
            triggers.add(str(episode["trigger_type"]))

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(args.cache_dir / "datasets")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    native_root = args.root / "lerobot_dataset"
    alignment = validate_native_lerobot_dataset(
        native_root, repo_id=args.repo_id, cache_dir=args.cache_dir
    )
    chunked = LeRobotDataset(
        repo_id=args.repo_id,
        root=native_root,
        video_backend="pyav",
        delta_timestamps={"action": [index / 30 for index in range(20)]},
    )
    batch = next(iter(DataLoader(
        chunked, batch_size=min(8, len(chunked)), shuffle=False, num_workers=0
    )))
    report = {
        "passed": bool(alignment["passed"] and maximum_hand_change <= 2e-6),
        "raw_episodes": len(paths),
        "raw_frames": total_frames,
        "raw_schema_compatible": True,
        "trigger_types": sorted(triggers),
        "maximum_wuji_target_change_rad": maximum_hand_change,
        "native_alignment": alignment,
        "act_chunk20_dataloader": {
            "observation.state": list(batch["observation.state"].shape),
            "action": list(batch["action"].shape),
            "action_is_pad": list(batch["action_is_pad"].shape),
            "observation.images.front": list(
                batch["observation.images.front"].shape
            ),
            "observation.images.wrist": list(
                batch["observation.images.wrist"].shape
            ),
        },
    }
    destination = args.root / "validation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
