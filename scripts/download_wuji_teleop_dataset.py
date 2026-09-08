from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "yeeeiii111/wuji-pick-and-place"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download Wuji teleoperation numeric trajectories without videos"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/external/wuji-pick-and-place",
    )
    parser.add_argument(
        "--cube-videos", action="store_true",
        help="also download both wrist videos for cube episodes 60-119",
    )
    args = parser.parse_args()

    info = HfApi().dataset_info(REPO_ID, files_metadata=True)
    revision = info.sha
    selected = [
        sibling for sibling in info.siblings
        if sibling.rfilename.startswith("teleop/meta/")
        or (
            sibling.rfilename.startswith("teleop/data/")
            and sibling.rfilename.endswith(".parquet")
        )
    ]
    allow_patterns = [
        "teleop/meta/*", "teleop/data/**/*.parquet", "README.md"
    ]
    if args.cube_videos:
        for camera in (
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            allow_patterns.extend(
                f"teleop/videos/chunk-000/{camera}/episode_{episode:06d}.mp4"
                for episode in range(60, 120)
            )
        cube_names = {
            f"episode_{episode:06d}.mp4" for episode in range(60, 120)
        }
        selected.extend(
            sibling for sibling in info.siblings
            if sibling.rfilename.startswith("teleop/videos/chunk-000/")
            and "wrist" in sibling.rfilename
            and Path(sibling.rfilename).name in cube_names
        )
        for camera in (
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            (
                args.output / ".cache/huggingface/download/teleop/videos/"
                / "chunk-000" / camera
            ).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=args.output,
        allow_patterns=allow_patterns,
        max_workers=4,
    )
    manifest = {
        "repo_id": REPO_ID,
        "revision": revision,
        "selection": (
            "teleop numeric trajectories and metadata; cube wrist videos included"
            if args.cube_videos else
            "teleop numeric trajectories and metadata; videos excluded"
        ),
        "selected_file_count": len(selected),
        "selected_bytes": sum(sibling.size or 0 for sibling in selected),
        "dataset_total_bytes": sum(
            sibling.size or 0 for sibling in info.siblings
        ),
        "cube_wrist_videos_included": args.cube_videos,
    }
    (args.output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
