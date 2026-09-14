"""Export successful coordinated recordings via LeRobotDataset 0.6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_wuji.dataset import export_native_lerobot_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=root / "outputs/coordinated_demos/raw/successful",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/coordinated_demos/lerobot_dataset",
    )
    parser.add_argument(
        "--repo-id", default="local/openarm-wuji-coordinated",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=root.parent / ".hf_cache",
        help="short local cache path; avoids Windows lock-file path overflow",
    )
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no raw episodes in {args.input}")
    manifest = export_native_lerobot_dataset(
        paths, args.output, repo_id=args.repo_id, cache_dir=args.cache_dir
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
