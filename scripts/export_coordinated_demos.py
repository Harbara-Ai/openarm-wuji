"""Rebuild LeRobot-shaped staging Parquet from successful raw episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_wuji.dataset import export_successful_episodes


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=root / "outputs/coordinated_demos/raw/successful",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/coordinated_demos/lerobot_staging",
    )
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no raw episodes in {args.input}")
    manifest = export_successful_episodes(paths, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
