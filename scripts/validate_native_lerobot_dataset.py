"""Validate episode boundaries and timestamps in a local native dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_wuji.dataset import validate_native_lerobot_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=root / "outputs/act_e2e_smoke/lerobot_dataset",
    )
    parser.add_argument("--repo-id", default="local/openarm-wuji-act-smoke")
    parser.add_argument(
        "--cache-dir", type=Path, default=root.parent / ".hf_cache"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_native_lerobot_dataset(
        args.dataset, repo_id=args.repo_id, cache_dir=args.cache_dir
    )
    destination = args.output or args.dataset / "alignment_validation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path = args.dataset / "conversion_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["alignment_validation"] = report
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
