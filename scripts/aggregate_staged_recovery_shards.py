from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from evaluate_staged_with_recovery import _plain, _summarize


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_field(summaries: list[dict[str, Any]], field: str) -> Any:
    values = {json.dumps(summary[field], sort_keys=True) for summary in summaries}
    if len(values) != 1:
        raise ValueError(f"shards disagree on {field}: {sorted(values)}")
    return summaries[0][field]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        type=Path,
        default=root / "outputs/staged_act_with_recovery/unseen_shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/staged_act_with_recovery/unseen_matched",
    )
    parser.add_argument("--seed-start", type=int, default=1400)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary_paths = sorted(args.shards.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"no completed shard summaries under {args.shards}")
    summaries = [_load(path) for path in summary_paths]
    for path, summary in zip(summary_paths, summaries, strict=True):
        if summary["completed_rollouts"] != summary["requested_rollouts"]:
            raise ValueError(f"incomplete shard: {path}")
        if summary["expert_action_used"]:
            raise ValueError(f"expert action was used in shard: {path}")
        if summary["router"]["maximum_recovery_attempts"] != 1:
            raise ValueError(f"recovery-loop mismatch in shard: {path}")

    reach_checkpoint = Path(_same_field(summaries, "reach_checkpoint"))
    approach_checkpoint = Path(_same_field(summaries, "approach_checkpoint"))
    recovery_checkpoint = Path(_same_field(summaries, "recovery_checkpoint"))
    router = _same_field(summaries, "router")

    rows = sorted(
        (row for summary in summaries for row in summary["episodes_detail"]),
        key=lambda row: row["seed"],
    )
    actual_seeds = [int(row["seed"]) for row in rows]
    expected_seeds = list(range(args.seed_start, args.seed_start + args.rollouts))
    if len(actual_seeds) != len(set(actual_seeds)):
        raise ValueError("duplicate seeds found across shards")
    if actual_seeds != expected_seeds:
        missing = sorted(set(expected_seeds) - set(actual_seeds))
        extra = sorted(set(actual_seeds) - set(expected_seeds))
        raise ValueError(f"seed coverage mismatch; missing={missing}, extra={extra}")
    for index, row in enumerate(rows):
        row["rollout_index"] = index

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for pattern in ("recovery_seed_*.npz", "summary*.json", "shard_manifest.json"):
            for path in args.output.glob(pattern):
                path.unlink()

    copied = []
    for summary_path in summary_paths:
        for source in summary_path.parent.glob("recovery_seed_*.npz"):
            destination = args.output / source.name
            if destination.exists():
                raise FileExistsError(f"duplicate recovery trajectory: {destination}")
            shutil.copy2(source, destination)
            copied.append(destination.name)

    combined = _summarize(
        rows,
        requested=args.rollouts,
        seed_start=args.seed_start,
        reach_checkpoint=reach_checkpoint,
        approach_checkpoint=approach_checkpoint,
        recovery_checkpoint=recovery_checkpoint,
        router_spec=router.get("candidate_spec"),
    )
    (args.output / "summary.json").write_text(
        json.dumps(_plain(combined), indent=2), encoding="utf-8"
    )
    manifest = {
        "summary_paths": [path.resolve() for path in summary_paths],
        "seed_start": args.seed_start,
        "seed_end_inclusive": expected_seeds[-1],
        "unique_seed_count": len(set(actual_seeds)),
        "copied_recovery_trajectories": len(copied),
    }
    (args.output / "shard_manifest.json").write_text(
        json.dumps(_plain(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "completed_rollouts": combined["completed_rollouts"],
        "seed_start": combined["seed_start"],
        "seed_end_inclusive": combined["seed_end_inclusive"],
        "recovery_trajectories": len(copied),
        "reach_successes": combined["reach_successes"],
        "baseline_successes": combined["baseline"]["approach_successes"],
        "staged_with_recovery_successes": combined["staged_with_recovery"][
            "approach_stage_successes"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
