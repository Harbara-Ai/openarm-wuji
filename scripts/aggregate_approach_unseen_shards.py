"""Merge disjoint metrics-only Approach rollout shards into one summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mine_approach_near_failures import _plain, _summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reach-checkpoint", type=Path, required=True)
    parser.add_argument("--approach-checkpoint", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--requested", type=int, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rollouts"])

    rows.sort(key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in rows]
    expected = list(range(args.seed_start, args.seed_start + args.requested))
    if seeds != expected:
        raise ValueError(
            "Shard seeds must be unique, contiguous, and cover the requested range; "
            f"got {len(seeds)} rows from {seeds[:1]} to {seeds[-1:]}."
        )

    summary = _summarize(
        rows,
        requested=args.requested,
        seed_start=args.seed_start,
        reach_checkpoint=args.reach_checkpoint,
        approach_checkpoint=args.approach_checkpoint,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(_plain(summary), indent=2), encoding="utf-8")
    print(json.dumps(_plain({
        key: summary[key]
        for key in (
            "completed_rollouts",
            "reach_successes",
            "approach_attempts",
            "approach_successes",
            "approach_failures",
            "candidate_snapshots",
            "rollouts_with_trigger",
            "trigger_type_distribution",
        )
    }), indent=2))


if __name__ == "__main__":
    main()
