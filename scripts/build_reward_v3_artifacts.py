"""Assemble compact machine-readable Reward V3 experiment artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)), "median": float(np.median(array)),
        "std": float(np.std(array)), "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def training_components(report: dict[str, Any]) -> dict[str, Any]:
    by_term: dict[str, list[float]] = {}
    for episode in report["training_episode_reward_component_sums"]:
        for name, value in episode.items():
            by_term.setdefault(name, []).append(float(value))
    return {name: stats(values) for name, values in by_term.items()}


def compact_training(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key] for key in (
            "algorithm", "reward_version", "action_version", "action_representation",
            "timesteps", "seed", "observation_dimension", "action_dimension",
            "hyperparameters", "replay_buffer_size", "gradient_updates",
            "actor_parameter_l2_change", "first_reward_window_mean",
            "last_reward_window_mean", "max_contact_fingers", "success_events",
            "initial_entropy_coefficient", "final_entropy_coefficient",
            "training_action_statistics", "policy_distribution_diagnostics",
            "completed_training_episodes", "training_episode_returns",
            "training_episode_lengths", "training_episode_max_contact_fingers",
            "training_episode_failure_reasons", "training_milestone_episode_counts",
            "contact_episode_fraction", "multi_contact_episode_fraction",
            "three_contact_episode_fraction", "deterministic_evaluation",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--v2-report", type=Path, required=True)
    parser.add_argument("--v3-report", type=Path, required=True)
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2_report.read_text(encoding="utf-8"))
    v3 = json.loads(args.v3_report.read_text(encoding="utf-8"))
    deterministic = json.loads(args.deterministic.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reward_ranking.json").write_text(
        json.dumps(ranking, indent=2), encoding="utf-8"
    )
    (args.output / "sac_5k_metrics.json").write_text(json.dumps({
        "comparison": "fresh PCA5+V2 5K vs fresh PCA5+V3 5K, seed 7",
        "v2": compact_training(v2), "v3": compact_training(v3),
    }, indent=2), encoding="utf-8")
    (args.output / "reward_component_stats.json").write_text(json.dumps({
        "training_episode_component_stats": {
            "v2": training_components(v2), "v3": training_components(v3),
        },
        "deterministic_seed_7_11_component_stats": {
            version: deterministic["aggregate"][version]["reward_component_sums"]
            for version in ("v2", "v3")
        },
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ranking_pass": ranking["pass"]}, indent=2))


if __name__ == "__main__":
    main()
