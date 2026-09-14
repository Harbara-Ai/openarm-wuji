"""Train ACT on original Approach and correction episodes with a fixed source mix.

The wrapper deliberately keeps LeRobot's policy, optimizer, preprocessing, and
checkpoint code unchanged.  It replaces only dataset construction and the
training sampler.  Source selection is exact-quota and deterministic; within a
source, episodes (and correction trigger/outcome strata) are sampled uniformly
before choosing a frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator

import numpy as np


SOURCE_NAMES = ("original", "correction")
SOURCE_TARGETS = {"original": 0.70, "correction": 0.30}


def _source_targets(correction_fraction: float) -> dict[str, float]:
    correction = float(correction_fraction)
    if not 0.0 < correction < 1.0:
        raise ValueError("correction fraction must be strictly between zero and one")
    return {"original": 1.0 - correction, "correction": correction}


def _largest_remainder(total: int, weights: dict[Any, float]) -> dict[Any, int]:
    if total < 1:
        raise ValueError("quota total must be positive")
    if not weights or any(value <= 0 for value in weights.values()):
        raise ValueError("all quota weights must be positive")
    normalized = {key: value / sum(weights.values()) for key, value in weights.items()}
    scaled = {key: total * value for key, value in normalized.items()}
    result = {key: int(np.floor(value)) for key, value in scaled.items()}
    remaining = total - sum(result.values())
    ranked = sorted(
        weights,
        key=lambda key: (scaled[key] - result[key], str(key)),
        reverse=True,
    )
    for key in ranked[:remaining]:
        result[key] += 1
    return result


def _episode_frame_indices(dataset) -> dict[int, np.ndarray]:
    episodes = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    return {
        int(episode): np.flatnonzero(episodes == episode).astype(np.int64)
        for episode in sorted(set(episodes.tolist()))
    }


def _correction_strata(metadata_path: Path) -> dict[int, tuple[str, str]]:
    source = json.loads(metadata_path.read_text(encoding="utf-8"))
    result: dict[int, tuple[str, str]] = {}
    pattern = re.compile(r"episode_(\d+)_seed_\d+\.npz$")
    for row in source["results"]:
        if not row.get("correction_success", False):
            continue
        match = pattern.search(str(row["raw_episode"]).replace("\\", "/"))
        if match is None:
            raise ValueError(f"cannot read correction episode index: {row['raw_episode']}")
        episode = int(match.group(1))
        if episode in result:
            raise ValueError(f"duplicate correction episode metadata: {episode}")
        result[episode] = (str(row["trigger_type"]), str(row["source_outcome"]))
    return result


class SourceBalancedSampler:
    """Deterministic episode-first sampler for a concatenated two-source dataset."""

    def __init__(
        self,
        original_episodes: dict[int, np.ndarray],
        correction_episodes: dict[int, np.ndarray],
        correction_strata: dict[int, tuple[str, str]],
        *,
        seed: int,
        start_sample: int = 0,
        source_targets: dict[str, float] | None = None,
    ) -> None:
        if set(correction_episodes) != set(correction_strata):
            raise ValueError("correction metadata does not match dataset episodes")
        self.original_episodes = original_episodes
        self.correction_episodes = correction_episodes
        self.correction_strata = correction_strata
        self.seed = int(seed)
        self.start_sample = int(start_sample)
        self.source_targets = dict(source_targets or SOURCE_TARGETS)
        if set(self.source_targets) != set(SOURCE_NAMES):
            raise ValueError(f"source targets must contain exactly {SOURCE_NAMES}")
        if any(value <= 0 for value in self.source_targets.values()):
            raise ValueError("source target weights must be positive")
        self.original_size = sum(map(len, original_episodes.values()))
        self.correction_size = sum(map(len, correction_episodes.values()))
        self.dataset_size = self.original_size + self.correction_size
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if not self.original_size or not self.correction_size:
            raise ValueError("both sources must contain frames")

    def __len__(self) -> int:
        return self.dataset_size

    @staticmethod
    def _sample_episode_balanced(
        rng: np.random.Generator,
        episodes: dict[int, np.ndarray],
        count: int,
        *,
        offset: int,
    ) -> tuple[np.ndarray, list[int]]:
        quotas = _largest_remainder(count, {episode: 1.0 for episode in episodes})
        indices: list[int] = []
        episode_labels: list[int] = []
        for episode, quota in quotas.items():
            chosen = rng.choice(episodes[episode], size=quota, replace=True)
            indices.extend((chosen + offset).tolist())
            episode_labels.extend([episode] * quota)
        order = rng.permutation(len(indices))
        return np.asarray(indices, dtype=np.int64)[order], [episode_labels[i] for i in order]

    def _sample_corrections(
        self, rng: np.random.Generator, count: int
    ) -> tuple[np.ndarray, list[int], list[tuple[str, str]]]:
        by_stratum: dict[tuple[str, str], list[int]] = {}
        for episode, stratum in self.correction_strata.items():
            by_stratum.setdefault(stratum, []).append(episode)
        stratum_quotas = _largest_remainder(
            count, {stratum: 1.0 for stratum in by_stratum}
        )
        indices: list[int] = []
        episode_labels: list[int] = []
        stratum_labels: list[tuple[str, str]] = []
        for stratum, stratum_quota in stratum_quotas.items():
            episode_quotas = _largest_remainder(
                stratum_quota,
                {episode: 1.0 for episode in by_stratum[stratum]},
            ) if stratum_quota else {episode: 0 for episode in by_stratum[stratum]}
            for episode, quota in episode_quotas.items():
                chosen = rng.choice(
                    self.correction_episodes[episode], size=quota, replace=True
                )
                indices.extend((chosen + self.original_size).tolist())
                episode_labels.extend([episode] * quota)
                stratum_labels.extend([stratum] * quota)
        order = rng.permutation(len(indices))
        return (
            np.asarray(indices, dtype=np.int64)[order],
            [episode_labels[i] for i in order],
            [stratum_labels[i] for i in order],
        )

    def epoch_records(self, epoch: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + int(epoch))
        quotas = _largest_remainder(self.dataset_size, self.source_targets)
        original, original_eps = self._sample_episode_balanced(
            rng, self.original_episodes, quotas["original"], offset=0
        )
        correction, correction_eps, correction_groups = self._sample_corrections(
            rng, quotas["correction"]
        )
        indices = np.concatenate([original, correction])
        sources = np.asarray(
            ["original"] * len(original) + ["correction"] * len(correction)
        )
        episodes = np.asarray(original_eps + correction_eps, dtype=np.int64)
        groups: list[tuple[str, str] | None] = (
            [None] * len(original) + correction_groups
        )
        order = rng.permutation(len(indices))
        return {
            "indices": indices[order],
            "sources": sources[order],
            "episodes": episodes[order],
            "correction_strata": [groups[i] for i in order],
        }

    def stream_records(self, start_sample: int, count: int) -> dict[str, Any]:
        if start_sample < 0 or count < 0:
            raise ValueError("stream start/count must be non-negative")
        records = {"indices": [], "sources": [], "episodes": [], "correction_strata": []}
        cursor = int(start_sample)
        remaining = int(count)
        while remaining:
            epoch = cursor // self.dataset_size
            offset = cursor % self.dataset_size
            epoch_data = self.epoch_records(epoch)
            take = min(remaining, self.dataset_size - offset)
            for key in records:
                records[key].extend(epoch_data[key][offset:offset + take])
            cursor += take
            remaining -= take
        records["indices"] = np.asarray(records["indices"], dtype=np.int64)
        records["sources"] = np.asarray(records["sources"])
        records["episodes"] = np.asarray(records["episodes"], dtype=np.int64)
        return records

    def __iter__(self) -> Iterator[int]:
        records = self.stream_records(self.start_sample, self.dataset_size)
        self.start_sample += self.dataset_size
        return iter(records["indices"].tolist())


def _weighted_stats(
    original: dict, correction: dict, source_targets: dict[str, float]
) -> dict:
    """Aggregate source statistics using the configured source weights."""
    from lerobot.datasets.compute_stats import aggregate_stats

    inputs = [copy.deepcopy(original), copy.deepcopy(correction)]
    weights = tuple(source_targets[name] for name in SOURCE_NAMES)
    for stats, weight in zip(inputs, weights, strict=True):
        for feature in stats.values():
            for key, value in feature.items():
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                feature[key] = np.asarray(value)
            feature["count"] = np.asarray([weight * 10000.0], dtype=np.float64)
    return aggregate_stats(inputs)


class SourceBalancedDataset:
    def __init__(self, original, correction, source_targets: dict[str, float]) -> None:
        self.original = original
        self.correction = correction
        self.meta = copy.deepcopy(original.meta)
        self.meta.stats = _weighted_stats(
            original.meta.stats, correction.meta.stats, source_targets
        )

    @property
    def num_frames(self) -> int:
        return self.original.num_frames + self.correction.num_frames

    @property
    def num_episodes(self) -> int:
        return self.original.num_episodes + self.correction.num_episodes

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self.original.num_frames:
            return self.original[index]
        return self.correction[index - self.original.num_frames]


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items()))


def _report(records: dict[str, Any], sampler: SourceBalancedSampler, cfg, step: int) -> dict:
    source_counts = _counts(records["sources"])
    total = sum(source_counts.values())
    original_eps = records["episodes"][records["sources"] == "original"]
    correction_eps = records["episodes"][records["sources"] == "correction"]
    correction_groups = [
        value for value in records["correction_strata"] if value is not None
    ]
    group_counts = _counts([f"{a}|{b}" for a, b in correction_groups])
    return {
        "schema_version": 1,
        "strategy": "deterministic source-balanced, episode-first sampling",
        "seed": int(cfg.seed if cfg.seed is not None else 0),
        "target_source_fractions": sampler.source_targets,
        "dataset": {
            "original_frames": sampler.original_size,
            "original_episodes": len(sampler.original_episodes),
            "correction_frames": sampler.correction_size,
            "correction_episodes": len(sampler.correction_episodes),
        },
        "training_steps": int(cfg.steps),
        "restored_step": int(step),
        "batch_size": int(cfg.batch_size),
        "samples": total,
        "actual_source_counts": source_counts,
        "actual_source_fractions": {
            key: value / total for key, value in source_counts.items()
        },
        "actual_original_episode_counts": _counts(original_eps),
        "actual_correction_episode_counts": _counts(correction_eps),
        "actual_correction_stratum_counts": group_counts,
        "normalization_stats": (
            f"{100 * sampler.source_targets['original']:.0f}/"
            f"{100 * sampler.source_targets['correction']:.0f} weighted source moments; "
            "ImageNet camera stats retained"
        ),
    }


def _parse_wrapper_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--correction-repo-id", required=True)
    parser.add_argument("--correction-metadata", type=Path, required=True)
    parser.add_argument("--source-sampling-report", type=Path, required=True)
    parser.add_argument("--correction-fraction", type=float, default=0.30)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    wrapper = _parse_wrapper_arguments()
    source_targets = _source_targets(wrapper.correction_fraction)
    project_root = Path(__file__).resolve().parents[1]

    import torch
    import lerobot.common.train_utils as train_utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.scripts.lerobot_train as train_module

    original_update = train_utils.update_last_checkpoint

    def update_last_checkpoint(checkpoint_dir: Path) -> None:
        try:
            original_update(checkpoint_dir)
        except OSError as error:
            if os.name != "nt" or getattr(error, "winerror", None) != 1314:
                raise
            pointer = checkpoint_dir.parent / "last_checkpoint.txt"
            pointer.write_text(checkpoint_dir.name + "\n", encoding="utf-8")
            logging.warning("Windows symlink unavailable; wrote %s", pointer)

    train_utils.update_last_checkpoint = update_last_checkpoint
    train_module.update_last_checkpoint = update_last_checkpoint
    original_make_datasets = train_module.make_train_eval_datasets
    original_make_dataloaders = train_module.make_dataloaders

    def make_source_datasets(cfg):
        original, eval_dataset = original_make_datasets(cfg)
        if eval_dataset is not None:
            raise ValueError("source-balanced experiment requires dataset.eval_split=0")
        correction_root = wrapper.correction_root
        if not correction_root.is_absolute():
            correction_root = project_root / correction_root
        correction = LeRobotDataset(
            wrapper.correction_repo_id,
            root=correction_root,
            delta_timestamps=original.delta_timestamps,
            video_backend=cfg.dataset.video_backend,
            return_uint8=True,
            tolerance_s=cfg.tolerance_s,
        )
        # Match make_dataset(use_imagenet_stats=True) for the second source.
        if cfg.dataset.use_imagenet_stats:
            from lerobot.datasets.factory import IMAGENET_STATS
            for key in correction.meta.camera_keys:
                if key in correction.meta.depth_keys:
                    continue
                correction.meta.stats.setdefault(key, {})
                for stat, value in IMAGENET_STATS.items():
                    correction.meta.stats[key][stat] = torch.tensor(value, dtype=torch.float32)
        if original.meta.features != correction.meta.features:
            raise ValueError("original and correction dataset schemas differ")
        return SourceBalancedDataset(original, correction, source_targets), None

    def make_source_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
        original_train, eval_loader = original_make_dataloaders(
            cfg, dataset.original, eval_dataset, step, parallel_dims
        )
        metadata = wrapper.correction_metadata
        if not metadata.is_absolute():
            metadata = project_root / metadata
        correction_root = wrapper.correction_root
        if not correction_root.is_absolute():
            correction_root = project_root / correction_root
        sampler = SourceBalancedSampler(
            _episode_frame_indices(dataset.original),
            _episode_frame_indices(dataset.correction),
            _correction_strata(metadata),
            seed=cfg.seed if cfg.seed is not None else 0,
            start_sample=int(step) * int(cfg.batch_size),
            source_targets=source_targets,
        )
        requested = (int(cfg.steps) - int(step)) * int(cfg.batch_size)
        records = sampler.stream_records(int(step) * int(cfg.batch_size), requested)
        report = _report(records, sampler, cfg, step)
        report["sources"] = {
            "original_root": Path(cfg.dataset.root).resolve().as_posix(),
            "original_repo_id": cfg.dataset.repo_id,
            "correction_root": correction_root.resolve().as_posix(),
            "correction_repo_id": wrapper.correction_repo_id,
            "correction_metadata": metadata.resolve().as_posix(),
        }
        report_path = wrapper.source_sampling_report
        if not report_path.is_absolute():
            report_path = project_root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logging.info("Actual source fractions: %s", report["actual_source_fractions"])
        train_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            pin_memory=parallel_dims.device_type == "cuda",
            drop_last=False,
            collate_fn=original_train.collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            multiprocessing_context=(
                cfg.dataloader_multiprocessing_context if cfg.num_workers > 0 else None
            ),
        )
        return train_loader, eval_loader

    train_module.make_train_eval_datasets = make_source_datasets
    train_module.make_dataloaders = make_source_dataloaders
    train_module.main()


if __name__ == "__main__":
    main()
