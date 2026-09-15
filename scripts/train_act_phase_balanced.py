"""Train LeRobot ACT with deterministic phase-balanced frame sampling.

This is a narrow experiment wrapper.  It leaves LeRobot's optimizer and
training loop intact, but replaces the normal episode-aware sampler with an
exact-quota sampler built from the coordinated demo recorder's raw ``phase``
arrays.  Custom arguments are consumed here; all remaining arguments are
forwarded unchanged to ``lerobot_train``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np


PHASE_ORDER = ("Reach", "Approach", "Grasp", "Preload", "Lift", "Hold")
PHASE_TARGETS = {
    "Reach": 0.25,
    "Approach": 0.25,
    "Grasp": 0.20,
    "Preload": 0.10,
    "Lift": 0.10,
    "Hold": 0.10,
}
RAW_PHASE_MAP = {
    "reach": "Reach",
    "approach": "Approach",
    "grasp_close": "Grasp",
    "freeze_synergy": "Grasp",
    "grasp_settle": "Grasp",
    "preload": "Preload",
    "preload_settle": "Preload",
    "lift": "Lift",
    "lift_s_curve": "Lift",
    "hold": "Hold",
}


def _largest_remainder_quotas(total: int) -> dict[str, int]:
    """Turn target fractions into integer quotas that sum exactly to total."""
    if total < 1:
        raise ValueError("sampler length must be positive")
    scaled = {phase: total * PHASE_TARGETS[phase] for phase in PHASE_ORDER}
    quotas = {phase: int(np.floor(scaled[phase])) for phase in PHASE_ORDER}
    remaining = total - sum(quotas.values())
    ranked = sorted(
        PHASE_ORDER,
        key=lambda phase: (scaled[phase] - quotas[phase], -PHASE_ORDER.index(phase)),
        reverse=True,
    )
    for phase in ranked[:remaining]:
        quotas[phase] += 1
    return quotas


def _counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    counter = Counter(str(labels[int(index)]) for index in indices)
    return {phase: int(counter.get(phase, 0)) for phase in PHASE_ORDER}


def _fractions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {
        phase: (counts[phase] / total if total else 0.0)
        for phase in PHASE_ORDER
    }


class PhaseBalancedSampler:
    """Deterministic exact-quota sampling with replacement within each phase."""

    def __init__(
        self,
        labels: np.ndarray,
        *,
        seed: int,
        start_sample: int = 0,
    ) -> None:
        self.labels = np.asarray(labels)
        self.seed = int(seed)
        self.start_sample = int(start_sample)
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        self.quotas = _largest_remainder_quotas(len(self.labels))
        self.phase_indices = {
            phase: np.flatnonzero(self.labels == phase).astype(np.int64)
            for phase in PHASE_ORDER
        }
        empty = [phase for phase, indices in self.phase_indices.items() if len(indices) == 0]
        if empty:
            raise ValueError(f"cannot balance phases with no frames: {empty}")

    def __len__(self) -> int:
        return len(self.labels)

    def indices_for_epoch(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + int(epoch))
        pieces = [
            rng.choice(self.phase_indices[phase], size=self.quotas[phase], replace=True)
            for phase in PHASE_ORDER
        ]
        indices = np.concatenate(pieces).astype(np.int64)
        rng.shuffle(indices)
        return indices

    def indices_for_stream(self, start_sample: int, count: int) -> np.ndarray:
        """Return a contiguous slice of the deterministic multi-epoch stream."""
        if start_sample < 0 or count < 0:
            raise ValueError("stream start/count must be non-negative")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        dataset_size = len(self.labels)
        cursor = int(start_sample)
        remaining = int(count)
        pieces: list[np.ndarray] = []
        while remaining:
            epoch = cursor // dataset_size
            offset = cursor % dataset_size
            epoch_indices = self.indices_for_epoch(epoch)
            take = min(remaining, dataset_size - offset)
            pieces.append(epoch_indices[offset:offset + take])
            cursor += take
            remaining -= take
        return np.concatenate(pieces).astype(np.int64)

    def __iter__(self) -> Iterator[int]:
        indices = self.indices_for_stream(self.start_sample, len(self.labels))
        self.start_sample += len(self.labels)
        return iter(indices.tolist())


def _load_phase_labels(
    manifest_path: Path,
    dataset,
    *,
    project_root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels: list[str] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    raw_counts: Counter[str] = Counter()

    rows = sorted(
        manifest["source_episodes"],
        key=lambda row: int(row["native_episode_index"]),
    )
    for expected_episode, row in enumerate(rows):
        episode_index = int(row["native_episode_index"])
        if episode_index != expected_episode:
            raise ValueError("manifest native episodes must be contiguous and ordered")
        source = Path(row["source"])
        if not source.is_absolute():
            source = project_root / source
        with np.load(source, allow_pickle=False) as episode:
            raw_phases = [str(value) for value in episode["phase"]]
        if len(raw_phases) != int(row["frames"]):
            raise ValueError(f"manifest phase length mismatch for {source}")
        unknown = sorted(set(raw_phases) - set(RAW_PHASE_MAP))
        if unknown:
            raise ValueError(f"unmapped phases in {source}: {unknown}")
        labels.extend(RAW_PHASE_MAP[value] for value in raw_phases)
        raw_counts.update(raw_phases)
        episode_indices.extend([episode_index] * len(raw_phases))
        frame_indices.extend(range(len(raw_phases)))

    global_labels = np.asarray(labels)
    native_episode = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    native_frame = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
    native_global = np.asarray(dataset.hf_dataset["index"], dtype=np.int64)
    expected_episode = np.asarray(episode_indices, dtype=np.int64)
    expected_frame = np.asarray(frame_indices, dtype=np.int64)
    if not np.array_equal(native_global, np.arange(len(global_labels))):
        raise ValueError("phase-balanced training requires the full contiguous dataset")
    if not np.array_equal(native_episode, expected_episode):
        raise ValueError("native episode order differs from the raw phase manifest")
    if not np.array_equal(native_frame, expected_frame):
        raise ValueError("native frame order differs from the raw phase manifest")
    if len(dataset) != len(global_labels):
        raise ValueError("native dataset and phase manifest have different lengths")
    return global_labels, {
        "manifest": manifest_path.resolve().as_posix(),
        "raw_phase_counts": dict(sorted(raw_counts.items())),
    }


def _parse_wrapper_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase-manifest", type=Path, required=True)
    parser.add_argument("--phase-sampling-report", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    wrapper_args = _parse_wrapper_arguments()
    project_root = Path(__file__).resolve().parents[1]

    import torch
    import lerobot.common.train_utils as train_utils
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
            logging.warning(
                "Windows symlink privilege is unavailable; wrote %s instead",
                pointer,
            )

    train_utils.update_last_checkpoint = update_last_checkpoint
    train_module.update_last_checkpoint = update_last_checkpoint
    original_make_dataloaders = train_module.make_dataloaders

    def make_phase_balanced_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
        if bool(cfg.resume) != (step > 0):
            raise ValueError("resume flag and restored training step disagree")
        original_train, eval_loader = original_make_dataloaders(
            cfg, dataset, eval_dataset, step, parallel_dims
        )
        manifest_path = wrapper_args.phase_manifest
        if not manifest_path.is_absolute():
            manifest_path = project_root / manifest_path
        labels, metadata = _load_phase_labels(
            manifest_path,
            dataset,
            project_root=project_root,
        )
        world_size = int(getattr(parallel_dims, "dp_world_size", 1))
        samples_per_step = int(cfg.batch_size) * world_size
        start_sample = int(step) * samples_per_step
        sampler = PhaseBalancedSampler(
            labels,
            seed=cfg.seed if cfg.seed is not None else 0,
            start_sample=start_sample,
        )
        epoch_zero = sampler.indices_for_epoch(0)
        requested_samples = (int(cfg.steps) - int(step)) * samples_per_step
        if requested_samples < 1:
            raise ValueError("training target step must be greater than restored step")
        consumed = sampler.indices_for_stream(start_sample, requested_samples)
        cumulative = sampler.indices_for_stream(0, int(cfg.steps) * samples_per_step)
        raw_canonical_counts = _counts(labels, np.arange(len(labels), dtype=np.int64))
        epoch_counts = _counts(labels, epoch_zero)
        consumed_counts = _counts(labels, consumed)
        report = {
            "schema_version": 1,
            "strategy": "deterministic exact-quota phase sampling with replacement",
            "seed": int(cfg.seed if cfg.seed is not None else 0),
            "dataset_frames": int(len(labels)),
            "training_steps": int(cfg.steps),
            "restored_step": int(step),
            "batch_size": int(cfg.batch_size),
            "dp_world_size": world_size,
            "stream_start_sample": start_sample,
            "training_samples_consumed": requested_samples,
            "phase_mapping": RAW_PHASE_MAP,
            "target_fractions": PHASE_TARGETS,
            "raw_canonical": {
                "counts": raw_canonical_counts,
                "fractions": _fractions(raw_canonical_counts),
            },
            "planned_epoch_zero": {
                "counts": epoch_counts,
                "fractions": _fractions(epoch_counts),
            },
            "actual_training_prefix": {
                "counts": consumed_counts,
                "fractions": _fractions(consumed_counts),
                "note": (
                    "Exact deterministic stream segment consumed by this run, "
                    "starting after restored_step*batch_size samples."
                ),
            },
            "cumulative_through_target_step": {
                "counts": _counts(labels, cumulative),
                "fractions": _fractions(_counts(labels, cumulative)),
            },
            **metadata,
        }
        report_path = wrapper_args.phase_sampling_report
        if not report_path.is_absolute():
            report_path = project_root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logging.info(
            "Phase-balanced actual training-prefix fractions: %s",
            report["actual_training_prefix"]["fractions"],
        )

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

    train_module.make_dataloaders = make_phase_balanced_dataloaders
    train_module.main()


if __name__ == "__main__":
    main()
