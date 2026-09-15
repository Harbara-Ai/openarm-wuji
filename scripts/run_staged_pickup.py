"""Run the frozen OpenArm + Wuji staged ACT pickup pipeline.

The single source of truth is ``configs/staged_pipeline.json``.  This runner
does not train policies, use expert actions, perform a retry, or execute the
experimental micro-lift probe.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from openarm_wuji.policy import ApproachPolicy, ReachPolicy, RecoveryPolicy
from openarm_wuji.policy.grasp_secure_controller import GraspSecurePolicy
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks.staged_pickup import (
    load_pipeline_manifest,
    missing_pipeline_paths,
    phase_outcomes,
    resolve_pipeline_paths,
    select_representative_rollouts,
    summarize_staged_pickup,
    verify_artifact_integrity,
)
from scripts.evaluate_grasp_secure_act import _set_grasp_deterministic
from scripts.evaluate_staged_with_recovery import _set_deterministic
from scripts.evaluate_staged_with_scripted_lift import run_one


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return value.resolve().as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2), encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_plain(value), separators=(",", ":")) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class RolloutCapture:
    """Collect a compact front+wrist GIF and an explicit phase timeline."""

    COLORS = {
        "REACH": (42, 111, 219),
        "APPROACH": (31, 161, 136),
        "RECOVERY": (237, 162, 52),
        "GRASP_SECURE": (145, 91, 181),
        "LIFT": (224, 91, 74),
        "HOLD": (93, 166, 57),
        "SUCCESS": (36, 150, 80),
        "FAILURE": (190, 55, 55),
    }

    def __init__(self, *, seed: int, frame_stride: int = 2):
        self.seed = seed
        self.frame_stride = max(1, int(frame_stride))
        self.frames: list[np.ndarray] = []
        self.phase_sequence: list[str] = []
        self._frame = 0
        self._last_image: np.ndarray | None = None

    def __call__(self, phase: str, observation: dict[str, Any], _telemetry) -> None:
        front = np.asarray(observation["front_rgb"], dtype=np.uint8)
        wrist = np.asarray(observation["wrist_rgb"], dtype=np.uint8)
        combined = np.concatenate([front, wrist], axis=1)
        self.phase_sequence.append(phase)
        self._last_image = combined
        if self._frame % self.frame_stride == 0:
            self.frames.append(self._annotate(combined, phase, self._frame))
        self._frame += 1

    def _annotate(self, frame: np.ndarray, phase: str, index: int) -> np.ndarray:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        color = self.COLORS.get(phase, (40, 40, 40))
        draw.rectangle((0, 0, image.width, 24), fill=color)
        draw.text((8, 6), f"seed {self.seed} | {phase} | frame {index}",
                  fill=(255, 255, 255))
        return np.asarray(image)

    def add_terminal(self, success: bool) -> None:
        phase = "SUCCESS" if success else "FAILURE"
        self.phase_sequence.append(phase)
        if self._last_image is not None:
            self.frames.extend([
                self._annotate(self._last_image, phase, self._frame)
            ] * 8)

    def save(self, output: Path) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        gif = output / "rollout.gif"
        if self.frames:
            imageio.mimsave(
                gif, self.frames,
                duration=self.frame_stride / 30.0,
                loop=0,
            )
        counts = dict(Counter(self.phase_sequence))
        timeline = output / "phase_timeline.png"
        self._save_timeline(timeline)
        return {
            "gif": gif if self.frames else None,
            "phase_timeline": timeline,
            "raw_phase_frame_counts": counts,
            "captured_gif_frames": len(self.frames),
            "frame_stride": self.frame_stride,
        }

    def _save_timeline(self, path: Path) -> None:
        width, height = 1200, 150
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        total = max(1, len(self.phase_sequence))
        start = 0
        runs: list[tuple[str, int]] = []
        for phase in self.phase_sequence:
            if runs and runs[-1][0] == phase:
                runs[-1] = (phase, runs[-1][1] + 1)
            else:
                runs.append((phase, 1))
        for phase, count in runs:
            end = start + count
            x0 = int(start / total * (width - 40)) + 20
            x1 = int(end / total * (width - 40)) + 20
            draw.rectangle(
                (x0, 45, max(x0 + 1, x1), 95),
                fill=self.COLORS.get(phase, (100, 100, 100)),
            )
            if x1 - x0 >= 55:
                draw.text((x0 + 4, 62), phase, fill="white")
            start = end
        draw.text((20, 15), f"seed {self.seed} staged pickup phase timeline",
                  fill="black")
        draw.text((20, 112), f"{total} control frames at 30 Hz", fill="black")
        image.save(path)


def _load_runtime(manifest_path: Path):
    manifest = load_pipeline_manifest(manifest_path)
    paths = resolve_pipeline_paths(manifest, ROOT)
    missing = missing_pipeline_paths(paths)
    if missing:
        details = "\n".join(f"  - {name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(
            "frozen staged-pipeline artifacts are missing:\n" + details
            + "\nSee README.md#prepare-frozen-artifacts."
        )
    integrity = verify_artifact_integrity(manifest, ROOT)
    bad_integrity = [
        item for item in integrity
        if not item["bytes_match"] or not item["sha256_match"]
    ]
    if bad_integrity:
        raise RuntimeError(
            "frozen staged-pipeline artifact integrity check failed:\n"
            + json.dumps(bad_integrity, indent=2)
        )
    task_config = json.loads(paths["task_config"].read_text(encoding="utf-8"))
    grasp_config = json.loads(paths["grasp_config"].read_text(encoding="utf-8"))
    router = json.loads(paths["router"].read_text(encoding="utf-8"))
    reach = ReachPolicy(paths["reach"])
    approach = ApproachPolicy(
        paths["approach"],
        max_cube_displacement_m=float(
            task_config["grasp"]["max_approach_cube_displacement_m"]
        ),
    )
    recovery = RecoveryPolicy(
        paths["recovery"],
        max_cube_displacement_m=float(
            task_config["grasp"]["max_approach_cube_displacement_m"]
        ),
    )
    grasp = GraspSecurePolicy(paths["grasp"])
    inference_seed = int(manifest["inference_seed"])
    for policy in (reach, approach, recovery):
        _set_deterministic(policy, inference_seed)
    _set_grasp_deterministic(grasp, inference_seed)
    policies = {
        "reach": reach,
        "approach": approach,
        "recovery": recovery,
        "grasp": grasp,
    }
    robot = MujocoOpenArmWuji(
        paths["model"], paths["synergies"],
        arm_side=task_config["arm_side"],
        control_hz=int(manifest["control_hz"]),
        image_height=240,
        image_width=320,
        front_camera=task_config["scene"]["front_camera_name"],
    )
    return manifest, paths, task_config, grasp_config, router, policies, robot


def _run(*, seed: int, index: int, output: Path, robot,
         task_config: dict[str, Any], grasp_config: dict[str, Any],
         router: dict[str, Any], policies: dict[str, Any],
         callback=None) -> dict[str, Any]:
    row = run_one(
        seed=seed,
        rollout_index=index,
        group="episodes",
        robot=robot,
        config=task_config,
        reach=policies["reach"],
        approach=policies["approach"],
        recovery=policies["recovery"],
        grasp=policies["grasp"],
        router=router,
        preload_gate_rad=float(
            grasp_config["gate"]["hand_preload_l2_min_rad"]
        ),
        grasp_timeout_frames=int(grasp_config["gate"]["timeout_frames"]),
        output=output,
        grasp_gate_mode="diagnostic_only",
        frame_callback=callback,
    )
    row["phase_outcomes"] = phase_outcomes(row)
    row["formal_pipeline_version"] = "openarm_wuji_staged_act_pickup_v1"
    return row


def _save_representatives(*, rows: list[dict[str, Any]], output: Path,
                          robot, task_config, grasp_config, router, policies,
                          seed_start: int, inference_seed: int
                          ) -> dict[str, Any]:
    selected = select_representative_rollouts(rows)
    result: dict[str, Any] = {}
    for name, source in selected.items():
        seed = int(source["seed"])
        destination = output / "representative_rollouts" / name
        if destination.exists():
            shutil.rmtree(destination)
        # ACT samples a latent at inference.  The formal benchmark seeds that
        # stream once before episode zero, so reproduce all preceding episodes
        # before rendering the selected row.  This makes the visual correspond
        # to the exact benchmark outcome rather than to a new latent sample.
        for policy in (
            policies["reach"], policies["approach"], policies["recovery"]
        ):
            _set_deterministic(policy, inference_seed)
        _set_grasp_deterministic(policies["grasp"], inference_seed)
        warmup_output = destination / "warmup"
        for warmup_index in range(int(source["rollout_index"])):
            _run(
                seed=seed_start + warmup_index,
                index=warmup_index,
                output=warmup_output,
                robot=robot,
                task_config=task_config,
                grasp_config=grasp_config,
                router=router,
                policies=policies,
            )
        capture = RolloutCapture(seed=seed)
        replay = _run(
            seed=seed,
            index=int(source["rollout_index"]),
            output=destination / "raw",
            robot=robot,
            task_config=task_config,
            grasp_config=grasp_config,
            router=router,
            policies=policies,
            callback=capture,
        )
        if warmup_output.exists():
            shutil.rmtree(warmup_output)
        capture.add_terminal(bool(replay["full_task_success"]))
        artifacts = capture.save(destination)
        matched = bool(
            replay["full_task_success"] == source["full_task_success"]
            and replay["failure_stage"] == source["failure_stage"]
            and replay["failure_reason"] == source["failure_reason"]
        )
        record = {
            "category": name,
            "seed": seed,
            "benchmark_rollout_index": source["rollout_index"],
            "benchmark_outcome": {
                "success": source["full_task_success"],
                "failure_stage": source["failure_stage"],
                "failure_reason": source["failure_reason"],
            },
            "replay_outcome": {
                "success": replay["full_task_success"],
                "failure_stage": replay["failure_stage"],
                "failure_reason": replay["failure_reason"],
            },
            "deterministic_outcome_match": matched,
            "artifacts": artifacts,
            "phase_outcomes": replay["phase_outcomes"],
        }
        _write_json(destination / "summary.json", record)
        result[name] = record
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-config", type=Path,
        default=ROOT / "configs/staged_pipeline.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/staged_pickup_formal",
    )
    parser.add_argument("--compact-summary", type=Path)
    parser.add_argument("--save-representatives", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate the manifest and print the resolved frozen artifacts",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be positive")
    manifest = load_pipeline_manifest(args.pipeline_config)
    if args.check_only:
        paths = resolve_pipeline_paths(manifest, ROOT)
        missing = missing_pipeline_paths(paths)
        integrity = verify_artifact_integrity(manifest, ROOT) if not missing else []
        report = {
            "pipeline": manifest["name"],
            "manifest": args.pipeline_config.resolve(),
            "artifacts": {
                name: {
                    "path": path,
                    "exists": path.exists(),
                }
                for name, path in paths.items()
            },
            "ready": not missing,
            "missing": missing,
            "integrity": integrity,
        }
        print(json.dumps(_plain(report), indent=2))
        if missing or any(
            not item["bytes_match"] or not item["sha256_match"]
            for item in integrity
        ):
            raise SystemExit(2)
        return
    if args.seed is not None and args.seed_start is not None:
        raise ValueError("use either --seed or --seed-start, not both")
    seed_start = (
        args.seed if args.seed is not None
        else args.seed_start if args.seed_start is not None
        else int(manifest["benchmark"]["seed_start"])
    )
    if args.seed is not None and args.num_runs != 1:
        raise ValueError("--seed runs exactly one rollout")
    output = args.output.resolve()
    if args.overwrite and output.exists():
        shutil.rmtree(output)
    if output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"output exists: {output}; use --resume/--overwrite")
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.jsonl"
    rows = _load_jsonl(progress) if args.resume else []
    if len(rows) > args.num_runs:
        raise ValueError("resume output contains more runs than requested")

    (manifest, paths, task_config, grasp_config,
     router, policies, robot) = _load_runtime(args.pipeline_config)
    robot.connect()
    try:
        for index in range(len(rows), args.num_runs):
            seed = seed_start + index
            row = _run(
                seed=seed,
                index=index,
                output=output,
                robot=robot,
                task_config=task_config,
                grasp_config=grasp_config,
                router=router,
                policies=policies,
            )
            rows.append(row)
            _append_jsonl(progress, row)
            compact = summarize_staged_pickup(
                rows,
                seed_start=seed_start,
                requested_runs=args.num_runs,
                pipeline_name=manifest["name"],
            )
            _write_json(output / "summary.partial.json", compact)
            print(
                f"{index + 1}/{args.num_runs} seed={seed} "
                f"reach={int(row['reach_success'])} "
                f"approach={int(row['approach_stage_success'])} "
                f"grasp_gate={int(row.get('graspsecure_gate_pass', False))} "
                f"lift={int(row['lift_success'])} "
                f"full={int(row['full_task_success'])} "
                f"failure={row['failure_stage']}:{row['failure_reason']}",
                flush=True,
            )
        representatives = None
        if args.save_representatives:
            representatives = _save_representatives(
                rows=rows,
                output=output,
                robot=robot,
                task_config=task_config,
                grasp_config=grasp_config,
                router=router,
                policies=policies,
                seed_start=seed_start,
                inference_seed=int(manifest["inference_seed"]),
            )
    finally:
        robot.disconnect()

    compact = summarize_staged_pickup(
        rows,
        seed_start=seed_start,
        requested_runs=args.num_runs,
        pipeline_name=manifest["name"],
    )
    compact["representative_rollouts"] = representatives
    _write_json(output / "summary.json", {**compact, "episodes": rows})
    _write_json(output / "summary.compact.json", compact)
    _write_json(output / "pipeline_manifest.snapshot.json", manifest)
    if args.compact_summary is not None:
        _write_json(args.compact_summary.resolve(), compact)
    print(json.dumps(_plain(compact), indent=2))


if __name__ == "__main__":
    main()
