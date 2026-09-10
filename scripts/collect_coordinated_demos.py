"""Batch-record scripted OpenArm + Wuji trajectories without changing control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openarm_wuji.dataset import (
    CoordinatedDemoRecorder,
    export_successful_episodes,
)
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_seeds(value: str | None, *, seed_start: int,
                attempts: int) -> list[int]:
    if value:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not result:
            raise ValueError("--seeds did not contain a seed")
        return result
    if attempts < 1:
        raise ValueError("--attempts must be positive")
    return list(range(seed_start, seed_start + attempts))


def make_robot(args, config: dict) -> MujocoOpenArmWuji:
    return MujocoOpenArmWuji(
        args.model,
        args.synergies,
        arm_side=config["arm_side"],
        control_hz=args.control_hz,
        image_height=args.image_height,
        image_width=args.image_width,
        front_camera=config["scene"]["front_camera_name"],
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb",
    )
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/reach_grasp_lift.json",
    )
    parser.add_argument(
        "--synergies", type=Path,
        default=root / "configs/wuji_hand_left_synergies.json",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "outputs/coordinated_demos"
    )
    parser.add_argument("--seeds", help="comma-separated explicit seeds")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--max-successes", type=int)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    seeds = parse_seeds(
        args.seeds, seed_start=args.seed_start, attempts=args.attempts
    )
    if args.max_successes is not None and args.max_successes < 1:
        raise ValueError("--max-successes must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    required_hold_s = float(config["post_settle"]["window_s"])
    successful_dir = args.output / "raw/successful"
    diagnostics_dir = args.output / "raw/diagnostics"
    successful_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    robot = make_robot(args, config)
    robot.connect()
    episode_rows = []
    try:
        for attempt_index, seed in enumerate(seeds):
            recorder = CoordinatedDemoRecorder(
                task_name="reach_grasp_lift",
                task_description=(
                    "Coordinate OpenArm and Wuji to grasp the cube, lift it, "
                    "and hold it unsupported."
                ),
                control_hz=args.control_hz,
                episode_index=attempt_index,
                required_hold_s=required_hold_s,
            )
            task = ReachGraspLiftTask(robot, config, recorder=recorder)
            result = task.run_lift(seed)
            recorder.finish(result.to_dict())
            category = "successful" if recorder.is_successful_demonstration else "diagnostics"
            has_trajectory = bool(recorder.transitions)
            suffix = ".npz" if has_trajectory else ".json"
            destination = (successful_dir if recorder.is_successful_demonstration
                           else diagnostics_dir) / (
                f"episode_{attempt_index:06d}_seed_{seed:06d}{suffix}"
            )
            if destination.exists() and not args.overwrite:
                raise FileExistsError(
                    f"refusing to overwrite {destination}; use --overwrite"
                )
            if has_trajectory:
                recorder.save(destination)
                validation = CoordinatedDemoRecorder.validate(destination)
            else:
                diagnostic = {
                    "summary": recorder.demo_summary,
                    "outcome": recorder.outcome,
                    "note": "episode failed before the first control transition",
                }
                destination.write_text(
                    json.dumps(diagnostic, indent=2), encoding="utf-8"
                )
                validation = {"samples": 0, "metadata_only": True}
            row = {
                **recorder.demo_summary,
                "category": category,
                "file": destination.relative_to(args.output).as_posix(),
                "validation": {
                    key: value for key, value in validation.items()
                    if key != "metadata"
                },
            }
            episode_rows.append(row)
            print(json.dumps({
                "attempt": attempt_index + 1,
                "seed": seed,
                "demonstration_success": recorder.is_successful_demonstration,
                "outcome": result.outcome,
                "samples": validation["samples"],
                "saved": str(destination),
            }), flush=True)
            if (args.max_successes is not None
                    and sum(item["demonstration_success"] for item in episode_rows)
                    >= args.max_successes):
                break
    finally:
        robot.disconnect()

    successful_paths = [
        args.output / row["file"] for row in episode_rows
        if row["demonstration_success"]
    ]
    export_manifest = None
    if successful_paths and not args.skip_export:
        export_manifest = export_successful_episodes(
            successful_paths, args.output / "lerobot_staging"
        )
    summary = {
        "format": CoordinatedDemoRecorder.FORMAT_NAME,
        "schema_version": CoordinatedDemoRecorder.SCHEMA_VERSION,
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "control_hz": args.control_hz,
        "state_dim": 27,
        "action_dim": 27,
        "action_source": (
            "MuJoCo data.ctrl position-actuator targets after arm clipping "
            "and Wuji synergy mapping"
        ),
        "attempts": len(episode_rows),
        "successful_demonstrations": len(successful_paths),
        "diagnostic_failures": len(episode_rows) - len(successful_paths),
        "successful_files": [
            path.relative_to(args.output).as_posix() for path in successful_paths
        ],
        "export": export_manifest,
        "episodes": episode_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "attempts": summary["attempts"],
        "successful_demonstrations": summary["successful_demonstrations"],
        "diagnostic_failures": summary["diagnostic_failures"],
        "lerobot_staging": (
            None if export_manifest is None
            else str(args.output / "lerobot_staging")
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
