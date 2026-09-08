from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from openarm_wuji.dataset import CausalEpisodeRecorder
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask
from openarm_wuji.tasks.se3 import pose_drift


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _lift_transient_frames(config: dict, control_hz: float) -> int:
    baseline = config["external_baseline"]
    distance = float(np.linalg.norm(config["lift"]["grasp_center_delta_m"]))
    first_distance = min(float(baseline["gripper_lift_m"]), distance)
    first_duration = float(baseline["lift_duration_s"])
    duration = first_duration
    if distance > first_distance:
        duration += (distance - first_distance) / (first_distance / first_duration)
    return int(round(duration * control_hz))


def _analysis_segment(phase: str, lift_index: int,
                      lift_transient_frames: int) -> str:
    if phase == "grasp_close":
        return "close"
    if phase in ("preload", "preload_settle"):
        return "preload"
    if phase == "lift_s_curve":
        return (
            "lift_transient" if lift_index <= lift_transient_frames else "lift_hold"
        )
    return phase


def _time_series(recorder: CausalEpisodeRecorder, config: dict) -> list[dict]:
    lift_start = next((
        transition["telemetry_t"] for transition in recorder.transitions
        if transition["phase"] == "lift_s_curve"
    ), None)
    lift_index = 0
    transient_frames = _lift_transient_frames(config, recorder.control_hz)
    rows = []
    for frame, transition in enumerate(recorder.transitions, start=1):
        phase = transition["phase"]
        if phase == "lift_s_curve":
            lift_index += 1
        telemetry = transition["telemetry_tp1"]
        if lift_start is None or phase != "lift_s_curve":
            translation_drift = None
            rotation_drift = None
        else:
            translation_drift, rotation_drift = pose_drift(
                lift_start["object_relative_position_m"],
                lift_start["object_relative_quaternion_wxyz"],
                telemetry["object_relative_position_m"],
                telemetry["object_relative_quaternion_wxyz"],
            )
        rows.append(_plain({
            "frame": frame,
            "sim_time_s": transition["observation_tp1"]["sim_time"],
            "controller_phase": phase,
            "analysis_segment": _analysis_segment(
                phase, lift_index, transient_frames
            ),
            "cube_position_world_m": telemetry["cube_position_m"],
            "cube_quaternion_world_wxyz": telemetry["cube_quaternion_wxyz"],
            "palm_position_world_m": telemetry["grasp_center_position_m"],
            "palm_quaternion_world_wxyz": (
                telemetry["grasp_center_quaternion_wxyz"]
            ),
            "commanded_palm_quaternion_wxyz": (
                telemetry["commanded_palm_quaternion_wxyz"]
            ),
            "ik_command_palm_quaternion_wxyz": (
                telemetry["ik_command_palm_quaternion_wxyz"]
            ),
            "palm_orientation_error_deg": telemetry["palm_orientation_error_deg"],
            "object_relative_position_palm_m": (
                telemetry["object_relative_position_m"]
            ),
            "object_relative_quaternion_palm_wxyz": (
                telemetry["object_relative_quaternion_wxyz"]
            ),
            "relative_translation_drift_from_lift_start_m": translation_drift,
            "relative_rotation_drift_from_lift_start_deg": rotation_drift,
            "cube_linear_velocity_world_m_s": (
                telemetry["cube_linear_velocity_world_m_s"]
            ),
            "cube_angular_velocity_world_rad_s": (
                telemetry["cube_angular_velocity_world_rad_s"]
            ),
            "palm_linear_velocity_world_m_s": (
                telemetry["grasp_center_linear_velocity_world_m_s"]
            ),
            "palm_angular_velocity_world_rad_s": (
                telemetry["grasp_center_angular_velocity_world_rad_s"]
            ),
            "contact_geometry": telemetry["contact_geometry"],
            "net_force_world_n": telemetry["contact_resultant_force_world_n"],
            "net_moment_about_cube_world_nm": (
                telemetry["contact_resultant_moment_about_cube_world_nm"]
            ),
        }))
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = (
        "frame", "sim_time_s", "controller_phase", "analysis_segment",
        "palm_orientation_error_deg",
        "relative_translation_drift_from_lift_start_m",
        "relative_rotation_drift_from_lift_start_deg",
        "cube_position_world_m", "cube_quaternion_world_wxyz",
        "palm_position_world_m", "palm_quaternion_world_wxyz",
        "object_relative_position_palm_m",
        "object_relative_quaternion_palm_wxyz",
        "cube_linear_velocity_world_m_s", "cube_angular_velocity_world_rad_s",
        "palm_linear_velocity_world_m_s", "palm_angular_velocity_world_rad_s",
        "net_force_world_n", "net_moment_about_cube_world_nm",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (json.dumps(row[key], separators=(",", ":"))
                      if isinstance(row[key], (dict, list)) else row[key])
                for key in columns
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synergies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    robot = MujocoOpenArmWuji(
        args.model,
        args.synergies,
        arm_side=config["arm_side"],
        control_hz=30,
        image_height=64,
        image_width=64,
        front_camera=config["scene"]["front_camera_name"],
    )
    summaries = []
    robot.connect()
    try:
        for seed in args.seeds:
            recorder = CausalEpisodeRecorder(
                task_name="reach_grasp_lift_diagnostics", control_hz=30
            )
            result = ReachGraspLiftTask(
                robot, config, recorder=recorder
            ).run_lift(seed)
            rows = _time_series(recorder, config)
            json_path = args.output / f"seed_{seed}_timeseries.json"
            csv_path = args.output / f"seed_{seed}_timeseries.csv"
            json_path.write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            _write_csv(csv_path, rows)
            summaries.append({
                "seed": seed,
                "result": result.to_dict(),
                "time_series_json": json_path.name,
                "time_series_csv": csv_path.name,
            })
    finally:
        robot.disconnect()
    report = {
        "controller_constants_held_fixed": {
            "preload_target_synergy": config["grasp"]["preload_target_synergy"],
            "lift_profile": "two_segment_quintic_minimum_jerk",
            "friction_and_randomization": "unchanged",
        },
        "post_settle_gate": config["post_settle"],
        "seeds": summaries,
    }
    report_path = args.output / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "seeds": [{
            "seed": item["seed"],
            "task_success": item["result"]["task_success"],
            "grasp_stable": item["result"]["grasp_stable"],
            "post_settle_stable": item["result"]["post_settle_stable"],
            "outcome": item["result"]["outcome"],
        } for item in summaries],
    }, indent=2))


if __name__ == "__main__":
    main()
