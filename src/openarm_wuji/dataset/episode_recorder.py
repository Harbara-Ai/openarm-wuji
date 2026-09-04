from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from ..tasks.se3 import relative_pose, rotation_geodesic_angle_deg


def _copy_observation(observation: dict) -> dict:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }


def _state(observation: dict) -> np.ndarray:
    arm = np.asarray(observation["arm_joint_position"], dtype=float)
    hand = np.asarray(observation["hand_joint_position"], dtype=float)
    if arm.shape != (7,) or hand.shape != (20,):
        raise ValueError(f"invalid policy state shapes: arm={arm.shape}, hand={hand.shape}")
    state = np.concatenate([arm, hand])
    if not np.isfinite(state).all():
        raise ValueError("policy state contains NaN or infinity")
    return state


def _contact_vector(telemetry: dict) -> np.ndarray:
    forces = telemetry["finger_normal_forces_n"]
    return np.asarray([forces.get(f"finger{index}", 0.0) for index in range(1, 6)])


def _pack_contacts(telemetry_samples: list[dict]) -> dict[str, np.ndarray]:
    offsets = [0]
    fingers: list[str] = []
    positions = []
    normals = []
    forces = []
    torques = []
    moments = []
    normal_forces = []
    for sample in telemetry_samples:
        for contact in sample["contacts"]:
            fingers.append(str(contact["finger"]))
            positions.append(contact["position_world_m"])
            normals.append(contact["normal_on_cube_world"])
            forces.append(contact["force_on_cube_world_n"])
            torques.append(contact["contact_torque_on_cube_world_nm"])
            moments.append(contact["moment_about_cube_center_world_nm"])
            normal_forces.append(contact["normal_force_n"])
        offsets.append(len(fingers))

    def vectors(values) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1, 3)

    return {
        "contact_sample_offsets": np.asarray(offsets, dtype=np.int64),
        "contact_finger": np.asarray(fingers, dtype="U16"),
        "contact_position_world_m": vectors(positions),
        "contact_normal_on_cube_world": vectors(normals),
        "contact_force_on_cube_world_n": vectors(forces),
        "contact_torque_on_cube_world_nm": vectors(torques),
        "contact_moment_about_cube_world_nm": vectors(moments),
        "contact_normal_force_n": np.asarray(normal_forces, dtype=float),
    }


def _contact_slice(episode, sample_index: int) -> slice:
    offsets = episode["contact_sample_offsets"]
    return slice(int(offsets[sample_index]), int(offsets[sample_index + 1]))


class CausalEpisodeRecorder:
    """Record obs_t, bounded action_t, and the exact resulting obs_t+1."""

    SCHEMA_VERSION = 2

    def __init__(self, *, task_name: str, control_hz: float,
                 task_description: str = "", episode_index: int = 0):
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        self.task_name = task_name
        self.task_description = task_description or task_name
        self.episode_index = int(episode_index)
        self.control_hz = float(control_hz)
        self.seed: int | None = None
        self.transitions: list[dict] = []
        self.outcome: dict | None = None

    def start(self, seed: int) -> None:
        self.seed = int(seed)
        self.transitions.clear()
        self.outcome = None

    def record_transition(self, *, phase: str, observation_t: dict,
                          action_t, observation_tp1: dict,
                          telemetry_t: dict, telemetry_tp1: dict) -> None:
        if self.seed is None:
            raise RuntimeError("start(seed) must be called before recording")
        if not phase:
            raise ValueError("phase must be non-empty")
        action = np.asarray(action_t, dtype=float)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("recorded action must be finite and 10-D")
        state_t = _state(observation_t)
        state_tp1 = _state(observation_tp1)
        frame_t = int(observation_t["frame_index"])
        frame_tp1 = int(observation_tp1["frame_index"])
        if frame_tp1 != frame_t + 1:
            raise ValueError(f"non-causal frame transition: {frame_t} -> {frame_tp1}")
        if float(observation_tp1["sim_time"]) <= float(observation_t["sim_time"]):
            raise ValueError("simulation time did not advance")
        for camera in ("front_rgb", "wrist_rgb"):
            image_t = np.asarray(observation_t[camera])
            image_tp1 = np.asarray(observation_tp1[camera])
            if image_t.shape != image_tp1.shape or image_t.dtype != np.uint8:
                raise ValueError(f"invalid {camera} transition")
        if self.transitions:
            previous = self.transitions[-1]
            if frame_t != int(previous["observation_tp1"]["frame_index"]):
                raise ValueError("frame chain is discontinuous")
            if not np.array_equal(state_t, _state(previous["observation_tp1"])):
                raise ValueError("state chain is discontinuous")
            for camera in ("front_rgb", "wrist_rgb"):
                if not np.array_equal(
                    observation_t[camera], previous["observation_tp1"][camera]
                ):
                    raise ValueError(f"{camera} chain is discontinuous")
        self.transitions.append({
            "phase": phase,
            "observation_t": _copy_observation(observation_t),
            "action_t": action.copy(),
            "observation_tp1": _copy_observation(observation_tp1),
            "telemetry_t": telemetry_t.copy(),
            "telemetry_tp1": telemetry_tp1.copy(),
        })

    def finish(self, outcome: dict) -> None:
        if self.seed is None:
            raise RuntimeError("episode was not started")
        if not self.transitions:
            raise RuntimeError("episode contains no transitions")
        self.outcome = json.loads(json.dumps(outcome))

    def save(self, path: str | Path) -> Path:
        if self.outcome is None:
            raise RuntimeError("finish(outcome) must be called before saving")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.transitions
        observations = [row["observation_t"] for row in rows]
        next_observations = [row["observation_tp1"] for row in rows]
        telemetry = [row["telemetry_t"] for row in rows]
        next_telemetry = [row["telemetry_tp1"] for row in rows]
        telemetry_boundaries = [telemetry[0], *next_telemetry]
        final_observation = next_observations[-1]
        arrays = {
            "schema_version": np.asarray(self.SCHEMA_VERSION),
            "task_name": np.asarray(self.task_name),
            "task_description": np.asarray(self.task_description),
            "episode_index": np.asarray(self.episode_index),
            "control_hz": np.asarray(self.control_hz),
            "episode_seed": np.asarray(self.seed),
            "outcome_success": np.asarray(bool(self.outcome.get("success", False))),
            "task_success": np.asarray(bool(self.outcome.get("task_success", False))),
            "grasp_stable": np.asarray(bool(self.outcome.get("grasp_stable", False))),
            "outcome": np.asarray(self.outcome.get("outcome") or ""),
            "failure_reason": np.asarray(self.outcome.get("failure_reason") or ""),
            "outcome_json": np.asarray(json.dumps(self.outcome, separators=(",", ":"))),
            "phase": np.asarray([row["phase"] for row in rows]),
            "observation_state": np.asarray([_state(item) for item in observations]),
            "observation_front_rgb": np.asarray([item["front_rgb"] for item in observations]),
            "observation_wrist_rgb": np.asarray([item["wrist_rgb"] for item in observations]),
            "action": np.asarray([row["action_t"] for row in rows]),
            "next_observation_state": np.asarray([_state(item) for item in next_observations]),
            "frame_index": np.asarray([item["frame_index"] for item in observations]),
            "next_frame_index": np.asarray([item["frame_index"] for item in next_observations]),
            "host_timestamp": np.asarray([item["timestamp"] for item in observations]),
            "next_host_timestamp": np.asarray([item["timestamp"] for item in next_observations]),
            "sim_time": np.asarray([item["sim_time"] for item in observations]),
            "next_sim_time": np.asarray([item["sim_time"] for item in next_observations]),
            "cube_position": np.asarray([item["cube_position_m"] for item in telemetry]),
            "next_cube_position": np.asarray([item["cube_position_m"] for item in next_telemetry]),
            "cube_quaternion_wxyz": np.asarray([
                item["cube_quaternion_wxyz"] for item in telemetry
            ]),
            "next_cube_quaternion_wxyz": np.asarray([
                item["cube_quaternion_wxyz"] for item in next_telemetry
            ]),
            "cube_height": np.asarray([item["cube_height_m"] for item in telemetry]),
            "next_cube_height": np.asarray([item["cube_height_m"] for item in next_telemetry]),
            "finger_normal_forces": np.asarray([_contact_vector(item) for item in telemetry]),
            "next_finger_normal_forces": np.asarray([
                _contact_vector(item) for item in next_telemetry
            ]),
            "grasp_center_position": np.asarray([
                item["grasp_center_position_m"] for item in telemetry
            ]),
            "next_grasp_center_position": np.asarray([
                item["grasp_center_position_m"] for item in next_telemetry
            ]),
            "grasp_center_quaternion_wxyz": np.asarray([
                item["grasp_center_quaternion_wxyz"] for item in telemetry
            ]),
            "next_grasp_center_quaternion_wxyz": np.asarray([
                item["grasp_center_quaternion_wxyz"] for item in next_telemetry
            ]),
            "object_relative_position": np.asarray([
                item["object_relative_position_m"] for item in telemetry
            ]),
            "next_object_relative_position": np.asarray([
                item["object_relative_position_m"] for item in next_telemetry
            ]),
            "object_relative_quaternion_wxyz": np.asarray([
                item["object_relative_quaternion_wxyz"] for item in telemetry
            ]),
            "next_object_relative_quaternion_wxyz": np.asarray([
                item["object_relative_quaternion_wxyz"] for item in next_telemetry
            ]),
            "contact_resultant_force_world_n": np.asarray([
                item["contact_resultant_force_world_n"] for item in telemetry
            ]),
            "next_contact_resultant_force_world_n": np.asarray([
                item["contact_resultant_force_world_n"] for item in next_telemetry
            ]),
            "contact_resultant_moment_about_cube_world_nm": np.asarray([
                item["contact_resultant_moment_about_cube_world_nm"] for item in telemetry
            ]),
            "next_contact_resultant_moment_about_cube_world_nm": np.asarray([
                item["contact_resultant_moment_about_cube_world_nm"]
                for item in next_telemetry
            ]),
            "final_observation_state": _state(final_observation),
            "final_front_rgb": np.asarray(final_observation["front_rgb"]),
            "final_wrist_rgb": np.asarray(final_observation["wrist_rgb"]),
        }
        arrays.update(_pack_contacts(telemetry_boundaries))
        np.savez_compressed(destination, **arrays)
        return destination

    @staticmethod
    def validate(path: str | Path) -> dict:
        with np.load(path, allow_pickle=False) as episode:
            samples = len(episode["action"])
            if samples == 0:
                raise ValueError("episode contains no transitions")
            if int(episode["schema_version"]) != CausalEpisodeRecorder.SCHEMA_VERSION:
                raise ValueError("unsupported episode schema")
            expected_shapes = {
                "observation_state": (samples, 27),
                "next_observation_state": (samples, 27),
                "action": (samples, 10),
                "cube_position": (samples, 3),
                "next_cube_position": (samples, 3),
                "cube_quaternion_wxyz": (samples, 4),
                "next_cube_quaternion_wxyz": (samples, 4),
                "grasp_center_position": (samples, 3),
                "next_grasp_center_position": (samples, 3),
                "grasp_center_quaternion_wxyz": (samples, 4),
                "next_grasp_center_quaternion_wxyz": (samples, 4),
                "object_relative_position": (samples, 3),
                "next_object_relative_position": (samples, 3),
                "object_relative_quaternion_wxyz": (samples, 4),
                "next_object_relative_quaternion_wxyz": (samples, 4),
                "contact_resultant_force_world_n": (samples, 3),
                "next_contact_resultant_force_world_n": (samples, 3),
                "contact_resultant_moment_about_cube_world_nm": (samples, 3),
                "next_contact_resultant_moment_about_cube_world_nm": (samples, 3),
                "finger_normal_forces": (samples, 5),
            }
            for key, shape in expected_shapes.items():
                if episode[key].shape != shape:
                    raise ValueError(f"invalid {key} shape: {episode[key].shape}")
                if not np.isfinite(episode[key]).all():
                    raise ValueError(f"{key} contains NaN or infinity")
            for key in ("observation_front_rgb", "observation_wrist_rgb"):
                if episode[key].shape[0] != samples or episode[key].dtype != np.uint8:
                    raise ValueError(f"invalid {key} array")
            if episode["phase"].shape != (samples,):
                raise ValueError("invalid phase array")
            for key in (
                "cube_quaternion_wxyz",
                "next_cube_quaternion_wxyz",
                "grasp_center_quaternion_wxyz",
                "next_grasp_center_quaternion_wxyz",
                "object_relative_quaternion_wxyz",
                "next_object_relative_quaternion_wxyz",
            ):
                norms = np.linalg.norm(episode[key], axis=1)
                if not np.allclose(norms, 1.0, atol=1e-9):
                    raise ValueError(f"{key} is not unit-normalized")
            offsets = episode["contact_sample_offsets"]
            if offsets.shape != (samples + 2,) or offsets[0] != 0:
                raise ValueError("invalid contact_sample_offsets")
            if np.any(np.diff(offsets) < 0):
                raise ValueError("contact offsets are not monotonic")
            contact_count = int(offsets[-1])
            contact_fields = {
                "contact_finger": (contact_count,),
                "contact_position_world_m": (contact_count, 3),
                "contact_normal_on_cube_world": (contact_count, 3),
                "contact_force_on_cube_world_n": (contact_count, 3),
                "contact_torque_on_cube_world_nm": (contact_count, 3),
                "contact_moment_about_cube_world_nm": (contact_count, 3),
                "contact_normal_force_n": (contact_count,),
            }
            for key, shape in contact_fields.items():
                if episode[key].shape != shape:
                    raise ValueError(f"invalid {key} shape: {episode[key].shape}")
                if key != "contact_finger" and not np.isfinite(episode[key]).all():
                    raise ValueError(f"{key} contains NaN or infinity")
            if contact_count:
                normal_norms = np.linalg.norm(
                    episode["contact_normal_on_cube_world"], axis=1
                )
                if not np.allclose(normal_norms, 1.0, atol=1e-9):
                    raise ValueError("contact normals are not unit vectors")

            for prefix in ("", "next_"):
                for index in range(samples):
                    relative_position, relative_quaternion = relative_pose(
                        episode[f"{prefix}grasp_center_position"][index],
                        episode[f"{prefix}grasp_center_quaternion_wxyz"][index],
                        episode[f"{prefix}cube_position"][index],
                        episode[f"{prefix}cube_quaternion_wxyz"][index],
                    )
                    if not np.allclose(
                        relative_position,
                        episode[f"{prefix}object_relative_position"][index],
                        atol=1e-12,
                    ):
                        raise ValueError("relative translation is inconsistent")
                    if rotation_geodesic_angle_deg(
                        relative_quaternion,
                        episode[f"{prefix}object_relative_quaternion_wxyz"][index],
                    ) > 1e-9:
                        raise ValueError("relative rotation is inconsistent")

            boundary_forces = np.concatenate([
                episode["contact_resultant_force_world_n"][:1],
                episode["next_contact_resultant_force_world_n"],
            ])
            boundary_moments = np.concatenate([
                episode["contact_resultant_moment_about_cube_world_nm"][:1],
                episode["next_contact_resultant_moment_about_cube_world_nm"],
            ])
            for sample_index in range(samples + 1):
                contact_slice = _contact_slice(episode, sample_index)
                force_sum = np.sum(
                    episode["contact_force_on_cube_world_n"][contact_slice], axis=0
                )
                moment_sum = np.sum(
                    episode["contact_moment_about_cube_world_nm"][contact_slice],
                    axis=0,
                )
                if not np.allclose(force_sum, boundary_forces[sample_index], atol=1e-12):
                    raise ValueError("contact resultant force is inconsistent")
                if not np.allclose(
                    moment_sum, boundary_moments[sample_index], atol=1e-12
                ):
                    raise ValueError("contact resultant moment is inconsistent")
            if not np.all(episode["next_frame_index"] == episode["frame_index"] + 1):
                raise ValueError("frame pairs are not consecutive")
            if samples > 1:
                if not np.array_equal(
                    episode["frame_index"][1:], episode["next_frame_index"][:-1]
                ):
                    raise ValueError("episode frame chain is discontinuous")
                if not np.array_equal(
                    episode["observation_state"][1:],
                    episode["next_observation_state"][:-1],
                ):
                    raise ValueError("episode state chain is discontinuous")
                for current_key, next_key in (
                    ("cube_position", "next_cube_position"),
                    ("cube_quaternion_wxyz", "next_cube_quaternion_wxyz"),
                    ("grasp_center_position", "next_grasp_center_position"),
                    (
                        "grasp_center_quaternion_wxyz",
                        "next_grasp_center_quaternion_wxyz",
                    ),
                    ("object_relative_position", "next_object_relative_position"),
                    (
                        "object_relative_quaternion_wxyz",
                        "next_object_relative_quaternion_wxyz",
                    ),
                ):
                    if not np.array_equal(
                        episode[current_key][1:], episode[next_key][:-1]
                    ):
                        raise ValueError(f"{current_key} chain is discontinuous")
            if not np.all(episode["next_sim_time"] > episode["sim_time"]):
                raise ValueError("episode simulation time is not increasing")
            phases, phase_counts = np.unique(episode["phase"], return_counts=True)
            return {
                "samples": samples,
                "duration_s": float(episode["next_sim_time"][-1] - episode["sim_time"][0]),
                "phases": sorted(phases.tolist()),
                "phase_counts": {
                    str(phase): int(count)
                    for phase, count in zip(phases, phase_counts, strict=True)
                },
                "contact_points": contact_count,
            }


def replay_causal_episode(path: str | Path, *, robot, reset_fn: Callable,
                          telemetry_fn: Callable[[], dict]) -> dict[str, float | int]:
    """Replay saved bounded actions after a seeded reset and compare transitions."""
    with np.load(path, allow_pickle=False) as episode:
        seed = int(episode["episode_seed"])
        actions = episode["action"].copy()
        expected_next_state = episode["next_observation_state"].copy()
        expected_next_cube = episode["next_cube_position"].copy()
        expected_next_cube_quaternion = episode[
            "next_cube_quaternion_wxyz"
        ].copy()
        expected_next_grasp_position = episode[
            "next_grasp_center_position"
        ].copy()
        expected_next_grasp_quaternion = episode[
            "next_grasp_center_quaternion_wxyz"
        ].copy()
        expected_next_relative_position = episode[
            "next_object_relative_position"
        ].copy()
        expected_next_relative_quaternion = episode[
            "next_object_relative_quaternion_wxyz"
        ].copy()
        expected_next_resultant_force = episode[
            "next_contact_resultant_force_world_n"
        ].copy()
        expected_next_resultant_moment = episode[
            "next_contact_resultant_moment_about_cube_world_nm"
        ].copy()
        expected_next_sim_time = episode["next_sim_time"].copy()
        expected_front = np.concatenate([
            episode["observation_front_rgb"][1:], episode["final_front_rgb"][None]
        ])
        expected_wrist = np.concatenate([
            episode["observation_wrist_rgb"][1:], episode["final_wrist_rgb"][None]
        ])
        expected_contacts = []
        for sample_index in range(1, len(actions) + 1):
            contact_slice = _contact_slice(episode, sample_index)
            expected_contacts.append({
                "finger": episode["contact_finger"][contact_slice].copy(),
                "position": episode["contact_position_world_m"][contact_slice].copy(),
                "normal": episode["contact_normal_on_cube_world"][contact_slice].copy(),
                "force": episode["contact_force_on_cube_world_n"][contact_slice].copy(),
                "moment": episode[
                    "contact_moment_about_cube_world_nm"
                ][contact_slice].copy(),
            })
    reset_fn(seed)
    max_state_error = 0.0
    max_cube_error = 0.0
    max_cube_rotation_error = 0.0
    max_grasp_position_error = 0.0
    max_grasp_rotation_error = 0.0
    max_relative_position_error = 0.0
    max_relative_rotation_error = 0.0
    max_resultant_force_error = 0.0
    max_resultant_moment_error = 0.0
    max_contact_position_error = 0.0
    max_contact_force_error = 0.0
    max_contact_normal_error = 0.0
    max_contact_moment_error = 0.0
    contact_mismatch_frames = 0
    max_sim_time_error = 0.0
    max_action_error = 0.0
    max_front_pixel_error = 0
    max_wrist_pixel_error = 0
    for index, action in enumerate(actions):
        sent = np.asarray(robot.send_action(action), dtype=float)
        observation = robot.latest_record
        telemetry = telemetry_fn()
        max_action_error = max(max_action_error, float(np.max(np.abs(sent - action))))
        max_state_error = max(max_state_error, float(np.max(np.abs(
            _state(observation) - expected_next_state[index]
        ))))
        max_cube_error = max(max_cube_error, float(np.max(np.abs(
            np.asarray(telemetry["cube_position_m"]) - expected_next_cube[index]
        ))))
        max_cube_rotation_error = max(
            max_cube_rotation_error,
            rotation_geodesic_angle_deg(
                telemetry["cube_quaternion_wxyz"],
                expected_next_cube_quaternion[index],
            ),
        )
        max_grasp_position_error = max(max_grasp_position_error, float(np.max(np.abs(
            np.asarray(telemetry["grasp_center_position_m"])
            - expected_next_grasp_position[index]
        ))))
        max_grasp_rotation_error = max(
            max_grasp_rotation_error,
            rotation_geodesic_angle_deg(
                telemetry["grasp_center_quaternion_wxyz"],
                expected_next_grasp_quaternion[index],
            ),
        )
        max_relative_position_error = max(
            max_relative_position_error,
            float(np.max(np.abs(
                np.asarray(telemetry["object_relative_position_m"])
                - expected_next_relative_position[index]
            ))),
        )
        max_relative_rotation_error = max(
            max_relative_rotation_error,
            rotation_geodesic_angle_deg(
                telemetry["object_relative_quaternion_wxyz"],
                expected_next_relative_quaternion[index],
            ),
        )
        max_resultant_force_error = max(max_resultant_force_error, float(np.max(np.abs(
            np.asarray(telemetry["contact_resultant_force_world_n"])
            - expected_next_resultant_force[index]
        ))))
        max_resultant_moment_error = max(
            max_resultant_moment_error,
            float(np.max(np.abs(
                np.asarray(telemetry["contact_resultant_moment_about_cube_world_nm"])
                - expected_next_resultant_moment[index]
            ))),
        )
        actual_contacts = telemetry["contacts"]
        expected = expected_contacts[index]
        actual_fingers = np.asarray(
            [item["finger"] for item in actual_contacts], dtype="U16"
        )
        if not np.array_equal(actual_fingers, expected["finger"]):
            contact_mismatch_frames += 1
        else:
            for error_name, actual_key, expected_key in (
                ("position", "position_world_m", "position"),
                ("normal", "normal_on_cube_world", "normal"),
                ("force", "force_on_cube_world_n", "force"),
                ("moment", "moment_about_cube_center_world_nm", "moment"),
            ):
                if actual_contacts:
                    error = float(np.max(np.abs(
                        np.asarray([item[actual_key] for item in actual_contacts])
                        - expected[expected_key]
                    )))
                else:
                    error = 0.0
                if error_name == "position":
                    max_contact_position_error = max(max_contact_position_error, error)
                elif error_name == "normal":
                    max_contact_normal_error = max(max_contact_normal_error, error)
                elif error_name == "force":
                    max_contact_force_error = max(max_contact_force_error, error)
                else:
                    max_contact_moment_error = max(max_contact_moment_error, error)
        max_sim_time_error = max(max_sim_time_error, abs(
            float(observation["sim_time"]) - expected_next_sim_time[index]
        ))
        max_front_pixel_error = max(max_front_pixel_error, int(np.max(np.abs(
            observation["front_rgb"].astype(np.int16) - expected_front[index].astype(np.int16)
        ))))
        max_wrist_pixel_error = max(max_wrist_pixel_error, int(np.max(np.abs(
            observation["wrist_rgb"].astype(np.int16) - expected_wrist[index].astype(np.int16)
        ))))
    return {
        "samples": len(actions),
        "max_action_error": max_action_error,
        "max_state_error": max_state_error,
        "max_cube_position_error": max_cube_error,
        "max_cube_rotation_error_deg": max_cube_rotation_error,
        "max_grasp_center_position_error": max_grasp_position_error,
        "max_grasp_center_rotation_error_deg": max_grasp_rotation_error,
        "max_relative_position_error": max_relative_position_error,
        "max_relative_rotation_error_deg": max_relative_rotation_error,
        "max_contact_resultant_force_error_n": max_resultant_force_error,
        "max_contact_resultant_moment_error_nm": max_resultant_moment_error,
        "max_contact_position_error_m": max_contact_position_error,
        "max_contact_normal_error": max_contact_normal_error,
        "max_contact_force_error_n": max_contact_force_error,
        "max_contact_moment_error_nm": max_contact_moment_error,
        "contact_mismatch_frames": contact_mismatch_frames,
        "max_sim_time_error": max_sim_time_error,
        "max_front_pixel_error": max_front_pixel_error,
        "max_wrist_pixel_error": max_wrist_pixel_error,
    }
