"""Joint-aligned OpenArm + Wuji demonstration recording and staging export."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .episode_recorder import CausalEpisodeRecorder, _state


ARM_DOF = 7
HAND_DOF = 20
EMBODIMENT_DOF = ARM_DOF + HAND_DOF
RAW_SCRIPT_ACTION_DOF = 10
ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    f"wuji_left_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]
JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _controller_target(observation_tp1: dict, raw_action: np.ndarray) -> np.ndarray:
    """Return targets held by all position actuators for this transition."""
    target = np.asarray(observation_tp1["controller_joint_target"], dtype=float)
    hand_target = np.asarray(observation_tp1["hand_joint_target"], dtype=float)
    if target.shape != (EMBODIMENT_DOF,):
        raise ValueError(f"controller target must be 27-D, got {target.shape}")
    if hand_target.shape != (HAND_DOF,):
        raise ValueError(f"hand target must be 20-D, got {hand_target.shape}")
    if not np.isfinite(target).all():
        raise ValueError("controller target contains NaN or infinity")
    if not np.array_equal(target[:ARM_DOF], raw_action[:ARM_DOF]):
        raise ValueError("arm controller target differs from bounded sent action")
    if not np.array_equal(target[ARM_DOF:], hand_target):
        raise ValueError("hand controller target differs from mapped hand target")
    return target.copy()


def _pose(telemetry: dict, *, relative: bool = False) -> list[float]:
    if relative:
        return [
            *np.asarray(telemetry["object_relative_position_m"], dtype=float),
            *np.asarray(
                telemetry["object_relative_quaternion_wxyz"], dtype=float
            ),
        ]
    return [
        *np.asarray(telemetry["cube_position_m"], dtype=float),
        *np.asarray(telemetry["cube_quaternion_wxyz"], dtype=float),
    ]


class CoordinatedDemoRecorder(CausalEpisodeRecorder):
    """Record vision/state -> 27-D actuator target demonstrations.

    The task still sends its original 10-D command (7 arm targets + 3 Wuji
    synergies).  For every transition, this recorder reads back the exact 27
    targets held by MuJoCo's position actuators after clipping and synergy
    mapping.  The original 10-D command is retained only for audit/replay.
    """

    SCHEMA_VERSION = 1
    FORMAT_NAME = "openarm_wuji_coordinated_demo"

    def __init__(self, *, task_name: str, control_hz: float,
                 task_description: str = "", episode_index: int = 0,
                 required_hold_s: float = 0.5):
        super().__init__(
            task_name=task_name,
            control_hz=control_hz,
            task_description=task_description,
            episode_index=episode_index,
        )
        if required_hold_s <= 0:
            raise ValueError("required_hold_s must be positive")
        self.required_hold_s = float(required_hold_s)
        self.demo_summary: dict[str, Any] | None = None

    def start(self, seed: int) -> None:
        super().start(seed)
        self.demo_summary = None

    def record_transition(self, *, phase: str, observation_t: dict,
                          action_t, observation_tp1: dict,
                          telemetry_t: dict, telemetry_tp1: dict) -> None:
        raw_action = np.asarray(action_t, dtype=float)
        super().record_transition(
            phase=phase,
            observation_t=observation_t,
            action_t=raw_action,
            observation_tp1=observation_tp1,
            telemetry_t=telemetry_t,
            telemetry_tp1=telemetry_tp1,
        )
        self.transitions[-1]["controller_action_t"] = _controller_target(
            observation_tp1, raw_action
        )

    def _semantic_phases(self) -> list[str]:
        phases = [str(row["phase"]) for row in self.transitions]
        if self.outcome is None:
            return phases
        duration = float(
            self.outcome.get("trajectory_diagnostics", {}).get(
                "trajectory_reference_duration_s", 0.0
            )
        )
        trajectory_frames = int(round(duration * self.control_hz))
        lift_index = 0
        result = []
        for phase in phases:
            if phase != "lift_s_curve":
                result.append(phase)
                continue
            lift_index += 1
            result.append("lift" if lift_index <= trajectory_frames else "hold")
        return result

    def _boundary_telemetry(self, phase: str) -> dict | None:
        for row in self.transitions:
            if row["phase"] == phase:
                return row["telemetry_t"]
        return None

    def _phase_cube_pose(self, phase: str) -> dict[str, Any] | None:
        telemetry = self._boundary_telemetry(phase)
        if telemetry is None:
            return None
        return {
            "world_xyz_wxyz": _pose(telemetry),
            "relative_to_palm_xyz_wxyz": _pose(telemetry, relative=True),
        }

    def _first_contact(self) -> dict[str, Any] | None:
        if not self.transitions:
            return None
        initial_contacts = self.transitions[0]["telemetry_t"]["contacts"]
        if initial_contacts:
            contact = initial_contacts[0]
            return self._contact_metadata(
                contact, phase="reset",
                observation=self.transitions[0]["observation_t"],
            )
        for row in self.transitions:
            contacts = row["telemetry_tp1"]["contacts"]
            if contacts:
                return self._contact_metadata(
                    contacts[0], phase=str(row["phase"]),
                    observation=row["observation_tp1"],
                )
        return None

    @staticmethod
    def _contact_metadata(contact: dict, *, phase: str,
                          observation: dict) -> dict[str, Any]:
        return {
            "phase": phase,
            "frame_index": int(observation["frame_index"]),
            "sim_time_s": float(observation["sim_time"]),
            "finger": str(contact["finger"]),
            "body": str(contact.get("other_body_name") or ""),
            "geom": str(contact.get("other_geom_name") or ""),
            "position_world_m": _plain(contact["position_world_m"]),
            "normal_force_n": float(contact["normal_force_n"]),
        }

    def finish(self, outcome: dict) -> None:
        # A reset/IK failure can happen before the first controller command.
        # Keep that attempt as diagnostic metadata so a long batch does not
        # abort, while still refusing to represent it as a demonstration.
        if self.transitions:
            super().finish(outcome)
        else:
            if self.seed is None:
                raise RuntimeError("episode was not started")
            self.outcome = json.loads(json.dumps(outcome))
        assert self.outcome is not None
        trajectory = self.outcome.get("trajectory_diagnostics", {})
        hold_frames = int(trajectory.get("trajectory_reference_hold_frames", 0))
        required_hold_frames = int(round(self.required_hold_s * self.control_hz))
        grasp = self.outcome.get("grasp", {})
        reach = grasp.get("reach", {})
        minimum_lift = float(
            self.outcome.get("external_baseline", {}).get(
                "min_object_lift_m", 0.025
            )
        )
        lift_success = bool(
            self.outcome.get("task_success", False)
            and float(self.outcome.get("peak_height_m", 0.0)) >= minimum_lift
        )
        hold_success = bool(
            lift_success
            and float(self.outcome.get("final_height_m", 0.0)) >= minimum_lift
            and hold_frames >= required_hold_frames
        )
        demonstration_success = bool(
            reach.get("success", False)
            and grasp.get("success", False)
            and lift_success
            and hold_success
        )
        reset_telemetry = (
            self.transitions[0]["telemetry_t"] if self.transitions else None
        )
        self.demo_summary = {
            "format_name": self.FORMAT_NAME,
            "schema_version": self.SCHEMA_VERSION,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "samples": len(self.transitions),
            "control_hz": self.control_hz,
            "duration_s": len(self.transitions) / self.control_hz,
            "state_dim": EMBODIMENT_DOF,
            "action_dim": EMBODIMENT_DOF,
            "action_semantics": (
                "exact MuJoCo position-actuator targets after arm clipping "
                "and Wuji synergy-to-20D mapping"
            ),
            "demonstration_success": demonstration_success,
            "failure_reason": (
                None if demonstration_success
                else self.outcome.get("failure_reason")
                    or self.outcome.get("outcome")
            ),
            "reach_success": bool(reach.get("success", False)),
            "grasp_success": bool(grasp.get("success", False)),
            "lift_success": lift_success,
            "hold_success": hold_success,
            "hold_frames": hold_frames,
            "required_hold_frames": required_hold_frames,
            "task_success": bool(self.outcome.get("task_success", False)),
            "grasp_stable": bool(self.outcome.get("grasp_stable", False)),
            "outcome": str(self.outcome.get("outcome") or ""),
            "cube_pose_at_reset": (
                None if reset_telemetry is None else {
                    "world_xyz_wxyz": _pose(reset_telemetry),
                    "relative_to_palm_xyz_wxyz": _pose(
                        reset_telemetry, relative=True
                    ),
                }
            ),
            "cube_pose_before_approach": self._phase_cube_pose("approach"),
            "cube_pose_before_grasp_close": self._phase_cube_pose("grasp_close"),
            "cube_pose_before_lift": self._phase_cube_pose("lift_s_curve"),
            "first_hand_cube_contact": self._first_contact(),
            "controller_phases": sorted({
                str(row["phase"]) for row in self.transitions
            }),
            "semantic_phases": sorted(set(self._semantic_phases())),
        }

    @property
    def is_successful_demonstration(self) -> bool:
        if self.demo_summary is None:
            raise RuntimeError("finish(outcome) must be called first")
        return bool(self.demo_summary["demonstration_success"])

    def save(self, path: str | Path) -> Path:
        if self.outcome is None or self.demo_summary is None:
            raise RuntimeError("finish(outcome) must be called before saving")
        if not self.transitions:
            raise RuntimeError("episode contains no control transitions")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.transitions
        observations = [row["observation_t"] for row in rows]
        next_observations = [row["observation_tp1"] for row in rows]
        telemetry = [row["telemetry_t"] for row in rows]
        start_sim_time = float(observations[0]["sim_time"])
        active_masks = []
        normal_forces = []
        for sample in telemetry:
            force_by_finger = sample["finger_normal_forces_n"]
            force = np.asarray([
                force_by_finger.get(f"finger{i}", 0.0) for i in range(1, 6)
            ], dtype=float)
            normal_forces.append(force)
            active_masks.append(force > 0.0)
        arrays = {
            "format_name": np.asarray(self.FORMAT_NAME),
            "schema_version": np.asarray(self.SCHEMA_VERSION),
            "task_name": np.asarray(self.task_name),
            "task_description": np.asarray(self.task_description),
            "episode_index": np.asarray(self.episode_index),
            "episode_seed": np.asarray(self.seed),
            "control_hz": np.asarray(self.control_hz),
            "joint_names": np.asarray(JOINT_NAMES),
            "state_dim": np.asarray(EMBODIMENT_DOF),
            "action_dim": np.asarray(EMBODIMENT_DOF),
            "timestamp": np.asarray([
                float(item["sim_time"]) - start_sim_time for item in observations
            ]),
            "host_timestamp": np.asarray([
                item["timestamp"] for item in observations
            ]),
            "sim_time": np.asarray([item["sim_time"] for item in observations]),
            "frame_index": np.asarray([
                item["frame_index"] for item in observations
            ], dtype=np.int64),
            "phase": np.asarray(self._semantic_phases()),
            "controller_phase": np.asarray([row["phase"] for row in rows]),
            "observation.state": np.asarray([
                _state(item) for item in observations
            ]),
            "action": np.asarray([
                row["controller_action_t"] for row in rows
            ]),
            "raw_script_action": np.asarray([
                row["action_t"] for row in rows
            ]),
            "next_observation.state": np.asarray([
                _state(item) for item in next_observations
            ]),
            "observation.images.front": np.asarray([
                item["front_rgb"] for item in observations
            ]),
            "observation.images.wrist": np.asarray([
                item["wrist_rgb"] for item in observations
            ]),
            "final_observation.images.front": np.asarray(
                next_observations[-1]["front_rgb"]
            ),
            "final_observation.images.wrist": np.asarray(
                next_observations[-1]["wrist_rgb"]
            ),
            "telemetry.cube_pose_world": np.asarray([
                _pose(item) for item in telemetry
            ]),
            "telemetry.cube_pose_relative_to_palm": np.asarray([
                _pose(item, relative=True) for item in telemetry
            ]),
            "telemetry.contact_count": np.asarray([
                len(item["contacts"]) for item in telemetry
            ], dtype=np.int32),
            "telemetry.active_finger_mask": np.asarray(active_masks),
            "telemetry.finger_normal_force_n": np.asarray(normal_forces),
            "demonstration_success": np.asarray(
                self.demo_summary["demonstration_success"]
            ),
            "reach_success": np.asarray(self.demo_summary["reach_success"]),
            "grasp_success": np.asarray(self.demo_summary["grasp_success"]),
            "lift_success": np.asarray(self.demo_summary["lift_success"]),
            "hold_success": np.asarray(self.demo_summary["hold_success"]),
            "outcome": np.asarray(self.demo_summary["outcome"]),
            "episode_metadata_json": np.asarray(json.dumps(
                _plain(self.demo_summary), separators=(",", ":")
            )),
            "outcome_json": np.asarray(json.dumps(
                _plain(self.outcome), separators=(",", ":")
            )),
        }
        np.savez_compressed(destination, **arrays)
        return destination

    @staticmethod
    def validate(path: str | Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as episode:
            if str(episode["format_name"]) != CoordinatedDemoRecorder.FORMAT_NAME:
                raise ValueError("not a coordinated demonstration")
            if int(episode["schema_version"]) != CoordinatedDemoRecorder.SCHEMA_VERSION:
                raise ValueError("unsupported coordinated demonstration schema")
            samples = len(episode["action"])
            expected = {
                "observation.state": (samples, EMBODIMENT_DOF),
                "next_observation.state": (samples, EMBODIMENT_DOF),
                "action": (samples, EMBODIMENT_DOF),
                "raw_script_action": (samples, RAW_SCRIPT_ACTION_DOF),
                "telemetry.cube_pose_world": (samples, 7),
                "telemetry.cube_pose_relative_to_palm": (samples, 7),
                "telemetry.active_finger_mask": (samples, 5),
                "telemetry.finger_normal_force_n": (samples, 5),
            }
            for key, shape in expected.items():
                if episode[key].shape != shape:
                    raise ValueError(f"invalid {key} shape: {episode[key].shape}")
                if not np.isfinite(episode[key]).all():
                    raise ValueError(f"{key} contains NaN or infinity")
            if not np.array_equal(
                episode["action"][:, :ARM_DOF],
                episode["raw_script_action"][:, :ARM_DOF],
            ):
                raise ValueError("saved arm action is not the bounded controller target")
            for key in ("observation.images.front", "observation.images.wrist"):
                if episode[key].shape[0] != samples or episode[key].dtype != np.uint8:
                    raise ValueError(f"invalid {key}")
            if episode["timestamp"].shape != (samples,):
                raise ValueError("invalid timestamp")
            if samples > 1 and not np.all(np.diff(episode["timestamp"]) > 0):
                raise ValueError("timestamps are not strictly increasing")
            if not np.array_equal(
                episode["frame_index"], np.arange(samples, dtype=np.int64)
            ):
                raise ValueError("frame indices are not a complete reset-origin chain")
            metadata = json.loads(str(episode["episode_metadata_json"]))
            if bool(episode["demonstration_success"]):
                if not all(bool(episode[key]) for key in (
                    "reach_success", "grasp_success", "lift_success", "hold_success"
                )):
                    raise ValueError("successful demo lacks a required stage")
                if "hold" not in set(episode["phase"].tolist()):
                    raise ValueError("successful demo has no hold frames")
            preload = episode["action"] - episode["observation.state"]
            return {
                "samples": samples,
                "duration_s": float(samples / float(episode["control_hz"])),
                "state_shape": list(episode["observation.state"].shape),
                "action_shape": list(episode["action"].shape),
                "front_image_shape": list(episode["observation.images.front"].shape),
                "wrist_image_shape": list(episode["observation.images.wrist"].shape),
                "semantic_phase_counts": {
                    str(name): int(count) for name, count in zip(
                        *np.unique(episode["phase"], return_counts=True), strict=True
                    )
                },
                "demonstration_success": bool(
                    episode["demonstration_success"]
                ),
                "max_abs_target_actual_preload_rad": float(np.max(np.abs(preload))),
                "metadata": metadata,
            }


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        stream, format="PNG", compress_level=3
    )
    return stream.getvalue()


def _stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
    }


def export_successful_episodes(paths: Iterable[str | Path],
                               output_dir: str | Path) -> dict[str, Any]:
    """Export successful raw NPZ episodes to a LeRobot-shaped Parquet staging set.

    Images use the Hugging Face Image-style ``{bytes, path}`` struct.  The
    result intentionally calls itself a staging set: producing a canonical
    LeRobotDataset still requires the optional ``datasets`` package and a final
    LeRobot API write/validation pass.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination = Path(output_dir)
    data_dir = destination / "data/chunk-000"
    meta_dir = destination / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[Any]] = {
        "observation.state": [], "action": [], "timestamp": [],
        "frame_index": [], "episode_index": [], "index": [],
        "task_index": [], "phase": [], "source_seed": [],
        "observation.images.front": [], "observation.images.wrist": [],
    }
    episodes = []
    all_states = []
    all_actions = []
    fps = None
    image_shape = None
    global_index = 0
    for source in sorted((Path(item) for item in paths), key=lambda item: item.name):
        validation = CoordinatedDemoRecorder.validate(source)
        if not validation["demonstration_success"]:
            continue
        with np.load(source, allow_pickle=False) as episode:
            current_fps = float(episode["control_hz"])
            current_shape = tuple(episode["observation.images.front"].shape[1:])
            if fps is None:
                fps, image_shape = current_fps, current_shape
            if current_fps != fps or current_shape != image_shape:
                raise ValueError("all exported episodes must share fps and image shape")
            export_index = len(episodes)
            states = episode["observation.state"].astype(np.float32)
            actions = episode["action"].astype(np.float32)
            timestamps = episode["timestamp"].astype(float)
            front = episode["observation.images.front"]
            wrist = episode["observation.images.wrist"]
            phases = episode["phase"].astype(str)
            seed = int(episode["episode_seed"])
            for frame in range(len(actions)):
                records["observation.state"].append(states[frame].tolist())
                records["action"].append(actions[frame].tolist())
                records["timestamp"].append(float(timestamps[frame]))
                records["frame_index"].append(frame)
                records["episode_index"].append(export_index)
                records["index"].append(global_index)
                records["task_index"].append(0)
                records["phase"].append(str(phases[frame]))
                records["source_seed"].append(seed)
                records["observation.images.front"].append({
                    "bytes": _png_bytes(front[frame]), "path": None
                })
                records["observation.images.wrist"].append({
                    "bytes": _png_bytes(wrist[frame]), "path": None
                })
                global_index += 1
            all_states.append(states)
            all_actions.append(actions)
            episodes.append({
                "episode_index": export_index,
                "tasks": [str(episode["task_description"])],
                "length": len(actions),
                "source_seed": seed,
                "source_raw_episode": source.as_posix(),
            })
    if not episodes:
        raise ValueError("no successful demonstrations were provided")
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({
        "observation.state": pa.array(
            records["observation.state"], type=pa.list_(pa.float32(), EMBODIMENT_DOF)
        ),
        "action": pa.array(
            records["action"], type=pa.list_(pa.float32(), EMBODIMENT_DOF)
        ),
        "timestamp": pa.array(records["timestamp"], type=pa.float64()),
        "frame_index": pa.array(records["frame_index"], type=pa.int64()),
        "episode_index": pa.array(records["episode_index"], type=pa.int64()),
        "index": pa.array(records["index"], type=pa.int64()),
        "task_index": pa.array(records["task_index"], type=pa.int64()),
        "phase": pa.array(records["phase"], type=pa.string()),
        "source_seed": pa.array(records["source_seed"], type=pa.int64()),
        "observation.images.front": pa.array(
            records["observation.images.front"], type=image_type
        ),
        "observation.images.wrist": pa.array(
            records["observation.images.wrist"], type=image_type
        ),
    })
    parquet_path = data_dir / "file-000.parquet"
    pq.write_table(table, parquet_path, compression="zstd")
    state_values = np.concatenate(all_states)
    action_values = np.concatenate(all_actions)
    height, width, channels = image_shape
    info = {
        "codebase_version": "v0.6",
        "robot_type": "openarm_wuji",
        "total_episodes": len(episodes),
        "total_frames": len(table),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/file-{file_chunk:03d}.parquet",
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [EMBODIMENT_DOF],
                "names": JOINT_NAMES,
            },
            "action": {
                "dtype": "float32", "shape": [EMBODIMENT_DOF],
                "names": JOINT_NAMES,
            },
            "observation.images.front": {
                "dtype": "image", "shape": [height, width, channels],
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "image", "shape": [height, width, channels],
                "names": ["height", "width", "channels"],
            },
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (meta_dir / "tasks.jsonl").write_text(json.dumps({
        "task_index": 0, "task": episodes[0]["tasks"][0]
    }) + "\n", encoding="utf-8")
    (meta_dir / "episodes.jsonl").write_text("".join(
        json.dumps(item) + "\n" for item in episodes
    ), encoding="utf-8")
    (meta_dir / "stats.json").write_text(json.dumps({
        "observation.state": _stats(state_values),
        "action": _stats(action_values),
    }, indent=2), encoding="utf-8")
    manifest = {
        "format": "lerobot_v0.6_parquet_staging",
        "native_lerobot_dataset": False,
        "successful_episodes_only": True,
        "episodes": len(episodes),
        "frames": len(table),
        "parquet": parquet_path.relative_to(destination).as_posix(),
        "image_storage": "Hugging Face Image-style PNG bytes/path structs",
        "remaining_for_native_lerobot": [
            "install the optional datasets dependency used by LeRobotDataset",
            "write/import through LeRobotDataset 0.6.2 and run its DataLoader check",
            "optionally encode image columns as MP4 for large-scale storage",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
