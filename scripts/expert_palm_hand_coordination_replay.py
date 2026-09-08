"""Deterministic expert palm/hand coordination replay.

The public LeRobot bundle contains absolute arm and hand joint arrays, but no
arm joint names, robot model, or palm pose.  This script therefore has two
explicit modes:

* always run the fixed-palm raw-20D and PCA5 hand baselines;
* run expert-relative-palm translation/rotation/full-6D replay only when the
  caller supplies the *expert* robot model, its seven joint names, and palm
  site.  OpenArm's model is never silently used as the expert FK model.

No training and no environment/reward/gate changes are performed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

from openarm_wuji.rl import WujiStaticGraspEnv
from openarm_wuji.tasks.ik import DampedLeastSquaresIK
from openarm_wuji.tasks.se3 import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    relative_pose,
    rotation_geodesic_angle_deg,
)


FPS = 30.0
DEFAULT_EPISODES = (69, 75, 87, 77, 84)
FINGERS = tuple(f"finger{i}" for i in range(1, 6))


def _load_prior(prior_dir: Path):
    mean = np.load(prior_dir / "mean.npy")
    basis = np.load(prior_dir / "basis.npy")
    scaling = json.loads((prior_dir / "latent_scaling.json").read_text(encoding="utf-8"))
    return (
        mean,
        basis,
        np.asarray(scaling["latent_center"], dtype=float),
        np.asarray(scaling["latent_half_range"], dtype=float),
    )


def _latent(q, mean, basis, center, half):
    z = (np.asarray(q, dtype=float) - mean) @ basis
    return np.clip((z - center) / half, -1.0, 1.0)


def _phase_rows(dataset_root: Path, phase_path: Path, episode_ids: Iterable[int], max_steps: int):
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    by_episode = {int(item["episode_id"]): item for item in phase["episodes"]}
    result = {}
    for episode_id in episode_ids:
        source = dataset_root / "teleop/data/chunk-000" / f"episode_{episode_id:06d}.parquet"
        table = pq.read_table(source).to_pydict()
        if episode_id not in by_episode:
            raise KeyError(f"episode {episode_id} is missing from {phase_path}")
        frame_col = np.asarray(table["frame_index"])
        indices = np.argsort(frame_col)
        metadata = by_episode[episode_id]
        start = int(metadata["start_frame"])
        end = min(int(metadata["end_frame_exclusive"]), start + int(max_steps))
        keep = (frame_col[indices] >= start) & (frame_col[indices] < end)
        indices = indices[keep]
        result[episode_id] = {
            "episode_id": episode_id,
            "frame_index": frame_col[indices].astype(int),
            "timestamp": np.asarray(table["timestamp"])[indices].astype(float),
            "state54": np.asarray([table["observation.state"][i] for i in indices], dtype=float),
            "action54": np.asarray([table["action"][i] for i in indices], dtype=float),
            "phase": metadata,
        }
    return result


def dataset_audit(root: Path, source: Path) -> dict:
    info_path = root / "data/external/wuji-pick-and-place/teleop/meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    schema = pq.read_schema(source)
    table = pq.read_table(source, columns=["observation.state", "action"])
    state_shape = [int(table["observation.state"].type.list_size)]
    action_shape = [int(table["action"].type.list_size)]
    columns = list(schema.names)
    return {
        "source": source.as_posix(),
        "dataset_info": info_path.as_posix(),
        "state_shape": state_shape,
        "action_shape": action_shape,
        "left_arm": {"slice_0_based": "[0:7]", "indices_inclusive": [0, 6], "names": None},
        "left_hand": {"slice_0_based": "[14:34]", "indices_inclusive": [14, 33]},
        "right_arm": {"slice_0_based": "[7:14]", "indices_inclusive": [7, 13], "names": None},
        "right_hand": {"slice_0_based": "[34:54]", "indices_inclusive": [34, 53]},
        "arm_joint_order": None,
        "hand_joint_order": {
            "provenance": "official Wuji flat-array convention; dataset info.json names=null",
            "left": [
                f"left_finger{finger}_joint{joint}"
                for finger in range(1, 6) for joint in range(1, 5)
            ],
        },
        "units": {
            "arm_state": "rad (dataset README: source arm degrees converted to radians)",
            "arm_action": "rad absolute target",
            "hand_state": "rad absolute encoder position",
            "hand_action": "rad absolute target",
            "timestamp": "s",
        },
        "recorder_pose_fields": [],
        "parquet_columns": columns,
        "expert_fk_available_in_bundle": False,
        "expert_fk_blocker": (
            "The bundle exposes only 7-D arm values in an unnamed flat array. "
            "It does not include expert arm joint names, URDF/MJCF, base/tool transform, "
            "or palm/wrist pose. OpenArm FK cannot be substituted without violating "
            "the embodiment-mismatch constraint."
        ),
    }


def _median_window(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    if stop <= start:
        return values[-1].copy()
    return np.median(values[start:stop], axis=0)


def motion_stats(episodes: dict[int, dict]) -> tuple[dict, list[dict]]:
    rows = []
    for episode_id, data in episodes.items():
        phase = data["phase"]
        q_arm = data["state54"][:, :7]
        a_arm = data["action54"][:, :7]
        q_hand = data["state54"][:, 14:34]
        close = max(0, min(len(q_arm) - 1, int(phase["close_start_frame"])))
        baseline_end = min(15, max(1, len(q_arm)))
        grasp_start = min(len(q_arm), close + 3)
        grasp_end = min(len(q_arm), grasp_start + 30)
        arm_start = _median_window(q_arm, 0, baseline_end)
        arm_grasp = _median_window(q_arm, grasp_start, grasp_end)
        action_start = _median_window(a_arm, 0, baseline_end)
        action_grasp = _median_window(a_arm, grasp_start, grasp_end)
        hand_start = _median_window(q_hand, 0, baseline_end)
        hand_excursion = np.linalg.norm(q_hand - hand_start, axis=1)
        arm_excursion = np.linalg.norm(q_arm - arm_start, axis=1)
        close_slice = slice(close, max(close + 1, len(q_arm)))
        rows.append({
            "episode_id": episode_id,
            "frames_replayed": len(q_arm),
            "close_start_frame_local": close,
            "arm_state_start_rad": arm_start.tolist(),
            "arm_state_grasp_rad": arm_grasp.tolist(),
            "delta_arm_state_rad": (arm_grasp - arm_start).tolist(),
            "arm_translation_proxy_joint_norm_rad": float(np.linalg.norm(arm_grasp - arm_start)),
            "delta_arm_action_rad": (action_grasp - action_start).tolist(),
            "hand_close_excursion_at_end_rad": float(hand_excursion[-1]),
            "arm_excursion_at_end_rad": float(arm_excursion[-1]),
            "hand_excursion_during_close_rad": {
                "max": float(np.max(hand_excursion[close_slice])),
                "mean": float(np.mean(hand_excursion[close_slice])),
            },
            "arm_excursion_during_close_rad": {
                "max": float(np.max(arm_excursion[close_slice])),
                "mean": float(np.mean(arm_excursion[close_slice])),
            },
            "palm_pose_available": False,
        })
    deltas = np.asarray([row["delta_arm_state_rad"] for row in rows], dtype=float)
    summary = {
        "status": "blocked_without_expert_fk",
        "episodes": [row["episode_id"] for row in rows],
        "palm_translation_m": None,
        "palm_rotation_deg": None,
        "arm_joint_delta_proxy_rad": {
            "mean": np.mean(deltas, axis=0).tolist(),
            "median": np.median(deltas, axis=0).tolist(),
            "std": np.std(deltas, axis=0).tolist(),
            "norm_mean": float(np.mean(np.linalg.norm(deltas, axis=1))),
        },
        "interpretation": (
            "These are expert arm-joint changes, not palm motion. A palm SE(3) "
            "statistic is intentionally not fabricated without the expert FK model."
        ),
        "per_episode": rows,
    }
    return summary, rows


def _finger_frame_stats(env, contacts, info):
    threshold = float(env.config["success"]["min_normal_force_n"])
    result = {}
    for finger in FINGERS:
        valid = [item for item in contacts if item["finger"] == finger
                 and float(item["normal_force_n"]) >= threshold]
        sliding = []
        tangential = []
        for item in valid:
            force = np.asarray(item["force_on_cube_cube_n"], dtype=float)
            normal = np.asarray(item["normal_on_cube_cube"], dtype=float)
            tangential.append(float(np.linalg.norm(force - np.dot(force, normal) * normal)))
            sliding.append(float(env._body_point_speed_relative_cube(
                item["other_body_id"], np.asarray(item["position_world_m"])
            )))
        result[finger] = {
            "valid": bool(valid),
            "normal_force_sum_n": float(sum(item["normal_force_n"] for item in valid)),
            "normal_force_max_n": float(max((item["normal_force_n"] for item in valid), default=0.0)),
            "sliding_speed_max_m_s": float(max(sliding, default=0.0)),
            "tangential_force_sum_n": float(sum(tangential)),
            "faces": sorted({item["cube_face"] for item in valid}),
            "contact_roles": sorted({item["contact_role"] for item in valid}),
            "edge_contact": bool(any(item["edge_contact"] for item in valid)),
            "corner_contact": bool(any(item["corner_contact"] for item in valid)),
            "edge_margin_min_m": float(min((item["distance_to_nearest_edge_m"] for item in valid), default=np.inf)),
            "corner_margin_min_m": float(min((item["distance_to_nearest_corner_m"] for item in valid), default=np.inf)),
            "contact_points_world_m": [np.asarray(item["position_world_m"]).tolist() for item in valid],
            "penetration_m": float(max(0.0, -float(info["finger_distances_m"].get(finger, 0.0)))),
        }
    return result


def _quality_summary(frames: list[dict], info_last: dict) -> dict:
    counts = np.asarray([frame["contact_count"] for frame in frames], dtype=int)
    dt = 1.0 / FPS

    def duration_at_least(k: int) -> dict:
        mask = counts >= k
        total = float(np.sum(mask) * dt)
        max_run = run = 0
        for value in mask:
            run = run + 1 if value else 0
            max_run = max(max_run, run)
        return {"total_s": total, "max_contiguous_s": float(max_run * dt), "frames": int(np.sum(mask))}

    per_finger = {}
    for finger in FINGERS:
        values = [frame["finger"][finger] for frame in frames if frame["finger"][finger]["valid"]]
        if not values:
            per_finger[finger] = {"contact_frames": 0, "contact_duration_s": 0.0}
            continue
        per_finger[finger] = {
            "contact_frames": len(values),
            "contact_duration_s": float(len(values) * dt),
            "max_normal_force_n": float(max(item["normal_force_sum_n"] for item in values)),
            "mean_normal_force_n": float(np.mean([item["normal_force_sum_n"] for item in values])),
            "max_sliding_speed_m_s": float(max(item["sliding_speed_max_m_s"] for item in values)),
            "mean_sliding_speed_m_s": float(np.mean([item["sliding_speed_max_m_s"] for item in values])),
            "min_edge_margin_m": float(min(item["edge_margin_min_m"] for item in values)),
            "edge_contact_frames": int(sum(item["edge_contact"] for item in values)),
            "corner_contact_frames": int(sum(item["corner_contact"] for item in values)),
            "faces": sorted({face for item in values for face in item["faces"]}),
            "contact_roles": sorted({role for item in values for role in item["contact_roles"]}),
            "max_penetration_m": float(max(item["penetration_m"] for item in values)),
        }
    first = {}
    for k in (1, 2, 3):
        indices = np.flatnonzero(counts >= k)
        first[str(k)] = None if len(indices) == 0 else float(indices[0] * dt)
    return {
        "steps": len(frames),
        "max_simultaneous_contacts": int(np.max(counts)) if len(counts) else 0,
        "time_to_first_contact_s": first["1"],
        "time_to_2_contacts_s": first["2"],
        "time_to_3_contacts_s": first["3"],
        "duration_ge_2": duration_at_least(2),
        "duration_ge_3": duration_at_least(3),
        "hold_ge_0.1s": bool(duration_at_least(2)["max_contiguous_s"] >= 0.1),
        "hold_ge_0.3s": bool(duration_at_least(2)["max_contiguous_s"] >= 0.3),
        "hold_ge_0.5s": bool(duration_at_least(2)["max_contiguous_s"] >= 0.5),
        "stable_grasp_success": bool(info_last.get("static_grasp_success", False)),
        "per_finger": per_finger,
        "max_relative_translation_drift_m": float(max((frame["relative_translation_drift_m"] for frame in frames), default=0.0)),
        "max_relative_rotation_drift_deg": float(max((frame["relative_rotation_drift_deg"] for frame in frames), default=0.0)),
        "max_relative_linear_speed_m_s": float(max((frame["relative_linear_speed_m_s"] for frame in frames), default=0.0)),
        "max_relative_angular_speed_deg_s": float(max((frame["relative_angular_speed_deg_s"] for frame in frames), default=0.0)),
        "max_cube_displacement_m": float(max((frame["cube_displacement_m"] for frame in frames), default=0.0)),
        "max_deepest_penetration_m": float(max((frame["deepest_penetration_m"] for frame in frames), default=0.0)),
    }


def _run_fixed(config: Path, root: Path, data: dict, targets: np.ndarray, actions: np.ndarray, label: str):
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    frames = []
    info_last = {}
    initial_relative = None
    try:
        env.reset(seed=7)
        for target, action in zip(targets, actions):
            _, _, terminated, truncated, info = env.step_absolute_target(target, reward_action=action)
            info_last = info
            contacts, _ = env._contacts()
            rel = env._relative_state()
            if initial_relative is None:
                initial_relative = (rel["cube_position"].copy(), rel["quaternion"].copy())
            position_drift = float(np.linalg.norm(rel["cube_position"] - initial_relative[0]))
            rotation_drift = float(rotation_geodesic_angle_deg(initial_relative[1], rel["quaternion"]))
            frames.append({
                "step": int(env._step_count),
                "time_s": float(env._step_count * env.control_dt),
                "contact_fingers": sorted({item["finger"] for item in contacts
                                             if item["normal_force_n"] >= env.config["success"]["min_normal_force_n"]}),
                "contact_count": int(info["contact_fingers"]),
                "finger": _finger_frame_stats(env, contacts, info),
                "relative_translation_drift_m": position_drift,
                "relative_rotation_drift_deg": rotation_drift,
                "relative_linear_speed_m_s": float(rel["linear_speed"]),
                "relative_angular_speed_deg_s": float(np.degrees(rel["angular_speed"])),
                "cube_displacement_m": float(info["cube_displacement_m"]),
                "deepest_penetration_m": float(info["deepest_finger_cube_penetration_m"]),
            })
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "episode_id": int(data["episode_id"]),
        "label": label,
        "quality": _quality_summary(frames, info_last),
        "timeseries": frames,
    }


class ExpertFK:
    """FK for the expert model only; no OpenArm fallback is permitted."""

    def __init__(self, model_path: Path, joint_names: list[str], palm_site: str):
        import mujoco

        if len(joint_names) != 7:
            raise ValueError("expert FK requires exactly seven expert joint names")
        self.model = (mujoco.MjModel.from_binary_path(str(model_path))
                      if model_path.suffix.lower() == ".mjb"
                      else mujoco.MjModel.from_xml_path(str(model_path)))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ], dtype=int)
        if np.any(self.joint_ids < 0):
            missing = [name for name, idx in zip(joint_names, self.joint_ids) if idx < 0]
            raise KeyError(f"expert FK joints missing: {missing}")
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, palm_site)
        if self.site_id < 0:
            raise KeyError(f"expert FK palm site missing: {palm_site}")

    def poses(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import mujoco

        positions, quaternions = [], []
        for row in np.asarray(q_arm, dtype=float):
            if row.shape != (7,):
                raise ValueError("expert arm trajectory must have shape (N, 7)")
            self.data.qpos[self.qpos_ids] = row
            mujoco.mj_forward(self.model, self.data)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
            positions.append(self.data.site_xpos[self.site_id].copy())
            quaternions.append(normalize_quaternion(quat))
        return np.asarray(positions), np.asarray(quaternions)


def _openarm_targets_from_relative(env, expert_position, expert_quaternion, mode: str):
    start_position, start_quaternion, _, _ = env._site_pose_velocity()
    relative_position = []
    relative_quaternion = []
    for position, quaternion in zip(expert_position, expert_quaternion):
        delta_position, delta_quaternion = relative_pose(
            expert_position[0], expert_quaternion[0], position, quaternion
        )
        relative_position.append(delta_position)
        relative_quaternion.append(delta_quaternion)
    relative_position = np.asarray(relative_position)
    relative_quaternion = np.asarray(relative_quaternion)
    start_rotation = quaternion_to_matrix(start_quaternion)
    target_position = start_position + (start_rotation @ relative_position)
    if mode == "translation_only":
        target_quaternion = start_quaternion.copy()
    elif mode == "rotation_only":
        target_position = np.repeat(start_position[None], len(target_position), axis=0)
        target_quaternion = np.asarray([
            quaternion_multiply(start_quaternion, quat) for quat in relative_quaternion
        ])
    elif mode == "full_6d":
        target_quaternion = np.asarray([
            quaternion_multiply(start_quaternion, quat) for quat in relative_quaternion
        ])
    else:
        raise ValueError(f"unknown palm mode: {mode}")
    if mode == "translation_only":
        target_position = np.asarray(target_position)
        target_quaternion = np.repeat(start_quaternion[None], len(target_position), axis=0)
    return np.asarray(target_position), np.asarray(target_quaternion)


def _run_moving(config: Path, root: Path, data: dict, targets: np.ndarray,
                actions: np.ndarray, expert_position: np.ndarray,
                expert_quaternion: np.ndarray, mode: str, label: str):
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    env.reset(seed=7)
    # The environment already owns the task IK settings. Reuse them and apply
    # the existing per-IK-step joint bound as the arm replay rate limit.
    ik = DampedLeastSquaresIK(
        env.model, env.data,
        site_name=env.task.config["scene"]["grasp_site_name"],
        joint_names=env.task.arm_joint_names,
        damping=float(env.task.config["reach"]["ik_damping"]),
        max_joint_step=float(env.task.config["reach"]["max_joint_step_rad"]),
    )
    frames = []
    info_last = {}
    previous_arm = env._locked_arm_qpos.copy()
    openarm_position, openarm_quaternion = _openarm_targets_from_relative(
        env, expert_position, expert_quaternion, mode
    )
    initial_relative = None
    try:
        for target, action, position, quaternion in zip(
                targets, actions, openarm_position, openarm_quaternion):
            goal, residual, orientation_residual, _ = ik.solve_pose(
                position, quaternion,
                max_iterations=int(env.task.config["grasp"]["ik_max_iterations"]),
                position_tolerance=float(env.task.config["grasp"]["ik_solve_tolerance_m"]),
                orientation_tolerance_deg=float(env.task.config["reach"]["ik_orientation_solve_tolerance_deg"]),
                orientation_weight=float(env.task.config["reach"]["ik_orientation_weight"]),
            )
            max_step = float(env.task.config["reach"]["max_joint_step_rad"])
            goal = previous_arm + np.clip(goal - previous_arm, -max_step, max_step)
            env._locked_arm_qpos = goal.copy()
            _, _, terminated, truncated, info = env.step_absolute_target(target, reward_action=action)
            previous_arm = goal.copy()
            info_last = info
            rel = env._relative_state()
            contacts, _ = env._contacts()
            if initial_relative is None:
                initial_relative = (rel["cube_position"].copy(), rel["quaternion"].copy())
            frames.append({
                "step": int(env._step_count),
                "time_s": float(env._step_count * env.control_dt),
                "ik_position_residual_m": float(residual),
                "ik_orientation_residual_deg": float(orientation_residual),
                "ik_unreachable": bool(
                    residual > float(env.task.config["grasp"]["ik_solve_tolerance_m"])
                    or orientation_residual > float(env.task.config["reach"]["ik_orientation_solve_tolerance_deg"])
                ),
                "desired_palm_position_m": position.tolist(),
                "actual_palm_position_m": env._site_pose_velocity()[0].tolist(),
                "desired_palm_quaternion_wxyz": quaternion.tolist(),
                "actual_palm_quaternion_wxyz": env._site_pose_velocity()[1].tolist(),
                "arm_joint_target_rad": goal.tolist(),
                "arm_qpos_rad": env.data.qpos[env.arm_qpos_ids].tolist(),
                "contact_count": int(info["contact_fingers"]),
                "finger": _finger_frame_stats(env, contacts, info),
                "relative_translation_drift_m": float(np.linalg.norm(rel["cube_position"] - initial_relative[0])),
                "relative_rotation_drift_deg": float(rotation_geodesic_angle_deg(initial_relative[1], rel["quaternion"])),
                "relative_linear_speed_m_s": float(rel["linear_speed"]),
                "relative_angular_speed_deg_s": float(np.degrees(rel["angular_speed"])),
                "cube_displacement_m": float(info["cube_displacement_m"]),
                "deepest_penetration_m": float(info["deepest_finger_cube_penetration_m"]),
            })
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "episode_id": int(data["episode_id"]),
        "label": label,
        "quality": _quality_summary(frames, info_last),
        "timeseries": frames,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--source", type=Path, default=Path("data/external/wuji-pick-and-place/teleop/data/chunk-000/episode_000069.parquet"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/wuji-pick-and-place"))
    parser.add_argument("--phase", type=Path, default=Path("outputs/expert_pca5/phase_metadata.json"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/expert_pca5"))
    parser.add_argument("--config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/expert_palm_replay"))
    parser.add_argument("--episode-ids", type=int, nargs="+", default=list(DEFAULT_EPISODES))
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--expert-fk-model", type=Path, default=None)
    parser.add_argument("--expert-arm-joint-names", type=str, default=None,
                        help="comma-separated seven names in the expert FK model")
    parser.add_argument("--expert-palm-site", type=str, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    phase = args.phase if args.phase.is_absolute() else root / args.phase
    prior = args.prior if args.prior.is_absolute() else root / args.prior
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    source_for_audit = args.source if args.source.is_absolute() else root / args.source
    audit = dataset_audit(root, source_for_audit)
    mean, basis, center, half = _load_prior(prior)
    episodes = {}
    for episode_id in args.episode_ids:
        episodes.update(_phase_rows(dataset_root, phase, [episode_id], args.max_steps))
    motion, motion_rows = motion_stats(episodes)
    comparison = []
    timeseries = {}
    for episode_id, data in episodes.items():
        q_hand = data["state54"][:, 14:34]
        actions = np.asarray([_latent(q, mean, basis, center, half) for q in q_hand])
        pca_targets = np.asarray([mean + basis @ (center + half * action) for action in actions])
        for label, targets in (("A_fixed_palm_raw20", q_hand), ("B_fixed_palm_pca5", pca_targets)):
            result = _run_fixed(config, root, data, targets, actions, label)
            comparison.append({"episode_id": episode_id, "label": label, "quality": result["quality"]})
            timeseries[f"episode_{episode_id}_{label}"] = result["timeseries"]
    blocked = {
        "status": "blocked_without_expert_fk",
        "reason": audit["expert_fk_blocker"],
        "required_inputs": ["expert FK model", "seven expert arm joint names", "expert palm/wrist site", "base/tool transform if not encoded in model"],
        "openarm_fallback": False,
    }
    moving_comparison = {
        "status": "blocked_without_expert_fk",
        "A_fixed_palm_raw20": blocked,
        "B_fixed_palm_pca5": blocked,
        "C_expert_palm_raw20": blocked,
        "D_expert_palm_pca5": blocked,
    }
    ablation = {
        "status": "blocked_without_expert_fk",
        "translation_only": blocked,
        "rotation_only": blocked,
        "full_6d": blocked,
    }
    if args.expert_fk_model is not None or args.expert_arm_joint_names is not None or args.expert_palm_site is not None:
        if not (args.expert_fk_model and args.expert_arm_joint_names and args.expert_palm_site):
            raise ValueError("expert FK replay requires --expert-fk-model, --expert-arm-joint-names, and --expert-palm-site together")
        expert_model = args.expert_fk_model if args.expert_fk_model.is_absolute() else root / args.expert_fk_model
        fk = ExpertFK(expert_model, [name.strip() for name in args.expert_arm_joint_names.split(",")], args.expert_palm_site)
        moving_results = []
        ablation_results = {"translation_only": [], "rotation_only": [], "full_6d": []}
        for episode_id, data in episodes.items():
            expert_position, expert_quaternion = fk.poses(data["state54"][:, :7])
            q_hand = data["state54"][:, 14:34]
            actions = np.asarray([_latent(q, mean, basis, center, half) for q in q_hand])
            pca_targets = np.asarray([mean + basis @ (center + half * action) for action in actions])
            for mode in ("translation_only", "rotation_only", "full_6d"):
                for hand_label, targets in (("raw20", q_hand), ("pca5", pca_targets)):
                    label = f"{mode}_{hand_label}"
                    result = _run_moving(
                        config, root, data, targets, actions,
                        expert_position, expert_quaternion, mode, label,
                    )
                    moving_results.append({"episode_id": episode_id, "label": label, "quality": result["quality"]})
                    if hand_label == "pca5":
                        ablation_results[mode].append({"episode_id": episode_id, "quality": result["quality"]})
                    timeseries[f"episode_{episode_id}_{label}"] = result["timeseries"]
        moving_comparison = {
            "status": "completed",
            "expert_fk_model": expert_model.as_posix(),
            "expert_arm_joint_names": [name.strip() for name in args.expert_arm_joint_names.split(",")],
            "expert_palm_site": args.expert_palm_site,
            "results": moving_results,
        }
        ablation = {"status": "completed", **ablation_results}
    payloads = {
        "expert_palm_motion_stats.json": {"dataset_audit": audit, **motion},
        "selected_episode_phase_boundaries.json": {str(ep): episodes[ep]["phase"] for ep in episodes},
        "replay_comparison.json": {
            "status": "partial_baseline_only",
            "replay_semantics": "fixed seed 7; same Reward V2 and 120-step horizon",
            "results": comparison,
            "moving_palm": moving_comparison,
        },
        "translation_vs_rotation_ablation.json": ablation,
        "palm_tracking_metrics.json": {
            "status": "blocked_without_expert_fk",
            "desired_actual_tracking": None,
            "ik_residuals": None,
            "unreachable_events": None,
        },
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        for filename, payload in payloads.items():
            (output / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (output / "timeseries.json").write_text(json.dumps(timeseries, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), "episodes": list(episodes), "status": "partial_baseline_only"}, indent=2))
    except PermissionError:
        print(json.dumps({
            "status": "partial_baseline_only",
            "audit": audit,
            "motion": motion,
            "comparison": comparison,
            "moving_palm": moving_comparison,
            "ablation": ablation,
        }, indent=2))


if __name__ == "__main__":
    main()
