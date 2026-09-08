"""Deterministic replay gate for the Wuji expert-PCA5 action prior.

This script deliberately does not train or modify the RL environment.  It
replays the same five Parquet phase trajectories as raw absolute postures and
as PCA5 reconstructed postures, then runs two deterministic hand baselines
through the same fixed-palm Reward V2 path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openarm_wuji.rl import WujiStaticGraspEnv
from openarm_wuji.tasks.se3 import rotation_geodesic_angle_deg
from openarm_wuji.teleop.synergies import HandSynergyMapper


REPRESENTATIVE_EPISODES = (77, 71, 70, 84, 74)


def _load_prior(prior_dir: Path):
    mean = np.load(prior_dir / "mean.npy")
    basis = np.load(prior_dir / "basis.npy")
    scaling = json.loads((prior_dir / "latent_scaling.json").read_text())
    center = np.asarray(scaling["latent_center"], dtype=float)
    half = np.asarray(scaling["latent_half_range"], dtype=float)
    return mean, basis, center, half


def _load_phase_rows(source: Path, phase_path: Path, episode_ids):
    table = pq.read_table(source)
    rows = table.to_pydict()
    phase = json.loads(phase_path.read_text())
    by_episode = {int(item["episode_id"]): item for item in phase["episodes"]}
    result = {}
    for episode_id in episode_ids:
        metadata = by_episode[episode_id]
        mask = np.asarray(rows["episode_index"]) == episode_id
        frame = np.asarray(rows["frame_index"])[mask]
        order = np.argsort(frame)
        q = np.asarray([rows["q_hand"][index] for index in np.flatnonzero(mask)[order]], dtype=float)
        action = np.asarray([rows["a_hand"][index] for index in np.flatnonzero(mask)[order]], dtype=float)
        timestamp = np.asarray(rows["timestamp"])[mask][order].astype(float)
        start = int(metadata["start_frame"])
        end = min(int(metadata["end_frame_exclusive"]), start + 120)
        keep = (frame[order] >= start) & (frame[order] < end)
        result[episode_id] = {
            "episode_id": episode_id,
            "frame_index": frame[order][keep].astype(int),
            "timestamp": timestamp[keep],
            "q_hand": q[keep],
            "a_hand": action[keep],
            "phase": metadata,
        }
    return result


def _latent(q, mean, basis, center, half):
    z = (np.asarray(q) - mean) @ basis
    return np.clip((z - center) / half, -1.0, 1.0)


def _run_sequence(config: Path, root: Path, targets, actions, *, label: str):
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    initial_position = None
    initial_quaternion = None
    total_reward = 0.0
    max_contacts = 0
    contacts_ge = {1: False, 2: False, 3: False}
    max_penetration = 0.0
    max_cube_displacement = 0.0
    max_cube_rotation_deg = 0.0
    rewards = []
    contact_curve = []
    finger_distances = {f"finger{i}": [] for i in range(1, 6)}
    last_info = {}
    terminated = False
    truncated = False
    try:
        env.reset(seed=7)
        for target, action in zip(targets, actions):
            _, reward, terminated, truncated, info = env.step_absolute_target(
                target, reward_action=action
            )
            last_info = info
            total_reward += float(reward)
            rewards.append(float(reward))
            count = int(info["contact_fingers"])
            contact_curve.append(count)
            max_contacts = max(max_contacts, count)
            for threshold in contacts_ge:
                contacts_ge[threshold] |= count >= threshold
            max_penetration = max(
                max_penetration,
                float(info["deepest_finger_cube_penetration_m"]),
            )
            max_cube_displacement = max(
                max_cube_displacement, float(info["cube_displacement_m"])
            )
            position, quaternion, _, _ = env._cube_pose_velocity()
            if initial_position is None:
                initial_position = position.copy()
                initial_quaternion = quaternion.copy()
            max_cube_rotation_deg = max(
                max_cube_rotation_deg,
                rotation_geodesic_angle_deg(initial_quaternion, quaternion),
            )
            for finger, distance in info["finger_distances_m"].items():
                finger_distances[finger].append(float(distance))
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "label": label,
        "steps": len(rewards),
        "return": float(total_reward),
        "mean_step_reward": float(np.mean(rewards)) if rewards else 0.0,
        "max_contact_fingers": max_contacts,
        "episodes_with_contact": bool(contacts_ge[1]),
        "episodes_with_2_contacts": bool(contacts_ge[2]),
        "episodes_with_3_contacts": bool(contacts_ge[3]),
        "contact_curve": contact_curve,
        "max_penetration_m": float(max_penetration),
        "max_cube_displacement_m": float(max_cube_displacement),
        "max_cube_rotation_deg": float(max_cube_rotation_deg),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(last_info.get("static_grasp_success", False)),
        "failure_reason": last_info.get("failure_reason"),
        "final_contact_fingers": int(last_info.get("contact_fingers", 0)),
        "third_closest_final_distance_m": float(
            last_info.get("reward_terms", {}).get(
                "third_closest_finger_distance_m", np.nan
            )
        ),
        "per_finger_distance_m": {
            finger: {
                "minimum": float(np.min(values)) if values else np.nan,
                "final": float(values[-1]) if values else np.nan,
            }
            for finger, values in finger_distances.items()
        },
    }


def _replay_expert(config, root, episodes, mean, basis, center, half):
    results = []
    for episode_id, data in episodes.items():
        q = data["q_hand"]
        actions = np.asarray([_latent(row, mean, basis, center, half) for row in q])
        raw = _run_sequence(config, root, q, actions, label=f"raw_episode_{episode_id}")
        raw["episode_id"] = episode_id
        pca_targets = np.asarray([
            mean + basis @ (center + half * action) for action in actions
        ])
        pca = _run_sequence(config, root, pca_targets, actions, label=f"pca5_episode_{episode_id}")
        pca["episode_id"] = episode_id
        results.extend([raw, pca])
    return results


def _replay_baselines(config, root, mapper):
    # Both probes use the same 120-step horizon and fixed reset.  The first is
    # the legacy 3D open/close synergy trajectory; the second is current
    # structured-per-finger +1 closing.  They are diagnostic, not new gates.
    progress = np.linspace(0.0, 1.0, 120)
    manual_targets = np.asarray([mapper.map(value, 0.0, 0.0) for value in progress])
    structured_targets = np.asarray([
        mapper.open_pose + value * (mapper.close_pose - mapper.open_pose)
        for value in progress
    ])
    zeros = np.zeros((120, 5), dtype=float)
    return [
        _run_sequence(config, root, manual_targets, zeros, label="manual3d_open_close"),
        _run_sequence(config, root, structured_targets, zeros, label="structured5_open_to_close"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5.json"))
    parser.add_argument("--source", type=Path, default=Path("outputs/wuji_teleop_analysis/cube_60_hand_trajectories.parquet"))
    parser.add_argument("--phase", type=Path, default=Path("outputs/expert_pca5/phase_metadata.json"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/expert_pca5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/expert_pca5"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    source = args.source if args.source.is_absolute() else root / args.source
    phase = args.phase if args.phase.is_absolute() else root / args.phase
    prior = args.prior if args.prior.is_absolute() else root / args.prior
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mean, basis, center, half = _load_prior(prior)
    episodes = _load_phase_rows(source, phase, REPRESENTATIVE_EPISODES)
    replay = _replay_expert(config, root, episodes, mean, basis, center, half)
    mapper = HandSynergyMapper.from_json(root / "configs/wuji_hand_left_synergies.json")
    baselines = _replay_baselines(config, root, mapper)
    all_results = replay + baselines
    pca_results = [item for item in replay if item["label"].startswith("pca5_")]
    gate = {
        "required_max_contact_fingers": 2,
        "pca5_max_contact_fingers": max(item["max_contact_fingers"] for item in pca_results),
        "pca5_max_penetration_m": max(item["max_penetration_m"] for item in pca_results),
        "pca5_max_cube_displacement_m": max(item["max_cube_displacement_m"] for item in pca_results),
        "penetration_limit_m": 0.008,
        "cube_motion_limit_m": 0.120,
    }
    gate["pass"] = bool(
        gate["pca5_max_contact_fingers"] >= gate["required_max_contact_fingers"]
        and gate["pca5_max_penetration_m"] <= gate["penetration_limit_m"]
        and gate["pca5_max_cube_displacement_m"] <= gate["cube_motion_limit_m"]
    )
    payload = {
        "source": str(source),
        "phase_metadata": str(phase),
        "representative_episode_ids": list(REPRESENTATIVE_EPISODES),
        "replay_semantics": "fixed seed 7; same Reward V2 and 120-step horizon; phase sequence truncated by horizon",
        "gate": gate,
        "results": all_results,
        "sac_training_allowed": gate["pass"],
        "sac_training_note": "No SAC run is authorized when this deterministic replay gate fails.",
    }
    try:
        (output / "replay_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (output / "reward_ranking.json").write_text(json.dumps({
        "ranking_metric": "total_return",
        "same_reset_seed": 7,
        "results": sorted(all_results, key=lambda item: item["return"], reverse=True),
        "gate": gate,
        }, indent=2), encoding="utf-8")
        print(json.dumps({"gate": gate, "results": len(all_results), "output": str(output)}, indent=2))
    except PermissionError:
        # Restricted desktop sandboxes may allow source edits but deny runtime
        # file creation.  Emit the complete machine-readable payload so the
        # caller can persist it with the workspace patch mechanism.
        compact = {
            **{key: value for key, value in payload.items() if key != "results"},
            "results": [
                {
                    key: value for key, value in item.items()
                    if key not in {"contact_curve", "per_finger_distance_m"}
                }
                for item in all_results
            ],
        }
        print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
