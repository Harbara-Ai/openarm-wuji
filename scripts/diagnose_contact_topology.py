"""Diagnose which Wuji fingers form and lose multi-contact.

This is an open-loop deterministic replay diagnostic.  It does not train SAC
and does not change Reward V2, success gates, or environment observations.
It replays raw and PCA5 reconstructed expert postures through the existing
fixed-palm MuJoCo environment and records per-finger force, sliding, and cube
contact geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openarm_wuji.rl import WujiStaticGraspEnv


REPRESENTATIVE_EPISODES = (77, 71, 70, 84, 74)
FINGERS = tuple(f"finger{i}" for i in range(1, 6))


def load_prior(prior_dir: Path):
    mean = np.load(prior_dir / "mean.npy")
    basis = np.load(prior_dir / "basis.npy")
    scaling = json.loads((prior_dir / "latent_scaling.json").read_text())
    return mean, basis, np.asarray(scaling["latent_center"]), np.asarray(scaling["latent_half_range"])


def load_episodes(source: Path, phase_path: Path):
    table = pq.read_table(source).to_pydict()
    phase = json.loads(phase_path.read_text())
    phase_by_id = {int(item["episode_id"]): item for item in phase["episodes"]}
    result = {}
    episode = np.asarray(table["episode_index"])
    frame = np.asarray(table["frame_index"])
    for episode_id in REPRESENTATIVE_EPISODES:
        indices = np.flatnonzero(episode == episode_id)
        indices = indices[np.argsort(frame[indices])]
        metadata = phase_by_id[episode_id]
        end = min(int(metadata["end_frame_exclusive"]), int(metadata["start_frame"]) + 120)
        indices = [index for index in indices if int(frame[index]) < end]
        result[episode_id] = {
            "q_hand": np.asarray([table["q_hand"][index] for index in indices], dtype=float),
            "a_hand": np.asarray([table["a_hand"][index] for index in indices], dtype=float),
            "frame_index": np.asarray([frame[index] for index in indices], dtype=int),
            "metadata": metadata,
        }
    return result


def finger_snapshot(env: WujiStaticGraspEnv, contacts: list[dict]) -> dict:
    threshold = float(env.config["success"]["min_normal_force_n"])
    by_finger = {}
    for finger in FINGERS:
        items = [item for item in contacts if item["finger"] == finger]
        valid = [item for item in items if float(item["normal_force_n"]) >= threshold]
        if not valid:
            by_finger[finger] = {
                "valid": False, "normal_force_sum_n": 0.0, "normal_force_max_n": 0.0,
                "sliding_speed_max_m_s": None, "tangential_force_sum_n": 0.0,
                "faces": [], "roles": [], "edge_contact": False, "corner_contact": False,
                "distance_to_nearest_edge_m": None, "distance_to_nearest_corner_m": None,
            }
            continue
        tangential = []
        sliding = []
        for item in valid:
            force = np.asarray(item["force_on_cube_cube_n"], dtype=float)
            normal = np.asarray(item["normal_on_cube_cube"], dtype=float)
            tangential.append(float(np.linalg.norm(force - np.dot(force, normal) * normal)))
            sliding.append(float(env._body_point_speed_relative_cube(
                item["other_body_id"], np.asarray(item["position_world_m"])
            )))
        by_finger[finger] = {
            "valid": True,
            "normal_force_sum_n": float(sum(float(item["normal_force_n"]) for item in valid)),
            "normal_force_max_n": float(max(float(item["normal_force_n"]) for item in valid)),
            "sliding_speed_max_m_s": float(max(sliding)),
            "tangential_force_sum_n": float(sum(tangential)),
            "faces": sorted({item["cube_face"] for item in valid}),
            "roles": sorted({item["contact_role"] for item in valid}),
            "edge_contact": bool(any(item["edge_contact"] for item in valid)),
            "corner_contact": bool(any(item["corner_contact"] for item in valid)),
            "distance_to_nearest_edge_m": float(min(item["distance_to_nearest_edge_m"] for item in valid)),
            "distance_to_nearest_corner_m": float(min(item["distance_to_nearest_corner_m"] for item in valid)),
        }
    return by_finger


def replay_episode(config: Path, root: Path, episode_id: int, data: dict, mean, basis, center, half):
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    env.reset(seed=7)
    snapshots = []
    total_reward = 0.0
    try:
        for q in data["q_hand"]:
            action = np.clip(((q - mean) @ basis - center) / half, -1.0, 1.0)
            _, reward, terminated, truncated, info = env.step_absolute_target(q, reward_action=action)
            contacts, _ = env._contacts()
            finger_data = finger_snapshot(env, contacts)
            valid_fingers = [finger for finger, value in finger_data.items() if value["valid"]]
            relative = env._relative_state()
            snapshots.append({
                "step": int(env._step_count),
                "time_s": float(env._step_count * env.control_dt),
                "contact_fingers": valid_fingers,
                "contact_count": len(valid_fingers),
                "finger": finger_data,
                "relative_position_m": relative["position"].tolist(),
                "relative_quaternion": relative["quaternion"].tolist(),
                "relative_linear_speed_m_s": float(relative["linear_speed"]),
                "relative_angular_speed_deg_s": float(np.degrees(relative["angular_speed"])),
                "cube_displacement_m": float(info["cube_displacement_m"]),
                "deepest_penetration_m": float(info["deepest_finger_cube_penetration_m"]),
                "reward": float(reward),
            })
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()

    first_three = next((index for index, item in enumerate(snapshots) if item["contact_count"] >= 3), None)
    drop_index = None
    dropped = []
    drop_events = []
    if first_three is not None:
        established = set(snapshots[first_three]["contact_fingers"])
        seen_loss = set()
        for index in range(first_three + 1, len(snapshots)):
            before = set(snapshots[index - 1]["contact_fingers"])
            after = set(snapshots[index]["contact_fingers"])
            lost = sorted(established - after - seen_loss)
            if lost:
                if drop_index is None:
                    drop_index = index
                for finger in lost:
                    drop_events.append({
                        "finger": finger,
                        "step": snapshots[index]["step"],
                        "time_s": snapshots[index]["time_s"],
                        "simultaneous_with": lost,
                        "contact_set_before": sorted(before),
                        "contact_set_after": sorted(after),
                        "finger_before": snapshots[index - 1]["finger"][finger],
                        "finger_after": snapshots[index]["finger"][finger],
                        "relative_linear_speed_m_s": snapshots[index]["relative_linear_speed_m_s"],
                        "relative_angular_speed_deg_s": snapshots[index]["relative_angular_speed_deg_s"],
                        "cube_displacement_m": snapshots[index]["cube_displacement_m"],
                    })
                seen_loss.update(lost)
                drop_index = index
                dropped = sorted(established - after)
                break
    pair_segment = []
    remaining = []
    if drop_index is not None and len(snapshots[drop_index]["contact_fingers"]) == 2:
        remaining = snapshots[drop_index]["contact_fingers"]
        for item in snapshots[drop_index:]:
            if set(item["contact_fingers"]) != set(remaining):
                break
            pair_segment.append(item)

    def force_stats(finger, key):
        values = [item["finger"][finger][key] for item in pair_segment if item["finger"][finger]["valid"]]
        return None if not values else {"min": float(np.min(values)), "mean": float(np.mean(values)), "max": float(np.max(values))}

    result = {
        "episode_id": episode_id,
        "trajectory_steps": len(snapshots),
        "return": float(total_reward),
        "first_three_contact_step": None if first_three is None else snapshots[first_three]["step"],
        "first_three_contact_time_s": None if first_three is None else snapshots[first_three]["time_s"],
        "first_three_contact_fingers": [] if first_three is None else snapshots[first_three]["contact_fingers"],
        "first_three_contact_snapshot": None if first_three is None else snapshots[first_three],
        "drop_step": None if drop_index is None else snapshots[drop_index]["step"],
        "drop_time_s": None if drop_index is None else snapshots[drop_index]["time_s"],
        "dropped_fingers": dropped,
        "pre_drop_window": [] if drop_index is None else snapshots[max(first_three, drop_index - 3):drop_index + 1],
        "remaining_fingers_after_drop": remaining,
        "drop_events": drop_events,
        "drop_is_simultaneous": bool(drop_events and len(drop_events[0]["simultaneous_with"]) > 1),
        "remaining_pair_frames": len(pair_segment),
        "remaining_pair_duration_s": max(0.0, (len(pair_segment) - 1) * env.control_dt),
        "remaining_pair_force_stats": {
            finger: {
                "normal_force_n": force_stats(finger, "normal_force_max_n"),
                "sliding_speed_m_s": force_stats(finger, "sliding_speed_max_m_s"),
                "tangential_force_n": force_stats(finger, "tangential_force_sum_n"),
            } for finger in remaining
        },
        "remaining_pair_contact_sequences": {
            finger: sorted({face for item in pair_segment for face in item["finger"][finger]["faces"]})
            for finger in remaining
        },
        "remaining_pair_relative_motion": {
            "linear_speed_m_s": None if not pair_segment else {
                "min": float(min(item["relative_linear_speed_m_s"] for item in pair_segment)),
                "mean": float(np.mean([item["relative_linear_speed_m_s"] for item in pair_segment])),
                "max": float(max(item["relative_linear_speed_m_s"] for item in pair_segment)),
            },
            "angular_speed_deg_s": None if not pair_segment else {
                "min": float(min(item["relative_angular_speed_deg_s"] for item in pair_segment)),
                "mean": float(np.mean([item["relative_angular_speed_deg_s"] for item in pair_segment])),
                "max": float(max(item["relative_angular_speed_deg_s"] for item in pair_segment)),
            },
        },
        "max_contact_fingers": max((item["contact_count"] for item in snapshots), default=0),
        "max_penetration_m": max((item["deepest_penetration_m"] for item in snapshots), default=0.0),
        "max_cube_displacement_m": max((item["cube_displacement_m"] for item in snapshots), default=0.0),
    }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5.json"))
    parser.add_argument("--source", type=Path, default=Path("outputs/wuji_teleop_analysis/cube_60_hand_trajectories.parquet"))
    parser.add_argument("--phase", type=Path, default=Path("outputs/expert_pca5/phase_metadata.json"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/expert_pca5"))
    parser.add_argument("--output", type=Path, default=Path("outputs/contact_topology"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    source = args.source if args.source.is_absolute() else root / args.source
    phase = args.phase if args.phase.is_absolute() else root / args.phase
    prior = args.prior if args.prior.is_absolute() else root / args.prior
    output = args.output if args.output.is_absolute() else root / args.output
    mean, basis, center, half = load_prior(prior)
    episodes = load_episodes(source, phase)
    results = [replay_episode(config, root, episode_id, data, mean, basis, center, half) for episode_id, data in episodes.items()]
    triplet_counts = {}
    drop_counts = {finger: 0 for finger in FINGERS}
    simultaneous_drop_count = 0
    pair_results = 0
    for item in results:
        triplet = tuple(item["first_three_contact_fingers"])
        if len(triplet) == 3:
            key = "+".join(triplet)
            triplet_counts[key] = triplet_counts.get(key, 0) + 1
        if item["drop_events"]:
            if item["drop_is_simultaneous"]:
                simultaneous_drop_count += 1
            for event in item["drop_events"]:
                drop_counts[event["finger"]] += 1
        if len(item["remaining_fingers_after_drop"]) == 2:
            pair_results += 1
    payload = {
        "method": "deterministic open-loop replay of expert cube_left phase q_hand through fixed-palm MuJoCo",
        "training": "none",
        "episodes": list(REPRESENTATIVE_EPISODES),
        "valid_contact_definition": "normal_force_n >= configs/rl/grasp_stage1_expert_pca5.json:success.min_normal_force_n",
        "results": results,
        "topology_summary": {
            "first_three_triplet_counts": triplet_counts,
            "first_loss_counts_by_finger": drop_counts,
            "simultaneous_first_loss_episodes": simultaneous_drop_count,
            "episodes_with_exact_two_contact_segment_after_drop": pair_results,
        },
        "interpretation": "Force/sliding/geometry are diagnostics. No new threshold is used to declare success.",
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        (output / "contact_topology_diagnosis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), "episodes": len(results)}, indent=2))
    except PermissionError:
        # Keep the full machine-readable result visible in restricted sandboxes.
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
