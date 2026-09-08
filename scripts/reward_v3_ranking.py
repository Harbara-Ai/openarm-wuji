"""Deterministic Reward V2/V3 ranking before any Reward V3 training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from stable_baselines3 import SAC

from openarm_wuji.rl import WujiStaticGraspEnv
from openarm_wuji.rl.grasp_env import (
    bounded_contact_slip_penalty,
    contact_interior_score,
    contact_persistence_score,
)
from openarm_wuji.teleop.synergies import HandSynergyMapper


EPISODES = (69, 75, 87, 77, 84, 74)


def load_prior(path: Path):
    mean = np.load(path / "mean.npy")
    basis = np.load(path / "basis.npy")
    scaling = json.loads((path / "latent_scaling.json").read_text(encoding="utf-8"))
    return mean, basis, np.asarray(scaling["latent_center"]), np.asarray(scaling["latent_half_range"])


def load_episodes(source: Path, phase_path: Path):
    table = pq.read_table(source).to_pydict()
    phases = json.loads(phase_path.read_text(encoding="utf-8"))
    by_id = {int(item["episode_id"]): item for item in phases["episodes"]}
    episode_col = np.asarray(table["episode_index"])
    frame_col = np.asarray(table["frame_index"])
    rows = {}
    for episode_id in EPISODES:
        indices = np.flatnonzero(episode_col == episode_id)
        indices = indices[np.argsort(frame_col[indices])]
        phase = by_id[episode_id]
        end = min(int(phase["end_frame_exclusive"]), int(phase["start_frame"]) + 120)
        indices = indices[frame_col[indices] < end]
        rows[episode_id] = np.asarray([table["q_hand"][index] for index in indices], dtype=float)
    return rows


def latent(q, mean, basis, center, half):
    return np.clip((((q - mean) @ basis) - center) / half, -1.0, 1.0)


def max_run(mask) -> int:
    result = current = 0
    for value in mask:
        current = current + 1 if value else 0
        result = max(result, current)
    return result


def run_sequence(config: Path, root: Path, label: str, targets, actions):
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    rewards = []
    components = {}
    contact_counts = []
    edge_margins = []
    slip_speeds = []
    persistence = []
    interior = []
    per_finger_frames = {f"finger{i}": 0 for i in range(1, 6)}
    last = {}
    try:
        env.reset(seed=7)
        for target, action in zip(targets, actions):
            _, reward, terminated, truncated, info = env.step_absolute_target(
                target, reward_action=action
            )
            last = info
            rewards.append(float(reward))
            contact_counts.append(int(info["contact_fingers"]))
            quality = info["contact_quality"]
            persistence.append(float(quality["persistence_score"]))
            interior.append(float(quality["contact_interior_reward"]))
            for finger, item in quality["per_finger"].items():
                if item["valid_contact"]:
                    per_finger_frames[finger] += 1
                    edge_margins.append(float(item["edge_margin_m"]))
                    slip_speeds.append(float(item["tangential_speed_m_s"]))
            for name, value in info["reward_terms"].items():
                components[name] = components.get(name, 0.0) + float(value)
            if terminated or truncated:
                break
    finally:
        env.close()
    counts = np.asarray(contact_counts)
    return {
        "label": label,
        "reward_version": json.loads(config.read_text(encoding="utf-8"))["reward_version"],
        "steps": len(rewards),
        "return": float(np.sum(rewards)),
        "max_contacts": int(np.max(counts)) if len(counts) else 0,
        "duration_ge2_s": float(np.sum(counts >= 2) / 30.0),
        "duration_ge3_s": float(np.sum(counts >= 3) / 30.0),
        "max_contiguous_ge2_s": float(max_run(counts >= 2) / 30.0),
        "max_contiguous_ge3_s": float(max_run(counts >= 3) / 30.0),
        "mean_edge_margin_m": None if not edge_margins else float(np.mean(edge_margins)),
        "median_edge_margin_m": None if not edge_margins else float(np.median(edge_margins)),
        "interior_contact_fraction": float(np.mean(np.asarray(edge_margins) >= 0.003)) if edge_margins else 0.0,
        "mean_tangential_slip_m_s": float(np.mean(slip_speeds)) if slip_speeds else 0.0,
        "max_tangential_slip_m_s": float(np.max(slip_speeds)) if slip_speeds else 0.0,
        "mean_persistence_score": float(np.mean(persistence)) if persistence else 0.0,
        "mean_contact_interior_reward": float(np.mean(interior)) if interior else 0.0,
        "per_finger_contact_duration_s": {
            finger: frames / 30.0 for finger, frames in per_finger_frames.items()
        },
        "stable_grasp_success": bool(last.get("static_grasp_success", False)),
        "reward_component_sums": components,
    }


def record_policy_targets(checkpoint: Path, config: Path, root: Path):
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    model = SAC.load(checkpoint, device="cpu")
    targets, actions = [], []
    try:
        observation, _ = env.reset(seed=7)
        for _ in range(env.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            targets.append(np.asarray(info["requested_hand_target_rad"], dtype=float))
            actions.append(np.asarray(action, dtype=float))
            if terminated or truncated:
                break
    finally:
        env.close()
    return np.asarray(targets), np.asarray(actions)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--v2-config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5.json"))
    parser.add_argument("--v3-config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5_reward_v3.json"))
    parser.add_argument("--source", type=Path, default=Path("outputs/wuji_teleop_analysis/cube_60_hand_trajectories.parquet"))
    parser.add_argument("--phase", type=Path, default=Path("outputs/expert_pca5/phase_metadata.json"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/expert_pca5"))
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/reward_v3_contact_quality/reward_ranking.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda p: p if p.is_absolute() else root / p
    v2, v3, source, phase, prior = map(resolve, (args.v2_config, args.v3_config, args.source, args.phase, args.prior))
    mean, basis, center, half = load_prior(prior)
    episodes = load_episodes(source, phase)
    mapper = HandSynergyMapper.from_json(root / "configs/wuji_hand_left_synergies.json")
    probes = []
    zeros = np.zeros((120, 5))
    probes.append(("no_contact_open_hold", np.repeat(mapper.open_pose[None], 120, axis=0), zeros))
    progress = np.linspace(0.0, 1.0, 120)
    probes.append(("scripted_coordinated_closing", np.asarray([
        mapper.open_pose + value * (mapper.close_pose - mapper.open_pose) for value in progress
    ]), zeros))
    thumb_target = mapper.open_pose.copy()
    thumb_target[:4] = mapper.close_pose[:4]
    probes.append(("thumb_only_hold_probe", np.repeat(thumb_target[None], 120, axis=0), zeros))
    for episode_id, q in episodes.items():
        actions = np.asarray([latent(row, mean, basis, center, half) for row in q])
        reconstructed = np.asarray([mean + basis @ (center + half * action) for action in actions])
        probes.append((f"raw_expert_ep{episode_id}", q, actions))
        probes.append((f"pca5_expert_ep{episode_id}", reconstructed, actions))
    policy_status = "checkpoint_not_provided"
    if args.policy_checkpoint is not None:
        checkpoint = resolve(args.policy_checkpoint)
        targets, actions = record_policy_targets(checkpoint, v2, root)
        probes.append(("pca5_sac_v2_5k_deterministic_seed7", targets, actions))
        policy_status = "loaded"
    results = []
    for label, targets, actions in probes:
        results.append(run_sequence(v2, root, label, targets, actions))
        results.append(run_sequence(v3, root, label, targets, actions))
    pairs = {}
    for row in results:
        pairs.setdefault(row["label"], {})[row["reward_version"]] = row
    multi = [pair for label, pair in pairs.items()
             if label not in {"no_contact_open_hold", "thumb_only_hold_probe"}
             and pair["v3"]["max_contacts"] >= 2]
    best_quality = max(multi, key=lambda pair: (
        (pair["v3"]["mean_edge_margin_m"] or 0.0)
        - pair["v3"]["mean_tangential_slip_m_s"]
    ))
    unstable = pairs["pca5_expert_ep74"]
    no_contact = pairs["no_contact_open_hold"]
    thumb_only = pairs["thumb_only_hold_probe"]
    contact_rich_best_return = max(pair["v3"]["return"] for pair in multi)
    v2_returns = np.asarray([pair["v2"]["return"] for pair in pairs.values()])
    v3_returns = np.asarray([pair["v3"]["return"] for pair in pairs.values()])
    scale_ratio = float(np.median(np.abs(v3_returns)) / max(np.median(np.abs(v2_returns)), 1e-9))
    def quality_delta(pair):
        return float(pair["v3"]["return"] - pair["v2"]["return"])

    stable_local = (
        0.30 * contact_interior_score([0.006] * 3, sigma_m=0.003, target_fingers=3)
        + 0.30 * contact_persistence_score([0.30] * 3, target_duration_s=0.30, target_fingers=3)
        - 0.20 * bounded_contact_slip_penalty([0.001] * 3, sigma_m_s=0.010)
    )
    unstable_local = (
        0.30 * contact_interior_score([0.00015] * 3, sigma_m=0.003, target_fingers=3)
        + 0.30 * contact_persistence_score([1.0 / 30.0] * 3, target_duration_s=0.30, target_fingers=3)
        - 0.20 * bounded_contact_slip_penalty([0.016] * 3, sigma_m_s=0.010)
    )
    thumb_only_local = (
        0.30 * contact_interior_score([0.006], sigma_m=0.003, target_fingers=3)
        + 0.30 * contact_persistence_score([0.30], target_duration_s=0.30, target_fingers=3)
        - 0.20 * bounded_contact_slip_penalty([0.001], sigma_m_s=0.010)
    )
    checks = {
        "contact_beats_no_contact": bool(contact_rich_best_return > no_contact["v3"]["return"]),
        "quality_increment_beats_ep74_unstable": bool(
            quality_delta(best_quality) > quality_delta(unstable)
        ),
        "matched_stable_quality_beats_edge_slip": bool(stable_local > unstable_local),
        "matched_multicontact_beats_thumb_only": bool(stable_local > thumb_only_local),
        "multicontact_beats_thumb_only": bool(contact_rich_best_return > thumb_only["v3"]["return"]),
        "reward_scale_same_order": bool(0.5 <= scale_ratio <= 2.0),
    }
    payload = {
        "policy_checkpoint_status": policy_status,
        "ranking_complete": policy_status == "loaded",
        "same_physics_different_reward": True,
        "results": results,
        "pairwise": pairs,
        "best_quality_label": best_quality["v3"]["label"],
        "unstable_reference": "pca5_expert_ep74",
        "cross_trajectory_total_return_diagnostic": {
            "best_quality_total_exceeds_ep74": bool(
                best_quality["v3"]["return"] > unstable["v3"]["return"]
            ),
            "note": (
                "Not a gate: different trajectories have different Reward V2 contact-transition "
                "and coverage totals. The matched local quality term and V3-minus-V2 increment "
                "are the controlled ranking checks."
            ),
        },
        "matched_local_quality": {
            "three_finger_interior_low_slip": stable_local,
            "three_finger_edge_high_slip": unstable_local,
            "thumb_only_interior_low_slip": thumb_only_local,
        },
        "median_abs_return_v3_over_v2": scale_ratio,
        "checks": checks,
        "pass": bool(all(checks.values()) and policy_status == "loaded"),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "pass": payload["pass"],
        "ranking_complete": payload["ranking_complete"], "checks": checks,
        "best_quality_label": payload["best_quality_label"],
        "median_abs_return_v3_over_v2": scale_ratio,
    }, indent=2))


if __name__ == "__main__":
    main()
