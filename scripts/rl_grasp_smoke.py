from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from openarm_wuji.rl import WujiStaticGraspEnv


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check and randomly step stage-1 grasp RL")
    parser.add_argument("--config", type=Path,
                        default=root / "configs/rl/grasp_stage1.json")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    env = WujiStaticGraspEnv.from_json(args.config, project_root=root)
    try:
        check_env(env, skip_render_check=True)
        rng = np.random.default_rng(args.seed)
        episodes = []
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            max_contacts = 0
            terminal_info = {}
            for step in range(env.max_episode_steps):
                action = rng.uniform(-1.0, 1.0, env.ACTION_DIM).astype(np.float32)
                observation, reward, terminated, truncated, terminal_info = env.step(action)
                total_reward += reward
                max_contacts = max(max_contacts, int(terminal_info["contact_fingers"]))
                if terminated or truncated:
                    break
            episodes.append({
                "seed": args.seed + episode,
                "steps": step + 1,
                "return": total_reward,
                "max_contact_fingers": max_contacts,
                "success": bool(terminal_info.get("static_grasp_success", False)),
                "failure_reason": terminal_info.get("failure_reason"),
                "observation_finite": bool(np.isfinite(observation).all()),
                "reset_cube_displacement_m": float(
                    reset_info["reset_cube_displacement_m"]
                ),
            })
        # A direction-only closing probe checks that contact is reachable from
        # the reset.  It still sends a 20-D independent-joint action and is not
        # an expert, a training initializer, or a success claim.
        observation, _ = env.reset(seed=args.seed)
        close_direction = np.sign(
            env.robot.mapper.close_pose - env.robot.mapper.open_pose
        ).astype(np.float32)
        probe_return = 0.0
        probe_max_contacts = 0
        first_contact_step = None
        probe_info = {}
        for probe_step in range(env.max_episode_steps):
            observation, reward, terminated, truncated, probe_info = env.step(
                close_direction
            )
            probe_return += reward
            contact_fingers = int(probe_info["contact_fingers"])
            probe_max_contacts = max(probe_max_contacts, contact_fingers)
            if contact_fingers and first_contact_step is None:
                first_contact_step = probe_step + 1
            if terminated or truncated:
                break
        report = {
            "environment": "WujiStaticGraspEnv-v0",
            "reward_version": env.config["reward_version"],
            "observation_dimension": env.observation_dim,
            "action_dimension": env.ACTION_DIM,
            "gymnasium_check_passed": True,
            "episodes": episodes,
            "direction_only_contact_reachability_probe": {
                "seed": args.seed,
                "steps": probe_step + 1,
                "return": probe_return,
                "first_contact_step": first_contact_step,
                "max_contact_fingers": probe_max_contacts,
                "final_contact_fingers": int(probe_info["contact_fingers"]),
                "success": bool(probe_info["static_grasp_success"]),
                "failure_reason": probe_info["failure_reason"],
                "deepest_penetration_m": float(
                    probe_info["deepest_finger_cube_penetration_m"]
                ),
            },
        }
        output = args.output or root / (
            "outputs/rl_grasp_stage1/"
            f"random_smoke_{env.config['reward_version']}.json"
        )
        print(json.dumps(report, indent=2))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
