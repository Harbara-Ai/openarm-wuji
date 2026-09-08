from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from openarm_wuji.rl import WujiStaticGraspEnv


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Diagnose a trained grasp SAC policy")
    parser.add_argument("model", type=Path)
    parser.add_argument("--config", type=Path,
                        default=root / "configs/rl/grasp_stage1.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/rl_grasp_stage1/sac_5000_eval.json")
    args = parser.parse_args()

    model = SAC.load(args.model)
    env = WujiStaticGraspEnv.from_json(args.config, project_root=root)
    episodes = []
    try:
        for seed in range(args.seed, args.seed + args.seeds):
            observation, _ = env.reset(seed=seed)
            initial_qpos = env.data.qpos[env.hand_qpos_ids].copy()
            initial_distances = env._finger_cube_distances()
            actions = []
            reward_terms = []
            max_contacts = 0
            info = {}
            for step in range(env.max_episode_steps):
                action, _ = model.predict(observation, deterministic=True)
                actions.append(action.copy())
                observation, _, terminated, truncated, info = env.step(action)
                reward_terms.append(info["reward_terms"])
                max_contacts = max(max_contacts, int(info["contact_fingers"]))
                if terminated or truncated:
                    break
            actions_array = np.asarray(actions)
            final_qpos = env.data.qpos[env.hand_qpos_ids].copy()
            episodes.append({
                "seed": seed,
                "steps": step + 1,
                "success": bool(info["static_grasp_success"]),
                "max_contact_fingers": max_contacts,
                "mean_abs_action": float(np.mean(np.abs(actions_array))),
                "max_abs_action": float(np.max(np.abs(actions_array))),
                "per_joint_mean_action": np.mean(actions_array, axis=0).tolist(),
                "hand_qpos_l2_change_rad": float(np.linalg.norm(
                    final_qpos - initial_qpos
                )),
                "initial_finger_distances_m": initial_distances,
                "final_finger_distances_m": env._finger_cube_distances(),
                "mean_reward_terms": {
                    key: float(np.mean([row[key] for row in reward_terms]))
                    for key in reward_terms[0]
                },
                "cube_displacement_m": float(info["cube_displacement_m"]),
                "failure_reason": info["failure_reason"],
            })
        report = {"model": str(args.model), "episodes": episodes}
        print(json.dumps(report, indent=2))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
