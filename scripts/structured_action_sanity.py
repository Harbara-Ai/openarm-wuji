from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from openarm_wuji.rl import WujiStaticGraspEnv


def tip_pad_distances(env: WujiStaticGraspEnv) -> dict[str, dict[str, float]]:
    import mujoco

    result = {}
    from_to = np.zeros(6)
    for finger in (f"finger{index}" for index in range(1, 6)):
        tip = env.fingertips[finger].tip_geom_id
        pad = env.pad_geom_ids[finger]
        result[finger] = {
            "tip": float(mujoco.mj_geomDistance(
                env.model, env.data, tip, env.cube_geom_id, 0.25, from_to
            )),
            "pad": float(mujoco.mj_geomDistance(
                env.model, env.data, pad, env.cube_geom_id, 0.25, from_to
            )),
        }
    return result


def finger_joint_rows(env: WujiStaticGraspEnv, values: np.ndarray) -> dict:
    return {
        f"finger{index + 1}": {
            name: float(value)
            for name, value in zip(
                env.hand_joint_names[4 * index:4 * (index + 1)],
                values[4 * index:4 * (index + 1)],
            )
        }
        for index in range(5)
    }


def all_finger_test(env: WujiStaticGraspEnv, seed: int) -> dict:
    env.reset(seed=seed)
    initial_qpos = env.data.qpos[env.hand_qpos_ids].copy()
    initial_distance = tip_pad_distances(env)
    distance_history = {
        finger: {role: [value] for role, value in roles.items()}
        for finger, roles in initial_distance.items()
    }
    contact_curve = []
    deepest_penetration = 0.0
    info = {}
    for step in range(env.max_episode_steps):
        _, _, terminated, truncated, info = env.step(
            np.ones(5, dtype=np.float32)
        )
        contact_curve.append(int(info["contact_fingers"]))
        deepest_penetration = max(
            deepest_penetration,
            float(info["deepest_finger_cube_penetration_m"]),
        )
        current = tip_pad_distances(env)
        for finger, roles in current.items():
            for role, value in roles.items():
                distance_history[finger][role].append(value)
        if terminated or truncated:
            break
    final_qpos = env.data.qpos[env.hand_qpos_ids].copy()
    return {
        "action": [1.0] * 5,
        "steps": step + 1,
        "joint_movement_rad": finger_joint_rows(
            env, final_qpos - initial_qpos
        ),
        "tip_pad_distance_m": {
            finger: {
                role: {
                    "initial": values[0],
                    "minimum": float(np.min(values)),
                    "final": values[-1],
                }
                for role, values in roles.items()
            }
            for finger, roles in distance_history.items()
        },
        "contact_count_curve": contact_curve,
        "max_simultaneous_contacts": max(contact_curve, default=0),
        "final_contacts": int(info.get("contact_fingers", 0)),
        "deepest_penetration_m": deepest_penetration,
        "failure_reason": info.get("failure_reason"),
        "success": bool(info.get("static_grasp_success", False)),
    }


def isolated_finger_tests(env: WujiStaticGraspEnv, seed: int) -> list[dict]:
    results = []
    for finger_index in range(5):
        env.reset(seed=seed)
        qpos_before = env.data.qpos[env.hand_qpos_ids].copy()
        action = np.zeros(5, dtype=np.float32)
        action[finger_index] = 1.0
        _, _, _, _, info = env.step(action)
        command_delta = np.asarray(info["applied_joint_delta_rad"])
        actual_delta = (
            env.data.qpos[env.hand_qpos_ids].copy() - qpos_before
        )
        section = slice(4 * finger_index, 4 * (finger_index + 1))
        outside = np.ones(20, dtype=bool)
        outside[section] = False
        isolated = bool(
            np.max(np.abs(command_delta[outside])) <= 1e-12
            and np.max(np.abs(command_delta[section])) > 0.0
        )
        results.append({
            "action": action.tolist(),
            "finger": f"finger{finger_index + 1}",
            "isolated_command_passed": isolated,
            "commanded_joint_delta_rad": finger_joint_rows(env, command_delta),
            "actual_joint_movement_rad": finger_joint_rows(env, actual_delta),
            "contact_fingers": int(info["contact_fingers"]),
            "deepest_penetration_m": float(
                info["deepest_finger_cube_penetration_m"]
            ),
        })
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate the 5-D per-finger structured grasp action"
    )
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/rl/grasp_stage1_structured5.json",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/rl_grasp_stage1/structured5_sanity.json",
    )
    args = parser.parse_args()

    env = WujiStaticGraspEnv.from_json(args.config, project_root=root)
    try:
        check_env(env, skip_render_check=True)
        test_a = all_finger_test(env, args.seed)
        test_b = isolated_finger_tests(env, args.seed)
        report = {
            "seed": args.seed,
            "observation_dimension": env.observation_dim,
            "action_dimension": env.ACTION_DIM,
            "action_representation": env.action_representation,
            "direction_source": env.config["action"]["direction_source"],
            "normalized_closing_directions": env.structured_directions.tolist(),
            "per_finger_delta_norm_scale_rad": (
                env.structured_scales_rad.tolist()
            ),
            "gymnasium_check_passed": True,
            "test_a_all_fingers_close": test_a,
            "test_b_isolated_fingers": test_b,
            "sanity_passed": bool(
                test_a["max_simultaneous_contacts"] >= 2
                and all(row["isolated_command_passed"] for row in test_b)
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not report["sanity_passed"]:
            raise SystemExit("structured action sanity test failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
