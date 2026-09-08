"""Compare current PCA5 policy with and without post-contact target latch.

The script accepts a PCA5 SB3 checkpoint.  In the present workspace the
previous checkpoint was not writable in the managed desktop sandbox, so the
default path trains the exact same fresh 5K seed-7 policy in memory and then
evaluates that *same* policy on the two environments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from openarm_wuji.rl import WujiStaticGraspEnv


def train_or_load(config: Path, root: Path, checkpoint: Path | None, seed: int, steps: int):
    if checkpoint is not None and checkpoint.exists():
        env = Monitor(WujiStaticGraspEnv.from_json(config, project_root=root))
        model = SAC.load(checkpoint, env=env, device="cpu")
        return model, env, "checkpoint"
    env = Monitor(WujiStaticGraspEnv.from_json(config, project_root=root))
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=50000, learning_starts=100,
        batch_size=64, tau=0.005, gamma=0.98, train_freq=1,
        gradient_steps=1, ent_coef="auto", policy_kwargs={"net_arch": [256, 256]},
        seed=seed, device="cpu", verbose=0,
    )
    model.learn(total_timesteps=steps, progress_bar=False)
    return model, env, "fresh_reconstruction"


def _safe_max(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite) if finite else None


def evaluate(model, config: Path, root: Path, seed: int, trace_dir: Path | None):
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    max_contacts = 0
    any_contact = False
    reward_sums = {
        "pre_latch": {},
        "post_latch": {},
    }
    traces = []
    gate_frames = 0
    established = False
    established_step = None
    established_count = None
    streak_ge2 = streak_ge3 = max_streak_ge2 = max_streak_ge3 = 0
    max_translation_drift = []
    max_rotation_drift = []
    max_linear_speed = []
    max_angular_speed = []
    max_penetration = 0.0
    max_cube_displacement = 0.0
    last_info = {}
    recent_trace = []
    post_trace_steps = 0
    try:
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            last_info = info
            count = int(info["contact_fingers"])
            any_contact |= count >= 1
            max_contacts = max(max_contacts, count)
            if count >= 3:
                gate_frames += 1
            else:
                gate_frames = 0
            if not established and gate_frames >= max(
                1, int(np.ceil(0.10 * env.control_hz))
            ):
                established = True
                established_step = step + 1
                established_count = count
            if count >= 2:
                streak_ge2 += 1
            else:
                streak_ge2 = 0
            if count >= 3:
                streak_ge3 += 1
            else:
                streak_ge3 = 0
            max_streak_ge2 = max(max_streak_ge2, streak_ge2)
            max_streak_ge3 = max(max_streak_ge3, streak_ge3)
            max_translation_drift.append(info.get("window_translation_drift_m", np.nan))
            max_rotation_drift.append(info.get("window_rotation_drift_deg", np.nan))
            max_linear_speed.append(info.get("relative_linear_speed_m_s", np.nan))
            max_angular_speed.append(info.get("relative_angular_speed_deg_s", np.nan))
            max_penetration = max(
                max_penetration, float(info["deepest_finger_cube_penetration_m"])
            )
            max_cube_displacement = max(
                max_cube_displacement, float(info["cube_displacement_m"])
            )
            phase = "post_latch" if established else "pre_latch"
            for name, value in info["reward_terms"].items():
                reward_sums[phase][name] = reward_sums[phase].get(name, 0.0) + float(value)
            # Keep a bounded event-window trace rather than a huge full episode dump.
            trace_row = {
                "step": step + 1,
                "time_s": (step + 1) * env.control_dt,
                "latent_action": np.asarray(action, dtype=float).tolist(),
                "q_target": np.asarray(info["desired_hand_target_rad"], dtype=float).tolist(),
                "requested_q_target": np.asarray(info["requested_hand_target_rad"], dtype=float).tolist(),
                "qpos": np.asarray(env.data.qpos[env.hand_qpos_ids], dtype=float).tolist(),
                "contact_count": count,
                "post_contact_latched": bool(info["post_contact_latched"]),
            }
            if not established:
                recent_trace.append(trace_row)
                recent_trace = recent_trace[-4:]
            if established and established_step == step + 1:
                traces.extend(recent_trace)
                post_trace_steps = 4
            elif post_trace_steps > 0:
                traces.append(trace_row)
                post_trace_steps -= 1
            if terminated or truncated:
                break
    finally:
        env.close()
    summary = {
        "seed": seed,
        "steps": step + 1,
        "return": float(total_return),
        "any_contact": bool(any_contact),
        "contacts_ge2": bool(max_streak_ge2 >= 1),
        "contacts_ge3": bool(max_streak_ge3 >= 1),
        "contact_established": bool(established),
        "established_time_s": None if established_step is None else established_step * env.control_dt,
        "contact_count_at_established": established_count,
        "hold_ge0p1s": bool(max_streak_ge2 >= int(np.ceil(0.1 * env.control_hz))),
        "hold_ge0p3s": bool(max_streak_ge2 >= int(np.ceil(0.3 * env.control_hz))),
        "hold_ge0p5s": bool(max_streak_ge2 >= int(np.ceil(0.5 * env.control_hz))),
        "hold_ge3_finger_0p1s": bool(max_streak_ge3 >= int(np.ceil(0.1 * env.control_hz))),
        "hold_ge3_finger_0p3s": bool(max_streak_ge3 >= int(np.ceil(0.3 * env.control_hz))),
        "hold_ge3_finger_0p5s": bool(max_streak_ge3 >= int(np.ceil(0.5 * env.control_hz))),
        "stable_grasp_success": bool(last_info.get("static_grasp_success", False)),
        "max_contact_fingers": int(max_contacts),
        "max_relative_translation_drift_m": _safe_max(max_translation_drift),
        "max_relative_rotation_drift_deg": _safe_max(max_rotation_drift),
        "max_relative_linear_speed_m_s": _safe_max(max_linear_speed),
        "max_relative_angular_speed_deg_s": _safe_max(max_angular_speed),
        "cube_displacement_m": float(max_cube_displacement),
        "deepest_penetration_m": float(max_penetration),
        "failure_reason": last_info.get("failure_reason"),
        "env_post_contact_latched": bool(last_info.get("post_contact_latched", False)),
        "reward_component_sums": reward_sums,
        "trace": traces,
    }
    if trace_dir is not None:
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"seed_{seed}.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except PermissionError:
            pass
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5.json"))
    parser.add_argument("--latch-config", type=Path, default=Path("configs/rl/grasp_stage1_expert_pca5_contact_latch.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contact_latch"))
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = args.baseline_config if args.baseline_config.is_absolute() else root / args.baseline_config
    latch = args.latch_config if args.latch_config.is_absolute() else root / args.latch_config
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    checkpoint = None if args.checkpoint is None else (
        args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    )
    model, training_env, source = train_or_load(baseline, root, checkpoint, 7, args.train_steps)
    results = {"baseline": [], "latch": []}
    try:
        for seed in range(7, 12):
            results["baseline"].append(evaluate(model, baseline, root, seed, output / "seed_7_11_timeseries" / "baseline"))
            results["latch"].append(evaluate(model, latch, root, seed, output / "seed_7_11_timeseries" / "latch"))
    finally:
        training_env.close()
    def aggregate(items):
        bool_fields = [
            "any_contact", "contacts_ge2", "contacts_ge3", "contact_established",
            "hold_ge0p1s", "hold_ge0p3s", "hold_ge0p5s",
            "hold_ge3_finger_0p1s", "hold_ge3_finger_0p3s", "hold_ge3_finger_0p5s",
            "stable_grasp_success",
        ]
        phase_sums = {"pre_latch": {}, "post_latch": {}}
        for item in items:
            for phase, terms in item["reward_component_sums"].items():
                for name, value in terms.items():
                    phase_sums[phase][name] = phase_sums[phase].get(name, 0.0) + float(value)
        return {
            "n": len(items),
            "counts": {field: sum(bool(item[field]) for item in items) for field in bool_fields},
            "max_contact_fingers": [item["max_contact_fingers"] for item in items],
            "established_time_s": [item["established_time_s"] for item in items],
            "returns": [item["return"] for item in items],
            "max_relative_translation_drift_m": [item["max_relative_translation_drift_m"] for item in items],
            "max_relative_rotation_drift_deg": [item["max_relative_rotation_drift_deg"] for item in items],
            "max_relative_linear_speed_m_s": [item["max_relative_linear_speed_m_s"] for item in items],
            "max_relative_angular_speed_deg_s": [item["max_relative_angular_speed_deg_s"] for item in items],
            "cube_displacement_m": [item["cube_displacement_m"] for item in items],
            "deepest_penetration_m": [item["deepest_penetration_m"] for item in items],
            "reward_component_sums": phase_sums,
        }
    payload = {
        "policy_source": source,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "training_steps_for_reconstruction": args.train_steps if source != "checkpoint" else None,
        "same_policy_and_reset_seeds": list(range(7, 12)),
        "baseline": aggregate(results["baseline"]),
        "latch": aggregate(results["latch"]),
        "per_seed": results,
        "interpretation": "The latch is a hybrid post-contact target hold; SAC remains responsible for acquisition.",
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        (output / "deterministic_baseline_vs_latch.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(json.dumps({"policy_source": source, "output": str(output), "baseline": payload["baseline"], "latch": payload["latch"]}, indent=2))
    except PermissionError:
        compact = {**payload, "per_seed": {
            mode: [
                {key: value for key, value in item.items() if key != "trace"}
                for item in items
            ] for mode, items in results.items()
        }}
        print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
