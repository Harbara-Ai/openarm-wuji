from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from openarm_wuji.rl import WujiStaticGraspEnv


class LearningSignalCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards: list[float] = []
        self.max_contact_fingers = 0
        self.successes = 0
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_max_contacts: list[int] = []
        self.episode_failure_reasons: list[str | None] = []
        self.episode_reward_component_sums: list[dict[str, float]] = []
        self.episode_milestones: list[dict[str, bool]] = []
        self.third_closest_distance_curve_m: list[float] = []
        self.contact_count_curve: list[int] = []
        self.coverage_potential_curve: list[float] = []
        self.finger_distance_curves_m = {
            f"finger{index}": [] for index in range(1, 6)
        }
        self.actions: list[np.ndarray] = []
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_max_contacts = 0
        self._reward_component_sums: dict[str, float] = {}
        self._milestones = self._empty_milestones()

    @staticmethod
    def _empty_milestones() -> dict[str, bool]:
        return {
            "multi_contact_acquired": False,
            "three_contact_acquired": False,
            "contact_held_0p1s": False,
            "contact_held_0p3s": False,
            "stable_grasp_success": False,
        }

    def _on_step(self) -> bool:
        rewards = [float(value) for value in self.locals["rewards"]]
        self.rewards.extend(rewards)
        actions = np.asarray(self.locals["actions"])
        for index, info in enumerate(self.locals["infos"]):
            self.actions.append(actions[index].copy())
            self._episode_return += rewards[index]
            self._episode_length += 1
            contacts = int(info.get("contact_fingers", 0))
            self.max_contact_fingers = max(
                self.max_contact_fingers, contacts
            )
            self._episode_max_contacts = max(self._episode_max_contacts, contacts)
            self.successes += int(info.get("static_grasp_success", False))
            terms = info["reward_terms"]
            self.third_closest_distance_curve_m.append(float(
                terms["third_closest_finger_distance_m"]
            ))
            self.coverage_potential_curve.append(float(terms["phi_coverage"]))
            self.contact_count_curve.append(contacts)
            for finger, distance in info["finger_distances_m"].items():
                self.finger_distance_curves_m[finger].append(float(distance))
            for name, value in terms.items():
                self._reward_component_sums[name] = (
                    self._reward_component_sums.get(name, 0.0) + float(value)
                )
            for name in self._milestones:
                key = "static_grasp_success" if name == "stable_grasp_success" else name
                self._milestones[name] = (
                    self._milestones[name] or bool(info.get(key, False))
                )
            if bool(self.locals["dones"][index]):
                self.episode_returns.append(self._episode_return)
                self.episode_lengths.append(self._episode_length)
                self.episode_max_contacts.append(self._episode_max_contacts)
                self.episode_failure_reasons.append(info.get("failure_reason"))
                self.episode_reward_component_sums.append(
                    self._reward_component_sums.copy()
                )
                self.episode_milestones.append(self._milestones.copy())
                self._episode_return = 0.0
                self._episode_length = 0
                self._episode_max_contacts = 0
                self._reward_component_sums = {}
                self._milestones = self._empty_milestones()
        return True


def parameter_vector(model: SAC) -> torch.Tensor:
    return torch.cat([
        parameter.detach().cpu().flatten() for parameter in model.actor.parameters()
    ])


def entropy_coefficient(model: SAC) -> float:
    if model.log_ent_coef is not None:
        return float(torch.exp(model.log_ent_coef.detach()).cpu())
    return float(model.ent_coef_tensor.detach().cpu())


def action_statistics(actions: np.ndarray) -> dict:
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] not in {5, 20}:
        raise ValueError("expected a batch of 5-D or 20-D grasp actions")
    if actions.shape[1] == 5:
        per_finger = {
            f"finger{finger + 1}": float(np.mean(np.abs(actions[:, finger])))
            for finger in range(5)
        }
    else:
        per_finger = {
            f"finger{finger + 1}": float(np.mean(np.abs(
                actions[:, 4 * finger:4 * (finger + 1)]
            ))) for finger in range(5)
        }
    return {
        "per_dimension_mean": np.mean(actions, axis=0).tolist(),
        "per_dimension_std": np.std(actions, axis=0).tolist(),
        "per_finger_mean_absolute": per_finger,
        "global_mean_absolute": float(np.mean(np.abs(actions))),
    }


def policy_distribution_diagnostics(model: SAC, config: Path, root: Path,
                                    seed: int) -> dict:
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    try:
        observation, _ = env.reset(seed=seed)
        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            _, log_std, _ = model.actor.get_action_dist_params(observation_tensor)
        deterministic_action, _ = model.predict(observation, deterministic=True)
        stochastic_actions = np.asarray([
            model.predict(observation, deterministic=False)[0]
            for _ in range(128)
        ])
        return {
            "seed": seed,
            "latent_gaussian_std": torch.exp(log_std).cpu().numpy()[0].tolist(),
            "deterministic_initial_action": deterministic_action.tolist(),
            "stochastic_initial_action": action_statistics(stochastic_actions),
        }
    finally:
        env.close()


def evaluate(model: SAC, config: Path, root: Path, seed: int,
             count: int) -> list[dict]:
    env = WujiStaticGraspEnv.from_json(config, project_root=root)
    episodes = []
    try:
        for evaluation_seed in range(seed, seed + count):
            observation, _ = env.reset(seed=evaluation_seed)
            total_reward = 0.0
            max_contacts = 0
            milestones = {
                "multi_contact_acquired": False,
                "three_contact_acquired": False,
                "contact_held_0p1s": False,
                "contact_held_0p3s": False,
            }
            actions = []
            distance_history = {f"finger{index}": [] for index in range(1, 6)}
            info = {}
            for step in range(env.max_episode_steps):
                action, _ = model.predict(observation, deterministic=True)
                actions.append(action.copy())
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                max_contacts = max(max_contacts, int(info["contact_fingers"]))
                for name in milestones:
                    milestones[name] = milestones[name] or bool(info[name])
                for finger, distance in info["finger_distances_m"].items():
                    distance_history[finger].append(float(distance))
                if terminated or truncated:
                    break
            episodes.append({
                "seed": evaluation_seed,
                "steps": step + 1,
                "return": total_reward,
                "max_contact_fingers": max_contacts,
                "final_contact_fingers": int(info["contact_fingers"]),
                "success": bool(info["static_grasp_success"]),
                "rigid_success_diagnostic": bool(
                    info["rigid_success_diagnostic"]
                ),
                "failure_reason": info["failure_reason"],
                "cube_displacement_m": float(info["cube_displacement_m"]),
                **milestones,
                "third_closest_final_distance_m": float(
                    info["reward_terms"]["third_closest_finger_distance_m"]
                ),
                "per_finger_distance_m": {
                    finger: {
                        "mean": float(np.mean(values)),
                        "minimum": float(np.min(values)),
                        "final": values[-1],
                    } for finger, values in distance_history.items()
                },
                "action_statistics": action_statistics(np.asarray(actions)),
            })
    finally:
        env.close()
    return episodes


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train stage-1 Wuji grasp with SAC")
    parser.add_argument("--config", type=Path,
                        default=root / "configs/rl/grasp_stage1.json")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/rl_grasp_stage1")
    args = parser.parse_args()

    env = Monitor(WujiStaticGraspEnv.from_json(args.config, project_root=root))
    environment_config = json.loads(args.config.read_text(encoding="utf-8"))
    reward_version = environment_config["reward_version"]
    action_version = environment_config.get(
        "action_version", f"action{env.action_space.shape[0]}"
    )
    hyperparameters = {
        "learning_rate": 3e-4,
        "buffer_size": 50000,
        "learning_starts": 100,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.98,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "policy_net_arch": [256, 256],
    }
    model = SAC(
        "MlpPolicy", env,
        learning_rate=hyperparameters["learning_rate"],
        buffer_size=hyperparameters["buffer_size"],
        learning_starts=hyperparameters["learning_starts"],
        batch_size=hyperparameters["batch_size"],
        tau=hyperparameters["tau"], gamma=hyperparameters["gamma"],
        train_freq=hyperparameters["train_freq"],
        gradient_steps=hyperparameters["gradient_steps"],
        ent_coef=hyperparameters["ent_coef"],
        policy_kwargs={"net_arch": hyperparameters["policy_net_arch"]},
        seed=args.seed, device=args.device, verbose=1,
    )
    before = parameter_vector(model)
    initial_entropy_coefficient = entropy_coefficient(model)
    callback = LearningSignalCallback()
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    progress_bar=False)
        after = parameter_vector(model)
        window = min(250, len(callback.rewards))
        evaluation = evaluate(
            model, args.config, root, args.seed, args.eval_seeds
        )
        training_actions = np.asarray(callback.actions)
        milestone_counts = {
            name: sum(int(episode[name]) for episode in callback.episode_milestones)
            for name in callback._empty_milestones()
        }
        report = {
            "algorithm": "Stable-Baselines3 SAC",
            "reward_version": reward_version,
            "action_version": action_version,
            "action_representation": env.unwrapped.action_representation,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "device": str(model.device),
            "observation_dimension": int(env.observation_space.shape[0]),
            "action_dimension": int(env.action_space.shape[0]),
            "action_mapping": list(env.unwrapped.action_mapping),
            "hyperparameters": hyperparameters,
            "replay_buffer_size": int(model.replay_buffer.size()),
            "gradient_updates": int(model._n_updates),
            "actor_parameter_l2_change": float(torch.linalg.vector_norm(after - before)),
            "first_reward_window_mean": float(np.mean(callback.rewards[:window])),
            "last_reward_window_mean": float(np.mean(callback.rewards[-window:])),
            "max_contact_fingers": callback.max_contact_fingers,
            "success_events": callback.successes,
            "initial_entropy_coefficient": initial_entropy_coefficient,
            "final_entropy_coefficient": entropy_coefficient(model),
            "training_action_statistics": action_statistics(training_actions),
            "policy_distribution_diagnostics": policy_distribution_diagnostics(
                model, args.config, root, args.seed
            ),
            "completed_training_episodes": len(callback.episode_returns),
            "training_episode_returns": callback.episode_returns,
            "training_episode_lengths": callback.episode_lengths,
            "training_episode_max_contact_fingers": (
                callback.episode_max_contacts
            ),
            "training_episode_failure_reasons": (
                callback.episode_failure_reasons
            ),
            "training_episode_reward_component_sums": (
                callback.episode_reward_component_sums
            ),
            "training_episode_milestones": callback.episode_milestones,
            "training_milestone_episode_counts": milestone_counts,
            "contact_episode_fraction": float(np.mean(
                np.asarray(callback.episode_max_contacts) >= 1
            )),
            "multi_contact_episode_fraction": float(np.mean(
                np.asarray(callback.episode_max_contacts) >= 2
            )),
            "three_contact_episode_fraction": float(np.mean(
                np.asarray(callback.episode_max_contacts) >= 3
            )),
            "third_closest_distance_curve_m": (
                callback.third_closest_distance_curve_m
            ),
            "contact_count_curve": callback.contact_count_curve,
            "coverage_potential_curve": callback.coverage_potential_curve,
            "finger_distance_curves_m": callback.finger_distance_curves_m,
            "deterministic_evaluation": evaluation,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        model_path = args.output / (
            f"sac_grasp_stage1_{reward_version}_{action_version}_"
            f"{args.timesteps}_steps"
        )
        report_path = args.output / (
            f"sac_{reward_version}_{action_version}_{args.timesteps}_steps.json"
        )
        try:
            model.save(model_path)
            report["model_saved"] = True
            report["model_path"] = str(model_path.with_suffix(".zip"))
        except PermissionError as error:
            # The desktop sandbox may deny binary artifact creation while
            # allowing source/report patches.  Keep the metrics auditable.
            report["model_saved"] = False
            report["model_save_error"] = str(error)
        try:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["report_saved"] = True
            report["report_path"] = str(report_path)
        except PermissionError as error:
            report["report_saved"] = False
            report["report_save_error"] = str(error)
        if report.get("report_saved", False):
            print(json.dumps({
                "action_version": action_version,
                "timesteps": args.timesteps,
                "max_contact_fingers": report["max_contact_fingers"],
                "success_events": report["success_events"],
                "deterministic_evaluation": report["deterministic_evaluation"],
                "report_path": report["report_path"],
            }, indent=2))
        else:
            print(json.dumps({
                "action_version": action_version,
                "timesteps": args.timesteps,
                "max_contact_fingers": report["max_contact_fingers"],
                "success_events": report["success_events"],
                "final_entropy_coefficient": report["final_entropy_coefficient"],
                "training_action_statistics": report["training_action_statistics"],
                "deterministic_evaluation": report["deterministic_evaluation"],
                "model_saved": report["model_saved"],
                "report_saved": report["report_saved"],
            }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
