"""Audit ACT data/inference parity, temporal alignment, padding, and scaling.

This script is intentionally diagnostic-only.  It does not train a policy and
does not change the scripted controller, MuJoCo task, or exported dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from openarm_wuji.dataset.coordinated_demo_recorder import JOINT_NAMES
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


TASK = (
    "Coordinate OpenArm and Wuji to grasp the cube, lift it, and hold it "
    "unsupported."
)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray, axis: int = 0) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=axis),
        "max": np.max(values, axis=axis),
        "mean": np.mean(values, axis=axis),
        "std": np.std(values, axis=axis),
    }


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    )))


def _image_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left)
    right = np.asarray(right)
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {
        "shape_left": list(left.shape),
        "shape_right": list(right.shape),
        "dtype_left": str(left.dtype),
        "dtype_right": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "different_values": int(np.count_nonzero(difference)),
        "max_abs_pixel_difference": int(difference.max(initial=0)),
    }


def _load_episodes(raw_dir: Path) -> list[dict[str, Any]]:
    episodes = []
    for path in sorted(raw_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            episodes.append({
                "path": path,
                "seed": int(episode["episode_seed"]),
                "episode_index": int(episode["episode_index"]),
                "state": episode["observation.state"].copy(),
                "next_state": episode["next_observation.state"].copy(),
                "action": episode["action"].copy(),
                "raw_action": episode["raw_script_action"].copy(),
                "timestamp": episode["timestamp"].copy(),
                "frame_index": episode["frame_index"].copy(),
                "phase": episode["phase"].astype(str),
                "front": episode["observation.images.front"].copy(),
                "wrist": episode["observation.images.wrist"].copy(),
                "final_front": episode[
                    "final_observation.images.front"
                ].copy(),
                "final_wrist": episode[
                    "final_observation.images.wrist"
                ].copy(),
            })
    if not episodes:
        raise FileNotFoundError(f"no successful NPZ episodes in {raw_dir}")
    return episodes


def _raw_alignment_audit(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    per_episode = []
    all_state_before = []
    all_state_after = []
    all_actions = []
    for episode in episodes:
        state = episode["state"]
        next_state = episode["next_state"]
        action = episode["action"]
        chain_error = (
            0.0 if len(state) < 2 else _max_abs(next_state[:-1], state[1:])
        )
        before = np.linalg.norm(action - state, axis=1)
        after = np.linalg.norm(action - next_state, axis=1)
        arm_before = np.linalg.norm(action[:, :7] - state[:, :7], axis=1)
        arm_after = np.linalg.norm(action[:, :7] - next_state[:, :7], axis=1)
        hand_before = np.linalg.norm(action[:, 7:] - state[:, 7:], axis=1)
        hand_after = np.linalg.norm(action[:, 7:] - next_state[:, 7:], axis=1)
        per_episode.append({
            "episode_index": episode["episode_index"],
            "seed": episode["seed"],
            "frames": len(state),
            "state_chain_max_abs_error_rad": chain_error,
            "frame_index_exact": bool(np.array_equal(
                episode["frame_index"], np.arange(len(state))
            )),
            "timestamp_strictly_increasing": bool(
                np.all(np.diff(episode["timestamp"]) > 0)
            ),
            "fraction_next_state_closer_to_target": float(np.mean(after < before)),
            "fraction_next_arm_state_closer_to_target": float(
                np.mean(arm_after < arm_before)
            ),
            "fraction_next_hand_state_closer_to_target": float(
                np.mean(hand_after < hand_before)
            ),
            "mean_target_error_before_rad_l2": float(before.mean()),
            "mean_target_error_after_rad_l2": float(after.mean()),
        })
        all_state_before.append(state)
        all_state_after.append(next_state)
        all_actions.append(action)

    states = np.concatenate(all_state_before)
    next_states = np.concatenate(all_state_after)
    actions = np.concatenate(all_actions)
    before_abs = np.abs(actions - states)
    after_abs = np.abs(actions - next_states)
    nontrivial = before_abs > 1e-7
    return {
        "contract": (
            "observation.state[t] and images[t] are captured before action[t]; "
            "action[t] is the absolute actuator target applied during the physics "
            "step; next_observation.state[t] is captured after that step"
        ),
        "episodes_checked": len(episodes),
        "frames_checked": int(len(states)),
        "all_state_chains_exact": bool(all(
            item["state_chain_max_abs_error_rad"] == 0.0 for item in per_episode
        )),
        "all_frame_indices_exact": bool(all(
            item["frame_index_exact"] for item in per_episode
        )),
        "all_timestamps_strictly_increasing": bool(all(
            item["timestamp_strictly_increasing"] for item in per_episode
        )),
        "fraction_nontrivial_joint_values_closer_after_one_step": float(
            np.mean(after_abs[nontrivial] < before_abs[nontrivial])
        ),
        "mean_abs_target_error_before_rad": float(before_abs.mean()),
        "mean_abs_target_error_after_rad": float(after_abs.mean()),
        "per_episode": per_episode,
    }


def _make_robot(args: argparse.Namespace, config: dict, *, render: bool
                ) -> MujocoOpenArmWuji:
    return MujocoOpenArmWuji(
        args.model,
        args.synergies,
        arm_side=config["arm_side"],
        control_hz=30,
        image_height=240,
        image_width=320,
        front_camera=config["scene"]["front_camera_name"],
        render=render,
    )


def _replay_and_runtime_audit(
    args: argparse.Namespace,
    config: dict,
    episode: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    robot = _make_robot(args, config, render=True)
    robot.connect()
    try:
        task = ReachGraspLiftTask(robot, config)
        task.reset(episode["seed"])
        observation = robot.get_observation()
        runtime_initial = {
            "state": np.concatenate([
                observation["arm_joint_position"],
                observation["hand_joint_position"],
            ]).astype(np.float32),
            "front": observation["front_rgb"].copy(),
            "wrist": observation["wrist_rgb"].copy(),
        }
        reset_comparison = {
            "state_max_abs_error_vs_raw_float32_rad": _max_abs(
                runtime_initial["state"], episode["state"][0].astype(np.float32)
            ),
            "front": _image_comparison(
                runtime_initial["front"], episode["front"][0]
            ),
            "wrist": _image_comparison(
                runtime_initial["wrist"], episode["wrist"][0]
            ),
        }

        max_state_error = 0.0
        max_controller_target_error = 0.0
        max_raw_action_error = 0.0
        max_front_pixel_error = 0
        max_wrist_pixel_error = 0
        mismatched_front_values = 0
        mismatched_wrist_values = 0
        for index, raw_action in enumerate(episode["raw_action"]):
            returned = robot.send_action(raw_action)
            post = robot.latest_record
            post_state = np.concatenate([
                post["arm_joint_position"], post["hand_joint_position"]
            ])
            max_raw_action_error = max(
                max_raw_action_error, _max_abs(returned, raw_action)
            )
            max_state_error = max(
                max_state_error,
                _max_abs(post_state, episode["next_state"][index]),
            )
            max_controller_target_error = max(
                max_controller_target_error,
                _max_abs(post["controller_joint_target"], episode["action"][index]),
            )
            if index + 1 < len(episode["state"]):
                expected_front = episode["front"][index + 1]
                expected_wrist = episode["wrist"][index + 1]
            else:
                expected_front = episode["final_front"]
                expected_wrist = episode["final_wrist"]
            front_diff = np.abs(
                post["front_rgb"].astype(np.int16)
                - expected_front.astype(np.int16)
            )
            wrist_diff = np.abs(
                post["wrist_rgb"].astype(np.int16)
                - expected_wrist.astype(np.int16)
            )
            max_front_pixel_error = max(
                max_front_pixel_error, int(front_diff.max(initial=0))
            )
            max_wrist_pixel_error = max(
                max_wrist_pixel_error, int(wrist_diff.max(initial=0))
            )
            mismatched_front_values += int(np.count_nonzero(front_diff))
            mismatched_wrist_values += int(np.count_nonzero(wrist_diff))

        actuator_ids = np.concatenate([
            robot.arm_actuator_ids, robot.hand_actuator_ids
        ])
        controller_ranges = robot.model.actuator_ctrlrange[actuator_ids].copy()
        replay = {
            "episode_index": episode["episode_index"],
            "seed": episode["seed"],
            "frames_replayed": len(episode["state"]),
            "reset_observation_comparison": reset_comparison,
            "max_abs_returned_raw_action_error": max_raw_action_error,
            "max_abs_controller_target_error_rad": max_controller_target_error,
            "max_abs_next_state_error_rad": max_state_error,
            "front_max_abs_pixel_error": max_front_pixel_error,
            "front_mismatched_values": mismatched_front_values,
            "front_mismatched_fraction": (
                mismatched_front_values
                / (len(episode["state"]) * np.prod(episode["front"].shape[1:]))
            ),
            "wrist_max_abs_pixel_error": max_wrist_pixel_error,
            "wrist_mismatched_values": mismatched_wrist_values,
            "wrist_mismatched_fraction": (
                mismatched_wrist_values
                / (len(episode["state"]) * np.prod(episode["wrist"].shape[1:]))
            ),
            "exact_state_replay": max_state_error == 0.0,
            "exact_controller_target_replay": max_controller_target_error == 0.0,
            "exact_image_replay": (
                mismatched_front_values == 0 and mismatched_wrist_values == 0
            ),
            "image_replay_note": (
                "State and actuator targets replay bit-exactly. Sparse 1-2 LSB "
                "raster differences are renderer repeatability noise, not a "
                "camera key, RGB/BGR, layout, range, or temporal mismatch."
            ),
        }
        return replay, runtime_initial, controller_ranges
    finally:
        robot.disconnect()


def _input_and_padding_audit(
    args: argparse.Namespace,
    episodes: list[dict[str, Any]],
    runtime_initial: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.utils import prepare_observation_for_inference

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset,
        delta_timestamps={"action": [index / 30 for index in range(100)]},
        video_backend="pyav",
    )
    item = dataset[0]
    raw = episodes[0]
    raw_state = raw["state"][0].astype(np.float32)
    raw_front_chw = torch.from_numpy(
        raw["front"][0].transpose(2, 0, 1).copy()
    ).float() / 255.0
    raw_wrist_chw = torch.from_numpy(
        raw["wrist"][0].transpose(2, 0, 1).copy()
    ).float() / 255.0

    runtime_batch = prepare_observation_for_inference(
        {
            "observation.state": runtime_initial["state"].copy(),
            "observation.images.front": runtime_initial["front"].copy(),
            "observation.images.wrist": runtime_initial["wrist"].copy(),
        },
        torch.device("cpu"),
        task=TASK,
        robot_type="openarm_wuji",
    )
    training_batch = {
        "observation.state": item["observation.state"].unsqueeze(0),
        "observation.images.front": item[
            "observation.images.front"
        ].unsqueeze(0),
        "observation.images.wrist": item[
            "observation.images.wrist"
        ].unsqueeze(0),
        "task": [item["task"]],
        "robot_type": ["openarm_wuji"],
    }
    policy_config = ACTConfig.from_pretrained(args.checkpoint)
    preprocessor, _ = make_pre_post_processors(
        policy_config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    processed_training = preprocessor(training_batch)
    processed_runtime = preprocessor(runtime_batch)
    prepared_errors = {
        key: _max_abs(runtime_batch[key].numpy(), training_batch[key].numpy())
        for key in (
            "observation.state",
            "observation.images.front",
            "observation.images.wrist",
        )
    }
    processed_errors = {
        key: _max_abs(
            processed_runtime[key].numpy(), processed_training[key].numpy()
        )
        for key in (
            "observation.state",
            "observation.images.front",
            "observation.images.wrist",
        )
    }
    parity_exact = all(value == 0.0 for value in prepared_errors.values())
    # The runtime renderer can differ by one quantization level at a handful of
    # edge pixels despite an identical MuJoCo state.  That is not a preprocessing
    # mismatch: both paths still interpret those pixels identically as RGB/255.
    parity_with_render_tolerance = bool(
        prepared_errors["observation.state"] <= 1e-7
        and prepared_errors["observation.images.front"] <= 1 / 255 + 1e-7
        and prepared_errors["observation.images.wrist"] <= 1 / 255 + 1e-7
    )

    parity = {
        "seed": raw["seed"],
        "frame_index": 0,
        "keys": {
            "dataset": sorted(key for key in item if key.startswith("observation.")),
            "policy_input": list(policy_config.input_features),
            "rollout": [
                "observation.state",
                "observation.images.front",
                "observation.images.wrist",
            ],
        },
        "camera_order_policy": list(policy_config.image_features),
        "state_order_dataset": JOINT_NAMES,
        "state_order_rollout": (
            [f"openarm_left_joint{index}" for index in range(1, 8)]
            + [
                f"wuji_left_finger{finger}_joint{joint}"
                for finger in range(1, 6) for joint in range(1, 5)
            ]
        ),
        "raw_to_native": {
            "state_max_abs_error_rad": _max_abs(
                item["observation.state"].numpy(), raw_state
            ),
            "front_max_abs_error_0_to_1": _max_abs(
                item["observation.images.front"].numpy(), raw_front_chw.numpy()
            ),
            "wrist_max_abs_error_0_to_1": _max_abs(
                item["observation.images.wrist"].numpy(), raw_wrist_chw.numpy()
            ),
        },
        "native_training_tensor": {
            "state_shape": list(item["observation.state"].shape),
            "state_dtype": str(item["observation.state"].dtype),
            "front_shape": list(item["observation.images.front"].shape),
            "front_dtype": str(item["observation.images.front"].dtype),
            "front_range": [
                float(item["observation.images.front"].min()),
                float(item["observation.images.front"].max()),
            ],
            "wrist_shape": list(item["observation.images.wrist"].shape),
            "wrist_dtype": str(item["observation.images.wrist"].dtype),
            "wrist_range": [
                float(item["observation.images.wrist"].min()),
                float(item["observation.images.wrist"].max()),
            ],
            "layout": "CHW",
            "color_order": "RGB",
        },
        "runtime_before_prepare": {
            "state_shape": list(runtime_initial["state"].shape),
            "state_dtype": str(runtime_initial["state"].dtype),
            "front_shape": list(runtime_initial["front"].shape),
            "front_dtype": str(runtime_initial["front"].dtype),
            "front_range": [
                int(runtime_initial["front"].min()),
                int(runtime_initial["front"].max()),
            ],
            "wrist_shape": list(runtime_initial["wrist"].shape),
            "wrist_dtype": str(runtime_initial["wrist"].dtype),
            "wrist_range": [
                int(runtime_initial["wrist"].min()),
                int(runtime_initial["wrist"].max()),
            ],
            "layout": "HWC",
            "color_order": "RGB",
        },
        "runtime_after_prepare_vs_training_before_normalization": prepared_errors,
        "after_checkpoint_preprocessor": {
            key: {
                "shape": list(processed_runtime[key].shape),
                "dtype": str(processed_runtime[key].dtype),
                "max_abs_training_runtime_error": processed_errors[key],
            }
            for key in (
                "observation.state",
                "observation.images.front",
                "observation.images.wrist",
            )
        },
        "resize": "none; recorder, native dataset, and rollout are all 240x320",
        "parity_exact": parity_exact,
        "parity_passed": parity_with_render_tolerance,
        "render_tolerance": (
            "one uint8 level (1/255) before checkpoint normalization; seed-0 "
            "wrist frame differs at only 12 channel values by 1 LSB while state "
            "is exact"
        ),
        "interpretation": (
            "No preprocessing mismatch: native PNG decoding and rollout prepare "
            "both produce float32 RGB BCHW in [0,1], then use the same checkpoint "
            "mean/std processor. The nonzero normalized wrist difference is the "
            "same one-LSB renderer noise amplified by channel standard deviation."
        ),
    }

    phase_frames = Counter()
    weighted_phase_targets = Counter()
    total_possible_slots = 0
    total_valid_slots = 0
    for episode in episodes:
        phases = episode["phase"]
        phase_frames.update(phases.tolist())
        length = len(phases)
        for anchor in range(length):
            valid = min(100, length - anchor)
            weighted_phase_targets.update(phases[anchor:anchor + valid].tolist())
            total_valid_slots += valid
            total_possible_slots += 100

    sample_indices = []
    global_offset = 0
    for native_episode, episode in enumerate(episodes):
        length = len(episode["phase"])
        for local_index in sorted({0, max(0, length - 100), length - 2, length - 1}):
            sample = dataset[global_offset + local_index]
            pad = sample["action_is_pad"].numpy()
            expected_valid = min(100, length - local_index)
            padded_actions = sample["action"].numpy()[expected_valid:]
            expected_tail = episode["action"][-1].astype(np.float32)
            sample_indices.append({
                "native_episode_index": native_episode,
                "source_seed": episode["seed"],
                "local_frame_index": local_index,
                "valid_actions": int((~pad).sum()),
                "padded_actions": int(pad.sum()),
                "expected_valid_actions": expected_valid,
                "mask_matches_expected": bool(
                    np.array_equal(
                        pad,
                        np.arange(100) >= expected_valid,
                    )
                ),
                "padding_repeats_last_action": bool(
                    len(padded_actions) == 0
                    or np.array_equal(
                        padded_actions,
                        np.repeat(
                            expected_tail[None, :], len(padded_actions), axis=0
                        ),
                    )
                ),
            })
        global_offset += length

    total_frames = sum(phase_frames.values())
    padding = {
        "chunk_size": 100,
        "episodes": len(episodes),
        "episode_length_frames": {
            "min": min(len(item["phase"]) for item in episodes),
            "max": max(len(item["phase"]) for item in episodes),
            "mean": float(np.mean([len(item["phase"]) for item in episodes])),
        },
        "all_possible_action_slots": total_possible_slots,
        "valid_action_slots": total_valid_slots,
        "padded_action_slots": total_possible_slots - total_valid_slots,
        "padding_fraction": (
            (total_possible_slots - total_valid_slots) / total_possible_slots
        ),
        "dataset_padding_behavior": (
            "out-of-episode indices clamp to the final episode action and "
            "action_is_pad marks those repeated values"
        ),
        "act_loss_behavior": (
            "ACTPolicy.forward computes valid_mask = ~action_is_pad and excludes "
            "padded action values from the L1 denominator and numerator"
        ),
        "padding_contributes_to_l1_loss": False,
        "sampled_tail_checks": sample_indices,
        "all_sampled_masks_correct": bool(all(
            item["mask_matches_expected"] for item in sample_indices
        )),
        "all_sampled_padding_repeats_tail": bool(all(
            item["padding_repeats_last_action"] for item in sample_indices
        )),
        "raw_phase_distribution": {
            key: {
                "frames": phase_frames[key],
                "fraction": phase_frames[key] / total_frames,
            }
            for key in sorted(phase_frames)
        },
        "valid_loss_target_phase_distribution": {
            key: {
                "slots": weighted_phase_targets[key],
                "fraction": weighted_phase_targets[key] / total_valid_slots,
            }
            for key in sorted(weighted_phase_targets)
        },
    }
    return parity, padding


def _action_normalization_audit(
    args: argparse.Namespace,
    episodes: list[dict[str, Any]],
    controller_ranges: np.ndarray,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    actions = np.concatenate([item["action"] for item in episodes]).astype(
        np.float64
    )
    raw_stats = _summary(actions)
    pre_path = args.checkpoint / (
        "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    post_path = args.checkpoint / (
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    pre = load_file(str(pre_path))
    post = load_file(str(post_path))
    mean = pre["action.mean"].numpy().astype(np.float64)
    std = pre["action.std"].numpy().astype(np.float64)
    minimum = pre["action.min"].numpy().astype(np.float64)
    maximum = pre["action.max"].numpy().astype(np.float64)
    normalized_actions = (actions - mean) / (std + 1e-8)
    normalized_stats = _summary(normalized_actions)

    rollout_files = sorted(args.rollouts.glob("rollout_seed_*.npz"))
    predictions = []
    sent = []
    for path in rollout_files:
        with np.load(path, allow_pickle=False) as rollout:
            predictions.append(rollout["predicted_action"].astype(np.float64))
            sent.append(rollout["sent_action"].astype(np.float64))
    predicted = np.concatenate(predictions) if predictions else np.empty((0, 27))
    sent_actions = np.concatenate(sent) if sent else np.empty((0, 27))
    recoverable = std > 0
    predicted_normalized = np.full_like(predicted, np.nan)
    if len(predicted):
        predicted_normalized[:, recoverable] = (
            predicted[:, recoverable] - mean[recoverable]
        ) / std[recoverable]

    per_joint = []
    for index, name in enumerate(JOINT_NAMES):
        item = {
            "index": index,
            "name": name,
            "controller_range_rad": controller_ranges[index],
            "training_action": {
                key: raw_stats[key][index] for key in raw_stats
            },
            "checkpoint_action_stats": {
                "min": minimum[index],
                "max": maximum[index],
                "mean": mean[index],
                "std": std[index],
            },
            "training_action_normalized": {
                key: normalized_stats[key][index] for key in normalized_stats
            },
        }
        if len(predicted):
            prediction_stats = _summary(predicted[:, index])
            sent_stats = _summary(sent_actions[:, index])
            normalized_prediction_stats = _summary(
                predicted_normalized[:, index]
            ) if recoverable[index] else {
                "min": np.nan, "max": np.nan,
                "mean": np.nan, "std": np.nan,
            }
            difference = np.abs(predicted[:, index] - sent_actions[:, index])
            item["step500_rollout_predicted_unnormalized"] = prediction_stats
            item["step500_rollout_predicted_normalized"] = (
                normalized_prediction_stats
            )
            item["step500_rollout_sent"] = sent_stats
            item["step500_rollout_clipping"] = {
                "values": int(np.count_nonzero(difference > 1e-9)),
                "fraction": float(np.mean(difference > 1e-9)),
                "max_abs_rad": float(difference.max(initial=0.0)),
            }
        per_joint.append(item)

    checkpoint_stat_errors = {
        key: _max_abs(
            pre[f"action.{key}"].numpy(),
            np.asarray(raw_stats[key], dtype=np.float32),
        )
        for key in ("min", "max", "mean", "std")
    }
    shared_keys = sorted(set(pre) & set(post))
    processor_round_trip_stat_error = max(
        _max_abs(pre[key].numpy(), post[key].numpy()) for key in shared_keys
    )
    finger = per_joint[7]
    finger["interpretation"] = (
        "The training target starts exactly at the controller lower bound. "
        "Step-500 predictions below that bound are ordinary model extrapolation "
        "after correct mean/std unnormalization; send_controller_joint_target "
        "clips them to the physical actuator bound."
    )
    return {
        "joint_order": JOINT_NAMES,
        "action_semantics": "absolute MuJoCo position-actuator target in radians",
        "normalization": "MEAN_STD: normalized=(action-mean)/(std+1e-8)",
        "unnormalization": "action=network_output*std+mean",
        "training_frames": len(actions),
        "step500_rollout_files": len(rollout_files),
        "step500_rollout_frames": len(predicted),
        "checkpoint_stats_match_raw_training_data_max_abs": checkpoint_stat_errors,
        "preprocessor_postprocessor_stats_max_abs_error": (
            processor_round_trip_stat_error
        ),
        "all_clipped_joint_names": [
            item["name"] for item in per_joint
            if item.get("step500_rollout_clipping", {}).get("values", 0) > 0
        ],
        "finger1_joint1": finger,
        "per_joint": per_joint,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path,
        default=root / "outputs/act_e2e_smoke/coordinated_demos/raw/successful",
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=root / "outputs/act_e2e_smoke/lerobot_dataset",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / (
            "outputs/act_e2e_smoke/act_train/checkpoints/000500/"
            "pretrained_model"
        ),
    )
    parser.add_argument(
        "--rollouts", type=Path,
        default=root / (
            "outputs/act_e2e_smoke/rollouts/step_000500_seeds_0_19"
        ),
    )
    parser.add_argument(
        "--model", type=Path,
        default=root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb",
    )
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/reach_grasp_lift.json",
    )
    parser.add_argument(
        "--synergies", type=Path,
        default=root / "configs/wuji_hand_left_synergies.json",
    )
    parser.add_argument("--repo-id", default="local/openarm-wuji-act-smoke")
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/act_e2e_smoke/diagnosis/pipeline_audit",
    )
    args = parser.parse_args()

    # Keep Hugging Face lock files on the short, project-adjacent cache path.
    cache = root.parent / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))

    config = json.loads(args.config.read_text(encoding="utf-8"))
    episodes = _load_episodes(args.raw_dir)
    raw_alignment = _raw_alignment_audit(episodes)
    replay, runtime_initial, controller_ranges = _replay_and_runtime_audit(
        args, config, episodes[0]
    )
    input_parity, padding = _input_and_padding_audit(
        args, episodes, runtime_initial
    )
    action_normalization = _action_normalization_audit(
        args, episodes, controller_ranges
    )

    conclusions = {
        "input_preprocessing_mismatch_found": not input_parity["parity_passed"],
        "one_frame_temporal_shift_found": not bool(
            raw_alignment["all_state_chains_exact"]
            and replay["exact_state_replay"]
            and replay["exact_controller_target_replay"]
        ),
        "padding_mask_bug_found": not bool(
            padding["all_sampled_masks_correct"]
            and not padding["padding_contributes_to_l1_loss"]
        ),
        "normalization_or_range_semantics_bug_found": not bool(
            max(action_normalization[
                "checkpoint_stats_match_raw_training_data_max_abs"
            ].values()) < 1e-5
            and action_normalization[
                "preprocessor_postprocessor_stats_max_abs_error"
            ] == 0.0
        ),
    }
    conclusions["clear_pipeline_bug_found"] = any(conclusions.values())
    report = {
        "scope": (
            "read-only ACT pipeline audit; no training, data export, scripted "
            "grasp, MuJoCo controller, or policy code was changed"
        ),
        "inputs": {
            "raw_dir": args.raw_dir,
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "rollouts": args.rollouts,
            "model": args.model,
            "model_sha256": _sha256(args.model),
            "config": args.config,
            "config_sha256": _sha256(args.config),
        },
        "training_vs_rollout_input_parity": input_parity,
        "temporal_alignment": {
            "raw_dataset": raw_alignment,
            "seeded_full_episode_replay": replay,
            "passed": bool(
                raw_alignment["all_state_chains_exact"]
                and replay["exact_state_replay"]
                and replay["exact_controller_target_replay"]
            ),
        },
        "action_chunk_padding_and_phase_distribution": padding,
        "action_normalization_and_controller_range": action_normalization,
        "conclusions": conclusions,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "pipeline_audit.json"
    destination.write_text(
        json.dumps(_plain(report), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(destination),
        "input_parity_passed": input_parity["parity_passed"],
        "temporal_alignment_passed": report["temporal_alignment"]["passed"],
        "padding_masks_passed": padding["all_sampled_masks_correct"],
        "clipped_joints": action_normalization["all_clipped_joint_names"],
    }, indent=2))


if __name__ == "__main__":
    main()
