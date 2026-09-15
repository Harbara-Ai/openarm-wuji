"""Measure whether ACT conditions its first action on reset observations.

The diagnostic reads frame 0 from the 20 successful raw demonstrations and
evaluates controlled input substitutions.  It is inference-only: expert
actions are comparison targets and are never passed to ACT's VAE encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TASK = "Coordinate OpenArm and Wuji to grasp the cube, lift it, and hold it unsupported."
CONDITION_ORDER = [
    "paired",
    "vary_images_fixed_seed0_state",
    "vary_state_fixed_seed0_images",
    "vary_front_fixed_seed0_state_and_wrist",
    "vary_wrist_fixed_seed0_state_and_front",
]
GROUPS = {
    "overall_27d": slice(0, 27),
    "arm_7d": slice(0, 7),
    "hand_20d": slice(7, 27),
}


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _checkpoint_label(checkpoint: Path) -> str:
    if checkpoint.name == "pretrained_model":
        return checkpoint.parent.name
    return checkpoint.name


def _image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _load_frame_zero(raw_dir: Path, expected_episodes: int) -> dict[str, Any]:
    rows = []
    joint_names: list[str] | None = None
    image_shape: tuple[int, int, int] | None = None
    for path in sorted(raw_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            if not bool(episode["demonstration_success"]):
                raise ValueError(f"non-successful episode found in successful directory: {path}")
            state = np.asarray(episode["observation.state"][0], dtype=np.float32)
            action = np.asarray(episode["action"][0], dtype=np.float32)
            front = np.asarray(episode["observation.images.front"][0])
            wrist = np.asarray(episode["observation.images.wrist"][0])
            cube_pose = np.asarray(episode["telemetry.cube_pose_world"][0], dtype=np.float64)
            names = [str(name) for name in episode["joint_names"]]
            if state.shape != (27,) or action.shape != (27,):
                raise ValueError(f"expected 27-D frame-0 state/action in {path}")
            if front.dtype != np.uint8 or wrist.dtype != np.uint8:
                raise ValueError(f"expected uint8 frame-0 images in {path}")
            if front.shape != wrist.shape or front.shape[-1] != 3:
                raise ValueError(f"front/wrist RGB shape mismatch in {path}")
            if cube_pose.shape != (7,):
                raise ValueError(f"expected a 7-D frame-0 cube pose in {path}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"non-finite frame-0 state/action in {path}")
            if joint_names is None:
                joint_names = names
                image_shape = front.shape
            elif names != joint_names or front.shape != image_shape:
                raise ValueError(f"episode contract differs in {path}")
            rows.append({
                "source": path.resolve().as_posix(),
                "episode_index": int(episode["episode_index"]),
                "seed": int(episode["episode_seed"]),
                "frame_index": int(episode["frame_index"][0]),
                "phase": str(episode["phase"][0]),
                "timestamp": float(episode["timestamp"][0]),
                "state": state,
                "action": action,
                "front": front.copy(),
                "wrist": wrist.copy(),
                "cube_xy": cube_pose[:2],
                "front_sha256": _image_digest(front),
                "wrist_sha256": _image_digest(wrist),
            })

    rows.sort(key=lambda row: (row["seed"], row["episode_index"]))
    if len(rows) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} successful episodes, found {len(rows)} in {raw_dir}"
        )
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("frame-0 reset-conditioning audit requires unique seeds")
    if 0 not in seeds:
        raise ValueError("seed 0 is required as the fixed-input reference")
    if any(row["frame_index"] != 0 for row in rows):
        raise ValueError("the first stored row must have frame_index == 0")
    if any(row["phase"] != "reach" for row in rows):
        raise ValueError("all frame-0 observations must be in the reach phase")

    return {
        "source": np.asarray([row["source"] for row in rows]),
        "episode_index": np.asarray([row["episode_index"] for row in rows], dtype=np.int64),
        "seed": np.asarray(seeds, dtype=np.int64),
        "frame_index": np.asarray([row["frame_index"] for row in rows], dtype=np.int64),
        "phase": np.asarray([row["phase"] for row in rows]),
        "timestamp": np.asarray([row["timestamp"] for row in rows]),
        "state": np.stack([row["state"] for row in rows]),
        "action": np.stack([row["action"] for row in rows]),
        "front": np.stack([row["front"] for row in rows]),
        "wrist": np.stack([row["wrist"] for row in rows]),
        "cube_xy": np.stack([row["cube_xy"] for row in rows]),
        "front_sha256": np.asarray([row["front_sha256"] for row in rows]),
        "wrist_sha256": np.asarray([row["wrist_sha256"] for row in rows]),
        "joint_names": joint_names,
        "image_shape_hwc": image_shape,
        "seed0_index": seeds.index(0),
    }


def _condition_inputs(data: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    count = len(data["seed"])
    reference = int(data["seed0_index"])
    state0 = np.repeat(data["state"][reference:reference + 1], count, axis=0)
    front0 = np.repeat(data["front"][reference:reference + 1], count, axis=0)
    wrist0 = np.repeat(data["wrist"][reference:reference + 1], count, axis=0)
    return {
        "paired": {
            "state": data["state"], "front": data["front"], "wrist": data["wrist"],
        },
        "vary_images_fixed_seed0_state": {
            "state": state0, "front": data["front"], "wrist": data["wrist"],
        },
        "vary_state_fixed_seed0_images": {
            "state": data["state"], "front": front0, "wrist": wrist0,
        },
        "vary_front_fixed_seed0_state_and_wrist": {
            "state": state0, "front": data["front"], "wrist": wrist0,
        },
        "vary_wrist_fixed_seed0_state_and_front": {
            "state": state0, "front": front0, "wrist": data["wrist"],
        },
    }


def _predict_first_actions(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    inputs: dict[str, np.ndarray],
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    predictions = []
    count = len(inputs["state"])
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            prepared = [
                prepare_observation_for_inference(
                    {
                        "observation.state": inputs["state"][index].copy(),
                        "observation.images.front": inputs["front"][index].copy(),
                        "observation.images.wrist": inputs["wrist"][index].copy(),
                    },
                    device,
                    task=TASK,
                    robot_type="openarm_wuji",
                )
                for index in range(start, stop)
            ]
            batch = {
                key: torch.cat([item[key] for item in prepared], dim=0)
                for key in (
                    "observation.state",
                    "observation.images.front",
                    "observation.images.wrist",
                )
            }
            batch["task"] = [str(item["task"]) for item in prepared]
            batch["robot_type"] = [str(item["robot_type"]) for item in prepared]
            # Do not add expert action: ACT would otherwise use it in the VAE encoder.
            normalized_chunk = policy.predict_action_chunk(preprocessor(batch))
            chunk = postprocessor(normalized_chunk)
            if tuple(chunk.shape[1:]) != (policy.config.chunk_size, 27):
                raise RuntimeError(f"unexpected ACT action chunk shape {tuple(chunk.shape)}")
            predictions.append(chunk[:, 0].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(predictions)


def _rms_spread(std: np.ndarray, joint_slice: slice) -> float:
    selected = np.asarray(std, dtype=np.float64)[joint_slice]
    return float(np.sqrt(np.mean(np.square(selected))))


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator / denominator) if denominator > eps else float("nan")


def _action_statistics(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    std = np.std(actions, axis=0)
    return {
        "per_joint_mean_rad": np.mean(actions, axis=0),
        "per_joint_std_rad": std,
        "per_joint_min_rad": np.min(actions, axis=0),
        "per_joint_max_rad": np.max(actions, axis=0),
        "rms_spread_rad": {
            name: _rms_spread(std, joint_slice)
            for name, joint_slice in GROUPS.items()
        },
    }


def _mae(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.abs(
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(target, dtype=np.float64)
    )
    return {
        "per_joint_mae_rad": np.mean(error, axis=0),
        "group_mae_rad": {
            name: float(np.mean(error[:, joint_slice]))
            for name, joint_slice in GROUPS.items()
        },
        "max_abs_error_rad": float(np.max(error)),
    }


def _correlation(one_dimensional: np.ndarray, actions: np.ndarray) -> np.ndarray:
    source = np.asarray(one_dimensional, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    result = np.full(actions.shape[1], np.nan, dtype=np.float64)
    if np.std(source) <= 1e-12:
        return result
    centered_source = source - np.mean(source)
    source_norm = np.linalg.norm(centered_source)
    for joint in range(actions.shape[1]):
        centered_action = actions[:, joint] - np.mean(actions[:, joint])
        denominator = source_norm * np.linalg.norm(centered_action)
        if denominator > 1e-12:
            result[joint] = float(np.dot(centered_source, centered_action) / denominator)
    return result


def _cube_action_correlation(
    cube_xy: np.ndarray,
    actions: np.ndarray,
    joint_names: list[str],
) -> dict[str, Any]:
    matrix = np.stack([
        _correlation(cube_xy[:, 0], actions),
        _correlation(cube_xy[:, 1], actions),
    ])
    top = {}
    for axis_index, axis_name in enumerate(("cube_x", "cube_y")):
        valid = np.flatnonzero(np.isfinite(matrix[axis_index]))
        order = valid[np.argsort(-np.abs(matrix[axis_index, valid]))][:5]
        top[axis_name] = [
            {
                "joint_index": int(index),
                "joint_name": joint_names[index],
                "pearson_r": float(matrix[axis_index, index]),
            }
            for index in order
        ]
    return {
        "axes": ["cube_x", "cube_y"],
        "pearson_r_by_axis_and_joint": matrix,
        "top_five_absolute_correlations": top,
        "note": "Population of 20 reset seeds; NaN/null means zero variance.",
    }


def _condition_report(
    *,
    prediction: np.ndarray,
    expert_action: np.ndarray,
    paired_prediction: np.ndarray,
    seed0_prediction: np.ndarray,
    expert_spread: dict[str, float],
    cube_xy: np.ndarray,
    joint_names: list[str],
) -> dict[str, Any]:
    statistics = _action_statistics(prediction)
    policy_std = np.asarray(statistics["per_joint_std_rad"])
    expert_std = np.std(expert_action, axis=0)
    statistics["per_joint_std_ratio_vs_expert"] = np.divide(
        policy_std,
        expert_std,
        out=np.full_like(policy_std, np.nan),
        where=expert_std > 1e-12,
    )
    statistics["rms_spread_ratio_vs_expert"] = {
        name: _safe_ratio(statistics["rms_spread_rad"][name], expert_spread[name])
        for name in GROUPS
    }
    return {
        "policy_action_statistics": statistics,
        "mae_vs_corresponding_expert_frame0_action": _mae(prediction, expert_action),
        "mae_vs_paired_policy_prediction": _mae(prediction, paired_prediction),
        "mae_vs_seed0_policy_action_repeated": _mae(prediction, seed0_prediction),
        "cube_xy_action_correlation": _cube_action_correlation(
            cube_xy, prediction, joint_names
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path,
        default=root / "outputs/act_e2e_smoke/coordinated_demos/raw/successful",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "outputs/act_e2e_smoke/act_train/checkpoints/001500/pretrained_model",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.expected_episodes < 2:
        raise ValueError("--expected-episodes must be at least two")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    checkpoint_label = _checkpoint_label(args.checkpoint)
    output = args.output or (
        root / f"outputs/act_e2e_smoke/diagnosis/reset_conditioning_step_{checkpoint_label}"
    )

    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy

    device = torch.device(args.device)
    data = _load_frame_zero(args.raw_dir, args.expected_episodes)
    inputs = _condition_inputs(data)
    if list(inputs) != CONDITION_ORDER:
        raise RuntimeError("condition order changed unexpectedly")

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    if tuple(policy.config.input_features["observation.state"].shape) != (27,):
        raise ValueError("checkpoint does not accept a 27-D observation.state")
    if tuple(policy.config.output_features["action"].shape) != (27,):
        raise ValueError("checkpoint does not emit a 27-D action")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    predictions = {
        condition: _predict_first_actions(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            inputs=condition_inputs,
            batch_size=args.batch_size,
            device=device,
        )
        for condition, condition_inputs in inputs.items()
    }
    expert_action = data["action"].astype(np.float64)
    paired = predictions["paired"]
    reference = int(data["seed0_index"])
    seed0_prediction = np.repeat(paired[reference:reference + 1], len(paired), axis=0)
    expert_statistics = _action_statistics(expert_action)
    expert_spread = expert_statistics["rms_spread_rad"]
    joint_names = list(data["joint_names"])
    condition_reports = {
        condition: _condition_report(
            prediction=prediction,
            expert_action=expert_action,
            paired_prediction=paired,
            seed0_prediction=seed0_prediction,
            expert_spread=expert_spread,
            cube_xy=data["cube_xy"],
            joint_names=joint_names,
        )
        for condition, prediction in predictions.items()
    }
    identical_seed0_input_error = {
        condition: float(np.max(np.abs(prediction[reference] - paired[reference])))
        for condition, prediction in predictions.items()
    }

    report = {
        "schema_version": 1,
        "diagnostic": "ACT frame-0 reset-conditioning input ablation",
        "checkpoint": args.checkpoint.resolve(),
        "checkpoint_label": checkpoint_label,
        "raw_directory": args.raw_dir.resolve(),
        "output_directory": output.resolve(),
        "episodes": len(data["seed"]),
        "seeds": data["seed"],
        "episode_indices": data["episode_index"],
        "joint_names": joint_names,
        "state_dim": 27,
        "action_dim": 27,
        "image_shape_hwc": data["image_shape_hwc"],
        "cube_xy_m": data["cube_xy"],
        "conditions": {
            "paired": "state_i + front_i + wrist_i",
            "vary_images_fixed_seed0_state": "state_seed0 + front_i + wrist_i",
            "vary_state_fixed_seed0_images": "state_i + front_seed0 + wrist_seed0",
            "vary_front_fixed_seed0_state_and_wrist": (
                "state_seed0 + front_i + wrist_seed0"
            ),
            "vary_wrist_fixed_seed0_state_and_front": (
                "state_seed0 + front_seed0 + wrist_i"
            ),
        },
        "inference_contract": {
            "policy_method": "ACTPolicy.predict_action_chunk",
            "preparation": "lerobot.policies.utils.prepare_observation_for_inference",
            "preprocessor": "checkpoint official policy preprocessor",
            "postprocessor": "checkpoint official policy postprocessor",
            "expert_action_passed_to_policy": False,
            "reported_policy_action": "first unnormalized 27-D absolute target from each chunk",
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "input_variation": {
            "frame0_state_statistics": _action_statistics(data["state"]),
            "front_unique_sha256": len(set(data["front_sha256"].tolist())),
            "wrist_unique_sha256": len(set(data["wrist_sha256"].tolist())),
        },
        "expert_frame0_action": {
            "statistics": expert_statistics,
            "cube_xy_action_correlation": _cube_action_correlation(
                data["cube_xy"], expert_action, joint_names
            ),
        },
        "policy_conditions": condition_reports,
        "sanity_checks": {
            "all_frame_indices_zero": bool(np.all(data["frame_index"] == 0)),
            "all_phases_reach": bool(np.all(data["phase"] == "reach")),
            "all_conditions_have_shape_20x27": bool(all(
                prediction.shape == (args.expected_episodes, 27)
                for prediction in predictions.values()
            )),
            "all_policy_actions_finite": bool(all(
                np.isfinite(prediction).all() for prediction in predictions.values()
            )),
            "seed0_identical_input_max_abs_action_error_rad": (
                identical_seed0_input_error
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "reset_conditioning_summary.json").write_text(
        json.dumps(_plain(report), indent=2), encoding="utf-8"
    )
    arrays = {
        "seed": data["seed"],
        "episode_index": data["episode_index"],
        "source": data["source"],
        "cube_xy_m": data["cube_xy"],
        "frame0_state": data["state"],
        "expert_frame0_action": expert_action,
        "front_sha256": data["front_sha256"],
        "wrist_sha256": data["wrist_sha256"],
        "joint_names": np.asarray(joint_names),
    }
    arrays.update({
        f"policy_first_action__{condition}": prediction
        for condition, prediction in predictions.items()
    })
    np.savez_compressed(output / "reset_conditioning_actions.npz", **arrays)
    print(json.dumps({
        "output": output.resolve().as_posix(),
        "checkpoint": checkpoint_label,
        "episodes": len(data["seed"]),
        "conditions": CONDITION_ORDER,
        "paired_mae": condition_reports["paired"][
            "mae_vs_corresponding_expert_frame0_action"
        ]["group_mae_rad"],
    }, indent=2))


if __name__ == "__main__":
    main()
