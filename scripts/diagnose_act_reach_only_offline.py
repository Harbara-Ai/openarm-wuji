"""Inference-only H=1 and frame-0 conditioning diagnostics for Reach-only ACT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASK = "Move OpenArm and the open Wuji hand to the cube pregrasp pose and hold it stable."
GROUPS = {"overall_27d": slice(0, 27), "arm_7d": slice(0, 7), "hand_20d": slice(7, 27)}


def _plain(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load(raw_dir: Path) -> dict[str, np.ndarray]:
    parts = {key: [] for key in ("state", "action", "front", "wrist", "phase", "seed", "frame")}
    for path in sorted(raw_dir.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            count = len(episode["action"])
            parts["state"].append(np.asarray(episode["observation.state"], dtype=np.float32))
            parts["action"].append(np.asarray(episode["action"], dtype=np.float32))
            parts["front"].append(np.asarray(episode["observation.images.front"], dtype=np.uint8))
            parts["wrist"].append(np.asarray(episode["observation.images.wrist"], dtype=np.uint8))
            parts["phase"].append(episode["phase"].astype(str))
            parts["seed"].append(np.full(count, int(episode["episode_seed"]), dtype=np.int64))
            parts["frame"].append(np.asarray(episode["frame_index"], dtype=np.int64))
    if len(parts["state"]) != 20:
        raise ValueError("expected 20 Reach-only episodes")
    return {key: np.concatenate(value) for key, value in parts.items()}


def _predict(*, policy, preprocessor, postprocessor, state: np.ndarray,
             front: np.ndarray, wrist: np.ndarray, batch_size: int, device) -> np.ndarray:
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(state), batch_size):
            stop = min(len(state), start + batch_size)
            prepared = [
                prepare_observation_for_inference(
                    {
                        "observation.state": state[index],
                        "observation.images.front": front[index],
                        "observation.images.wrist": wrist[index],
                    },
                    device, task=TASK, robot_type="openarm_wuji",
                )
                for index in range(start, stop)
            ]
            batch = {
                key: torch.cat([item[key] for item in prepared], dim=0)
                for key in ("observation.state", "observation.images.front", "observation.images.wrist")
            }
            batch["task"] = [str(item["task"]) for item in prepared]
            batch["robot_type"] = [str(item["robot_type"]) for item in prepared]
            # No expert action is present in batch, so ACT uses its zero latent inference path.
            chunk = postprocessor(policy.predict_action_chunk(preprocessor(batch)))
            outputs.append(chunk[:, 0].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs)


def _mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    error = np.abs(prediction[mask] - target[mask])
    return {
        name: float(np.mean(error[:, group])) for name, group in GROUPS.items()
    } | {"samples": int(np.count_nonzero(mask)), "per_joint_mae_rad": np.mean(error, axis=0)}


def _rms_spread(actions: np.ndarray) -> dict[str, float]:
    std = np.std(np.asarray(actions, dtype=np.float64), axis=0)
    return {
        name: float(np.sqrt(np.mean(np.square(std[group]))))
        for name, group in GROUPS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy

    data = _load(args.raw)
    device = torch.device("cpu")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    prediction = _predict(
        policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
        state=data["state"], front=data["front"], wrist=data["wrist"],
        batch_size=args.batch_size, device=device,
    )
    real_mask = data["phase"] == "reach"
    hold_mask = data["phase"] == "reach_hold"
    all_mask = np.ones(len(prediction), dtype=bool)
    frame0 = data["frame"] == 0
    frame0_state = data["state"][frame0]
    frame0_front = data["front"][frame0]
    frame0_wrist = data["wrist"][frame0]
    paired = prediction[frame0]
    image_only = _predict(
        policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
        state=np.repeat(frame0_state[:1], len(frame0_state), axis=0),
        front=frame0_front, wrist=frame0_wrist,
        batch_size=args.batch_size, device=device,
    )
    expert_frame0 = data["action"][frame0]
    expert_spread = _rms_spread(expert_frame0)
    paired_spread = _rms_spread(paired)
    image_only_spread = _rms_spread(image_only)
    ratio = lambda numerator: {
        name: float(numerator[name] / expert_spread[name]) if expert_spread[name] > 1e-12 else None
        for name in GROUPS
    }
    report = {
        "diagnostic": "Reach-only ACT inference-only H=1 reconstruction and frame-0 conditioning",
        "checkpoint": args.checkpoint.resolve(),
        "frames": len(prediction),
        "episodes": int(np.count_nonzero(frame0)),
        "chunk_size": int(policy.config.chunk_size),
        "h1_mae_rad": {
            "real_reach": _mae(prediction, data["action"], real_mask),
            "terminal_hold": _mae(prediction, data["action"], hold_mask),
            "all": _mae(prediction, data["action"], all_mask),
        },
        "frame0_conditioning": {
            "expert_action_rms_spread_rad": expert_spread,
            "paired_policy_rms_spread_rad": paired_spread,
            "paired_policy_spread_ratio_vs_expert": ratio(paired_spread),
            "image_only_fixed_seed0_state_rms_spread_rad": image_only_spread,
            "image_only_spread_ratio_vs_expert": ratio(image_only_spread),
            "unique_seeds": data["seed"][frame0],
        },
        "inference_contract": {
            "expert_action_passed_to_policy": False,
            "prediction": "first unnormalized 27D action of ACT chunk20",
            "task": TASK,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(_plain(report), indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "predictions.npz",
        seed=data["seed"], frame_index=data["frame"], phase=data["phase"],
        expert_action=data["action"], predicted_h1_action=prediction,
        frame0_paired_action=paired, frame0_image_only_action=image_only,
    )
    print(json.dumps(_plain({
        "h1_real_reach": report["h1_mae_rad"]["real_reach"],
        "frame0_spread_ratio": report["frame0_conditioning"]["paired_policy_spread_ratio_vs_expert"],
    }), indent=2))


if __name__ == "__main__":
    main()
