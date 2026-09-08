from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not read video frame {index}")
    return frame


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract wrist-camera keyframes at inferred grasp phases"
    )
    parser.add_argument("episode", type=int)
    parser.add_argument(
        "--dataset", type=Path,
        default=root / "data/external/wuji-pick-and-place",
    )
    parser.add_argument(
        "--analysis", type=Path,
        default=root / "outputs/wuji_teleop_analysis/cube_hand_analysis.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/wuji_teleop_analysis",
    )
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    row = next(item for item in analysis["episodes"]
               if item["episode_index"] == args.episode)
    camera = f"observation.images.cam_{row['side']}_wrist"
    video = (
        args.dataset / "teleop/videos/chunk-000" / camera
        / f"episode_{args.episode:06d}.mp4"
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [
        row["inferred_onset_frame"],
        (row["inferred_onset_frame"] + row["inferred_close_frame"]) // 2,
        row["inferred_close_frame"],
        min(row["frames"] - 1, row["inferred_close_frame"] + 15),
        min(row["frames"] - 1, row["inferred_release_frame"]),
    ]
    labels = ("onset", "mid-close", "inferred close", "+0.5 s", "release")
    frames = []
    for label, index in zip(labels, indices):
        frame = read_frame(capture, min(index, video_frames - 1))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            frame, f"{label}  frame {index}  t={index / 30:.2f}s",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            cv2.LINE_AA,
        )
        frames.append(frame)
    capture.release()
    sheet = np.concatenate(frames, axis=1)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"episode_{args.episode:06d}_{row['side']}_keyframes.jpg"
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"could not write {output}")
    print(json.dumps({
        "episode": args.episode,
        "side": row["side"],
        "parquet_frames": row["frames"],
        "video_frames": video_frames,
        "indices": dict(zip(labels, indices)),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
