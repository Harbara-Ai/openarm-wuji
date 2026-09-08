from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot a SAC grasp training report")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/rl_grasp_stage1/sac_5000_training.png")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    returns = np.asarray(report["training_episode_returns"], dtype=float)
    contacts = np.asarray(
        report["training_episode_max_contact_fingers"], dtype=float
    )
    width, height = 1200, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right = 90, width - 40
    panel_top = (90, 390)
    panel_bottom = (330, 630)
    grid = "#d9e1e8"
    navy = "#1f4e79"
    pale_blue = "#8da0cb"
    orange = "#d95f02"
    draw.text((left, 28),
              f"Wuji fixed-palm SAC — {report['timesteps']} steps, seed {report['seed']}",
              fill="#17202a")

    x = np.linspace(left, right, len(returns))
    return_min = float(np.floor(returns.min()))
    return_max = float(np.ceil(returns.max()))
    if return_max <= return_min:
        return_max = return_min + 1.0

    def return_y(value: float) -> float:
        fraction = (value - return_min) / (return_max - return_min)
        return panel_bottom[0] - fraction * (panel_bottom[0] - panel_top[0])

    for fraction in np.linspace(0.0, 1.0, 5):
        value = return_min + fraction * (return_max - return_min)
        y = return_y(value)
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((20, y - 7), f"{value:.1f}", fill="#52616b")
    draw.line((left, panel_top[0], left, panel_bottom[0]), fill="#52616b", width=2)
    draw.line((left, panel_bottom[0], right, panel_bottom[0]), fill="#52616b", width=2)
    draw.line([(float(px), return_y(float(value)))
               for px, value in zip(x, returns)], fill=pale_blue, width=3)
    average = moving_average(returns)
    average_x = x[4:] if len(returns) >= 5 else x
    draw.line([(float(px), return_y(float(value)))
               for px, value in zip(average_x, average)], fill=navy, width=5)
    draw.text((left + 10, panel_top[0] + 8),
              "Episode return (blue) / 5-episode mean (navy)", fill="#17202a")

    for count in range(6):
        y = panel_bottom[1] - count / 5.0 * (panel_bottom[1] - panel_top[1])
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((55, y - 7), str(count), fill="#52616b")
    draw.line((left, panel_top[1], left, panel_bottom[1]), fill="#52616b", width=2)
    draw.line((left, panel_bottom[1], right, panel_bottom[1]), fill="#52616b", width=2)
    bar_width = max(3.0, (right - left) / max(len(contacts), 1) * 0.65)
    for px, count in zip(x, contacts):
        y = panel_bottom[1] - count / 5.0 * (panel_bottom[1] - panel_top[1])
        draw.rectangle((px - bar_width / 2, y, px + bar_width / 2,
                        panel_bottom[1]), fill=orange)
    draw.text((left + 10, panel_top[1] + 8),
              "Maximum simultaneous contact fingers", fill="#17202a")
    draw.text((width // 2 - 65, height - 38), "Training episode", fill="#17202a")
    draw.text((right - 170, panel_top[0] + 8),
              f"successes: {report['success_events']}", fill="#a93226")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
