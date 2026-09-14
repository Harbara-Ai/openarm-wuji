"""Run LeRobot training with a Windows-safe latest-checkpoint pointer."""
from __future__ import annotations

import logging
import os
from pathlib import Path


def main() -> None:
    import lerobot.common.train_utils as train_utils
    import lerobot.scripts.lerobot_train as train_module

    original_update = train_utils.update_last_checkpoint

    def update_last_checkpoint(checkpoint_dir: Path) -> None:
        try:
            original_update(checkpoint_dir)
        except OSError as error:
            if os.name != "nt" or getattr(error, "winerror", None) != 1314:
                raise
            pointer = checkpoint_dir.parent / "last_checkpoint.txt"
            pointer.write_text(checkpoint_dir.name + "\n", encoding="utf-8")
            logging.warning(
                "Windows symlink privilege is unavailable; wrote %s instead",
                pointer,
            )

    train_utils.update_last_checkpoint = update_last_checkpoint
    train_module.update_last_checkpoint = update_last_checkpoint
    train_module.main()


if __name__ == "__main__":
    main()
