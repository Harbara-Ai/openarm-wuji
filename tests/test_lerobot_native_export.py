import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from openarm_wuji.dataset import export_native_lerobot_dataset
import test_coordinated_demo_recorder as recorder_fixtures


@unittest.skipUnless(
    importlib.util.find_spec("datasets") is not None,
    "LeRobot dataset optional dependencies are not installed",
)
class NativeLeRobotExportTests(unittest.TestCase):
    def test_native_round_trip_and_dataloader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "episode.npz"
            recorder_fixtures.CoordinatedDemoRecorderTests().successful_episode(
                source
            )
            destination = root / "native"
            manifest = export_native_lerobot_dataset(
                [source], destination, repo_id="local/test-openarm-wuji"
            )

            self.assertTrue(manifest["native_lerobot_dataset"])
            self.assertTrue(manifest["dataloader_smoke_test"]["passed"])
            self.assertEqual(manifest["state_dim"], 27)
            self.assertEqual(manifest["action_dim"], 27)
            self.assertEqual(manifest["episodes"], 1)
            self.assertEqual(manifest["frames"], 1)
            self.assertTrue(manifest["alignment_validation"]["passed"])
            self.assertTrue((destination / "meta/info.json").exists())
            self.assertTrue((destination / "conversion_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
