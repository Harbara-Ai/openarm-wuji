import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


LEROBOT_AVAILABLE = importlib.util.find_spec("lerobot") is not None


@unittest.skipUnless(LEROBOT_AVAILABLE, "install the pinned LeRobot environment")
class LeRobotAdapterTests(unittest.TestCase):
    def test_mock_contract_and_feature_conversion(self):
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import hw_to_dataset_features
        from lerobot.robots import make_robot_from_config
        from lerobot_robot_openarm_wuji import (
            OpenArmWujiFollower,
            OpenArmWujiFollowerConfig,
        )

        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            config = OpenArmWujiFollowerConfig(
                id="contract_test",
                calibration_dir=Path(directory),
                backend="mock",
                synergy_config=root / "configs/wuji_hand_left_synergies.json",
                image_height=64,
                image_width=80,
            )
            self.assertEqual(config.type, "openarm_wuji_follower")
            robot = make_robot_from_config(config)
            self.assertIsInstance(robot, OpenArmWujiFollower)
            self.assertEqual(len(robot.observation_features), 29)
            self.assertEqual(len(robot.action_features), 10)
            robot.connect()
            observation = robot.get_observation()
            self.assertEqual(set(observation), set(robot.observation_features))
            self.assertNotIn("timestamp", observation)
            self.assertNotIn("sim_time", observation)
            for telemetry_key in (
                "cube_position_m",
                "cube_quaternion_wxyz",
                "grasp_center_position_m",
                "grasp_center_quaternion_wxyz",
                "object_relative_position_m",
                "object_relative_quaternion_wxyz",
                "contacts",
                "contact_resultant_force_world_n",
            ):
                self.assertNotIn(telemetry_key, observation)
            action = dict.fromkeys(robot.action_features, 0.5)
            action["hand.open_close"] = 2.0
            sent = robot.send_action(action)
            self.assertEqual(set(sent), set(robot.action_features))
            self.assertTrue(np.isfinite(list(sent.values())).all())
            self.assertEqual(sent["hand.open_close"], 1.0)
            robot.disconnect()

            observation_ds = hw_to_dataset_features(
                robot.observation_features, OBS_STR, use_video=False
            )
            action_ds = hw_to_dataset_features(robot.action_features, ACTION)
            self.assertEqual(observation_ds["observation.state"]["shape"], (27,))
            self.assertEqual(observation_ds["observation.images.front_rgb"]["shape"], (64, 80, 3))
            self.assertEqual(observation_ds["observation.images.wrist_rgb"]["shape"], (64, 80, 3))
            self.assertEqual(action_ds["action"]["shape"], (10,))

    def test_mujoco_backend_through_lerobot_adapter(self):
        from lerobot_robot_openarm_wuji import (
            OpenArmWujiFollower,
            OpenArmWujiFollowerConfig,
        )

        root = Path(__file__).resolve().parents[1]
        model = root / "outputs/combined/openarm_v2_wuji_left.mjb"
        if not model.exists():
            self.skipTest("generate the combined model before integration testing")
        with TemporaryDirectory() as directory:
            config = OpenArmWujiFollowerConfig(
                id="mujoco_contract_test",
                calibration_dir=Path(directory),
                backend="mujoco",
                model_path=model,
                synergy_config=root / "configs/wuji_hand_left_synergies.json",
                image_height=64,
                image_width=80,
            )
            robot = OpenArmWujiFollower(config)
            robot.connect()
            observation = robot.get_observation()
            action = {
                name: observation[name] if name in observation else 0.0
                for name in robot.action_features
            }
            action["joint_1.pos"] += 0.05
            action["hand.open_close"] = 0.8
            action["hand.pinch"] = 0.4
            action["hand.spread"] = 0.2
            sent = robot.send_action(action)
            final = robot.get_observation()
            self.assertEqual(set(final), set(robot.observation_features))
            self.assertEqual(set(sent), set(robot.action_features))
            self.assertNotEqual(final["joint_1.pos"], observation["joint_1.pos"])
            self.assertNotEqual(final["hand_joint_01.pos"], observation["hand_joint_01.pos"])
            robot.disconnect()


if __name__ == "__main__":
    unittest.main()
