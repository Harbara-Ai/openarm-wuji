import unittest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from openarm_wuji.robot.mock import MockOpenArmWuji
from openarm_wuji.safety.actions import ActionSafety, UnsafeAction
from openarm_wuji.teleop.synergies import HandSynergyMapper, synergy_to_joints
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji


class CoreTests(unittest.TestCase):
    def test_mock_state_and_velocity_limit(self):
        robot = MockOpenArmWuji(control_hz=10)
        robot.connect()
        robot.send_action(np.ones(27))
        np.testing.assert_allclose(robot.get_observation()["joint_position"], 0.2)

    def test_nan_enters_safe_failure(self):
        safety = ActionSafety([-1], [1], [1])
        with self.assertRaises(UnsafeAction):
            safety.validate([np.nan])

    def test_synergy_is_20d_and_finite(self):
        target = synergy_to_joints(1, 0.5, -0.2)
        self.assertEqual(target.shape, (20,))
        self.assertTrue(np.isfinite(target).all())

    def test_synergy_velocity_limit_and_nan(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        mapper = HandSynergyMapper.from_json(root / "configs/wuji_hand_left_synergies.json")
        target = mapper.next((1, 0, 0), previous=np.zeros(20), dt=0.1)
        self.assertLessEqual(np.max(np.abs(target)), 0.2 + 1e-12)
        with self.assertRaises(ValueError):
            mapper.map(np.nan, 0, 0)

    def test_combined_mujoco_arm_and_synergy_control(self):
        root = Path(__file__).resolve().parents[1]
        model = root / "outputs/combined/openarm_v2_wuji_left.mjb"
        if not model.exists():
            self.skipTest("generate the combined model before integration testing")
        synergy_config = root / "configs/wuji_hand_left_synergies.json"
        robot = MujocoOpenArmWuji(
            model, synergy_config, control_hz=30, image_height=64, image_width=80
        )
        robot.connect()
        initial = robot.get_observation()
        self.assertEqual(initial["front_rgb"].shape, (64, 80, 3))
        self.assertEqual(initial["wrist_rgb"].shape, (64, 80, 3))
        self.assertEqual(initial["front_rgb"].dtype, np.uint8)
        action = np.r_[initial["arm_joint_position"], [1.0, 0.5, 0.2]]
        action[0] += 0.05
        for _ in range(10):
            robot.send_action(action)
        final = robot.get_observation()
        self.assertEqual(final["sent_action"].shape, (10,))
        self.assertEqual(final["hand_joint_position"].shape, (20,))
        self.assertGreater(np.max(np.abs(final["arm_joint_position"] - initial["arm_joint_position"])), 1e-4)
        self.assertGreater(np.max(np.abs(final["hand_joint_position"] - initial["hand_joint_position"])), 1e-4)
        self.assertEqual(len(robot.records), 10)
        with TemporaryDirectory() as directory:
            recording = Path(directory) / "episode.npz"
            robot.save_recording(recording)
            with np.load(recording) as episode:
                self.assertEqual(episode["front_rgb"].shape, (10, 64, 80, 3))
                self.assertEqual(episode["sent_action"].shape, (10, 10))
                self.assertEqual(episode["hand_joint_target"].shape, (10, 20))
                self.assertTrue(np.all(np.diff(episode["timestamp"]) > 0))
                self.assertTrue(np.all(np.diff(episode["sim_time"]) > 0))
            replay = MujocoOpenArmWuji(
                model, synergy_config, control_hz=30, image_height=64, image_width=80
            )
            replay.connect()
            report = replay.replay_recording(recording)
            self.assertEqual(report["samples"], 10)
            self.assertLess(report["max_arm_position_error"], 1e-12)
            self.assertLess(report["max_hand_position_error"], 1e-12)
            replay.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    unittest.main()
