import json
import unittest
from pathlib import Path

import numpy as np

from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask


class ReachTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.model = cls.root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb"
        cls.config_path = cls.root / "configs/reach_grasp_lift.json"
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def make_robot(self):
        if not self.model.exists():
            self.skipTest("build the Reach-Grasp-Lift model before integration testing")
        return MujocoOpenArmWuji(
            self.model,
            self.root / "configs/wuji_hand_left_synergies.json",
            control_hz=30,
            image_height=48,
            image_width=64,
            front_camera=self.config["scene"]["front_camera_name"],
        )

    def test_reset_is_seeded_and_reach_converges(self):
        robot = self.make_robot()
        robot.connect()
        try:
            task = ReachGraspLiftTask.from_json(robot, self.config_path)
            first = task.reset(11)["cube_position_m"]
            repeated = task.reset(11)["cube_position_m"]
            different = task.reset(12)["cube_position_m"]
            np.testing.assert_allclose(first, repeated, atol=1e-12)
            self.assertGreater(np.linalg.norm(first[:2] - different[:2]), 1e-4)
            result = task.run_reach(11)
            self.assertTrue(result.success, result.to_dict())
            self.assertLessEqual(
                result.final_error_m, self.config["reach"]["position_tolerance_m"]
            )
            self.assertGreater(result.initial_error_m, result.final_error_m)
            self.assertEqual(robot.records[-1]["front_rgb"].shape, (48, 64, 3))
            self.assertGreater(int(np.ptp(robot.records[-1]["front_rgb"])), 80)
        finally:
            robot.disconnect()


if __name__ == "__main__":
    unittest.main()
