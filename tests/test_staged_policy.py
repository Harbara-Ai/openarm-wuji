import unittest
from unittest.mock import patch

import numpy as np

from openarm_wuji.policy.staged_controller import (
    ApproachPolicy,
    ReachPolicy,
    RecoveryPolicy,
)


class _FakeController:
    def __init__(self, *args, **kwargs):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def predict(self, **kwargs):
        return np.arange(27, dtype=float)


class StagedPolicyTest(unittest.TestCase):
    def _observation(self):
        return {
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(20),
            "front_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
        }

    @patch("openarm_wuji.policy.staged_controller.ACTController", _FakeController)
    def test_reach_uses_position_only_five_frame_gate(self):
        policy = ReachPolicy("unused")
        for _ in range(4):
            status = policy.observe(
                position_error_m=0.012,
                orientation_error_deg=180.0,
            )
            self.assertFalse(status.success)
        status = policy.observe(
            position_error_m=0.012,
            orientation_error_deg=180.0,
        )
        self.assertTrue(status.success)

    @patch("openarm_wuji.policy.staged_controller.ACTController", _FakeController)
    def test_approach_rejects_unsafe_cube_displacement(self):
        policy = ApproachPolicy("unused", max_cube_displacement_m=0.025)
        status = policy.observe(
            position_error_m=0.005,
            orientation_error_deg=1.0,
            cube_displacement_m=0.026,
        )
        self.assertFalse(status.success)
        self.assertEqual(status.failure_reason, "cube_displacement")

    @patch("openarm_wuji.policy.staged_controller.ACTController", _FakeController)
    def test_select_action_discards_unused_chunk_each_frame(self):
        policy = ApproachPolicy("unused")
        initial_resets = policy.controller.reset_count
        action = policy.select_action(self._observation())
        np.testing.assert_array_equal(action, np.arange(27, dtype=float))
        self.assertEqual(policy.controller.reset_count, initial_resets + 1)

    @patch("openarm_wuji.policy.staged_controller.ACTController", _FakeController)
    def test_recovery_has_independent_phase_and_same_safety_gate(self):
        policy = RecoveryPolicy("unused", max_cube_displacement_m=0.025)
        self.assertEqual(policy.status.phase, "recovery")
        for _ in range(4):
            self.assertFalse(policy.observe(
                position_error_m=0.005,
                orientation_error_deg=1.0,
                cube_displacement_m=0.0,
            ).success)
        self.assertTrue(policy.observe(
            position_error_m=0.005,
            orientation_error_deg=1.0,
            cube_displacement_m=0.0,
        ).success)
        policy.reset()
        status = policy.observe(
            position_error_m=0.005,
            orientation_error_deg=1.0,
            cube_displacement_m=0.026,
        )
        self.assertEqual(status.failure_reason, "cube_displacement")


if __name__ == "__main__":
    unittest.main()
