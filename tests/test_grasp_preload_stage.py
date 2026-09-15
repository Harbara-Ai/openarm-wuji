import unittest
from unittest.mock import patch

import numpy as np

from openarm_wuji.policy.grasp_secure_controller import GraspSecurePolicy
from openarm_wuji.tasks.grasp_preload_stage import (
    grasp_window_metrics,
    window_passes,
)


class _FakeController:
    def __init__(self, *args, **kwargs):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def predict(self, **kwargs):
        return np.arange(27, dtype=float)


def _sample(*, position=(0.0, 0.0, 0.0), forces=None,
            resultant_force=(1.0, 0.0, 0.0),
            resultant_moment=(0.01, 0.0, 0.0)):
    if forces is None:
        forces = {"finger1": 1.0, "finger2": 1.0}
    return {
        "object_relative_position_m": np.asarray(position, dtype=float),
        "object_relative_quaternion_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "finger_normal_forces_n": forces,
        "contact_resultant_force_world_n": np.asarray(resultant_force),
        "contact_resultant_moment_about_cube_world_nm": np.asarray(
            resultant_moment
        ),
    }


class GraspPreloadStageTest(unittest.TestCase):
    def test_window_accepts_any_two_fingers_without_topology_rule(self):
        samples = [
            _sample(
                position=(index * 0.0001, 0.0, 0.0),
                forces={"finger1": 1.0, "finger4": 0.7},
                resultant_force=(1.0 - index * 0.01, 0.0, 0.0),
                resultant_moment=(0.01 - index * 0.0001, 0.0, 0.0),
            )
            for index in range(8)
        ]
        metrics = grasp_window_metrics(
            samples, minimum_fingers=2, minimum_force_n=0.1
        )
        self.assertTrue(metrics["contacts_sustained"])
        self.assertTrue(metrics["wrench_stable"])
        self.assertTrue(window_passes(metrics, baseline={
            "max_relative_translation_drift_m": 0.008,
            "max_relative_rotation_drift_deg": 6.0,
        }))

    def test_window_rejects_contact_loss(self):
        samples = [_sample() for _ in range(7)] + [
            _sample(forces={"finger1": 1.0})
        ]
        metrics = grasp_window_metrics(
            samples, minimum_fingers=2, minimum_force_n=0.1
        )
        self.assertFalse(metrics["contacts_sustained"])

    def test_terminal_hold_can_treat_wrench_as_diagnostic(self):
        samples = [
            _sample(
                position=(index * 0.0001, 0.0, 0.0),
                resultant_force=(1.0, 0.0, 0.0),
                resultant_moment=(0.01 + index * 0.0001, 0.0, 0.0),
            )
            for index in range(8)
        ]
        metrics = grasp_window_metrics(
            samples, minimum_fingers=2, minimum_force_n=0.1
        )
        self.assertFalse(metrics["wrench_stable"])
        baseline = {
            "max_relative_translation_drift_m": 0.008,
            "max_relative_rotation_drift_deg": 6.0,
        }
        self.assertFalse(window_passes(metrics, baseline=baseline))
        self.assertTrue(window_passes(
            metrics, baseline=baseline, require_wrench_stable=False
        ))

    @patch(
        "openarm_wuji.policy.grasp_secure_controller.ACTController",
        _FakeController,
    )
    def test_grasp_policy_reobserves_and_discards_chunk(self):
        policy = GraspSecurePolicy("unused")
        observation = {
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(20),
            "front_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
        }
        action = policy.select_action(observation)
        self.assertEqual(action.shape, (27,))
        self.assertEqual(policy.controller.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
