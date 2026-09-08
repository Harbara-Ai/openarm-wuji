import unittest

import numpy as np

from openarm_wuji.tasks import (
    evaluate_lift_outcome,
    relative_pose,
    rotation_geodesic_angle_deg,
)
from openarm_wuji.tasks.se3 import (
    quaternion_error_rotvec,
    rotation_vector_to_quaternion,
)


BASELINE = {
    "name": "CD-WM external baseline",
    "source_url": "https://huggingface.co/datasets/BWangCN/cdwm-grasp-dataset",
    "lift_duration_s": 0.5,
    "gripper_lift_m": 0.05,
    "min_object_lift_m": 0.025,
    "stability_anchor_margin_frames": 2,
    "max_relative_translation_drift_m": 0.008,
    "max_relative_rotation_drift_deg": 6.0,
}


def z_quaternion(angle_deg: float) -> np.ndarray:
    half = np.radians(angle_deg) / 2
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)])


def sample(*, relative_x=0.0, relative_angle_deg=0.0,
           cube_height=0.09, grasp_z=0.1) -> dict:
    return {
        "object_relative_position_m": np.asarray([relative_x, 0.0, -0.05]),
        "object_relative_quaternion_wxyz": z_quaternion(relative_angle_deg),
        "cube_height_m": cube_height,
        "grasp_center_position_m": np.asarray([0.4, 0.1, grasp_z]),
    }


def evaluate(samples, *, task_success=True, approach_push=False):
    return evaluate_lift_outcome(
        task_success=task_success,
        approach_push=approach_push,
        grasp_close_start=sample(cube_height=0.0, grasp_z=0.0),
        lift_samples=samples,
        control_hz=30,
        success_hold_frames=15,
        baseline=BASELINE,
    )


class SE3AndOutcomeTests(unittest.TestCase):
    def test_relative_pose_and_rotation_geodesic(self):
        quarter_turn = z_quaternion(90)
        position, quaternion = relative_pose(
            [1.0, 0.0, 0.0], quarter_turn,
            [1.0, 1.0, 0.0], quarter_turn,
        )
        np.testing.assert_allclose(position, [1.0, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(
            rotation_geodesic_angle_deg([1, 0, 0, 0], quaternion), 0.0
        )
        self.assertAlmostEqual(
            rotation_geodesic_angle_deg([1, 0, 0, 0], quarter_turn), 90.0
        )
        self.assertAlmostEqual(
            rotation_geodesic_angle_deg(quarter_turn, -quarter_turn), 0.0
        )
        rotation_vector = quaternion_error_rotvec(
            quarter_turn, [1.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(rotation_vector, [0.0, 0.0, np.pi / 2])
        np.testing.assert_allclose(
            rotation_vector_to_quaternion(rotation_vector), quarter_turn
        )

    def test_stable_grasp_is_rigid_success(self):
        samples = [
            sample(cube_height=0.006 * index, grasp_z=0.006 * index)
            for index in range(20)
        ]
        result = evaluate(samples)
        self.assertTrue(result["task_success"])
        self.assertTrue(result["grasp_stable"])
        self.assertTrue(result["post_settle_stable"])
        self.assertEqual(result["outcome"], "rigid_success")

    def test_translation_slip_is_not_stable(self):
        samples = [
            sample(
                relative_x=0.0 if index < 4 else 0.02,
                cube_height=0.006 * index,
                grasp_z=0.006 * index,
            )
            for index in range(20)
        ]
        result = evaluate(samples)
        self.assertFalse(result["grasp_stable"])
        self.assertTrue(result["post_settle_stable"])
        self.assertGreater(result["max_relative_translation_drift_m"], 0.008)
        self.assertEqual(result["outcome"], "settled_after_slip")

    def test_rotation_slip_is_not_stable(self):
        samples = [
            sample(
                relative_angle_deg=0.0 if index < 4 else 10.0,
                cube_height=0.006 * index,
                grasp_z=0.006 * index,
            )
            for index in range(20)
        ]
        result = evaluate(samples)
        self.assertFalse(result["grasp_stable"])
        self.assertTrue(result["post_settle_stable"])
        self.assertGreater(result["max_relative_rotation_drift_deg"], 6.0)
        self.assertEqual(result["outcome"], "settled_after_slip")

    def test_drop_is_distinct_from_never_lift(self):
        dropped = evaluate([
            sample(cube_height=0.0, grasp_z=0.0),
            sample(cube_height=0.03, grasp_z=0.05),
            sample(cube_height=0.0, grasp_z=0.06),
        ], task_success=False)
        never_lifted = evaluate([
            sample(cube_height=0.0, grasp_z=0.0),
            sample(cube_height=0.02, grasp_z=0.05),
        ], task_success=False)
        approach_push = evaluate([
            sample(cube_height=0.0, grasp_z=0.0)
        ], task_success=False, approach_push=True)
        self.assertEqual(dropped["outcome"], "drop")
        self.assertEqual(never_lifted["outcome"], "never_lift")
        self.assertEqual(approach_push["outcome"], "approach_push")

    def test_continuing_motion_is_persistent_slip(self):
        samples = [
            sample(
                relative_x=0.0015 * index,
                cube_height=0.006 * index,
                grasp_z=0.006 * index,
            )
            for index in range(20)
        ]
        result = evaluate(samples)
        self.assertFalse(result["grasp_stable"])
        self.assertFalse(result["final_window_stable"])
        self.assertFalse(result["post_settle_stable"])
        self.assertTrue(result["continuous_slip"])
        self.assertEqual(result["outcome"], "persistent_slip")


if __name__ == "__main__":
    unittest.main()
