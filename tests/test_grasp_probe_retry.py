import unittest

import numpy as np

from openarm_wuji.tasks.grasp_probe_retry import (
    MicroLiftProbeSpec,
    evaluate_probe_arrays,
    grasp_state_delta,
    probe_reference_frame,
)


def probe_arrays(*, cube_lift=0.012, palm_lift=0.016,
                 supported=False, contact_points=4,
                 translation_drift=0.004, relative_speed=0.002):
    frames = 15
    phase = np.asarray(
        ["micro_lift_probe"] * 6 + ["micro_lift_probe_hold"] * 9
    )
    return {
        "phase": phase,
        "controller_target": np.zeros((frames, 27)),
        "actual_state": np.zeros((frames, 27)),
        "cube_pose_world": np.zeros((frames, 7)),
        "cube_pose_relative_to_palm": np.zeros((frames, 7)),
        "cube_height_m": np.linspace(0.0, cube_lift, frames),
        "palm_lift_m": np.linspace(0.0, palm_lift, frames),
        "environment_support": np.full(frames, supported, dtype=bool),
        "hand_contact_points": np.full(frames, contact_points),
        "relative_translation_drift_m": np.linspace(
            0.0, translation_drift, frames
        ),
        "relative_rotation_drift_deg": np.linspace(0.0, 2.0, frames),
        "relative_linear_speed_m_s": np.full(frames, relative_speed),
        "contact_slip_max_m_s": np.full(frames, 0.001),
    }


class GraspProbeRetryTest(unittest.TestCase):
    def test_probe_uses_a_prefix_of_existing_lift(self):
        frame = probe_reference_frame(
            full_delta_m=[0.0, 0.0, 0.12],
            probe_height_m=0.015,
            control_hz=30,
            baseline={
                "lift_duration_s": 0.5,
                "gripper_lift_m": 0.05,
            },
            limits={
                "max_velocity_m_s": 0.19,
                "max_acceleration_m_s2": 1.16,
                "max_jerk_m_s3": 24.1,
            },
        )
        self.assertEqual(frame, 6)

    def test_probe_pass_is_load_and_support_based(self):
        result = evaluate_probe_arrays(
            probe_arrays(), spec=MicroLiftProbeSpec(), hold_frames=9,
            trajectory_complete=True, ik_failure_frame=None,
        )
        self.assertTrue(result["pass"])
        self.assertTrue(result["criteria"]["cube_followed"])
        self.assertTrue(result["criteria"]["environment_support_released"])

    def test_probe_rejects_support_and_contact_collapse(self):
        result = evaluate_probe_arrays(
            probe_arrays(supported=True, contact_points=0),
            spec=MicroLiftProbeSpec(), hold_frames=9,
            trajectory_complete=True, ik_failure_frame=None,
        )
        self.assertFalse(result["pass"])
        self.assertIn("environment_support_released", result["failure_reasons"])
        self.assertIn("contact_retained", result["failure_reasons"])

    def test_settle_frames_do_not_count_as_the_hold_window(self):
        arrays = probe_arrays()
        arrays = {
            key: np.concatenate(([value[0]] * 5, value))
            for key, value in arrays.items()
        }
        arrays["phase"][:5] = "micro_lift_probe_settle"
        arrays["environment_support"][:5] = True
        result = evaluate_probe_arrays(
            arrays, spec=MicroLiftProbeSpec(), hold_frames=9,
            trajectory_complete=True, ik_failure_frame=None,
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["supported_hold_fraction"], 0.0)

    def test_missing_physical_hold_is_not_trajectory_complete(self):
        arrays = probe_arrays()
        arrays["phase"][:] = "micro_lift_probe_settle"
        result = evaluate_probe_arrays(
            arrays, spec=MicroLiftProbeSpec(), hold_frames=9,
            trajectory_complete=True, ik_failure_frame=None,
        )
        self.assertFalse(result["pass"])
        self.assertIn("trajectory_complete", result["failure_reasons"])

    def test_regrasp_delta_reports_continuous_changes(self):
        first = {
            "actual_hand_qpos_rad": np.zeros(20),
            "target_hand_qpos_rad": np.zeros(20),
            "preload_l2_rad": 0.5,
            "per_finger_normal_force_n": np.zeros(5),
            "cube_pose_relative_to_palm": np.r_[np.zeros(3), [1, 0, 0, 0]],
            "contact_topology": "finger1+finger2",
        }
        second = {
            "actual_hand_qpos_rad": np.full(20, 0.1),
            "target_hand_qpos_rad": np.full(20, 0.2),
            "preload_l2_rad": 0.7,
            "per_finger_normal_force_n": np.ones(5),
            "cube_pose_relative_to_palm": np.r_[
                [0.003, 0, 0], [np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12)]
            ],
            "contact_topology": "finger1+finger3",
        }
        delta = grasp_state_delta(first, second)
        self.assertAlmostEqual(delta["target_hand_l2_change_rad"], np.sqrt(0.8))
        self.assertAlmostEqual(delta["cube_palm_translation_change_m"], 0.003)
        self.assertAlmostEqual(delta["cube_palm_rotation_change_deg"], 30.0)
        self.assertTrue(delta["topology_changed"])


if __name__ == "__main__":
    unittest.main()
