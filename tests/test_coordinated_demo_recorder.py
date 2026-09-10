import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from openarm_wuji.dataset import (
    CoordinatedDemoRecorder,
    export_successful_episodes,
)


def observation(frame, state, *, controller_target=None, hand_target=None):
    state = np.asarray(state, dtype=float)
    if controller_target is None:
        controller_target = np.zeros(27)
    if hand_target is None:
        hand_target = np.asarray(controller_target)[7:]
    return {
        "frame_index": frame,
        "timestamp": 100.0 + frame / 30,
        "sim_time": 0.5 + frame / 30,
        "front_rgb": np.full((4, 5, 3), frame, dtype=np.uint8),
        "wrist_rgb": np.full((4, 5, 3), frame + 1, dtype=np.uint8),
        "arm_joint_position": state[:7],
        "hand_joint_position": state[7:],
        "hand_joint_target": np.asarray(hand_target, dtype=float),
        "controller_joint_target": np.asarray(controller_target, dtype=float),
    }


def telemetry(contact=False):
    contacts = [] if not contact else [{
        "finger": "finger1",
        "other_body_name": "wuji_left_finger1_link4",
        "other_geom_name": "thumb_pad",
        "position_world_m": np.array([0.48, 0.15, 0.47]),
        "normal_force_n": 1.2,
    }]
    return {
        "cube_position_m": np.array([0.48, 0.15, 0.435]),
        "cube_quaternion_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "object_relative_position_m": np.array([0.05, 0.0, 0.0]),
        "object_relative_quaternion_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "finger_normal_forces_n": {"finger1": 1.2} if contact else {},
        "contacts": contacts,
    }


class CoordinatedDemoRecorderTests(unittest.TestCase):
    def test_precontrol_failure_is_retained_as_diagnostic(self):
        recorder = CoordinatedDemoRecorder(
            task_name="reach_grasp_lift", control_hz=30.0
        )
        recorder.start(13)
        recorder.finish({"failure_reason": "ik_failed", "outcome": "never_lift"})

        self.assertFalse(recorder.is_successful_demonstration)
        self.assertEqual(recorder.demo_summary["samples"], 0)
        self.assertEqual(recorder.demo_summary["failure_reason"], "ik_failed")

    def successful_episode(self, path: Path) -> None:
        recorder = CoordinatedDemoRecorder(
            task_name="reach_grasp_lift",
            task_description="Pick up and hold the cube.",
            control_hz=30,
            required_hold_s=1 / 30,
        )
        recorder.start(7)
        state_t = np.zeros(27)
        state_tp1 = np.linspace(0.01, 0.27, 27)
        raw_action = np.r_[np.linspace(0.2, 0.8, 7), [0.72, 0.0, 0.0]]
        target = np.r_[raw_action[:7], np.linspace(0.3, 1.1, 20)]
        recorder.record_transition(
            phase="lift_s_curve",
            observation_t=observation(0, state_t),
            action_t=raw_action,
            observation_tp1=observation(
                1, state_tp1, controller_target=target,
                hand_target=target[7:],
            ),
            telemetry_t=telemetry(False),
            telemetry_tp1=telemetry(True),
        )
        recorder.finish({
            "task_success": True,
            "grasp_stable": False,
            "outcome": "settled_after_slip",
            "peak_height_m": 0.09,
            "final_height_m": 0.085,
            "grasp": {"success": True, "reach": {"success": True}},
            "external_baseline": {"min_object_lift_m": 0.025},
            "trajectory_diagnostics": {
                "trajectory_reference_duration_s": 0.0,
                "trajectory_reference_hold_frames": 1,
            },
        })
        self.assertTrue(recorder.is_successful_demonstration)
        recorder.save(path)

    def test_action_is_27d_controller_target_not_next_actual_state(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            self.successful_episode(path)
            validation = CoordinatedDemoRecorder.validate(path)
            self.assertEqual(validation["state_shape"], [1, 27])
            self.assertEqual(validation["action_shape"], [1, 27])
            self.assertTrue(validation["demonstration_success"])
            with np.load(path, allow_pickle=False) as episode:
                self.assertEqual(episode["action"].shape, (1, 27))
                self.assertEqual(episode["raw_script_action"].shape, (1, 10))
                self.assertFalse(np.array_equal(
                    episode["action"], episode["next_observation.state"]
                ))
                np.testing.assert_array_equal(
                    episode["action"][0, :7],
                    episode["raw_script_action"][0, :7],
                )
                metadata = json.loads(str(episode["episode_metadata_json"]))
                self.assertTrue(metadata["hold_success"])
                self.assertEqual(
                    metadata["first_hand_cube_contact"]["phase"],
                    "lift_s_curve",
                )

    def test_successful_export_has_lerobot_named_columns(self):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            episode_path = directory / "episode.npz"
            self.successful_episode(episode_path)
            manifest = export_successful_episodes(
                [episode_path], directory / "staging"
            )
            self.assertEqual(manifest["episodes"], 1)
            self.assertFalse(manifest["native_lerobot_dataset"])
            table = pq.read_table(
                directory / "staging/data/chunk-000/file-000.parquet"
            )
            self.assertEqual(table.num_rows, 1)
            self.assertIn("observation.state", table.column_names)
            self.assertIn("observation.images.front", table.column_names)
            self.assertIn("action", table.column_names)
            self.assertEqual(len(table["observation.state"][0].as_py()), 27)
            self.assertEqual(len(table["action"][0].as_py()), 27)
            image = table["observation.images.front"][0].as_py()
            self.assertTrue(image["bytes"].startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
