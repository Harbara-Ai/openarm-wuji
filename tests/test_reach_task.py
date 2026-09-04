import json
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from openarm_wuji.dataset import CausalEpisodeRecorder, replay_causal_episode
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji
from openarm_wuji.tasks import ReachGraspLiftTask
from openarm_wuji.tasks.reach_grasp_lift import paced_lift_waypoint


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

    def test_lift_waypoints_follow_external_50mm_in_half_second_protocol(self):
        start = np.array([0.4, 0.2, 0.5])
        delta = np.array([0.0, 0.0, 0.12])
        baseline = self.config["external_baseline"]
        halfway = paced_lift_waypoint(
            start_position=start,
            total_delta=delta,
            step=15,
            control_hz=30,
            baseline=baseline,
        )
        final = paced_lift_waypoint(
            start_position=start,
            total_delta=delta,
            step=120,
            control_hz=30,
            baseline=baseline,
        )
        np.testing.assert_allclose(halfway - start, [0.0, 0.0, 0.05])
        np.testing.assert_allclose(final - start, delta)

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
            import mujoco

            site_id = mujoco.mj_name2id(
                robot.model,
                mujoco.mjtObj.mjOBJ_SITE,
                self.config["scene"]["grasp_site_name"],
            )
            self.assertEqual(robot.model.site_group[site_id], 4)
        finally:
            robot.disconnect()

    def test_power_grasp_has_sustained_multifinger_contact(self):
        robot = self.make_robot()
        robot.connect()
        try:
            task = ReachGraspLiftTask(robot, copy.deepcopy(self.config))
            result = task.run_grasp(7)
            self.assertTrue(result.success, result.to_dict())
            self.assertGreaterEqual(result.max_contact_groups, 2)
            self.assertGreaterEqual(result.contact_hold_frames, 8)
            self.assertGreaterEqual(len(result.final_contact_groups), 2)
            self.assertLess(result.approach_cube_displacement_m, 0.025)
            self.assertTrue(result.synergy_frozen)
            self.assertAlmostEqual(result.final_synergy, 0.72, places=12)
            self.assertEqual(result.close_steps, 24)
            self.assertEqual(result.settle_steps, result.settle_window_frames)
            self.assertEqual(len(task._phase_samples["grasp_close"]), 24)
            self.assertEqual(len(task._phase_samples["grasp_settle"]), 8)
            self.assertTrue(result.settle_stable)
            self.assertLess(result.settle_max_translation_drift_m, 0.008)
            self.assertLess(result.settle_max_rotation_drift_deg, 6.0)
        finally:
            robot.disconnect()

    def test_impossible_contact_requirement_reports_grasp_empty(self):
        robot = self.make_robot()
        robot.connect()
        try:
            config = copy.deepcopy(self.config)
            config["grasp"]["min_finger_groups"] = 6
            result = ReachGraspLiftTask(robot, config).run_grasp(7)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_reason, "grasp_empty")
            self.assertLessEqual(result.max_contact_groups, 5)
        finally:
            robot.disconnect()

    def test_paced_lift_seed7_reaches_height_but_slips(self):
        robot = self.make_robot()
        robot.connect()
        try:
            result = ReachGraspLiftTask(
                robot, copy.deepcopy(self.config)
            ).run_lift(7)
            self.assertTrue(result.task_success, result.to_dict())
            self.assertFalse(result.grasp_stable, result.to_dict())
            self.assertFalse(result.success, result.to_dict())
            self.assertEqual(result.outcome, "settled_after_slip")
            self.assertGreaterEqual(result.peak_height_m, 0.08)
            self.assertGreaterEqual(result.final_height_m, 0.025)
            self.assertAlmostEqual(
                result.gripper_lift_within_baseline_window_m,
                self.config["external_baseline"]["gripper_lift_m"],
                delta=0.001,
            )
            self.assertGreater(result.max_relative_translation_drift_m, 0.008)
            self.assertGreater(result.max_relative_rotation_drift_deg, 6.0)
        finally:
            robot.disconnect()

    def test_contact_diagnostics_include_world_wrench(self):
        robot = self.make_robot()
        robot.connect()
        try:
            result = ReachGraspLiftTask(
                robot, copy.deepcopy(self.config)
            ).run_lift(7)
            diagnostics = result.contact_diagnostics
            self.assertEqual(
                diagnostics["role"], "evaluation_only_not_policy_observation"
            )
            self.assertGreater(diagnostics["peak_resultant_force_magnitude_n"], 0)
            self.assertEqual(len(diagnostics["final_resultant_force_world_n"]), 3)
            self.assertEqual(
                len(diagnostics["final_resultant_moment_about_cube_world_nm"]), 3
            )
        finally:
            robot.disconnect()

    def test_unstable_settle_rejects_lift_and_requests_regrasp(self):
        robot = self.make_robot()
        robot.connect()
        try:
            config = copy.deepcopy(self.config)
            config["external_baseline"]["max_relative_translation_drift_m"] = 0.0
            config["external_baseline"]["max_relative_rotation_drift_deg"] = 0.0
            task = ReachGraspLiftTask(robot, config)
            result = task.run_lift(7)
            self.assertFalse(result.grasp.success)
            self.assertTrue(result.grasp.synergy_frozen)
            self.assertFalse(result.grasp.settle_stable)
            self.assertTrue(result.grasp.regrasp_required)
            self.assertEqual(result.grasp.failure_reason, "grasp_unstable")
            self.assertEqual(result.steps, 0)
            self.assertNotIn("lift", task._phase_samples)
        finally:
            robot.disconnect()

    def test_complete_state_machine_records_causal_episode_and_replays(self):
        recorder = CausalEpisodeRecorder(
            task_name="reach_grasp_lift", control_hz=30
        )
        robot = self.make_robot()
        robot.connect()
        try:
            task = ReachGraspLiftTask(
                robot, copy.deepcopy(self.config), recorder=recorder
            )
            result = task.run_lift(7)
            self.assertTrue(result.task_success, result.to_dict())
            self.assertFalse(result.grasp_stable, result.to_dict())
            self.assertEqual(result.outcome, "settled_after_slip")
            recorder.finish(result.to_dict())
            with TemporaryDirectory() as temporary_directory:
                episode_path = Path(temporary_directory) / "episode.npz"
                recorder.save(episode_path)
                validation = CausalEpisodeRecorder.validate(episode_path)
                self.assertGreater(validation["samples"], 0)
                self.assertEqual(
                    validation["phases"],
                    ["approach", "grasp_close", "grasp_settle", "lift", "reach"],
                )
                with np.load(episode_path, allow_pickle=False) as episode:
                    samples = validation["samples"]
                    self.assertEqual(episode["observation_state"].shape, (samples, 27))
                    self.assertEqual(episode["action"].shape, (samples, 10))
                    self.assertEqual(episode["cube_quaternion_wxyz"].shape, (samples, 4))
                    self.assertEqual(
                        episode["grasp_center_quaternion_wxyz"].shape, (samples, 4)
                    )
                    self.assertEqual(
                        episode["object_relative_position"].shape, (samples, 3)
                    )
                    self.assertEqual(
                        episode["object_relative_quaternion_wxyz"].shape,
                        (samples, 4),
                    )
                    self.assertEqual(
                        episode["contact_sample_offsets"].shape, (samples + 2,)
                    )
                    self.assertGreater(len(episode["contact_finger"]), 0)
                    self.assertEqual(episode["observation_front_rgb"].dtype, np.uint8)
                    self.assertEqual(episode["observation_wrist_rgb"].dtype, np.uint8)
                    self.assertEqual(int(episode["frame_index"][0]), 0)
                    np.testing.assert_array_equal(
                        episode["frame_index"][1:],
                        episode["next_frame_index"][:-1],
                    )
                    np.testing.assert_array_equal(
                        episode["observation_state"][1:],
                        episode["next_observation_state"][:-1],
                    )
                    self.assertIn("host_timestamp", episode.files)
                    self.assertNotIn("timestamp", episode.files)

                replay_robot = self.make_robot()
                replay_robot.connect()
                try:
                    replay_task = ReachGraspLiftTask(
                        replay_robot, copy.deepcopy(self.config)
                    )
                    replay = replay_causal_episode(
                        episode_path,
                        robot=replay_robot,
                        reset_fn=replay_task.reset,
                        telemetry_fn=replay_task.task_telemetry,
                    )
                finally:
                    replay_robot.disconnect()
                self.assertEqual(replay["samples"], validation["samples"])
                self.assertEqual(replay["max_action_error"], 0.0)
                self.assertEqual(replay["max_state_error"], 0.0)
                self.assertEqual(replay["max_cube_position_error"], 0.0)
                self.assertEqual(replay["max_cube_rotation_error_deg"], 0.0)
                self.assertEqual(replay["max_grasp_center_position_error"], 0.0)
                self.assertEqual(replay["max_grasp_center_rotation_error_deg"], 0.0)
                self.assertEqual(replay["max_relative_position_error"], 0.0)
                self.assertEqual(replay["max_relative_rotation_error_deg"], 0.0)
                self.assertEqual(replay["max_contact_resultant_force_error_n"], 0.0)
                self.assertEqual(replay["max_contact_resultant_moment_error_nm"], 0.0)
                self.assertEqual(replay["max_contact_position_error_m"], 0.0)
                self.assertEqual(replay["max_contact_force_error_n"], 0.0)
                self.assertEqual(replay["contact_mismatch_frames"], 0)
                self.assertEqual(replay["max_sim_time_error"], 0.0)
                self.assertLessEqual(replay["max_front_pixel_error"], 2)
                self.assertLessEqual(replay["max_wrist_pixel_error"], 2)
        finally:
            robot.disconnect()


if __name__ == "__main__":
    unittest.main()
