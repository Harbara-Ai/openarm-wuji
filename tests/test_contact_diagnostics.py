import unittest

import numpy as np

from openarm_wuji.tasks.metrics import (
    classify_cube_contact,
    summarize_contact_geometry,
)


def contact(*, finger, position, face, force):
    position = np.asarray(position, dtype=float)
    force = np.asarray(force, dtype=float)
    return {
        "finger": finger,
        "position_cube_m": position,
        "force_on_cube_world_n": force,
        "force_on_cube_cube_n": force,
        "moment_about_cube_center_world_nm": np.cross(position, force),
        "normal_force_n": float(np.linalg.norm(force)),
        "cube_face": face,
        "distance_to_nearest_edge_m": 0.02,
        "edge_contact": False,
        "corner_contact": False,
    }


class ContactDiagnosticTests(unittest.TestCase):
    def test_cube_face_and_edge_classification(self):
        face = classify_cube_contact(
            [-0.035, 0.0, 0.0], [1.0, 0.0, 0.0], [0.035] * 3
        )
        edge = classify_cube_contact(
            [-0.035, -0.035, 0.0], [1.0, 1.0, 0.0], [0.035] * 3
        )
        corner = classify_cube_contact(
            [-0.035, -0.035, 0.035], [1.0, 1.0, -1.0], [0.035] * 3
        )
        self.assertEqual(face["cube_face"], "-X")
        self.assertFalse(face["edge_contact"])
        self.assertTrue(edge["edge_contact"])
        self.assertFalse(edge["corner_contact"])
        self.assertTrue(corner["corner_contact"])

    def test_opposed_forces_and_wrench_are_reported(self):
        contacts = [
            contact(
                finger="finger1", position=[-0.035, 0.0, 0.0],
                face="-X", force=[2.0, 0.0, 0.0],
            ),
            contact(
                finger="finger2", position=[0.035, 0.0, 0.0],
                face="+X", force=[-2.0, 0.0, 0.0],
            ),
        ]
        result = summarize_contact_geometry(contacts)
        self.assertTrue(result["opposition_faces"])
        self.assertAlmostEqual(result["thumb_other_force_angle_deg"], 180.0)
        self.assertAlmostEqual(result["horizontal_net_force_n"], 0.0)
        self.assertAlmostEqual(result["net_moment_magnitude_nm"], 0.0)
        self.assertAlmostEqual(result["thumb_force_line_distance_to_com_m"], 0.0)
        self.assertAlmostEqual(result["other_force_line_distance_to_com_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
