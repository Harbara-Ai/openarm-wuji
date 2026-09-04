from __future__ import annotations

import re

import numpy as np


class FingerContactMonitor:
    """Extract per-finger cube contacts from MuJoCo's active contact buffer."""

    def __init__(self, model, data, *, cube_body_id: int, hand_side: str = "left"):
        self.model = model
        self.data = data
        self.cube_body_id = int(cube_body_id)
        self.pattern = re.compile(rf"wuji_{re.escape(hand_side)}_finger(\d+)_")

    def sample(self, min_normal_force_n: float = 0.0) -> dict[str, float]:
        groups: dict[str, float] = {}
        for contact in self.contacts(min_normal_force_n):
            finger = contact["finger"]
            groups[finger] = max(groups.get(finger, 0.0), contact["normal_force_n"])
        return groups

    def contacts(self, min_normal_force_n: float = 0.0) -> list[dict]:
        """Return hand-object contact points and wrenches in the world frame.

        Force and torque are canonicalized as acting on the cube. MuJoCo returns
        force:torque in the contact frame acting on geom2; contact.frame stores
        its axes as rows, so transpose maps the vectors to world coordinates.
        """
        import mujoco

        if min_normal_force_n < 0:
            raise ValueError("min_normal_force_n must be non-negative")
        contacts: list[dict] = []
        cube_position = self.data.xpos[self.cube_body_id]
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if self.cube_body_id not in (body1, body2):
                continue
            other_geom = contact.geom2 if body1 == self.cube_body_id else contact.geom1
            other_body = self.model.geom_bodyid[other_geom]
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, other_body
            ) or ""
            match = self.pattern.search(body_name)
            if match is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(
                self.model, self.data, contact_index, force
            )
            normal_force = float(force[0])
            if normal_force < min_normal_force_n:
                continue
            finger = f"finger{match.group(1)}"
            contact_frame = np.asarray(contact.frame).reshape(3, 3)
            force_world_on_geom2 = contact_frame.T @ force[:3]
            torque_world_on_geom2 = contact_frame.T @ force[3:]
            cube_is_geom2 = body2 == self.cube_body_id
            sign = 1.0 if cube_is_geom2 else -1.0
            force_on_cube = sign * force_world_on_geom2
            torque_on_cube = sign * torque_world_on_geom2
            position = np.asarray(contact.pos).copy()
            moment_about_cube = (
                np.cross(position - cube_position, force_on_cube) + torque_on_cube
            )
            normal_on_cube = sign * contact_frame[0]
            contacts.append({
                "finger": finger,
                "position_world_m": position,
                "normal_on_cube_world": normal_on_cube.copy(),
                "force_on_cube_world_n": force_on_cube,
                "contact_torque_on_cube_world_nm": torque_on_cube,
                "moment_about_cube_center_world_nm": moment_about_cube,
                "normal_force_n": normal_force,
            })
        return contacts
