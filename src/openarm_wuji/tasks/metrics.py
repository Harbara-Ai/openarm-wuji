from __future__ import annotations

import re

import numpy as np

from .se3 import quaternion_to_matrix


_AXES = "XYZ"


def classify_cube_contact(position_cube_m, normal_cube,
                          half_size_m, *, tolerance_m: float = 1e-3) -> dict:
    """Classify a contact on an oriented box and retain edge distances."""
    position = np.asarray(position_cube_m, dtype=float)
    normal = np.asarray(normal_cube, dtype=float)
    half_size = np.asarray(half_size_m, dtype=float)
    if (position.shape != (3,) or normal.shape != (3,)
            or half_size.shape != (3,) or np.any(half_size <= 0)
            or tolerance_m < 0):
        raise ValueError("invalid cube contact geometry")
    surface_distance = np.abs(np.abs(position) - half_size)
    nearest = float(np.min(surface_distance))
    candidates = np.flatnonzero(surface_distance <= nearest + tolerance_m)
    axis = int(candidates[np.argmax(np.abs(normal[candidates]))])
    sign = "+" if position[axis] >= 0.0 else "-"
    face = f"{sign}{_AXES[axis]}"
    tangential_axes = [index for index in range(3) if index != axis]
    margins = np.maximum(half_size[tangential_axes] - np.abs(
        position[tangential_axes]
    ), 0.0)
    on_surface_axes = int(np.sum(surface_distance <= tolerance_m))
    return {
        "cube_face": face,
        "face_plane_error_m": float(surface_distance[axis]),
        "distance_to_nearest_edge_m": float(np.min(margins)),
        "distance_to_nearest_corner_m": float(np.linalg.norm(margins)),
        "edge_contact": on_surface_axes >= 2,
        "corner_contact": on_surface_axes >= 3,
    }


def _force_line_distance_to_origin(point, force) -> float | None:
    force = np.asarray(force, dtype=float)
    magnitude = float(np.linalg.norm(force))
    if magnitude <= 1e-12:
        return None
    return float(np.linalg.norm(np.cross(np.asarray(point), force)) / magnitude)


def summarize_contact_geometry(contacts: list[dict]) -> dict:
    """Summarize opposition, contact layout, and the hand-on-object wrench."""
    faces = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    opposite = {"+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y",
                "+Z": "-Z", "-Z": "+Z"}
    thumb = [item for item in contacts if item["finger"] == "finger1"]
    others = [item for item in contacts if item["finger"] != "finger1"]

    def centroid(items):
        return (np.mean([item["position_cube_m"] for item in items], axis=0)
                if items else None)

    def resultant(items):
        return (np.sum([item["force_on_cube_world_n"] for item in items], axis=0)
                if items else np.zeros(3))

    def resultant_cube(items):
        return (np.sum([item["force_on_cube_cube_n"] for item in items], axis=0)
                if items else np.zeros(3))

    def distribution(items):
        return {face: sum(item["cube_face"] == face for item in items)
                for face in faces}

    def dominant_face(items):
        weights = {face: 0.0 for face in faces}
        for item in items:
            weights[item["cube_face"]] += float(item["normal_force_n"])
        return max(weights, key=weights.get) if items else None

    all_centroid = centroid(contacts)
    thumb_centroid = centroid(thumb)
    other_centroid = centroid(others)
    thumb_force = resultant(thumb)
    other_force = resultant(others)
    thumb_force_cube = resultant_cube(thumb)
    other_force_cube = resultant_cube(others)
    net_force = thumb_force + other_force
    net_moment = (np.sum([
        item["moment_about_cube_center_world_nm"] for item in contacts
    ], axis=0) if contacts else np.zeros(3))
    thumb_face = dominant_face(thumb)
    other_face = dominant_face(others)
    denominator = float(np.linalg.norm(thumb_force) * np.linalg.norm(other_force))
    force_angle = (float(np.degrees(np.arccos(np.clip(
        np.dot(thumb_force, other_force) / denominator, -1.0, 1.0
    )))) if denominator > 1e-12 else None)
    return {
        "contact_count": len(contacts),
        "thumb_contact_points_cube_m": [
            item["position_cube_m"].tolist() for item in thumb
        ],
        "finger_contact_centroids_cube_m": {
            finger: centroid([item for item in contacts if item["finger"] == finger]).tolist()
            for finger in sorted({item["finger"] for item in contacts})
        },
        "face_distribution": distribution(contacts),
        "thumb_face_distribution": distribution(thumb),
        "other_finger_face_distribution": distribution(others),
        "thumb_dominant_face": thumb_face,
        "other_finger_dominant_face": other_face,
        "opposition_faces": bool(
            thumb_face is not None and other_face == opposite[thumb_face]
        ),
        "contact_centroid_cube_m": (
            all_centroid.tolist() if all_centroid is not None else None
        ),
        "contact_centroid_offset_from_com_m": (
            float(np.linalg.norm(all_centroid)) if all_centroid is not None else None
        ),
        "thumb_edge_contact_rate": (
            sum(item["edge_contact"] for item in thumb) / len(thumb) if thumb else None
        ),
        "thumb_corner_contact_rate": (
            sum(item["corner_contact"] for item in thumb) / len(thumb) if thumb else None
        ),
        "thumb_mean_edge_distance_m": (
            float(np.mean([item["distance_to_nearest_edge_m"] for item in thumb]))
            if thumb else None
        ),
        "thumb_resultant_force_world_n": thumb_force.tolist(),
        "other_fingers_resultant_force_world_n": other_force.tolist(),
        "thumb_other_force_angle_deg": force_angle,
        "thumb_force_line_distance_to_com_m": (
            _force_line_distance_to_origin(thumb_centroid, thumb_force_cube)
            if thumb_centroid is not None else None
        ),
        "other_force_line_distance_to_com_m": (
            _force_line_distance_to_origin(other_centroid, other_force_cube)
            if other_centroid is not None else None
        ),
        "net_force_world_n": net_force.tolist(),
        "horizontal_net_force_n": float(np.linalg.norm(net_force[:2])),
        "net_moment_world_nm": net_moment.tolist(),
        "net_moment_magnitude_nm": float(np.linalg.norm(net_moment)),
        "edge_classification_tolerance_m": 1e-3,
    }


class FingerContactMonitor:
    """Extract per-finger cube contacts from MuJoCo's active contact buffer."""

    def __init__(self, model, data, *, cube_body_id: int,
                 cube_half_size_m, hand_side: str = "left"):
        import mujoco

        self.model = model
        self.data = data
        self.cube_body_id = int(cube_body_id)
        self.cube_half_size_m = np.asarray(cube_half_size_m, dtype=float)
        self.pattern = re.compile(rf"wuji_{re.escape(hand_side)}_finger(\d+)_")
        self.designated_tip_geom_ids = {}
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body
            ) or ""
            match = self.pattern.search(body_name)
            if match is None or not body_name.endswith("_link4"):
                continue
            if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
                continue
            finger = f"finger{match.group(1)}"
            current = self.designated_tip_geom_ids.get(finger)
            if current is None or float(np.linalg.norm(model.geom_pos[geom])) > float(np.linalg.norm(model.geom_pos[current])):
                self.designated_tip_geom_ids[finger] = geom

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
        cube_rotation = quaternion_to_matrix(self.data.xquat[self.cube_body_id])
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if self.cube_body_id not in (body1, body2):
                continue
            other_geom = contact.geom2 if body1 == self.cube_body_id else contact.geom1
            other_body = self.model.geom_bodyid[other_geom]
            other_geom_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            )
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
            position_cube = cube_rotation.T @ (position - cube_position)
            normal_cube = cube_rotation.T @ normal_on_cube
            force_cube = cube_rotation.T @ force_on_cube
            geometry = classify_cube_contact(
                position_cube, normal_cube, self.cube_half_size_m
            )
            contacts.append({
                "finger": finger,
                "other_geom_id": int(other_geom),
                "other_geom_name": other_geom_name,
                "other_body_id": int(other_body),
                "other_body_name": body_name,
                # The model has multiple collision geoms on distal link4;
                # this flag is intentionally weaker than fingertip identity.
                "is_distal_link": bool(body_name.endswith("_link4")),
                "is_designated_tip": bool(
                    self.designated_tip_geom_ids.get(finger) == int(other_geom)
                ),
                "is_distal_non_tip": bool(
                    body_name.endswith("_link4")
                    and self.designated_tip_geom_ids.get(finger) != int(other_geom)
                ),
                "contact_role": (
                    "designated_tip_geom"
                    if self.designated_tip_geom_ids.get(finger) == int(other_geom)
                    else "distal_non_tip_geom"
                    if body_name.endswith("_link4")
                    else "other_finger_link_geom"
                ),
                "position_world_m": position,
                "position_cube_m": position_cube,
                "normal_on_cube_world": normal_on_cube.copy(),
                "normal_on_cube_cube": normal_cube,
                "force_on_cube_world_n": force_on_cube,
                "force_on_cube_cube_n": force_cube,
                "contact_torque_on_cube_world_nm": torque_on_cube,
                "moment_about_cube_center_world_nm": moment_about_cube,
                "normal_force_n": normal_force,
                **geometry,
            })
        return contacts
