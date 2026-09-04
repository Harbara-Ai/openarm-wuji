from __future__ import annotations

import numpy as np


def normalize_quaternion(quaternion) -> np.ndarray:
    """Return a unit scalar-first (w, x, y, z) quaternion."""
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must be finite and have shape (4,)")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    return value / norm


def quaternion_conjugate(quaternion) -> np.ndarray:
    value = normalize_quaternion(quaternion)
    return value * np.asarray([1.0, -1.0, -1.0, -1.0])


def quaternion_multiply(left, right) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quaternion(left)
    w2, x2, y2, z2 = normalize_quaternion(right)
    return normalize_quaternion(np.asarray([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]))


def quaternion_to_matrix(quaternion) -> np.ndarray:
    w, x, y, z = normalize_quaternion(quaternion)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def relative_pose(reference_position, reference_quaternion,
                  target_position, target_quaternion) -> tuple[np.ndarray, np.ndarray]:
    """Express the target SE(3) pose in the reference coordinate frame."""
    reference_position = np.asarray(reference_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    if reference_position.shape != (3,) or target_position.shape != (3,):
        raise ValueError("pose positions must have shape (3,)")
    reference_quaternion = normalize_quaternion(reference_quaternion)
    target_quaternion = normalize_quaternion(target_quaternion)
    position = quaternion_to_matrix(reference_quaternion).T @ (
        target_position - reference_position
    )
    quaternion = quaternion_multiply(
        quaternion_conjugate(reference_quaternion), target_quaternion
    )
    return position, quaternion


def rotation_geodesic_angle_deg(first_quaternion, second_quaternion) -> float:
    """Shortest SO(3) geodesic angle in degrees, invariant to quaternion sign."""
    first = normalize_quaternion(first_quaternion)
    second = normalize_quaternion(second_quaternion)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    if dot >= 1.0 - 1e-12:
        return 0.0
    return float(np.degrees(2.0 * np.arccos(dot)))


def pose_drift(reference_position, reference_quaternion,
               position, quaternion) -> tuple[float, float]:
    translation = float(np.linalg.norm(
        np.asarray(position, dtype=float) - np.asarray(reference_position, dtype=float)
    ))
    rotation_deg = rotation_geodesic_angle_deg(reference_quaternion, quaternion)
    return translation, rotation_deg
