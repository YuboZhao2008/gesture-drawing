"""Rotation-invariant geometry in the coordinate units supplied by the caller."""

from math import acos, degrees, dist

from gesture_control.core.types import Landmark3D


def landmark_distance(a: Landmark3D, b: Landmark3D) -> float:
    """Euclidean 3D distance (palm units for normalized landmarks)."""
    return dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def joint_angle(a: Landmark3D, joint: Landmark3D, b: Landmark3D) -> float:
    """Interior angle in degrees; a collapsed bone returns zero."""
    u = (a.x - joint.x, a.y - joint.y, a.z - joint.z)
    v = (b.x - joint.x, b.y - joint.y, b.z - joint.z)
    denominator = landmark_distance(a, joint) * landmark_distance(b, joint)
    if denominator <= 1e-12:
        return 0.0
    cosine = sum(x * y for x, y in zip(u, v, strict=True)) / denominator
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def finger_extension_score(
    mcp: Landmark3D, pip: Landmark3D, dip: Landmark3D, tip: Landmark3D
) -> float:
    """Straightness in [0, 1], combining joint angles and chain reach.

    Reach prevents a folded-back fingertip with one straight joint from looking
    extended. Uses no image axes, handedness, or absolute bone lengths.
    """
    length = landmark_distance(mcp, pip) + landmark_distance(pip, dip) + landmark_distance(dip, tip)
    if length <= 1e-12:
        return 0.0
    reach = landmark_distance(mcp, tip) / length
    bend = min(joint_angle(mcp, pip, dip), joint_angle(pip, dip, tip)) / 180.0
    return min(reach, bend)
