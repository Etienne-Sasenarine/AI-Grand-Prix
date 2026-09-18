"""Course map, gate corners, and the camera projection — NumPy only.

This mirrors the projection the policy was *trained* against
(``isaac_drone_racer/utils/aigp_obs.py``), so that what the network sees in
flight is assembled the same way it was in the simulator. Any difference here
is a silent sim-to-real gap, so ``_self_test`` cross-checks every function
against that torch implementation directly when torch is importable.

Frames, stated once
-------------------
* **World**: x east, y north, z up. This is the simulator's frame. The official
  gate table is converted into it by ``x = X_ft·0.3048``,
  ``y = (165 − Y_ft)·0.3048``.
* **Body FLU**: x forward, y left, z up. Isaac's convention.
* **Body NED**: x forward, y right, z down. Betaflight/MSP's convention.
* **Camera optical**: x right, y down, z forward along the optical axis.

Gate rotation in the official table is clockwise from "toward the top edge of
the map", which in world terms is +y. So the through-axis of a gate with
rotation ``r`` is ``(sin r, cos r, 0)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- Image and lens ---------------------------------------------------------
FRAME_W = 640.0
FRAME_H = 360.0
OFF_FRAME_MARGIN = 0.15

#: Trained-against values, inherited from the virtual qualifier's 90-degree
#: camera. Kept as the default so the parity test can compare like with like.
TRAINING_FX = TRAINING_FY = 320.0
TRAINING_CX, TRAINING_CY = 320.0, 180.0
TRAINING_TILT_DEG = 20.0

#: What the real camera actually is. Three independent calibrations of this
#: model give fx in 423.6-426.7 at 640x360; the spec's nominal 75 deg would be
#: 417.0. Use the measurements until we calibrate our own.
REAL_FX = REAL_FY = 425.0
REAL_CX, REAL_CY = 322.0, 168.0

#: **Camera tilt is locked at 20 degrees up.** Decided 17 Sep 2026; set it on
#: the airframe and do not change it without re-running the visibility sweep.
#:
#: The tilt is adjustable on the real drone, and another team runs 0 degrees on
#: all three of theirs, so this was an open question. It is now measured. What
#: decides it is that a racing quadrotor flies **nose-down** — it has to, because
#: tilting the thrust vector forward is the only way it accelerates — and that
#: pitch subtracts directly from the camera's up-tilt.
#:
#: Corners still in frame at racing attitude (0.35 m below the gate centre,
#: 12 degrees nose-down); four are needed for a pose:
#:
#:     range     tilt 0     tilt 10    tilt 20
#:      3.0 m    4          6          8
#:      4.0 m    4          8          8
#:      6.0 m    6          8          8
#:      8.0 m    8          8          8
#:
#: At 0 degrees the gate climbs out of the top of the frame under exactly the
#: attitude we spend the race in. 20 degrees holds all eight corners from 3 m
#: out. It is also what the policy was trained with, so keeping it removes a
#: sim-to-real gap rather than adding one.
#:
#: The cost is at the other end: a gate *below* the aircraft sits outside the
#: cone, which is why the double gate's lower opening needs a stand-off
#: approach. That is handled in guidance, and it is the cheaper problem.
CAMERA_TILT_DEG = 20.0
#: The real camera sits forward of the body origin. The simulator puts it at
#: the origin, which is a real projection error inside a couple of metres.
REAL_CAMERA_OFFSET_BODY_M = (0.12, 0.02, 0.013)  # forward, right, up

GATE_OUTER_M = 2.7
GATE_INNER_M = 1.5
GATE_DEPTH_M = 0.08
_HALF_OUT = GATE_OUTER_M / 2.0
_HALF_IN = GATE_INNER_M / 2.0

#: Corner order is part of the observation contract. Do not reorder.
#: Outer ring 0-3, then inner ring 4-7, as (X, Y, Z) in the gate's own frame
#: where the world offset is ``X*right + Y*down + Z*through``.
KEYPOINT_OBJECT_POINTS = np.array(
    [
        [-_HALF_OUT, -_HALF_OUT, 0.0],
        [+_HALF_OUT, -_HALF_OUT, 0.0],
        [+_HALF_OUT, +_HALF_OUT, 0.0],
        [-_HALF_OUT, +_HALF_OUT, 0.0],
        [-_HALF_IN, -_HALF_IN, 0.0],
        [+_HALF_IN, -_HALF_IN, 0.0],
        [+_HALF_IN, +_HALF_IN, 0.0],
        [-_HALF_IN, +_HALF_IN, 0.0],
    ],
    dtype=np.float64,
)

FT = 0.3048
TRACK_WIDTH_M = 85 * FT    # 25.908
TRACK_LENGTH_M = 165 * FT  # 50.292

#: Floor-standing 2.7 m frames put the opening centre here.
GATE_CENTRE_HEIGHT_M = 1.35
#: The double gate's upper opening. Third-party figure; measure it.
DOUBLE_GATE_UPPER_HEIGHT_M = 4.05


@dataclass(frozen=True)
class Gate:
    """One gate opening. ``number`` is the organizer's, 1-based."""

    number: int
    x: float
    y: float
    height: float
    yaw: float  # radians, direction of flight, world frame
    is_double_upper: bool = False

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.x, self.y, self.height], dtype=np.float64)

    @property
    def through(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])

    @property
    def right(self) -> np.ndarray:
        return np.array([math.sin(self.yaw), -math.cos(self.yaw), 0.0])

    def corners_world(self) -> np.ndarray:
        """(8, 3) world positions, in the trained corner order."""
        return gate_corners_world(self.centre, self.yaw)


# Official table: X ft, Y ft, rotation degrees clockwise from "toward top edge".
# Source: organizer gate-coordinate reference. Gate 9 is the double gate.
_OFFICIAL = [
    (1, 12.0, 71.0, 0),
    (2, 16.0, 39.0, 0),
    (3, 40.0, 17.0, 90),
    (4, 72.0, 40.0, 180),
    (5, 66.0, 70.0, 215),
    (6, 41.0, 86.0, 90),
    (7, 70.5, 106.6, 180),
    (8, 60.0, 129.0, 215),
    (9, 39.7, 147.7, 180),
    (10, 13.0, 110.0, 0),
]
DOUBLE_GATE_NUMBER = 9


def _table_to_world(x_ft: float, y_ft: float, rot_deg: float) -> tuple[float, float, float]:
    """Official table entry to world x, y and yaw (radians).

    Rotation is clockwise from "toward the top edge", which is +y in world.
    So the through-axis is (sin r, cos r); as a yaw measured from +x that is
    ``yaw = pi/2 - r``.
    """
    x = x_ft * FT
    y = (165.0 - y_ft) * FT
    yaw = math.radians(90.0 - rot_deg)
    return x, y, yaw


def official_gates(*, double_upper_height: float = DOUBLE_GATE_UPPER_HEIGHT_M) -> list[Gate]:
    """The ten published gate openings, in world coordinates."""
    out = []
    for number, x_ft, y_ft, rot in _OFFICIAL:
        x, y, yaw = _table_to_world(x_ft, y_ft, rot)
        out.append(Gate(number, x, y, GATE_CENTRE_HEIGHT_M, yaw))
    return out


def race_sequence(*, laps: int = 2, double_upper_height: float = DOUBLE_GATE_UPPER_HEIGHT_M) -> list[Gate]:
    """The ordered list of *crossings* for a run.

    This is not the same as the list of gates. Gate 9 is a stacked double gate
    and **counts twice per lap** — once through the upper opening and once
    through the lower — so it appears twice, at two different heights. Getting
    this wrong is the single likeliest way for a gate counter to lose its place.

    **The two crossings are flown in opposite directions.** The organizers'
    coordinate reference draws the racing line, and at gate 9 that line comes
    down from gate 8, passes through the structure heading down the map, makes
    a tight hairpin immediately beyond it, and comes back up through the
    structure before running on to gate 10. The gate table lists gate 9 once,
    with the direction of the *first* crossing only.

    This file previously flew both crossings in the table's direction, which
    would have meant three direction reversals and an approach to gate 10 from
    entirely the wrong side. The hairpin exits already pointing at gate 10,
    which is why the course is drawn that way.

    Still unverified, and both need eyes on the structure: the two opening
    heights, and that the upper one is taken first. Upper-then-lower is what
    the geometry favours — gate 8 and gate 10 are both at 1.35 m, so taking the
    high opening first leaves the aircraft low and pointing at gate 10 — but
    the map is a plan view and cannot show height.
    """
    gates = {g.number: g for g in official_gates()}
    seq: list[Gate] = []
    for _ in range(laps):
        for n in range(1, 11):
            g = gates[n]
            if n == DOUBLE_GATE_NUMBER:
                # Upper opening, in the direction the gate table gives.
                seq.append(Gate(g.number, g.x, g.y, double_upper_height, g.yaw,
                                is_double_upper=True))
                # Hairpin, then back the other way through the lower opening.
                back = (g.yaw + math.pi + math.pi) % (2 * math.pi) - math.pi
                seq.append(Gate(g.number, g.x, g.y, g.height, back))
            else:
                seq.append(g)
    return seq


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def gate_corners_world(gate_pos_w: np.ndarray, gate_yaw: float,
                       object_points: np.ndarray | None = None) -> np.ndarray:
    """(8, 3) world corner positions. Mirrors ``gate_corners_world`` in training.

    ``offset = X*right + Y*down + Z*through`` with ``down = (0, 0, -1)``.
    """
    op = KEYPOINT_OBJECT_POINTS if object_points is None else np.asarray(object_points, dtype=np.float64)
    cy, sy = math.cos(gate_yaw), math.sin(gate_yaw)
    through = np.array([cy, sy, 0.0])
    right = np.array([sy, -cy, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    offset = (op[:, 0:1] * right + op[:, 1:2] * down + op[:, 2:3] * through)
    return np.asarray(gate_pos_w, dtype=np.float64).reshape(1, 3) + offset


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the body frame of a wxyz quaternion."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-8)
    w, x, y, z = q
    # R(q)^T applied to v.
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
    ])
    return np.asarray(v, dtype=np.float64) @ r.T


def flu_to_ned(v: np.ndarray) -> np.ndarray:
    """Isaac body FLU to MAVLink/Betaflight body NED. An involution."""
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    out[..., 1] = -v[..., 1]
    out[..., 2] = -v[..., 2]
    return out


def camera_body_rotation(tilt_up_deg: float) -> np.ndarray:
    """Body NED to camera-optical rotation."""
    t = math.radians(tilt_up_deg)
    st, ct = math.sin(t), math.cos(t)
    return np.array([[0.0, 1.0, 0.0], [st, 0.0, ct], [ct, 0.0, -st]])


def yaw_from_quat(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body FLU orientation as wxyz, from ZYX Euler angles."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_from_ahrs(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rebuild a body-FLU quaternion from gravity-referenced roll and pitch.

    The inverse of :func:`roll_pitch_from_quat`. It exists because the two
    conventions differ by a sign on pitch, and rebuilding an attitude with
    ``quat_from_euler(roll, pitch, yaw)`` silently gives the aircraft the wrong
    lean — which in closed-loop testing showed up as the guidance controller
    demanding maximum nose-down rate for ever.

    AHRS roll equals the FLU Euler roll; AHRS pitch is its negation, because
    the AHRS convention is defined in NED (z down) and ours in FLU (z up).
    """
    return quat_from_euler(roll, -pitch, yaw)


def body_to_world_matrix(q_wxyz) -> np.ndarray:
    """3x3 rotation taking body-FLU vectors to world: ``v_world = R @ v_body``."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def body_up_from_quat(q_wxyz) -> np.ndarray:
    """Body FLU +z (the thrust direction) in world coordinates."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([2.0 * (x * z + w * y),
                     2.0 * (y * z - w * x),
                     1.0 - 2.0 * (x * x + y * y)])


def body_up_from_ahrs(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Thrust direction in world, from gravity-referenced roll and pitch.

    Use this anywhere the attitude arrives as AHRS roll/pitch — which is how the
    flight controller reports it, and what the policy reads. Building the vector
    inline from a Euler formula is how this went wrong three separate times: the
    AHRS pitch is the negation of the FLU Euler pitch, so an inline formula
    mirrors the horizontal components and the predicted acceleration points the
    wrong way across the ground while looking perfectly fine vertically.
    """
    return body_up_from_quat(quat_from_ahrs(roll, pitch, yaw))


def level_bearing_from_ahrs(vec_body_flu, roll: float, pitch: float) -> float:
    """Horizontal bearing of a body-frame direction, measured from our heading.

    Adding this to the vehicle's yaw gives the bearing in the world. That is
    the whole point: it is the quantity a gate sighting can be compared against
    the map to *observe heading*.

    **Tilt compensation is not optional here.** ``atan2(y, x)`` of a body-frame
    vector is an angle in the *body* frame, and the body is rolled and pitched.
    Feeding it straight into a yaw residual reports a heading error that is not
    there, proportional to bank angle times how far the target sits off the
    optical axis. Measured with a perfect pose, a perfect map and zero true yaw
    error: 4.3 degrees of phantom residual at 25 degrees of bank, 11.8 degrees
    at 35 degrees of bank with the nose 15 degrees down.

    That is not noise — it is *signed with the turn*. This course turns
    predominantly one way, so the yaw filter integrated it, and by the seventh
    gate the estimated heading bias had walked to **+21 degrees**. At fx = 425
    that puts the predicted gate 163 px from where it actually is, the tracker
    stops recognising the gate in front of it, and the run unravels from there.

    So: rotate into a frame that shares our heading but is level, then take the
    angle. With the rotation applied, the residual is exactly zero in every case
    above.
    """
    v = np.asarray(vec_body_flu, dtype=np.float64).reshape(3)
    # Rotating by (roll, pitch, yaw = 0) lands in a frame that is level with the
    # world and still aligned with our nose, which is exactly what we want.
    w = body_to_world_matrix(quat_from_ahrs(roll, pitch, 0.0)) @ v
    return math.atan2(float(w[1]), float(w[0]))


def roll_pitch_from_quat(q_wxyz) -> tuple[float, float]:
    """Gravity-referenced roll and pitch in the body NED frame, in radians.

    This must match ``attitude_roll_pitch_ned`` in the training code exactly,
    because it is what the policy reads in observation slots 24 and 25.

    It is deliberately *not* a ZYX Euler extraction. The training code computes
    the attitude the way an AHRS does — by rotating the gravity direction into
    the body frame and taking angles from it:

        roll  = atan2(g_y, g_z)
        pitch = atan2(-g_x, hypot(g_y, g_z))

    An Euler extraction agrees on roll but gives pitch with the opposite sign,
    which would tell the policy the aircraft is leaning the other way. That bug
    was live in this file until a closed-loop test contradicted the physics;
    ``test_parity`` now checks this function directly so it cannot come back.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    g_world = np.array([[0.0, 0.0, -1.0]])
    g_flu = quat_rotate_inverse(q, g_world)
    g = flu_to_ned(g_flu)[0]
    roll = math.atan2(g[1], g[2])
    pitch = math.atan2(-g[0], max(math.hypot(g[1], g[2]), 1e-8))
    return roll, pitch


@dataclass
class Camera:
    """Pinhole camera on the drone.

    ``offset_body_flu`` is the camera's position relative to the body origin,
    in FLU metres. Training used (0, 0, 0); the real camera is 12 cm forward.
    Leaving it at zero reproduces the trained projection exactly.
    """

    fx: float = TRAINING_FX
    fy: float = TRAINING_FY
    cx: float = TRAINING_CX
    cy: float = TRAINING_CY
    tilt_up_deg: float = TRAINING_TILT_DEG
    frame_w: float = FRAME_W
    frame_h: float = FRAME_H
    off_frame_margin: float = OFF_FRAME_MARGIN
    offset_body_flu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _r_cb: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._r_cb = camera_body_rotation(self.tilt_up_deg)

    @classmethod
    def training(cls) -> "Camera":
        """Exactly what the policy was trained against."""
        return cls()

    @classmethod
    def real(cls, *, tilt_up_deg: float = CAMERA_TILT_DEG) -> "Camera":
        """Best current estimate of the real camera. Replace after calibration.

        ``tilt_up_deg`` defaults to the locked 20 degrees — see
        ``CAMERA_TILT_DEG``. It stays an argument only so the visibility sweep
        can ask "what if", not because the value is open.
        """
        return cls(fx=REAL_FX, fy=REAL_FY, cx=REAL_CX, cy=REAL_CY,
                   tilt_up_deg=tilt_up_deg,
                   offset_body_flu=REAL_CAMERA_OFFSET_BODY_M)

    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.frame_w / 2.0) / self.fx))

    def vfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.frame_h / 2.0) / self.fy))

    def project(self, points_w: np.ndarray, drone_pos_w: np.ndarray,
                drone_quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points to pixels. Returns (uv (K,2), visible (K,) bool)."""
        pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
        rel_w = pts - np.asarray(drone_pos_w, dtype=np.float64).reshape(1, 3)
        p_flu = quat_rotate_inverse(drone_quat_wxyz, rel_w)
        if any(self.offset_body_flu):
            p_flu = p_flu - np.asarray(self.offset_body_flu, dtype=np.float64).reshape(1, 3)
        p_ned = flu_to_ned(p_flu)
        p_cam = p_ned @ self._r_cb.T

        z = p_cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.fx * (p_cam[:, 0] / z_safe) + self.cx
        v = self.fy * (p_cam[:, 1] / z_safe) + self.cy

        mx = self.off_frame_margin * self.frame_w
        my = self.off_frame_margin * self.frame_h
        in_front = z > 1e-4
        in_frame = (u >= -mx) & (u <= self.frame_w + mx) & (v >= -my) & (v <= self.frame_h + my)
        return np.stack([u, v], axis=-1), in_front & in_frame


def _attitude_round_trip_check() -> None:
    """The attitude conventions must be mutually consistent."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        r, p, y = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-math.pi, math.pi)
        q = quat_from_ahrs(r, p, y)
        rr, pp = roll_pitch_from_quat(q)
        worst = max(worst, abs(rr - r), abs(pp - p))
        assert abs(np.linalg.norm(body_up_from_ahrs(r, p, y)) - 1.0) < 1e-12
        assert np.allclose(body_up_from_ahrs(r, p, y), body_up_from_quat(q), atol=1e-12)
    assert worst < 1e-9, f"attitude round trip is broken by {worst}"

    # Nose down must lean the thrust vector forward, not backward. This is the
    # assertion that would have caught the mirrored-acceleration bug.
    fwd = body_up_from_ahrs(0.0, -0.3, 0.0)   # AHRS pitch negative = nose down
    assert fwd[0] > 0.2, f"nose down must lean thrust forward, got {fwd}"
    back = body_up_from_ahrs(0.0, +0.3, 0.0)
    assert back[0] < -0.2, f"nose up must lean thrust backward, got {back}"
    # Yawed 90 degrees (facing +y), nose down must lean thrust along +y.
    left = body_up_from_ahrs(0.0, -0.3, math.pi / 2)
    assert left[1] > 0.2, f"yaw must rotate the lean direction, got {left}"

    _bearing_consistency_check()


def _bearing_consistency_check() -> None:
    """A level bearing plus our heading must equal the true world bearing.

    Stated as a physical fact: where a landmark *is* does not depend on how the
    aircraft is banked. So for a known landmark and a known attitude, the
    recovered world bearing must come out the same at every roll and pitch.
    The naive body-frame angle fails this by up to 11.8 degrees, which is the
    bug that walked the yaw estimate to +21 degrees in closed-loop testing.
    """
    rng = np.random.default_rng(3)
    worst_level, worst_naive = 0.0, 0.0
    for _ in range(400):
        yaw = rng.uniform(-math.pi, math.pi)
        roll = rng.uniform(-0.7, 0.7)
        pitch = rng.uniform(-0.4, 0.4)
        # A landmark somewhere ahead, in world coordinates.
        target_world = np.array([rng.uniform(2.0, 12.0), rng.uniform(-6.0, 6.0),
                                 rng.uniform(-2.0, 2.0)])
        truth = math.atan2(target_world[1], target_world[0])

        q = quat_from_ahrs(roll, pitch, yaw)
        in_body = quat_rotate_inverse(q, target_world.reshape(1, 3))[0]

        got = yaw + level_bearing_from_ahrs(in_body, roll, pitch)
        worst_level = max(worst_level, abs((got - truth + math.pi) % (2 * math.pi) - math.pi))

        naive = yaw + math.atan2(in_body[1], in_body[0])
        worst_naive = max(worst_naive, abs((naive - truth + math.pi) % (2 * math.pi) - math.pi))

    assert worst_level < 1e-9, (
        f"level_bearing_from_ahrs is not tilt-compensated: {math.degrees(worst_level):.3f} deg")
    # And confirm the naive version really is as wrong as the docstring claims,
    # so nobody "simplifies" this back to an atan2 on body components.
    assert worst_naive > math.radians(5.0), (
        "the naive body-frame bearing should be badly wrong under bank; if this "
        "fails the test has stopped exercising the case it exists for")


def signed_distance_through(gate: Gate, pos_w: np.ndarray) -> float:
    """Signed distance along the gate's through-axis. Negative is before it."""
    return float(np.dot(np.asarray(pos_w, dtype=np.float64) - gate.centre, gate.through))


def lateral_offsets(gate: Gate, pos_w: np.ndarray) -> tuple[float, float]:
    """(sideways, vertical) offset from the opening centre, in the gate plane."""
    d = np.asarray(pos_w, dtype=np.float64) - gate.centre
    return float(np.dot(d, gate.right)), float(d[2])


def passes_opening(gate: Gate, pos_w: np.ndarray, *,
                   half_span: float = 0.30, half_height: float = 0.12) -> bool:
    """Would an airframe of this size clear the 1.5 m opening at this point?"""
    side, vert = lateral_offsets(gate, pos_w)
    limit = GATE_INNER_M / 2.0
    return abs(side) + half_span <= limit and abs(vert) + half_height <= limit
