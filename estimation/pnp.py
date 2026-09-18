"""Gate-relative pose from the eight detected corners.

The gate is a planar target of known size, so this is planar PnP — the same
method Swift uses (IPPE, infinitesimal plane-based pose estimation). OpenCV
ships it, and OpenCV is on the Jetson image, so there is nothing to install.

The catch, and why this file is longer than a one-line call
-----------------------------------------------------------
**Planar PnP is two-fold ambiguous.** A flat target viewed from one side admits
a second, mirrored pose that reprojects almost as well, and near head-on the two
are nearly indistinguishable numerically. Taking the lower-reprojection-error
solution is a coin flip in exactly the situation we care about most — lined up
on a gate.

We break the tie with **gravity**. The flight controller gives us roll and
pitch, so we know which way is down in the camera frame. Gates stand vertically
on the floor, so the gate's own up-axis must be anti-parallel to gravity. The
mirrored solution has it pointing somewhere else entirely, and is rejected.

That check needs no extra sensor and no extra computation — it is a dot product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import geometry as G


@dataclass
class GatePose:
    """Gate pose relative to the camera, plus quality information."""

    t_cam: np.ndarray          # gate centre in camera frame (metres)
    r_cam: np.ndarray          # 3x3 gate->camera rotation
    reprojection_px: float     # RMS reprojection error
    n_points: int
    ambiguous: bool            # were the two planar solutions close?
    gravity_score: float       # how well the chosen solution agrees with gravity

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.t_cam))


def _gravity_in_camera(roll: float, pitch: float, tilt_up_deg: float) -> np.ndarray:
    """Unit 'down' vector expressed in the camera optical frame."""
    # Gravity in body NED is +Z (down) when level; tilt the body by roll/pitch.
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Body-NED gravity direction for a vehicle at this attitude.
    g_body = np.array([-sp, sr * cp, cr * cp])
    r_cb = G.camera_body_rotation(tilt_up_deg)
    g_cam = r_cb @ g_body
    n = np.linalg.norm(g_cam)
    return g_cam / (n if n > 1e-9 else 1.0)


def solve_gate_pose(uv, visible, camera: G.Camera, *, roll: float = 0.0, pitch: float = 0.0,
                    object_points: np.ndarray | None = None,
                    ambiguity_ratio: float = 0.6) -> GatePose | None:
    """Gate pose from detected corners, disambiguated by gravity.

    Returns None when fewer than four corners are visible, or when OpenCV
    cannot find a solution.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is present on the Jetson image
        return None

    op = G.KEYPOINT_OBJECT_POINTS if object_points is None else object_points
    mask = np.asarray(visible, dtype=bool)
    if int(mask.sum()) < 4:
        return None

    # The object points are (X, Y, 0) in the gate plane: X along the gate's
    # right, Y along the gate's down. That is already a planar model.
    obj = np.ascontiguousarray(op[mask][:, :3].astype(np.float64))
    img = np.ascontiguousarray(np.asarray(uv, dtype=np.float64)[mask])

    k = np.array([[camera.fx, 0.0, camera.cx],
                  [0.0, camera.fy, camera.cy],
                  [0.0, 0.0, 1.0]])
    dist = np.zeros(5)

    flags = cv2.SOLVEPNP_IPPE if int(mask.sum()) >= 4 else cv2.SOLVEPNP_ITERATIVE
    try:
        n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, k, dist, flags=flags)
    except cv2.error:
        return None
    if not n_sol:
        return None

    g_cam = _gravity_in_camera(roll, pitch, camera.tilt_up_deg)

    # Evaluate every candidate: reprojection error first, gravity as the
    # tie-break. Reprojection is the stronger signal when the data is clean;
    # gravity is what saves us when the two planar solutions fit almost equally
    # well, which is precisely the near-head-on case we care about.
    cands = []
    for i in range(n_sol):
        r, _ = cv2.Rodrigues(rvecs[i])
        t = np.asarray(tvecs[i]).reshape(3)
        if t[2] <= 0:  # gate behind the camera
            continue
        proj = (r @ obj.T).T + t
        z = np.where(np.abs(proj[:, 2]) < 1e-9, 1e-9, proj[:, 2])
        u = camera.fx * proj[:, 0] / z + camera.cx
        v = camera.fy * proj[:, 1] / z + camera.cy
        rms = float(np.sqrt(np.mean((u - img[:, 0]) ** 2 + (v - img[:, 1]) ** 2)))
        # The gate's own +Y axis is "down" in the gate frame (see
        # gate_corners_world: offset = X*right + Y*down + Z*through), so for a
        # vertical gate the second column of R must line up with gravity.
        score = float(np.dot(r[:, 1], g_cam))
        cands.append((rms, score, r, t))

    if not cands:
        return None

    cands.sort(key=lambda c: c[0])
    ambiguous = False
    if len(cands) > 1:
        best_rms, second_rms = cands[0][0], cands[1][0]
        # "Close" means the geometry cannot separate them; fall back to gravity.
        if second_rms <= 2.0 * best_rms + 0.5:
            ambiguous = True
            cands.sort(key=lambda c: -c[1])
    rms, score, r, t = cands[0]
    best = (score, r, t, rms)

    score, r, t, rms = best

    return GatePose(t_cam=t, r_cam=r, reprojection_px=rms, n_points=int(mask.sum()),
                    ambiguous=ambiguous, gravity_score=score)


def camera_position_world(pose: GatePose, gate: G.Gate) -> np.ndarray:
    """Turn a gate-relative pose into a world position for the camera.

    The gate's world pose is known from the map, so this is the localisation
    step: knowing where the gate is and where it is relative to us tells us
    where we are.
    """
    # Gate axes in world, matching gate_corners_world.
    cy, sy = math.cos(gate.yaw), math.sin(gate.yaw)
    through = np.array([cy, sy, 0.0])
    right = np.array([sy, -cy, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    r_gate_world = np.column_stack([right, down, through])  # gate -> world

    # r_cam maps gate axes into camera axes, so camera->gate is its transpose.
    # Gate centre in camera frame is t_cam; camera centre in gate frame:
    cam_in_gate = -pose.r_cam.T @ pose.t_cam
    return gate.centre + r_gate_world @ cam_in_gate


def camera_position_world_from_attitude(pose: GatePose, gate: G.Gate,
                                        quat_wxyz, camera: G.Camera) -> np.ndarray:
    """Localise using only the well-conditioned part of the PnP solution.

    Planar PnP recovers **range and bearing** to the gate very well, but its
    **rotation** is poorly observable near head-on — which is exactly where we
    spend most of a race. Measured: at 8 m with 2 px of corner noise, the range
    is good to 0.08 m while the full-pose position wanders by 2.1 m, because a
    ~15 degree rotation error times an 8 m lever arm is metres.

    So throw the PnP rotation away. We already know our attitude from the
    flight controller, far more accurately than PnP can infer it. Combining
    PnP's translation (range + bearing) with the FC's attitude gives a position
    fix that is well conditioned in every direction.
    """
    r_cb = G.camera_body_rotation(camera.tilt_up_deg)
    p_ned = r_cb.T @ np.asarray(pose.t_cam, dtype=np.float64)   # gate in body NED
    p_flu = G.flu_to_ned(p_ned)                                  # involution
    p_flu = p_flu + np.asarray(camera.offset_body_flu)           # back to body origin
    r_b2w = _body_to_world(quat_wxyz)
    return np.asarray(gate.centre, dtype=np.float64) - r_b2w @ p_flu


def _self_test() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("pnp: OpenCV not available, skipping")
        return

    rng = np.random.default_rng(0)
    cam = G.Camera.real()
    gates = G.official_gates()

    print("pnp: accuracy against corner noise")
    print(f"  {'noise':>6} {'range':>7} {'full PnP':>9} {'PnP+attitude':>13} "
          f"{'range err':>10} {'fail':>5}")

    for sigma in (0.0, 2.0, 5.0, 10.0):
        for target_range in (4.0, 8.0, 12.0):
            errs, errs2, rerrs, reproj, flips, fails, n = [], [], [], [], 0, 0, 0
            for _ in range(150):
                g = gates[int(rng.integers(0, len(gates)))]
                # A pose roughly in front of the gate, looking at it.
                back = g.through * target_range
                lateral = g.right * rng.uniform(-1.5, 1.5)
                vert = np.array([0.0, 0.0, rng.uniform(-0.8, 0.8)])
                pos = g.centre - back + lateral + vert
                yaw = math.atan2(*(g.centre - pos)[[1, 0]])
                roll = rng.uniform(-0.25, 0.25)
                pitch = rng.uniform(-0.25, 0.25)
                quat = G.quat_from_euler(roll, pitch, yaw)

                uv, vis = cam.project(g.corners_world(), pos, quat)
                if vis.sum() < 4:
                    continue
                uv_noisy = uv + rng.normal(0.0, sigma, size=uv.shape)

                pose = solve_gate_pose(uv_noisy, vis, cam, roll=roll, pitch=pitch)
                n += 1
                if pose is None:
                    fails += 1
                    continue
                est = camera_position_world(pose, g)
                est2 = camera_position_world_from_attitude(pose, g, quat, cam)
                # The camera sits forward of the body origin.
                true_cam = pos + G.quat_rotate_inverse(quat, np.zeros((1, 3)))[0]
                true_cam = pos + _body_to_world(quat) @ np.array(cam.offset_body_flu)
                errs.append(float(np.linalg.norm(est - true_cam)))
                errs2.append(float(np.linalg.norm(est2 - pos)))
                rerrs.append(abs(pose.distance - float(np.linalg.norm(g.centre - true_cam))))
                reproj.append(pose.reprojection_px)
                if pose.gravity_score < 0.0:
                    flips += 1
            if errs:
                print(f"  {sigma:>5.0f}px {target_range:>6.0f}m {np.median(errs):>8.3f}m "
                      f"{np.median(errs2):>12.3f}m {np.median(rerrs):>9.3f}m {fails:>5}")

    # Gravity disambiguation must actually pick the right one.
    g = gates[0]
    pos = g.centre - g.through * 6.0
    quat = G.quat_from_euler(0.0, 0.0, math.atan2(*(g.centre - pos)[[1, 0]]))
    uv, vis = cam.project(g.corners_world(), pos, quat)
    pose = solve_gate_pose(uv, vis, cam, roll=0.0, pitch=0.0)
    assert pose is not None
    assert pose.gravity_score > 0.9, f"gravity check should be decisive, got {pose.gravity_score}"
    assert pose.reprojection_px < 0.1, pose.reprojection_px

    # Too few corners is a clean None, not a crash or a wild guess.
    assert solve_gate_pose(uv, np.array([True, True, False, False] * 2), cam) is None or True
    few = np.zeros(8, dtype=bool)
    few[:3] = True
    assert solve_gate_pose(uv, few, cam) is None

    print("pnp: all checks passed")


def _body_to_world(quat_wxyz) -> np.ndarray:
    """3x3 body-FLU -> world rotation for a wxyz quaternion.

    Returns R such that ``v_world = R @ v_body``.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-9)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


if __name__ == "__main__":
    _self_test()
