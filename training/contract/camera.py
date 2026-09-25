"""The FPV camera, as both ends of the pipeline must agree it behaves.

Every number here was measured on the aircraft's own camera with an on-site
ChArUco board at 1920x1080 and scaled to the 640x360 frame the policy sees.
Nothing in this file is a guess, and nothing in it should be randomised -- a
calibration you have performed is not uncertainty, and training a policy to be
robust to a focal length you already know wastes capacity that the detector's
real error could have used.

Two things this file exists to stop:

1. **Scale drift.** The principal point used to be derived two different ways --
   ``CY = 529.035788 * (HEIGHT / 1080)`` on the client and
   ``CY = 529.035788 * (FRAME_W / 1920)`` in the simulator. Both equal one third
   at 640x360 and agree today by coincidence of aspect ratio. Change the frame
   to 640x480 and they silently disagree. Here the calibration is stored at its
   native resolution and scaled once, per axis.

2. **Distortion divergence.** The client packs Brown-Conrady in OpenCV's
   ``[k1, k2, p1, p2, k3]`` order; the simulator splits it into radial and
   tangential tuples. Same five numbers, two spellings, and no way to notice if
   one is edited. They are defined once here.
"""

from __future__ import annotations

import math

# --- the frame the policy sees ---------------------------------------------
FRAME_W = 640.0
FRAME_H = 360.0

# --- calibration, at the resolution it was measured -------------------------
CALIB_W = 1920.0
CALIB_H = 1080.0
CALIB_FX = 1302.941063
CALIB_FY = 1303.105865
CALIB_CX = 952.287144
CALIB_CY = 529.035788

_SCALE_X = FRAME_W / CALIB_W
_SCALE_Y = FRAME_H / CALIB_H

# Each axis scales by its own ratio: fx and cx are pixel counts across the
# width, fy and cy across the height. Both existing implementations scale fy by
# the *width* ratio, which is right only while the aspect ratio is preserved --
# and 1920x1080 and 640x360 are both 16:9, so today every convention produces
# 434.37 and nobody has been bitten. Resize to 640x480 and scaling fy by width
# gives 434.37 where the correct answer is 579.16.
FX = CALIB_FX * _SCALE_X
FY = CALIB_FY * _SCALE_Y
CX = CALIB_CX * _SCALE_X
CY = CALIB_CY * _SCALE_Y

# Horizontal field of view implied by the above: about 72.8 degrees. The VQ
# simulator's camera was fx = 320, a 90 degree HFOV, which is why checkpoints
# trained against it do not transfer -- a stress run at the real focal length
# scored 42 crashes per 100 gates against a baseline of 2.6.
HFOV_DEG = 2.0 * math.degrees(math.atan((FRAME_W / 2.0) / FX))

# --- Brown-Conrady ----------------------------------------------------------
# Defined once, in OpenCV's order. Use the helpers below rather than
# re-slicing this tuple by index at the call site.
DIST_OPENCV = (0.047494607, -0.182658343, 0.001303410, 0.000583514, 0.153277219)

DIST_K = (DIST_OPENCV[0], DIST_OPENCV[1], DIST_OPENCV[4])  # k1, k2, k3
DIST_P = (DIST_OPENCV[2], DIST_OPENCV[3])                  # p1, p2

# Brown-Conrady is only meaningful near the domain it was fitted over. Past
# this, the r^6 term stops describing a lens and starts describing float
# overflow -- in float32 a gate ten metres off to one side was enough, and the
# resulting NaN propagated into the reward and poisoned a whole 4096-env batch.
#
# tan(55 deg)^2 is about 2.0. That is already far outside a 72.8 degree lens,
# so the clamp only ever touches points that are off-frame and marked unseen.
# The client does not clamp: it works in float64 where the overflow does not
# occur, and it never projects points it cannot see. The divergence is real but
# confined to points neither end uses.
R2_DIST_MAX = 2.0

# --- mounting ---------------------------------------------------------------
# Optical axis is cos(tilt) forward + sin(tilt) up, i.e. pitched up from the
# body x-axis. A racing quad flies nose-down, so an up-tilted camera looks
# level in the attitude that matters.
#
# Consequence worth knowing: at 20 degrees up-tilt and this focal length, a
# gate at the aircraft's own altitude projects near the bottom of the frame,
# and at 10 m its lower corners fall outside the image entirely. That is
# correct for a racing attitude but not for sitting level on the start pad.
CAMERA_TILT_UP_DEG = 20.0

# --- gate geometry ----------------------------------------------------------
# Outer square and flyable opening, metres. The eight keypoints are the outer
# square (ids 0-3) then the opening (ids 4-7), each clockwise from top-left as
# seen looking along the gate's through-axis.
GATE_OUTER_M = 2.7
GATE_INNER_M = 1.5


def distortion_opencv() -> tuple[float, float, float, float, float]:
    """``(k1, k2, p1, p2, k3)`` -- the order ``cv2`` expects."""
    return DIST_OPENCV


def distortion_split() -> tuple[tuple[float, float, float], tuple[float, float]]:
    """``((k1, k2, k3), (p1, p2))`` -- radial and tangential, for the sim."""
    return DIST_K, DIST_P


def intrinsics_at(width: float, height: float) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` for a frame of any size, scaled from calibration.

    Use this rather than rescaling ``FX``/``CY`` by hand -- doing it by hand at
    two call sites is how the two ends drifted apart in the first place.
    """
    sx, sy = width / CALIB_W, height / CALIB_H
    return CALIB_FX * sx, CALIB_FY * sy, CALIB_CX * sx, CALIB_CY * sy
