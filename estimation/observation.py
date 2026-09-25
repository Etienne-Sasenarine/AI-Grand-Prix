"""Assemble the 51-number observation frame and the 32-frame history.

This must match ``AI_GP/race_obs.py`` and the Isaac training path exactly. The
layout, in order:

    corners   16   u,v per keypoint, normalised to [0, 1] by frame width/height
    visible    8   1.0 seen, 0.0 not
    state      5   roll, pitch (rad), then gyro x, y, z (NED, rad/s, clipped +-8)
    velocity   3   commanded body velocity (NED, m/s, clipped +-20)
    context   19   18-slot one-hot of the active gate, then lap fraction
    -----------
    total     51   x 32 frames of history = 1632

An unseen keypoint is written as ``-1.0`` in **both** u and v, and 0.0 in its
visibility slot.

A keypoint counts as unseen when it is non-finite, exactly (0, 0) (the
detector's not-seen convention), outside the frame by more than 15 % of a
dimension, or below the confidence threshold. That last rule matters: a pose
model predicts all eight corners whether or not they are in shot, and clamping
an off-screen guess to the image border tells the network a corner is at the
edge when it is really outside the picture.

History is oldest-first, newest-last, left-padded by repeating the oldest frame
— never with zeros, because an all-zero row is a valid-looking observation with
every corner at the image origin.
"""

from __future__ import annotations

import math

import numpy as np

KEYPOINT_COUNT = 8
N_GATES = 18
FRAME_W = 640.0
FRAME_H = 360.0
OFF_FRAME_MARGIN = 0.15
NOT_SEEN = -1.0
GYRO_CLIP = 8.0
VEL_CLIP = 20.0
DEFAULT_HISTORY = 32
MIN_CONFIDENCE = 0.25

FRAME_DIM = 2 * KEYPOINT_COUNT + KEYPOINT_COUNT + 5 + 3 + N_GATES + 1  # 51
OBS_DIM = FRAME_DIM * DEFAULT_HISTORY  # 1632


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def context_features(gate_index) -> list[float]:
    """18-slot one-hot of the active gate, then fractional progress."""
    out = [0.0] * (N_GATES + 1)
    if gate_index is None:
        return out
    g = float(gate_index)
    if math.isnan(g):
        return out
    idx = int(max(0, min(N_GATES - 1, int(g))))
    out[idx] = 1.0
    out[N_GATES] = idx / float(max(1, N_GATES - 1))
    return out


def build_frame(
    uv,
    visible=None,
    *,
    confidences=None,
    roll: float = 0.0,
    pitch: float = 0.0,
    gyro_ned=(0.0, 0.0, 0.0),
    vel_body_ned=(0.0, 0.0, 0.0),
    gate_index=None,
    frame_w: float = FRAME_W,
    frame_h: float = FRAME_H,
    min_confidence: float = MIN_CONFIDENCE,
) -> np.ndarray:
    """One 51-number observation frame.

    ``uv`` is (8, 2) in pixels. ``visible`` is an optional (8,) mask; when it is
    given, a False entry is unseen regardless of the coordinates. The
    non-finite / (0,0) / off-frame / low-confidence rules still apply on top.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if uv.shape[0] < KEYPOINT_COUNT:
        pad = np.full((KEYPOINT_COUNT - uv.shape[0], 2), np.nan)
        uv = np.vstack([uv, pad])

    mx = OFF_FRAME_MARGIN * frame_w
    my = OFF_FRAME_MARGIN * frame_h

    corners: list[float] = []
    vis: list[float] = []
    for i in range(KEYPOINT_COUNT):
        u, v = float(uv[i, 0]), float(uv[i, 1])
        conf = 1.0 if confidences is None else float(confidences[i])
        in_frame = (-mx <= u <= frame_w + mx) and (-my <= v <= frame_h + my)
        seen = (
            math.isfinite(u) and math.isfinite(v)
            and not (u == 0.0 and v == 0.0)
            and in_frame
            and (math.isnan(conf) or conf >= min_confidence)
        )
        if visible is not None:
            seen = seen and bool(visible[i])
        if seen:
            corners.append(min(1.0, max(0.0, u / frame_w)))
            corners.append(min(1.0, max(0.0, v / frame_h)))
            vis.append(1.0)
        else:
            corners.append(NOT_SEEN)
            corners.append(NOT_SEEN)
            vis.append(0.0)

    state = [0.0 if math.isnan(float(roll)) else float(roll),
             0.0 if math.isnan(float(pitch)) else float(pitch)]
    for g in gyro_ned:
        g = float(g)
        state.append(0.0 if math.isnan(g) else _clip(g, GYRO_CLIP))
    for v in vel_body_ned:
        v = float(v)
        state.append(0.0 if math.isnan(v) else _clip(v, VEL_CLIP))

    return np.asarray(corners + vis + state + context_features(gate_index), dtype=np.float64)


class ObservationBuffer:
    """Rolling history of frames, flattened for the policy.

    Left-pads by repeating the oldest frame, matching ``stack_history`` in the
    training code.
    """

    def __init__(self, history: int = DEFAULT_HISTORY, frame_dim: int = FRAME_DIM) -> None:
        self.history = int(history)
        self.frame_dim = int(frame_dim)
        self._rows: list[np.ndarray] = []

    def reset(self) -> None:
        self._rows.clear()

    def push(self, frame) -> None:
        f = np.asarray(frame, dtype=np.float64).reshape(-1)
        if f.shape[0] != self.frame_dim:
            raise ValueError(f"frame must be {self.frame_dim} numbers, got {f.shape[0]}")
        self._rows.append(f)
        # Keep a little slack so `vector()` never has to reallocate much.
        if len(self._rows) > self.history * 2:
            self._rows = self._rows[-self.history:]

    @property
    def filled(self) -> int:
        return len(self._rows)

    @property
    def ready(self) -> bool:
        """True once the buffer holds a full history of real frames.

        Before this, ``vector()`` still returns a valid observation, but the
        oldest frames are repeats of the first one rather than real history.
        """
        return len(self._rows) >= self.history

    def vector(self) -> np.ndarray:
        if not self._rows:
            return np.zeros(self.history * self.frame_dim, dtype=np.float64)
        rows = self._rows[-self.history:]
        if len(rows) < self.history:
            rows = [rows[0]] * (self.history - len(rows)) + rows
        return np.concatenate(rows)


def _self_test() -> None:
    # Layout.
    f = build_frame(np.full((8, 2), np.nan))
    assert f.shape == (FRAME_DIM,) == (51,)
    assert np.all(f[:16] == NOT_SEEN) and np.all(f[16:24] == 0.0)

    # A seen corner normalises by the frame size.
    uv = np.zeros((8, 2))
    uv[:] = np.nan
    uv[0] = (320.0, 180.0)
    f = build_frame(uv)
    assert abs(f[0] - 0.5) < 1e-12 and abs(f[1] - 0.5) < 1e-12
    assert f[16] == 1.0 and f[17] == 0.0

    # Exactly (0, 0) is the detector's not-seen convention, not a real corner.
    uv2 = np.full((8, 2), np.nan)
    uv2[0] = (0.0, 0.0)
    assert build_frame(uv2)[0] == NOT_SEEN

    # Just outside the frame is still seen (15 % margin), far outside is not.
    for u, expect_seen in ((FRAME_W + 0.10 * FRAME_W, True),
                           (FRAME_W + 0.20 * FRAME_W, False),
                           (-0.10 * FRAME_W, True),
                           (-0.20 * FRAME_W, False)):
        uv3 = np.full((8, 2), np.nan)
        uv3[0] = (u, 180.0)
        assert bool(build_frame(uv3)[16] == 1.0) == expect_seen, u

    # Clamping: a corner inside the margin but outside the image clamps to [0,1].
    uv4 = np.full((8, 2), np.nan)
    uv4[0] = (FRAME_W + 10.0, -5.0)
    f = build_frame(uv4)
    assert f[0] == 1.0 and f[1] == 0.0

    # Low confidence is unseen.
    uv5 = np.full((8, 2), np.nan)
    uv5[0] = (320.0, 180.0)
    assert build_frame(uv5, confidences=[0.1] * 8)[16] == 0.0
    assert build_frame(uv5, confidences=[0.9] * 8)[16] == 1.0

    # An explicit visibility mask can only remove, never add.
    assert build_frame(uv5, visible=[False] * 8)[16] == 0.0

    # Gyro and velocity clip.
    f = build_frame(np.full((8, 2), np.nan), gyro_ned=(99, -99, 0), vel_body_ned=(99, -99, 0))
    assert f[26] == GYRO_CLIP and f[27] == -GYRO_CLIP
    assert f[29] == VEL_CLIP and f[30] == -VEL_CLIP

    # NaN state becomes zero rather than poisoning the network.
    f = build_frame(np.full((8, 2), np.nan), roll=float("nan"), gyro_ned=(float("nan"), 0, 0))
    assert f[24] == 0.0 and f[26] == 0.0

    # Context one-hot and lap fraction.
    f = build_frame(np.full((8, 2), np.nan), gate_index=0)
    assert f[32] == 1.0 and f[-1] == 0.0
    f = build_frame(np.full((8, 2), np.nan), gate_index=17)
    assert f[32 + 17] == 1.0 and abs(f[-1] - 1.0) < 1e-12
    f = build_frame(np.full((8, 2), np.nan), gate_index=None)
    assert f[32:].sum() == 0.0
    # Out-of-range indices clamp rather than throwing.
    assert build_frame(np.full((8, 2), np.nan), gate_index=99)[32 + 17] == 1.0

    # Buffer: padding repeats the oldest frame, never zeros.
    buf = ObservationBuffer()
    a = np.arange(51, dtype=np.float64)
    buf.push(a)
    v = buf.vector()
    assert v.shape == (1632,)
    assert np.array_equal(v[:51], a) and np.array_equal(v[-51:], a)
    assert not buf.ready

    for i in range(1, 32):
        buf.push(a + i)
    assert buf.ready
    v = buf.vector()
    assert np.array_equal(v[:51], a), "oldest frame must come first"
    assert np.array_equal(v[-51:], a + 31), "newest frame must come last"

    # Overflow keeps the newest 32.
    for i in range(32, 100):
        buf.push(a + i)
    v = buf.vector()
    assert np.array_equal(v[-51:], a + 99)
    assert np.array_equal(v[:51], a + 68)

    buf.reset()
    assert buf.filled == 0 and not buf.ready
    assert np.array_equal(buf.vector(), np.zeros(1632))

    try:
        buf.push(np.zeros(50))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-width frame must be rejected")

    print("observation: all checks passed")


if __name__ == "__main__":
    _self_test()
