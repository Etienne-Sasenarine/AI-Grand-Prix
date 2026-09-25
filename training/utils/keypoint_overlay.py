"""Draw the policy's derived gate keypoints on an FPV frame.

Uses the same 51-D ``race_obs`` layout as ``aigp_race_observation`` (outer 0–3,
inner 4–7, vis flags). History stacks put the current frame in the last 51 of
the 1632-D vector; a 1639-D lookahead model still stores that history first.

Drawing is NumPy-only so play.py does not depend on OpenCV inside Isaac Sim.
"""

from __future__ import annotations

import numpy as np

from utils.aigp_obs import FEATURE_DIM_VEL_CTX, KEYPOINT_COUNT, NOT_SEEN

HISTORY_FLAT = 32 * FEATURE_DIM_VEL_CTX  # 1632

# 5x7 bitmaps for the corner IDs we actually paint (0–7).
_DIGIT = {
    0: (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    1: (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    2: (0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111),
    3: (0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110),
    4: (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    5: (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    6: (0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110),
    7: (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    8: (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
}


def current_race_obs_frame(flat, frame_dim: int = FEATURE_DIM_VEL_CTX) -> np.ndarray:
    """Last (current) race_obs frame from a stacked policy observation."""
    x = np.asarray(flat, dtype=np.float32).reshape(-1)
    if x.size >= HISTORY_FLAT:
        return x[HISTORY_FLAT - frame_dim : HISTORY_FLAT].copy()
    if x.size >= frame_dim:
        return x[-frame_dim:].copy()
    out = np.full(frame_dim, float(NOT_SEEN), dtype=np.float32)
    out[: x.size] = x
    return out


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(rgb)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img = img[..., :3].copy()
    if img.dtype != np.uint8:
        peak = float(np.nanmax(img)) if img.size else 0.0
        if peak <= 1.0 + 1e-5:
            img = np.clip(img, 0.0, 1.0) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _blend(img: np.ndarray, ys: np.ndarray, xs: np.ndarray, colour: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    if not np.any(inside):
        return
    img[ys[inside], xs[inside]] = colour


def _fill_circle(img: np.ndarray, cx: int, cy: int, radius: int, colour: tuple[int, int, int]) -> None:
    rr = np.arange(-radius, radius + 1, dtype=np.int32)
    yy, xx = np.meshgrid(rr, rr, indexing="ij")
    mask = xx * xx + yy * yy <= radius * radius
    _blend(img, yy[mask] + cy, xx[mask] + cx, colour)


def _draw_line(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], colour: tuple[int, int, int], width: int = 2) -> None:
    x0, y0 = p0
    x1, y1 = p1
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).round().astype(np.int32)
    ys = np.linspace(y0, y1, steps + 1).round().astype(np.int32)
    if width <= 1:
        _blend(img, ys, xs, colour)
        return
    off = np.arange(-(width // 2), width // 2 + 1, dtype=np.int32)
    for dy in off:
        for dx in off:
            _blend(img, ys + dy, xs + dx, colour)


def _draw_digit(img: np.ndarray, digit: int, x: int, y: int, colour: tuple[int, int, int], scale: int = 2) -> None:
    bits = _DIGIT.get(int(digit))
    if bits is None:
        return
    for row, row_bits in enumerate(bits):
        for col in range(5):
            if row_bits & (1 << (4 - col)):
                ys = np.arange(scale, dtype=np.int32) + y + row * scale
                xs = np.arange(scale, dtype=np.int32) + x + col * scale
                yy, xx = np.meshgrid(ys, xs, indexing="ij")
                _blend(img, yy.ravel(), xx.ravel(), colour)


def annotate_rgb(rgb: np.ndarray, obs_frame: np.ndarray) -> np.ndarray:
    """Copy ``rgb`` and paint numbered outer/inner rings from ``obs_frame``."""
    img = _as_uint8_rgb(rgb)
    h, w = img.shape[:2]
    obs = np.asarray(obs_frame, dtype=np.float32).reshape(-1)

    cx, cy = w // 2, h // 2
    _draw_line(img, (cx - 10, cy), (cx + 10, cy), (90, 90, 90), 1)
    _draw_line(img, (cx, cy - 10), (cx, cy + 10), (90, 90, 90), 1)

    outer = (250, 200, 80)
    inner = (80, 220, 120)
    placed: dict[int, tuple[int, int]] = {}
    for i in range(KEYPOINT_COUNT):
        if obs.size < 2 * KEYPOINT_COUNT + i + 1:
            break
        u_n, v_n = float(obs[2 * i]), float(obs[2 * i + 1])
        visible = float(obs[2 * KEYPOINT_COUNT + i]) > 0.5
        if not visible or u_n <= NOT_SEEN + 1e-3:
            continue
        px = int(round(u_n * w))
        py = int(round(v_n * h))
        placed[i] = (px, py)
        colour = outer if i < 4 else inner
        _fill_circle(img, px, py, 5, colour)
        _draw_digit(img, i, px + 7, py - 12, colour)

    for ring, colour in ((range(4), outer), (range(4, 8), inner)):
        pts = [placed[i] for i in ring if i in placed]
        for a, b in zip(pts, pts[1:]):
            _draw_line(img, a, b, colour, 2)
        if len(pts) == 4:
            _draw_line(img, pts[-1], pts[0], colour, 2)

    n_seen = len(placed)
    tag = (80, 220, 120) if n_seen == 8 else ((90, 200, 250) if n_seen else (80, 80, 240))
    img[4:28, 6:220] = (20, 20, 20)
    _draw_digit(img, n_seen, 12, 8, tag)
    # slash + 8
    _draw_line(img, (28, 22), (36, 8), tag, 1)
    _draw_digit(img, 8, 40, 8, tag)
    return img
