#!/usr/bin/env python3
"""Extract the VQ2 gate face from real competition frames (no PIL fonts).

Finds a clean frontal-ish gate still under ``AI_GP/frames/``, perspective-warps
it to a square, cleans HUD bleed on the frame ring, and writes:

* ``assets/gate/textures/gate_face_vq2.png`` — right-side-up artwork
* ``assets/gate/textures/material_0.png`` — flipped for the current OBJ UV
  convention (vertical: AI-GP was on the bottom; horizontal: L/R mirror)

Run from repo root::

    python tools/make_gate_texture.py
    python tools/make_gate_asset.py   # only needed if the USD mesh changed
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUTER, INNER = 2.7, 1.5
SIZE = 2048  # high-res screenshot is ~1400px across the gate; keep detail

# Preferred sources. High-res VQ2 screenshot first (sharp pixel font); camera
# frames as fallbacks. Overhead lighting in the screenshot is flattened out
# during clean() so the albedo stays usable under our own hangar lights.
DEFAULT_FRAMES = [
    "AI_GP/reference/LARGE_GATE_IMAGE_HIGHDEF.png",
    "AI_GP/frames/run_20260901_155240/gate_20260901_155240_0000000859.jpg",
    "AI_GP/frames/run_20260810_221635/gate_20260810_221635_0000011235.jpg",
    "AI_GP/frames/run_20260810_221105/gate_20260810_221105_0000010084.jpg",
]


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    rem = np.array([p for p in pts if not (np.allclose(p, tl) or np.allclose(p, br))])
    if rem[0, 0] > rem[1, 0]:
        tr, bl = rem[0], rem[1]
    else:
        tr, bl = rem[1], rem[0]
    return np.stack([tl, tr, br, bl])


def quad_from_cnt(cnt) -> np.ndarray:
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.06, 40):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    pts = hull.reshape(-1, 2).astype(np.float32)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    picks = []
    for lo, hi in [(-np.pi, -np.pi / 2), (-np.pi / 2, 0), (0, np.pi / 2), (np.pi / 2, np.pi + 0.01)]:
        mask = (ang >= lo) & (ang < hi)
        if mask.any():
            sub = pts[mask]
            picks.append(sub[np.linalg.norm(sub - c, axis=1).argmax()])
    return np.array(picks, np.float32)


def warp_gate(path: str) -> tuple[np.ndarray, float] | None:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 90, 110), (16, 255, 255)),
        cv2.inRange(hsv, (168, 90, 110), (180, 255, 255)),
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    areas = sorted([(cv2.contourArea(c), i) for i, c in enumerate(cnts)], reverse=True)
    if areas[0][0] < 8000:
        return None
    outer = order_quad(quad_from_cnt(cnts[areas[0][1]]))
    dst = np.array([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]], np.float32)
    M = cv2.getPerspectiveTransform(outer, dst)
    warped = cv2.warpPerspective(rgb, M, (SIZE, SIZE), flags=cv2.INTER_CUBIC)

    m = int(round(SIZE * (OUTER - INNER) / (2 * OUTER)))
    r, g, b = warped[:, :, 0].astype(float), warped[:, :, 1].astype(float), warped[:, :, 2].astype(float)
    frame = np.zeros((SIZE, SIZE), bool)
    frame[:m, :] = frame[SIZE - m :, :] = frame[:, :m] = frame[:, SIZE - m :] = True
    white = frame & ((r + g + b) / 3 > 180) & (g > 150)
    top_white = white[:m, m : SIZE - m].mean()
    orange = frame & (r > 140) & (r - g > 50) & (r - b > 50)
    cyan = frame & (b > r + 20) & (b > 80)
    score = orange.mean() * 2 + white.mean() * 3 + top_white * 2 - cyan.mean() * 4
    return warped, float(score)


def clean(warped: np.ndarray) -> np.ndarray:
    """Remove HUD bleed; keep real VQ2 lettering and dither."""
    m = int(round(SIZE * (OUTER - INNER) / (2 * OUTER)))
    face = warped.astype(np.float32)
    r, g, b = face[:, :, 0], face[:, :, 1], face[:, :, 2]
    lum = (r + g + b) / 3.0
    frame = np.zeros((SIZE, SIZE), bool)
    frame[:m, :] = frame[SIZE - m :, :] = frame[:, :m] = frame[:, SIZE - m :] = True
    yy = np.arange(SIZE)[:, None]

    top = frame & (yy > 20) & (yy < m - 10)
    orange_m = top & (r > 160) & (r - g > 70) & (r - b > 70) & (g < 140)
    base = face[orange_m].mean(0) if orange_m.any() else np.array([255.0, 50.0, 20.0])

    cyan = frame & (b > r + 15) & (b > 70)
    dark = frame & (lum < 55)
    white = frame & (lum > 175) & (g > 145) & (np.abs(r - g) < 60)
    pink = frame & (yy > SIZE - m) & (b > g + 12) & (g < 150) & (lum < 200) & (r > 140) & ~white
    face[cyan | dark | pink] = base
    face[~frame] = 0

    soft_white = frame & (lum > 155) & (g > 130) & (np.abs(r - g) < 70)
    orange_field = frame & ~soft_white
    face[orange_field] = 0.55 * face[orange_field] + 0.45 * base
    strong = frame & (lum > 190) & (g > 160)
    face[strong] = np.clip(warped[strong].astype(np.float32) * 1.05, 0, 255)
    return face.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=None, help="Explicit source jpg (skips auto-pick)")
    ap.add_argument(
        "--repo",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    ap.add_argument(
        "--frames-root",
        default=None,
        help="Competitions root containing AI_GP/frames (default: parent of repo)",
    )
    args = ap.parse_args()
    repo = Path(args.repo)
    competitions = Path(args.frames_root) if args.frames_root else repo.parent

    best = None
    if args.frame:
        got = warp_gate(str(Path(args.frame)))
        if got is None:
            raise SystemExit(f"failed to warp {args.frame}")
        best = (got[1], got[0], args.frame)
    else:
        for rel in DEFAULT_FRAMES:
            p = competitions / rel if not Path(rel).is_absolute() else Path(rel)
            # also try relative to competitions with AI_GP prefix already in DEFAULT
            if not p.is_file():
                p = competitions / rel
            if not p.is_file():
                # DEFAULT_FRAMES already include AI_GP/...
                p = Path(rel)
                if not p.is_file():
                    p = competitions / rel
            if not p.is_file():
                print(f"skip missing {rel}")
                continue
            got = warp_gate(str(p))
            if got is None:
                continue
            warped, score = got
            print(f"score={score:6.3f}  {p}")
            if best is None or score > best[0]:
                best = (score, warped, str(p))

    if best is None:
        raise SystemExit("no usable gate frame found")

    score, warped, src = best
    print(f"using {src}  (score={score:.3f})")
    face = clean(warped)
    # OBJ UVs in this asset are mirrored relative to image space.
    albedo = np.fliplr(np.flipud(face))

    tex_dir = repo / "assets" / "gate" / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(face).save(tex_dir / "gate_face_vq2.png")
    Image.fromarray(albedo).save(tex_dir / "material_0.png")
    mesh_dir = repo / "assets" / "gate" / "vq2_mesh"
    if mesh_dir.is_dir():
        Image.fromarray(albedo).save(mesh_dir / "material_0.png")
    print(f"wrote {tex_dir / 'gate_face_vq2.png'}")
    print(f"wrote {tex_dir / 'material_0.png'}  (UV-corrected albedo)")


if __name__ == "__main__":
    main()
