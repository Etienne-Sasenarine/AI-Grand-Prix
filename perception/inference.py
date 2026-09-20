"""Gate inference with the trained pose model (best.pt).

Two ways to use it:

1. As a library from the flight-control code:

       from inference.inference import GateDetector
       det = GateDetector()                     # loads inference/best.pt once, warms up
       gates = det.detect(frame_bgr)            # list[Gate], sorted best-first
       if gates:
           x, y = gates[0].aim_point            # pixel to steer towards
           dx, dy = gates[0].aim_offset(frame_bgr.shape)   # -1..1 offset from image centre

2. From the command line:

       python inference/inference.py --source frames/train --show
       python inference/inference.py --source flight.mp4 --save out.mp4
       python inference/inference.py --source 0 --show                      # webcam
       python inference/inference.py --source img.jpg --json gates.jsonl    # machine-readable output

Keypoint order (fixed by the dataset):
    0 out_UL, 1 out_UR, 2 out_BR, 3 out_BL, 4 in_UL, 5 in_UR, 6 in_BR, 7 in_BL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

KPT_NAMES = ["out_UL", "out_UR", "out_BR", "out_BL", "in_UL", "in_UR", "in_BR", "in_BL"]
OUTER = slice(0, 4)
INNER = slice(4, 8)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


@dataclass
class Gate:
    """One detected gate. All coordinates are pixels in the original frame."""

    conf: float
    box: tuple[float, float, float, float]            # x1, y1, x2, y2
    keypoints: np.ndarray                             # (8, 2) float32, (0, 0) if not predicted
    kpt_conf: np.ndarray                              # (8,)   float32
    kpt_visible: np.ndarray = field(init=False)       # (8,)   bool

    def __post_init__(self):
        self.kpt_visible = (self.kpt_conf >= self.KPT_CONF_THRES) & (self.keypoints.sum(axis=1) > 0)

    KPT_CONF_THRES = 0.5

    @property
    def outer(self) -> np.ndarray:
        return self.keypoints[OUTER]

    @property
    def inner(self) -> np.ndarray:
        return self.keypoints[INNER]

    @property
    def box_centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def aim_point(self) -> tuple[float, float]:
        """Point to fly towards: centroid of the visible inner corners.

        Falls back to the outer corners, then the box centre, so it always returns something.
        """
        for pts, vis in ((self.inner, self.kpt_visible[INNER]), (self.outer, self.kpt_visible[OUTER])):
            if vis.sum() >= 2:
                cx, cy = pts[vis].mean(axis=0)
                return float(cx), float(cy)
        return self.box_centre

    def aim_offset(self, frame_shape: Sequence[int]) -> tuple[float, float]:
        """Aim point relative to the image centre, normalised to [-1, 1] (x right, y down)."""
        h, w = frame_shape[:2]
        ax, ay = self.aim_point
        return (ax - w / 2) / (w / 2), (ay - h / 2) / (h / 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["keypoints"] = {n: [round(float(x), 1), round(float(y), 1)] for n, (x, y) in zip(KPT_NAMES, self.keypoints)}
        d["kpt_conf"] = {n: round(float(c), 3) for n, c in zip(KPT_NAMES, self.kpt_conf)}
        d["kpt_visible"] = {n: bool(v) for n, v in zip(KPT_NAMES, self.kpt_visible)}
        d["box"] = [round(float(v), 1) for v in self.box]
        d["conf"] = round(self.conf, 3)
        d["aim_point"] = [round(v, 1) for v in self.aim_point]
        return d


class GateDetector:
    """Thin wrapper around the YOLO pose model that returns Gate objects."""

    def __init__(self, weights: str | Path = Path(__file__).parent / "best.pt", imgsz: int = 640, conf: float = 0.4,
                 iou: float = 0.5, device: Optional[str] = None, half: bool = False, warmup: bool = True):
        self.model = YOLO(str(weights))
        if self.model.task != "pose":
            raise ValueError(f"{weights} is a '{self.model.task}' model; this script expects a pose model")
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device != "cpu"
        if warmup:
            self.detect(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> list[Gate]:
        """Run the model on one BGR frame. Returns gates sorted best-first (see `rank`)."""
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device,
                               half=self.half, verbose=False)[0]
        n = len(r.boxes)
        if n == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        xy = r.keypoints.xy.cpu().numpy().astype(np.float32)
        kc = r.keypoints.conf
        kc = kc.cpu().numpy().astype(np.float32) if kc is not None else np.ones((n, len(KPT_NAMES)), np.float32)
        gates = [Gate(float(confs[i]), tuple(map(float, boxes[i])), xy[i], kc[i]) for i in range(n)]
        gates.sort(key=self.rank, reverse=True)
        return gates

    @staticmethod
    def rank(g: Gate) -> float:
        """Bigger + more confident gates first: the nearest gate is the one to fly through."""
        return g.conf * np.sqrt(g.area)


# --------------------------------------------------------------------------- drawing

C_OUTER, C_INNER, C_AIM, C_TEXT = (0, 200, 255), (0, 255, 90), (255, 80, 255), (255, 255, 255)


def draw_gates(frame: np.ndarray, gates: list[Gate], target_idx: int = 0, labels: bool = True) -> np.ndarray:
    """Overlay corners, gate outlines and the aim point. Modifies and returns `frame`."""
    t = max(1, round(min(frame.shape[:2]) / 400))
    for i, g in enumerate(gates):
        is_target = i == target_idx
        for sl, colour in ((OUTER, C_OUTER), (INNER, C_INNER)):
            pts, vis = g.keypoints[sl], g.kpt_visible[sl]
            if vis.all():
                cv2.polylines(frame, [pts.astype(np.int32).reshape(-1, 1, 2)], True, colour, t)
            for (x, y), v in zip(pts, vis):
                if v:
                    cv2.circle(frame, (int(x), int(y)), 3 * t, colour, -1)
        if not is_target:
            x1, y1, x2, y2 = map(int, g.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), t)
        ax, ay = map(int, g.aim_point)
        r = 8 * t if is_target else 4 * t
        cv2.circle(frame, (ax, ay), r, C_AIM, t)
        cv2.line(frame, (ax - r, ay), (ax + r, ay), C_AIM, t)
        cv2.line(frame, (ax, ay - r), (ax, ay + r), C_AIM, t)
        if labels:
            x1, y1 = int(g.box[0]), int(g.box[1])
            tag = f"{'TARGET ' if is_target else ''}gate {g.conf:.2f}"
            cv2.putText(frame, tag, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, C_TEXT, t, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, fps: float, gates: list[Gate]) -> np.ndarray:
    t = max(1, round(min(frame.shape[:2]) / 400))
    line = f"{fps:5.1f} FPS  {len(gates)} gate(s)"
    if gates:
        dx, dy = gates[0].aim_offset(frame.shape)
        line += f"  aim dx={dx:+.2f} dy={dy:+.2f}"
    cv2.putText(frame, line, (8, 20 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * t, C_TEXT, t, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- sources

def iter_frames(source: str) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, BGR frame) from an image, a folder, a video file, or a webcam index."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            sys.exit(f"could not open webcam {source}")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"cam{source}_{i:06d}", frame
            i += 1
        cap.release()
        return

    p = Path(source)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
        if not files:
            sys.exit(f"no images in {p}")
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                yield f.stem, img
    elif p.suffix.lower() in IMG_EXTS:
        img = cv2.imread(str(p))
        if img is None:
            sys.exit(f"could not read {p}")
        yield p.stem, img
    elif p.suffix.lower() in VID_EXTS or p.exists():
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            sys.exit(f"could not open video {p}")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"{p.stem}_{i:06d}", frame
            i += 1
        cap.release()
    else:
        sys.exit(f"source not found: {source}")


def source_fps(source: str) -> float:
    if source.isdigit() or Path(source).suffix.lower() not in VID_EXTS:
        return 30.0
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


# --------------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(Path(__file__).parent / "best.pt"))
    ap.add_argument("--source", required=True, help="image, folder, video, or webcam index")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4, help="box confidence threshold")
    ap.add_argument("--kpt-conf", type=float, default=0.5, help="keypoint visibility threshold")
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'cpu', or omit for auto")
    ap.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    ap.add_argument("--show", action="store_true", help="display annotated frames (q / Esc to quit)")
    ap.add_argument("--save", default=None,
                    help="output path: a .mp4 for video/webcam sources, a folder for image sources")
    ap.add_argument("--json", default=None, help="write one JSON line per frame with all gate data")
    ap.add_argument("--quiet", action="store_true", help="don't print per-frame summary")
    args = ap.parse_args()

    Gate.KPT_CONF_THRES = args.kpt_conf
    det = GateDetector(args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device, half=args.half)
    print(f"loaded {args.weights} on {det.device}  imgsz={args.imgsz} conf={args.conf}")

    is_stream = args.source.isdigit() or Path(args.source).suffix.lower() in VID_EXTS
    writer = None
    out_dir = None
    if args.save:
        if is_stream:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(args.save)
            out_dir.mkdir(parents=True, exist_ok=True)
    jf = open(args.json, "w") if args.json else None

    n_frames, t_start = 0, time.perf_counter()
    fps_smooth = 0.0
    try:
        for name, frame in iter_frames(args.source):
            t0 = time.perf_counter()
            gates = det.detect(frame)
            dt = time.perf_counter() - t0
            fps_smooth = 1 / dt if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 / dt
            n_frames += 1

            if not args.quiet:
                msg = f"{name}: {len(gates)} gate(s) {dt * 1000:.0f} ms"
                if gates:
                    ax, ay = gates[0].aim_point
                    msg += f"  target conf={gates[0].conf:.2f} aim=({ax:.0f}, {ay:.0f})"
                print(msg)
            if jf:
                jf.write(json.dumps({"frame": name, "gates": [g.to_dict() for g in gates]}) + "\n")

            if args.show or args.save:
                vis = draw_hud(draw_gates(frame.copy(), gates), fps_smooth, gates)
                if args.save:
                    if is_stream:
                        if writer is None:
                            h, w = vis.shape[:2]
                            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                                     source_fps(args.source), (w, h))
                        writer.write(vis)
                    else:
                        cv2.imwrite(str(out_dir / f"{name}.jpg"), vis)
                if args.show:
                    cv2.imshow("gates", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        if writer is not None:
            writer.release()
        if jf:
            jf.close()
        if args.show:
            cv2.destroyAllWindows()

    total = time.perf_counter() - t_start
    print(f"\n{n_frames} frame(s) in {total:.1f}s  ({n_frames / max(total, 1e-9):.1f} FPS end-to-end)")
    if args.save:
        print(f"saved to {args.save}")
    if args.json:
        print(f"json to {args.json}")


if __name__ == "__main__":
    main()
