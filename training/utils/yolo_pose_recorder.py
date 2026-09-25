"""Dump Isaac FPV frames + YOLO-pose keypoint labels (AI_GP 8-corner layout).

Label format matches ``AI_GP/datasets/AIGP_8keypoints.v5i.yolov8``:

    class cx cy w h  (kpt_x kpt_y v)×8

All values normalised to [0, 1]. Keypoint order is outer 0–3 then inner 4–7,
clockwise from top-left (same as ``utils.aigp_obs.KEYPOINT_OBJECT_POINTS``).
Visibility ``v`` is 2 when the projector marks the corner in-frame, else 0.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.aigp_obs import (
    FRAME_H,
    FRAME_W,
    KEYPOINT_COUNT,
    gate_keypoints_world,
    project_points_aigp_camera,
)


class YoloPoseRecorder:
    """Write ``images/``, ``labels/``, and a Ultralytics ``data.yaml``."""

    def __init__(
        self,
        root: str | Path,
        *,
        every_n: int = 2,
        max_frames: int = 2000,
        min_visible: int = 2,
        class_id: int = 0,
        class_name: str = "gate",
        val_fraction: float = 0.1,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = Path(root) / f"isaac_gate_pose_{stamp}"
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.every_n = max(1, int(every_n))
        self.max_frames = int(max_frames)
        self.min_visible = int(min_visible)
        self.class_id = int(class_id)
        self.class_name = class_name
        self.val_fraction = float(val_fraction)

        self._step = 0
        self._saved = 0
        self._skipped_empty = 0
        self._names: list[str] = []

    @property
    def done(self) -> bool:
        return self._saved >= self.max_frames

    @property
    def saved(self) -> int:
        return self._saved

    def maybe_record(
        self,
        rgb: torch.Tensor | np.ndarray,
        drone_pos_w: torch.Tensor,
        drone_quat_w: torch.Tensor,
        gate_pos_w: torch.Tensor,
        gate_quat_w: torch.Tensor,
    ) -> bool:
        """Record one frame if the stride allows. Returns True when the cap is hit."""
        self._step += 1
        if self.done or (self._step % self.every_n) != 0:
            return self.done

        if isinstance(rgb, torch.Tensor):
            img = rgb.detach().cpu().numpy()
        else:
            img = np.asarray(rgb)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        lines = self._label_lines(
            drone_pos_w=drone_pos_w,
            drone_quat_w=drone_quat_w,
            gate_pos_w=gate_pos_w,
            gate_quat_w=gate_quat_w,
            width=float(w),
            height=float(h),
        )
        if not lines:
            self._skipped_empty += 1
            return self.done

        stem = f"frame_{self._saved:06d}"
        cv2.imwrite(str(self.images_dir / f"{stem}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        (self.labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        self._names.append(stem)
        self._saved += 1
        return self.done

    def _label_lines(
        self,
        *,
        drone_pos_w: torch.Tensor,
        drone_quat_w: torch.Tensor,
        gate_pos_w: torch.Tensor,
        gate_quat_w: torch.Tensor,
        width: float,
        height: float,
    ) -> list[str]:
        # Accept (3,)/(4,) or (1,3)/(1,4); gates as (G,3)/(G,4).
        if drone_pos_w.ndim == 1:
            drone_pos_w = drone_pos_w.unsqueeze(0)
        if drone_quat_w.ndim == 1:
            drone_quat_w = drone_quat_w.unsqueeze(0)
        n_gates = gate_pos_w.shape[0]
        lines: list[str] = []
        for i in range(n_gates):
            kps_w = gate_keypoints_world(gate_pos_w[i : i + 1], gate_quat_w[i : i + 1])
            uv, vis = project_points_aigp_camera(
                kps_w,
                drone_pos_w,
                drone_quat_w,
                frame_w=width,
                frame_h=height,
            )
            uv = uv[0].detach().cpu().numpy()  # (8, 2) pixels
            vis = vis[0].detach().cpu().numpy().astype(bool)
            if int(vis.sum()) < self.min_visible:
                continue

            vis_uv = uv[vis]
            x0, y0 = vis_uv.min(axis=0)
            x1, y1 = vis_uv.max(axis=0)
            # Pad slightly so the orange bar is inside the box.
            pad_x = 0.02 * width
            pad_y = 0.02 * height
            x0 = max(0.0, x0 - pad_x)
            y0 = max(0.0, y0 - pad_y)
            x1 = min(width - 1.0, x1 + pad_x)
            y1 = min(height - 1.0, y1 + pad_y)
            bw = max(x1 - x0, 1.0)
            bh = max(y1 - y0, 1.0)
            cx = float(np.clip((x0 + x1) * 0.5 / width, 0.0, 1.0))
            cy = float(np.clip((y0 + y1) * 0.5 / height, 0.0, 1.0))
            nw = float(np.clip(bw / width, 1e-6, 1.0))
            nh = float(np.clip(bh / height, 1e-6, 1.0))
            # Keep the box fully inside the image after clamping centre/size.
            nw = min(nw, 2.0 * min(cx, 1.0 - cx))
            nh = min(nh, 2.0 * min(cy, 1.0 - cy))

            parts = [str(self.class_id), f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
            for k in range(KEYPOINT_COUNT):
                u = float(np.clip(uv[k, 0] / width, 0.0, 1.0))
                v = float(np.clip(uv[k, 1] / height, 0.0, 1.0))
                conf = 2 if vis[k] else 0
                parts.extend([f"{u:.6f}", f"{v:.6f}", str(conf)])
            lines.append(" ".join(parts))
        return lines

    def finalize(self) -> Path:
        """Write data.yaml with a train/val split over the saved frames."""
        n = len(self._names)
        n_val = max(1, int(round(n * self.val_fraction))) if n > 10 else max(0, min(1, n // 10))
        # Deterministic split: last val_fraction as val.
        val_names = set(self._names[-n_val:]) if n_val else set()
        train_img = self.root / "train" / "images"
        train_lbl = self.root / "train" / "labels"
        val_img = self.root / "valid" / "images"
        val_lbl = self.root / "valid" / "labels"
        for d in (train_img, train_lbl, val_img, val_lbl):
            d.mkdir(parents=True, exist_ok=True)

        import shutil

        for stem in self._names:
            split_img = val_img if stem in val_names else train_img
            split_lbl = val_lbl if stem in val_names else train_lbl
            shutil.copy2(self.images_dir / f"{stem}.jpg", split_img / f"{stem}.jpg")
            shutil.copy2(self.labels_dir / f"{stem}.txt", split_lbl / f"{stem}.txt")

        yaml_text = (
            f"path: {self.root.as_posix()}\n"
            f"train: train/images\n"
            f"val: valid/images\n"
            f"\n"
            f"kpt_shape: [8, 3]\n"
            f"flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]\n"
            f"\n"
            f"nc: 1\n"
            f"names: ['{self.class_name}']\n"
        )
        (self.root / "data.yaml").write_text(yaml_text)
        meta = {
            "frames": self._saved,
            "skipped_empty": self._skipped_empty,
            "every_n": self.every_n,
            "frame_size": [int(FRAME_W), int(FRAME_H)],
            "keypoint_count": KEYPOINT_COUNT,
            "note": "Projected analytic corners; v=2 in-frame, v=0 otherwise.",
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        return self.root
