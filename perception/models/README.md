# Gate-pose model: `gate_pose_hand497`

**The one to fly.** `gate_pose_hand497.onnx` (deploy) and `gate_pose_hand497.pt`,
copied unchanged from `aigp-perception/models/` (commit `04b429c`).

| | |
|---|---|
| Files | `gate_pose_hand497.onnx` (sha256 `1200ca4ec0c7ed08...`), `gate_pose_hand497.pt` (`9700770b546e39f3...`) |
| Architecture | `yolov8n-pose`, 8 keypoints, imgsz 640 |
| Training set | 1265 frames / 2348 gates, **497 human-annotated**, 58 of the newest being close-ups with the gate cut off by the frame edge |
| Initialised from | `gate_pose_hybrid_v1.pt`; 150 epochs |
| Keypoint confidence | **0.25** everywhere |

Measured against its predecessor `gate_pose_hand434` **at a real gate on
2026-09-21**, 1345 frames, both at keypoint confidence 0.25:

| | hand497 | hand434 |
|---|---|---|
| corner flicker | **4.2 %** | 6.0 % |
| usable fragments (fewer = steadier) | **37** | 78 |
| longest unbroken run | **178 frames** | 142 |
| policy-ready frames | **59 %** | 56 % |

It does best with all eight corners clearly in frame and degrades once the gate
is close enough to be cut off. Keypoint error against human labels is quoted in
**1920x1080 pixels**.

`inference/best.pt` in this repo is an older model (`gate_pose_teammate`). To fly
this one, pass `--weights models/gate_pose_hand497.pt` (Plan B `onboard.py`) or
set `GATE_WEIGHTS` for the camera runner.
