# AI_GP_YOLO
Trained YOLO pose model for the AI Grand Prix Physical Qualifier.

Detects racing gates from the drone's camera and localises their 8 corners
(4 outer + 4 inner) so the flight controller can aim through the opening.

## Contents
* `best.pt` – fine-tuned YOLOv8n-pose, 1 class (`gate`), 8 keypoints per gate
* `inference.py` – `GateDetector` library class + CLI for images / video / webcam
* `AI_GP_Yolo_Model.ipynb` – notebook used to build the dataset and train the model

Keypoint order: `out_UL, out_UR, out_BR, out_BL, in_UL, in_UR, in_BR, in_BL`.

## Setup
```
pip install -r requirements.txt
```
A CUDA build of torch is recommended (see https://pytorch.org/get-started/locally/); the script falls back to CPU.

## Usage
From flight-control code:
```python
from inference.inference import GateDetector
det = GateDetector()                     # loads inference/best.pt once, warms up, auto GPU/CPU
gates = det.detect(frame_bgr)            # list[Gate], nearest/most-confident first
if gates:
    x, y   = gates[0].aim_point          # pixel to fly towards (inner-corner centroid)
    dx, dy = gates[0].aim_offset(frame_bgr.shape)   # -1..1 offset from image centre
```

From the command line:
```
python inference/inference.py --source img.jpg --show
python inference/inference.py --source flight.mp4 --save out.mp4
python inference/inference.py --source 0 --show                    # webcam, q / Esc quits
python inference/inference.py --source frames/ --json gates.jsonl  # one JSON line per frame
```
Options: `--imgsz`, `--conf`, `--kpt-conf`, `--device`, `--half`, `--quiet`.
