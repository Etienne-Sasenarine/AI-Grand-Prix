# Perception quantization (TODO)

Home for quantizing the YOLOv8n-pose gate detector for fast inference on the
Jetson Orin NX. This is **not implemented yet** — the tracked artifacts today
are the FP32 PyTorch weights (`../models/*.pt`) and an ONNX export
(`../models/*.onnx`), which is the on-ramp to the steps below.

## Planned pipeline
1. **Export** — `best.pt` → ONNX (opset pinned; static input size). *(ONNX export exists.)*
2. **Calibrate** — build an INT8 calibration set from representative gate frames
   (the Roboflow dataset; fetch via `tools/download_models.sh`). *(TODO)*
3. **Build engine** — ONNX → TensorRT `.engine` with INT8 (FP16 fallback),
   targeting the Orin's GPU. *(TODO)*
4. **Validate** — compare quantized vs FP32 keypoint accuracy and per-frame
   latency **on the Orin**, and record both in the paper (`docs/paper/`) with the
   hardware named. *(TODO)*

## Notes
- Keep engines out of git — they are hardware/TensorRT-version specific and
  regenerable. Add them to `tools/download_models.sh` if they need distributing.
- Do not commit measured latency/accuracy numbers here; put them in the
  paper, against the exact hardware they were measured on.
