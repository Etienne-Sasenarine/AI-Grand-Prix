# IMX477 camera calibration

`calibrate_camera.py` estimates a single camera's intrinsic matrix and five OpenCV pinhole distortion coefficients using a flat checkerboard. It targets the Jetson Orin NX / IMX477 setup in your guides. It does not estimate camera-to-IMU alignment, time offset, or position on the race track. Gate-center coordinates alone are not a calibration target.

## Prepare

Use a rigid, flat checkerboard with a white border. The default **9 by 6 inner corners** means **10 by 7 squares**. Measure the actual printed square side; the commands below assume 25 mm and must be changed to match your board. Keep focus fixed. Use the same sensor mode, resolution, crop, orientation and lens settings that your application will use. Changing those can invalidate calibration.

Use the Jetson's system `python3`, with its existing OpenCV/NumPy and GStreamer support. The supplied guides advise against replacing that OpenCV with a pip wheel. Stop other applications using the camera before direct capture.

## Capture on the Jetson with a display

Copy the script to the Jetson and run from its containing directory:

```bash
python3 calibrate_camera.py --cols 9 --rows 6 --square-mm 25 \
  --width 1920 --height 1080 --fps 30 --output calibration_1080p
```

This uses `nvarguscamerasrc`. Select a supported mode/resolution; use `--sensor-mode N` if needed. The guides describe EGL/display requirements for their Argus setup. `--headless` only disables the script's window; it does not supply an EGL context.

Hold the board still for each capture. Press **Space** when the full checkerboard is detected. Collect about 25 sharp views covering the center, edges and corners, different distances, and substantial tilts in both axes. Keep the entire board visible. Do not collect only front-facing views. Press **Enter** to finish early after at least 12 accepted views; **Q** cancels. Collection normally stops after 25 accepted views or 300 seconds. Camera/backend reads can block independently of this collection timeout.

The preview is resized for display only; calibration and saved photos use full-resolution frames.

## Without a local window

The guide's existing viewer exposes an MJPEG stream. Start that viewer separately and calibrate its stream:

```bash
python3 calibrate_camera.py --source http://127.0.0.1:8080/stream \
  --headless --square-mm 25 --interval 3 --timeout 300 \
  --output calibration_stream
```

This samples automatically when the board is detected. Move to varied poses between samples, then hold still. Stream support depends on OpenCV's available video backends. The stream must be clean, with no overlays on the board, and must have the same image geometry as your eventual input. JPEG compression or a fallback viewer that scales/crops frames can affect the result. Width/height flags do not resize a URL stream; the actual decoded dimensions are saved.

Alternatively, process existing full-resolution photographs, with no display required:

```bash
python3 calibrate_camera.py --images 'checkerboard_photos/*.png' \
  --cols 9 --rows 6 --square-mm 25 --output calibration_photos
```

Use `--source 0` for a USB camera, or `--pipeline '... ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'` for a custom GStreamer input. Directly opening the IMX477 raw Bayer device as a USB webcam is not supported.

## Outputs and usage

Each run requires a new output directory to protect earlier results:

- `calibration.json`: camera matrix, distortion coefficients, actual image size, board dimensions, capture settings and per-view RMS errors in pixels.
- `captures/`: original accepted live frames, for inspection and recalibration.
- `undistorted_preview.png`: first accepted image corrected with the fitted model; black borders are possible.

```python
import json
import cv2
import numpy as np

with open('calibration_1080p/calibration.json') as f:
    calibration = json.load(f)
K = np.array(calibration['camera_matrix'], dtype=np.float64)
D = np.array(calibration['distortion_coefficients'], dtype=np.float64)
# frame is a BGR image from the same camera configuration used for calibration.
assert (frame.shape[1], frame.shape[0]) == (
    calibration['image_width'], calibration['image_height'])
corrected = cv2.undistort(frame, K, D, None, K)
```

`K` contains focal lengths and principal point in pixels. `D` is ordered `k1, k2, p1, p2, k3`. For repeated real-time frames, precompute maps with `cv2.initUndistortRectifyMap` and use `cv2.remap`. For pose estimation on original distorted images, pass `K` and `D` to `solvePnP`; for these corrected images, use `K` and zero distortion.

Inspect the corrected image and validate straight lines and pose/distance estimates on fresh images. RMS above 1 pixel or any view above 2 pixels triggers an advisory warning, not a universal acceptance criterion. A small fit error alone does not guarantee a useful calibration. Review bad views and recapture them; the script does not silently discard fit outliers. Very wide-angle/fisheye lenses may require OpenCV's separate fisheye model, which this script does not implement. The sensor name alone does not identify the lens model.

Algorithm reference: [OpenCV camera calibration tutorial](https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html).

Validation performed during authoring: Python syntax, CLI help and invalid-argument checks. OpenCV execution and physical camera calibration were not tested because the authoring environment has neither OpenCV nor the Jetson camera.
