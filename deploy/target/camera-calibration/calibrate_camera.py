#!/usr/bin/env python3
"""Checkerboard intrinsic calibration for Jetson IMX477 or saved photographs."""
import argparse
import glob
import json
import math
from pathlib import Path
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--images', help='Quoted image glob; no camera or display needed')
    source.add_argument('--source', help='Video file, MJPEG URL, or USB camera index')
    source.add_argument('--pipeline', help='Custom GStreamer pipeline ending in BGR appsink')
    p.add_argument('--cols', type=int, default=9, help='Inner corners across (default 9)')
    p.add_argument('--rows', type=int, default=6, help='Inner corners down (default 6)')
    p.add_argument('--square-mm', type=float, required=True, help='Measured checker square side in mm')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--sensor-id', type=int, default=0)
    p.add_argument('--sensor-mode', type=int, help='Argus sensor mode index')
    p.add_argument('--views', type=int, default=25, help='Target number of live captures')
    p.add_argument('--min-views', type=int, default=12)
    p.add_argument('--headless', action='store_true', help='Auto-capture without an OpenCV window')
    p.add_argument('--interval', type=float, default=2.0, help='Seconds between headless samples')
    p.add_argument('--timeout', type=float, default=300.0, help='Live collection time limit in seconds')
    p.add_argument('--output', type=Path, default=Path('camera_calibration'))
    a = p.parse_args()
    if a.cols < 3 or a.rows < 3 or not math.isfinite(a.square_mm) or a.square_mm <= 0:
        p.error('Use at least 3 x 3 inner corners and a finite positive square size')
    if min(a.width, a.height, a.fps) <= 0 or a.min_views < 10 or a.views < a.min_views:
        p.error('Dimensions/FPS must be positive; views >= min-views >= 10')
    if not all(math.isfinite(v) and v > 0 for v in (a.interval, a.timeout)):
        p.error('Interval and timeout must be finite and positive')
    if a.sensor_id < 0 or (a.sensor_mode is not None and a.sensor_mode < 0):
        p.error('Sensor identifiers must be nonnegative')
    return a


def jetson_pipeline(a):
    mode = '' if a.sensor_mode is None else f' sensor-mode={a.sensor_mode}'
    return (f'nvarguscamerasrc sensor-id={a.sensor_id}{mode} ! '
            f'video/x-raw(memory:NVMM),width={a.width},height={a.height},framerate={a.fps}/1 ! '
            'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! '
            'video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false')


def detect(frame, a):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (a.cols, a.rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))


def collect(a):
    points, names = [], []
    size = None
    first_frame = None

    def accept(frame, corners, name):
        nonlocal size, first_frame
        current = (frame.shape[1], frame.shape[0])
        if size is not None and current != size:
            raise ValueError(f'Mixed image sizes: {current} versus {size}; do not resize the dataset')
        # Compare both checkerboard orderings; prevent repeated near-identical poses.
        diagonal = math.hypot(*current)
        for previous in points:
            distance = min(np.mean(np.linalg.norm(corners - previous, axis=2)),
                           np.mean(np.linalg.norm(corners[::-1] - previous, axis=2)))
            if distance / diagonal < 0.015:
                print(f'Skipped near-duplicate: {name}', flush=True)
                return False
        size = current
        if first_frame is None:
            first_frame = frame.copy()
        points.append(corners)
        names.append(name)
        print(f'Accepted {len(points)}: {name}', flush=True)
        return True

    if a.images:
        files = sorted(glob.glob(a.images))
        if not files:
            raise ValueError(f'No images matched {a.images!r}')
        for name in files:
            frame = cv2.imread(name)
            if frame is None:
                raise ValueError(f'Cannot read image: {name}')
            if size is not None and (frame.shape[1], frame.shape[0]) != size:
                raise ValueError(f'Mixed image sizes in {name}')
            corners = detect(frame, a)
            if corners is None:
                print(f'No full checkerboard: {name}')
            else:
                accept(frame, corners, name)
    else:
        pipeline = a.pipeline or (jetson_pipeline(a) if a.source is None else None)
        if pipeline:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            source = int(a.source) if a.source.isdecimal() else a.source
            cap = cv2.VideoCapture(source)
            if isinstance(source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
                cap.set(cv2.CAP_PROP_FPS, a.fps)
        try:
            if not cap.isOpened():
                raise ValueError('Cannot open camera/stream. For Jetson use system Python with '
                                 'GStreamer and a working Argus/EGL context, or use --images / --source.')
            captures = a.output / 'captures'
            captures.mkdir()
            print('Move/tilt the board across the image. SPACE: capture; ENTER: finish; Q: cancel.', flush=True)
            start = time.monotonic()
            last = start - a.interval
            while len(points) < a.views and time.monotonic() - start < a.timeout:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Stream ended or camera read failed')
                corners = detect(frame, a)
                key = -1
                if not a.headless:
                    display = frame.copy()
                    if corners is not None:
                        cv2.drawChessboardCorners(display, (a.cols, a.rows), corners, True)
                    cv2.putText(display, f'{len(points)}/{a.views}  SPACE save | ENTER finish | Q cancel',
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                    scale = min(1.0, 1280 / display.shape[1])
                    cv2.imshow('Calibration', cv2.resize(display, None, fx=scale, fy=scale))
                    key = cv2.waitKey(1) & 255
                    if key in (ord('q'), 27):
                        raise ValueError('Cancelled; collected photos remain available')
                    if key in (10, 13):
                        break
                now = time.monotonic()
                capture = (a.headless and now - last >= a.interval) or key == 32
                if capture and corners is not None:
                    last = now
                    name = str(captures / f'frame_{len(points):03d}.png')
                    if accept(frame, corners, name) and not cv2.imwrite(name, frame):
                        raise ValueError(f'Cannot save {name}')
        finally:
            cap.release()
            if not a.headless:
                cv2.destroyAllWindows()
    if len(points) < a.min_views:
        raise ValueError(f'Only {len(points)} distinct valid views; need {a.min_views}. '
                         'Vary board tilt, distance and position; any captured images were retained.')
    return points, names, size, first_frame


def calibrate(a, points, names, size):
    board = np.zeros((a.rows * a.cols, 3), np.float32)
    board[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * (a.square_mm / 1000.0)
    objects = [board.copy() for _ in points]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objects, points, size, None, None)
    if not all(np.all(np.isfinite(x)) for x in (K, D)) or not math.isfinite(rms):
        raise ValueError('Calibration produced non-finite values')
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('Calibration produced invalid focal lengths')
    errors = []
    for measured, r, t in zip(points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(board, r, t, K, D)
        errors.append(float(np.sqrt(np.mean(np.sum((measured - projected) ** 2, axis=2)))))
    result = dict(model='opencv_pinhole', distortion_order=['k1', 'k2', 'p1', 'p2', 'k3'],
                  image_width=size[0], image_height=size[1], camera_matrix=K.tolist(),
                  distortion_coefficients=D.ravel().tolist(), rms_reprojection_error_px=float(rms),
                  board_inner_corners=[a.cols, a.rows], square_size_m=a.square_mm / 1000,
                  opencv_version=cv2.__version__, views=[dict(image=n, rms_px=e) for n, e in zip(names, errors)],
                  capture_settings=dict(width=a.width, height=a.height, fps=a.fps,
                                        sensor_id=a.sensor_id, sensor_mode=a.sensor_mode,
                                        source=a.images or a.source or a.pipeline or jetson_pipeline(a)))
    (a.output / 'calibration.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return K, D, rms, errors


def main():
    a = arguments()
    global cv2, np
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit('OpenCV and NumPy are required. On the supplied Jetson use system python3.') from exc
    try:
        # Never overwrite a previous calibration or its source images.
        a.output.mkdir(parents=True, exist_ok=False)
        points, names, size, first = collect(a)
        K, D, rms, errors = calibrate(a, points, names, size)
        corrected = cv2.undistort(first, K, D, None, K)
        if not cv2.imwrite(str(a.output / 'undistorted_preview.png'), corrected):
            raise ValueError('Unable to write undistorted preview')
        print(f'Saved {a.output / "calibration.json"}; RMS {rms:.3f} px; '
              f'worst view {max(errors):.3f} px; {len(points)} views.')
        if rms > 1.0 or max(errors) > 2.0:
            print('WARNING: high reprojection error. Inspect per-view errors and recapture blurry/poor views.')
        print('Inspect the preview and validate on new images; low training error alone does not prove accuracy.')
    except (ValueError, OSError, cv2.error) as exc:
        raise SystemExit(f'Error: {exc}') from exc
    except KeyboardInterrupt:
        raise SystemExit('Interrupted; captured images remain available.')


if __name__ == '__main__':
    main()
