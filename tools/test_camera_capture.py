"""TEST 3 — camera capture for labelling. Nothing flies.

    python3 test_camera_capture.py --verify              # prove the clock domain
    python3 test_camera_capture.py --modes               # what the sensor offers
    python3 test_camera_capture.py --seconds 120 --fps 2 --out capture/

The safest test on the list
---------------------------
Props off, and the aircraft never leaves the ground or arms. If the Orin is on
its own supply the **flight battery need not be connected at all**, which makes
this the only measurement with no stored energy near the propellers.

What it produces
----------------
JPEG frames plus ``frames.csv`` carrying ``frame, pts_ns, mono_s, utc`` — the
same columns the organizers' own ``frame-timestamps.py`` writes, so the two are
directly comparable.

Why python3-gi and not cv2.VideoCapture
---------------------------------------
Because **cv2.VideoCapture throws the hardware timestamp away.** The software
guide is explicit about it: absolute capture time is
``pipeline.base_time + buffer.pts``, and an OpenCV capture can only stamp
*arrival*, which folds in tens of milliseconds of pipeline latency and jitter.

That timestamp is latched in hardware at the frame boundary by NVCSI/VI and
passed through untouched — CPU scheduling and ISP work move the *latency*, never
the timestamp. It is the one honest clock we have for pairing a frame with an
IMU sample, and latency is the most sensitive quantity in the whole system.

An earlier version of this file used ``cv2.VideoCapture(..., CAP_GSTREAMER)``
while its own docstring claimed otherwise. Naming the pipeline GStreamer does
not preserve the PTS; only reading ``GstBuffer.pts`` does.

Three cautions the guide raises, worth knowing before trusting these stamps
--------------------------------------------------------------------------
1. It is the **SoC's** clock, not the camera's. Crystals differ by tens of ppm,
   so a nominal 30.000 fps measures about 29.9997 and the error accumulates.
2. It marks a **frame boundary**, which on a rolling shutter is *after* the
   first row's exposure ended. The exposure midpoint you want to pair with an
   IMU sample is behind the timestamp, not ahead of it.
3. Over a plain SSH session there is no EGL context, so Argus cannot run and the
   colour is flat and un-exposed **by design**. Geometry is unaffected — corners
   are still corners — but a detector trained only on flat frames meets
   different-looking ones on race day.

Safety
------
This program never opens the flight-controller port. It cannot arm and cannot
command a motor. Props off is still the rule, because the aircraft is powered.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: Real camera. ``appsink`` is named so we can pull buffers and read their PTS.
#: ``drop=true max-buffers=1`` is not optional for anything live: without it
#: GStreamer queues frames and you fall further behind every second.
PIPELINE = ("nvarguscamerasrc sensor-id={sid} ! "
            "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false")

#: No camera needed. The organizers' own tooling offers the same escape hatch.
TEST_PIPELINE = ("videotestsrc pattern=smpte is-live=true ! "
                 "video/x-raw,width={w},height={h},framerate={fps}/1,format=BGR ! "
                 "appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false")


class MissingGst(RuntimeError):
    pass


def _gst():
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except Exception as e:                       # pragma: no cover - drone has it
        raise MissingGst(
            f"python3-gi / GStreamer not available here ({e}). This is installed "
            f"on the drone image; run this test there.") from e
    if not Gst.is_initialized():
        Gst.init(None)
    return Gst


def list_modes() -> str:
    """Ask the driver what it actually offers. The field of view belongs to the mode."""
    exe = shutil.which("v4l2-ctl")
    if not exe:
        return "v4l2-ctl not found. On the drone: v4l2-ctl -d /dev/video0 --list-formats-ext"
    try:
        return subprocess.run([exe, "-d", "/dev/video0", "--list-formats-ext"],
                              capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        return f"could not list modes: {e}"


def has_egl() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def verify_clock(test_source: bool = False, width: int = 640, height: int = 360,
                 fps: int = 30) -> dict:
    """Query the pipeline's clock domain rather than assuming it.

    The guide is firm that this is queried, not assumed, because the whole value
    of the timestamp is knowing which clock it is in.
    """
    Gst = _gst()
    src = (TEST_PIPELINE if test_source else PIPELINE).format(
        sid=0, w=width, h=height, fps=fps)
    pipeline = Gst.parse_launch(src)
    pipeline.set_state(Gst.State.PLAYING)
    pipeline.get_state(Gst.CLOCK_TIME_NONE)
    clock = pipeline.get_clock()
    info = {
        "clock_type": type(clock).__name__ if clock else None,
        "base_time_ns": int(pipeline.get_base_time()),
        "clock_now_ns": int(clock.get_time()) if clock else None,
        "monotonic_now_s": time.monotonic(),
        "is_system_clock": bool(clock and "System" in type(clock).__name__),
    }
    pipeline.set_state(Gst.State.NULL)
    return info


def capture(out_dir: Path, *, seconds: float, fps: float, width: int, height: int,
            sensor_id: int, test_source: bool, jpeg_quality: int = 92,
            source_fps: int = 30, verbose: bool = True) -> dict:
    Gst = _gst()
    import numpy as np

    try:
        import cv2
        def encode(arr, path):
            cv2.imwrite(str(path), arr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    except ImportError:                          # pragma: no cover
        from PIL import Image
        def encode(arr, path):
            Image.fromarray(arr[:, :, ::-1]).save(path, quality=jpeg_quality)

    src = (TEST_PIPELINE if test_source else PIPELINE).format(
        sid=sensor_id, w=width, h=height, fps=source_fps)
    pipeline = Gst.parse_launch(src)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    if pipeline.get_state(5 * Gst.SECOND)[0] != Gst.StateChangeReturn.SUCCESS:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError(
            "pipeline would not reach PLAYING. If this is the real camera, check "
            "~/target/camera-bind-check.sh; if Argus failed, there may be no EGL "
            "context (a plain SSH session has none).")

    base_time = int(pipeline.get_base_time())
    clock = pipeline.get_clock()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows, index, dropped = [], 0, 0
    t0 = time.monotonic()
    save_period = 1.0 / max(fps, 1e-6)
    next_save = 0.0
    try:
        while time.monotonic() - t0 < seconds:
            sample = sink.emit("try-pull-sample", int(1.0 * Gst.SECOND))
            if sample is None:
                dropped += 1
                if dropped > 20:
                    break
                continue
            buf = sample.get_buffer()
            pts = int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else -1
            mono = time.monotonic()
            elapsed = mono - t0
            if elapsed < next_save:
                continue                          # keep the newest, save on cadence
            next_save = elapsed + save_period

            caps = sample.get_caps().get_structure(0)
            w, h = caps.get_value("width"), caps.get_value("height")
            ok, mi = buf.map(Gst.MapFlags.READ)
            if not ok:
                dropped += 1
                continue
            try:
                arr = np.frombuffer(mi.data, dtype=np.uint8).reshape(h, w, 3).copy()
            finally:
                buf.unmap(mi)

            name = f"frame_{index:06d}.jpg"
            encode(arr, frames_dir / name)
            rows.append({
                "frame": index, "file": name,
                "pts_ns": pts,
                "abs_capture_ns": (base_time + pts) if pts >= 0 else -1,
                "mono_s": round(mono, 6),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "width": w, "height": h,
            })
            index += 1
            if verbose and index % 10 == 0:
                print(f"  {index} frames, {elapsed:.0f}s", flush=True)
    finally:
        pipeline.set_state(Gst.State.NULL)

    with open(out_dir / "frames.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["frame", "file", "pts_ns", "abs_capture_ns",
                                             "mono_s", "utc", "width", "height"])
        wtr.writeheader()
        wtr.writerows(rows)

    # Inter-frame timing straight from the hardware stamps.
    dts = [(b["pts_ns"] - a["pts_ns"]) / 1e6
           for a, b in zip(rows, rows[1:]) if a["pts_ns"] >= 0 and b["pts_ns"] >= 0]
    meta = {
        "frames": len(rows), "missed_pulls": dropped,
        "requested": {"width": width, "height": height, "save_fps": fps,
                      "source_fps": source_fps, "seconds": seconds,
                      "sensor_id": sensor_id},
        "actual_frame_size": [rows[0]["width"], rows[0]["height"]] if rows else None,
        "test_source": test_source,
        "pts_available": bool(dts),
        "pts_dt_ms_median": round(sorted(dts)[len(dts) // 2], 3) if dts else None,
        "base_time_ns": base_time,
        "clock_type": type(clock).__name__ if clock else None,
        "egl_context": has_egl(),
        "exposure_note": (
            "Captured WITH a graphics context; ISP colour should be correct."
            if has_egl() else
            "NO graphics context (plain SSH). Argus cannot run, so colour is flat "
            "and un-exposed BY DESIGN. Geometry is unaffected; colour is not "
            "representative of race day."),
        "modes_reported_by_driver": None if test_source else list_modes(),
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "capture_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def report(meta: dict, out_dir: Path) -> None:
    print()
    print("CAMERA CAPTURE RESULT")
    print("=" * 68)
    print(f"  frames saved     {meta['frames']}  -> {out_dir / 'frames'}")
    print(f"  frame size       {meta['actual_frame_size']}")
    print(f"  missed pulls     {meta['missed_pulls']}")
    print(f"  hardware PTS     {'yes' if meta['pts_available'] else 'NO'}"
          + (f", median gap {meta['pts_dt_ms_median']} ms" if meta["pts_available"] else ""))
    print(f"  pipeline clock   {meta['clock_type']}")
    print(f"  graphics context {'yes' if meta['egl_context'] else 'NO'}")
    if meta["actual_frame_size"] and meta["requested"]["width"] != meta["actual_frame_size"][0]:
        print()
        print("  NOTE: the driver gave a different frame size than requested.")
        print("        The field of view belongs to the MODE, so record which one")
        print("        you got -- calibration is only valid for that mode.")
    print()
    print(f"  {meta['exposure_note']}")
    print()
    print("  For labelling, variety beats volume: every gate, 1-15 m, angles you")
    print("  will actually fly, gates half out of frame, two gates at once,")
    print("  backlit, plus negatives with no gate at all.")
    print("=" * 68)


BRIEF = """
TEST 3 - CAMERA CAPTURE                              NOTHING FLIES
------------------------------------------------------------------------
This program never opens the flight-controller port. It cannot arm and
cannot command a motor.

SETUP:
  * PROPS OFF. The aircraft is powered; props off is still the rule.
  * If the Orin has its own supply you do NOT need the flight battery.
    That is the safest configuration on the whole measurement list.
  * Carry the aircraft around the track by hand, pointing the camera at
    gates from the ranges and angles you will actually fly.

WHAT MAKES A USEFUL FRAME SET:
  * Every gate on the course, not just the nearest.
  * A spread of distance: close, mid, and far enough to be small.
  * Gates half out of frame, and two gates in shot at once - those are
    exactly the cases the tracker has to disambiguate.
  * Some frames with NO gate at all. A detector needs negatives.
  * The double gate from several angles; it is the one counted twice.

Slow and steady: walking blur is not 16 m/s blur, and sharp frames are
easier to label.
------------------------------------------------------------------------
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Camera capture for labelling; nothing flies")
    ap.add_argument("--out", default="capture")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--fps", type=float, default=2.0, help="frames SAVED per second")
    ap.add_argument("--source-fps", type=int, default=30, help="sensor mode frame rate")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--modes", action="store_true")
    ap.add_argument("--verify", action="store_true", help="prove the clock domain, then exit")
    ap.add_argument("--test-source", action="store_true",
                    help="videotestsrc instead of the camera, to prove the path")
    args = ap.parse_args()

    if args.modes:
        print(list_modes())
        return 0
    try:
        if args.verify:
            info = verify_clock(test_source=args.test_source,
                                width=args.width, height=args.height, fps=args.source_fps)
            print(json.dumps(info, indent=2))
            print("\n  absolute capture time = base_time + buffer.pts")
            print(f"  clock is {'a system clock' if info['is_system_clock'] else info['clock_type']}")
            return 0

        print(BRIEF)
        if not args.test_source and not has_egl():
            print("  NOTE: no graphics context. Frames will be flat and un-exposed.")
            print()
        meta = capture(Path(args.out), seconds=args.seconds, fps=args.fps,
                       width=args.width, height=args.height, sensor_id=args.sensor_id,
                       test_source=args.test_source, source_fps=args.source_fps)
    except MissingGst as e:
        print(f"\n  {e}")
        return 2
    except KeyboardInterrupt:
        print("\n  stopped by operator")
        return 0
    except RuntimeError as e:
        print(f"\n  {e}")
        return 1
    report(meta, Path(args.out))
    return 0


def _self_test() -> int:
    """Exercise the real capture path through videotestsrc — no camera needed."""
    try:
        _gst()
    except MissingGst as e:
        print(f"test_camera_capture: SKIPPED -- {e}")
        print("  The capture path was NOT exercised. Run on the drone:")
        print("    python3 test_camera_capture.py --self-test")
        return 2

    out = Path("/tmp/_camtest")
    shutil.rmtree(out, ignore_errors=True)
    meta = capture(out, seconds=3.0, fps=4.0, width=640, height=360, sensor_id=0,
                   test_source=True, source_fps=30, verbose=False)

    assert meta["frames"] >= 6, meta
    assert (out / "frames.csv").exists() and (out / "capture_meta.json").exists()
    files = sorted((out / "frames").glob("*.jpg"))
    assert len(files) == meta["frames"], (len(files), meta["frames"])
    assert all(f.stat().st_size > 512 for f in files), "a frame came out empty"

    rows = list(csv.DictReader(open(out / "frames.csv")))
    pts = [int(r["pts_ns"]) for r in rows]
    assert all(p >= 0 for p in pts), "hardware PTS missing — this is the whole point"
    assert all(b > a for a, b in zip(pts, pts[1:])), "PTS must increase"
    mono = [float(r["mono_s"]) for r in rows]
    assert all(b > a for a, b in zip(mono, mono[1:])), "arrival times must increase"
    # PTS and arrival must agree on elapsed time to within a sane margin.
    span_pts = (pts[-1] - pts[0]) / 1e9
    span_mono = mono[-1] - mono[0]
    assert abs(span_pts - span_mono) < 0.5, (span_pts, span_mono)
    assert meta["pts_available"] and meta["pts_dt_ms_median"] > 0

    info = verify_clock(test_source=True)
    assert info["base_time_ns"] >= 0 and info["clock_type"]

    print("test_camera_capture: all checks passed")
    print(f"  {meta['frames']} frames via videotestsrc, {meta['actual_frame_size']}")
    print(f"  hardware PTS present and monotonic; median gap "
          f"{meta['pts_dt_ms_median']} ms")
    print(f"  PTS span {span_pts:.2f}s vs arrival span {span_mono:.2f}s")
    print(f"  pipeline clock: {info['clock_type']}")
    shutil.rmtree(out, ignore_errors=True)
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
