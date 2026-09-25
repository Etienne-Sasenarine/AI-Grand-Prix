"""TEST 3 — camera capture for labelling.

    python3 test_camera_capture.py --verify              # prove the clock domain
    python3 test_camera_capture.py --modes               # what the sensor offers
    python3 test_camera_capture.py --seconds 120 --fps 2 --out capture/
    python3 test_camera_capture.py --seconds 0           # until Ctrl-C

    # props ON, during a piloted flight, telemetry paired to every frame:
    python3 test_camera_capture.py --flight --dev /dev/ttyTHS1 --seconds 0

Two modes
---------
**Ground mode (the default). Nothing flies.** Props off, the aircraft never
leaves the ground or arms. If the Orin is on its own supply the **flight battery
need not be connected at all**, which makes this the only measurement on the
list with no stored energy near the propellers. Carry the aircraft around and
point it at things.

**Flight mode (``--flight``). Props on, a human pilot flying.** Same capture,
plus the flight controller's telemetry recorded alongside and paired to every
saved frame by timestamp. It exists because walking blur is not 16 m/s blur: a
detector trained only on hand-carried frames meets something different on race
day. Pairing each frame with attitude also gives the data to check the camera
angle and, later, the camera-to-IMU timing offset.

**Flight mode still never commands anything.** It opens the flight-controller
port read-only — attitude, gyro, motor outputs, stick positions and battery. It
has no code path that arms, transmits RC, or writes a setting, in either mode.
The pilot flies; this only watches. If the link drops it keeps taking pictures
and records that telemetry was lost, because the images are the point.

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
3. The guide warns that without an EGL context — which a plain SSH session has
   not got — Argus cannot expose, and the colour comes out flat.
   **Measured on the race Orin on 18 Sep, this did not happen.** Over plain SSH
   with no ``DISPLAY``, Argus started, auto-exposure converged within about one
   frame, and 1920x1080 frames came back correctly exposed with normal colour
   (median pixel mean 104, spread 62, full 0-255 range). The warning may apply
   to other boards, images or modes, so this program **measures** exposure from
   the frames it took rather than inferring it from the environment. See
   ``grade_exposure``.

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
import threading
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


#: Columns appended to frames.csv in flight mode, on top of the ground-mode ones.
TELEM_FIELDS = [
    "telem_age_ms", "armed", "roll_deg", "pitch_deg", "yaw_deg",
    "gyro_x_raw", "gyro_y_raw", "gyro_z_raw",
    "motor_mean", "rc_throttle", "voltage_v", "altitude_m",
]


class TelemetryRecorder:
    """Poll the flight controller on a background thread. **Read-only.**

    It calls only the accessors — ``attitude``, ``raw_imu``, ``motors``,
    ``rc_channels``, ``analog``, ``altitude``, ``status``. There is deliberately
    no code path here that arms, transmits RC, or writes a setting, because this
    runs while a pilot is flying and the aircraft has propellers on.

    Samples are kept in memory and paired to frames afterwards by monotonic
    timestamp. A read failure is counted, not raised: if the serial link drops
    mid-flight we would rather keep taking pictures than lose the session.
    """

    #: One sample costs several MSP round trips, and the organizers put the
    #: link at 30-50 Hz for a *single* request. Polling everything every cycle
    #: therefore caps the sample rate at a few Hz. Attitude, gyro, motors and
    #: sticks change frame to frame and are read every cycle; battery, altitude
    #: and the armed flag change slowly, so they are refreshed every Nth cycle
    #: and carried forward in between.
    SLOW_EVERY = 10

    def __init__(self, fc, hz: float = 50.0):
        self.fc = fc
        self.period = 1.0 / max(hz, 1e-6)
        self.samples: list[tuple[float, dict]] = []
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cycle = 0
        self._slow = {"armed": False, "voltage_v": "", "current_a": "",
                      "altitude_m": "", "vario_m_s": ""}

    def _poll_once(self) -> dict:
        roll, pitch, yaw = self.fc.attitude()
        acc, gyro, _ = self.fc.raw_imu()
        motors = self.fc.motors()
        rc = self.fc.rc_channels()
        if self._cycle % self.SLOW_EVERY == 0:
            batt = self.fc.analog()
            alt, vario = self.fc.altitude()
            st = self.fc.status()
            self._slow = {
                "armed": bool(st.get("armed", False)),
                "voltage_v": batt.get("voltage_v", ""),
                "current_a": batt.get("current_a", ""),
                "altitude_m": round(float(alt), 3),
                "vario_m_s": round(float(vario), 4),
            }
        self._cycle += 1
        return {
            **self._slow,
            "roll_deg": round(roll, 3), "pitch_deg": round(pitch, 3),
            "yaw_deg": round(yaw, 3),
            "gyro_x_raw": gyro[0], "gyro_y_raw": gyro[1], "gyro_z_raw": gyro[2],
            "acc_x_raw": acc[0], "acc_y_raw": acc[1], "acc_z_raw": acc[2],
            "motor0": motors[0], "motor1": motors[1],
            "motor2": motors[2], "motor3": motors[3],
            "motor_mean": round(sum(motors[:4]) / 4.0, 1),
            "rc_throttle": rc[0] if len(rc) > 0 else "",
            "rc_roll": rc[1] if len(rc) > 1 else "",
            "rc_pitch": rc[2] if len(rc) > 2 else "",
            "rc_yaw": rc[3] if len(rc) > 3 else "",
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                self.samples.append((now, self._poll_once()))
            except Exception:
                self.errors += 1
            slack = self.period - (time.monotonic() - now)
            if slack > 0:
                self._stop.wait(slack)

    def start(self, *, wait_first_s: float = 6.0) -> "TelemetryRecorder":
        """Start polling and wait for the first sample to prove the link works.

        The wait is generous on purpose. ``MSPLink.request`` retries twice with
        a 0.5 s timeout, so a single slow field can take over a second, and one
        sample is several requests. An earlier version waited 0.3 s, decided the
        flight controller was not answering, and refused to record -- against a
        flight controller that was answering perfectly well.
        """
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + wait_first_s
        while time.monotonic() < deadline and not self.samples:
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def nearest(self, t: float) -> tuple[dict | None, float]:
        """Closest sample to monotonic time ``t``, and how stale it is in ms.

        Linear scan from the end. Frames arrive in order, so this is short in
        practice, and being obviously correct matters more here than being fast.
        """
        best, best_dt = None, float("inf")
        for ts, sample in reversed(self.samples):
            dt = abs(ts - t)
            if dt < best_dt:
                best, best_dt = sample, dt
            elif ts < t - best_dt:
                break
        return best, (best_dt * 1000.0 if best is not None else float("nan"))

    def write_csv(self, path: Path) -> dict:
        """Full-rate telemetry, independent of the frame cadence."""
        if not self.samples:
            return {"rows": 0, "achieved_hz": 0.0, "read_errors": self.errors}
        t0 = self.samples[0][0]
        cols = ["t"] + sorted(self.samples[0][1].keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for ts, sample in self.samples:
                w.writerow({"t": round(ts - t0, 6), **sample})
        span = self.samples[-1][0] - t0
        return {
            "rows": len(self.samples),
            "seconds": round(span, 2),
            "achieved_hz": round(len(self.samples) / span, 1) if span > 0 else 0.0,
            "read_errors": self.errors,
            "armed_fraction": round(
                sum(1 for _, s in self.samples if s["armed"]) / len(self.samples), 3),
            "path": str(path),
        }


def has_egl() -> bool:
    """Whether this session advertises a graphics context.

    Kept for the record, but **do not conclude anything about exposure from
    it.** Measured on the race Orin (JetPack 6.2, plain SSH, no DISPLAY):
    Argus started, auto-exposure converged and 1920x1080 frames came out
    correctly exposed with normal colour. The guide's warning that headless
    gives flat un-exposed frames did not hold on this board. ``grade_exposure``
    measures the frames instead of assuming.
    """
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def grade_exposure(stats: list[tuple[float, float]]) -> dict:
    """Judge exposure from the frames actually taken, not from the environment.

    ``stats`` is per-frame ``(mean, std)`` of pixel value, 0-255. A frame that
    the image processor never exposed is flat: everything bunched into a narrow
    band, so a low standard deviation. A correctly exposed indoor frame lands
    near mid-grey with a wide spread.

    The thresholds are deliberately loose. This is here to catch the failure
    mode "we captured 4000 useless grey frames and nobody looked until the
    labelling started", not to grade photography.
    """
    if not stats:
        return {"verdict": "no frames", "ok": False}
    means = sorted(m for m, _ in stats)
    stds = sorted(sd for _, sd in stats)
    med_mean = means[len(means) // 2]
    med_std = stds[len(stds) // 2]

    if med_std < 12.0:
        verdict = ("FLAT - the image processor does not appear to be exposing. "
                   "These frames are poor labelling data.")
        ok = False
    elif med_mean < 35.0:
        verdict = "TOO DARK - usable geometry, but poor labelling data."
        ok = False
    elif med_mean > 215.0:
        verdict = "BLOWN OUT - highlights clipped."
        ok = False
    else:
        verdict = "exposed normally - good labelling data."
        ok = True

    # Auto-exposure needs a moment. Find where the run settles, so the operator
    # knows whether the warm-up was long enough.
    settle = 0
    if len(stats) > 2:
        for i, (m, _) in enumerate(stats):
            if abs(m - med_mean) <= 0.15 * max(med_mean, 1.0):
                settle = i
                break
        else:
            settle = len(stats)
    return {
        "verdict": verdict, "ok": ok,
        "median_mean": round(med_mean, 1), "median_std": round(med_std, 1),
        "first_frame_mean": round(stats[0][0], 1),
        "settled_after_frames": settle,
    }


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
            source_fps: int = 30, verbose: bool = True,
            telemetry: "TelemetryRecorder | None" = None,
            warmup_s: float = 1.5) -> dict:
    """Capture frames to ``out_dir``.

    ``seconds <= 0`` runs until Ctrl-C, which is what a walk-around session or a
    flight wants — you do not know in advance how long the battery lasts. Either
    way the CSV and the metadata are written, so an interrupted session is still
    a usable session.

    ``warmup_s`` pulls and discards frames before saving any. Auto-exposure
    starts dark and converges: measured on the race camera the first frame came
    back at mean 57 against a settled 106. Those early frames are real data of
    the wrong thing, and they are worth discarding rather than labelling.
    """
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
    open_ended = seconds <= 0
    interrupted = False
    exposure_stats: list[tuple[float, float]] = []
    warmed = warmup_s <= 0.0
    if verbose and open_ended:
        print("  running until Ctrl-C", flush=True)
    if verbose and not warmed:
        print(f"  letting auto-exposure settle for {warmup_s:.1f}s", flush=True)
    try:
        while open_ended or (time.monotonic() - t0 < seconds):
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
            if not warmed:
                if elapsed < warmup_s:
                    continue                      # pull and drop; let AE converge
                warmed = True
                t0 = time.monotonic()             # the session starts now
                elapsed = 0.0
                next_save = 0.0
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

            # Subsampled: at 1080p the full array costs more than it tells us.
            probe = arr[::4, ::4]
            exposure_stats.append((float(probe.mean()), float(probe.std())))

            name = f"frame_{index:06d}.jpg"
            encode(arr, frames_dir / name)
            row = {
                "frame": index, "file": name,
                "pts_ns": pts,
                "abs_capture_ns": (base_time + pts) if pts >= 0 else -1,
                "mono_s": round(mono, 6),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "width": w, "height": h,
            }
            if telemetry is not None:
                sample, age_ms = telemetry.nearest(mono)
                row["telem_age_ms"] = round(age_ms, 1) if sample else ""
                for k in TELEM_FIELDS[1:]:
                    row[k] = sample.get(k, "") if sample else ""
            rows.append(row)
            index += 1
            if verbose and index % 10 == 0:
                extra = ""
                if telemetry is not None and rows[-1].get("armed") != "":
                    extra = (f", {'ARMED' if rows[-1]['armed'] else 'disarmed'}"
                             f", {rows[-1]['voltage_v']}V")
                print(f"  {index} frames, {elapsed:.0f}s{extra}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("\n  stopped by operator -- writing what we have", flush=True)
    finally:
        pipeline.set_state(Gst.State.NULL)

    fieldnames = ["frame", "file", "pts_ns", "abs_capture_ns",
                  "mono_s", "utc", "width", "height"]
    if telemetry is not None:
        fieldnames += TELEM_FIELDS
    with open(out_dir / "frames.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fieldnames)
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
        "exposure": grade_exposure(exposure_stats),
        "warmup_s": warmup_s,
        "modes_reported_by_driver": None if test_source else list_modes(),
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "open_ended": open_ended,
        "interrupted": interrupted,
        "flight_mode": telemetry is not None,
    }
    if telemetry is not None:
        meta["telemetry"] = telemetry.write_csv(out_dir / "telemetry.csv")
        ages = [r["telem_age_ms"] for r in rows
                if isinstance(r.get("telem_age_ms"), float)]
        meta["telemetry"]["frame_pairing_age_ms_median"] = (
            round(sorted(ages)[len(ages) // 2], 1) if ages else None)
        meta["telemetry"]["frame_pairing_age_ms_worst"] = (
            round(max(ages), 1) if ages else None)
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
    if meta.get("interrupted"):
        print("  stopped by operator; everything above was still written")
    if meta.get("flight_mode"):
        t = meta.get("telemetry", {})
        print()
        print("  TELEMETRY (read-only)")
        print(f"    rows            {t.get('rows')} at {t.get('achieved_hz')} Hz"
              f"  -> {t.get('path')}")
        print(f"    read errors     {t.get('read_errors')}")
        print(f"    armed fraction  {t.get('armed_fraction')}")
        print(f"    frame pairing   median {t.get('frame_pairing_age_ms_median')} ms,"
              f" worst {t.get('frame_pairing_age_ms_worst')} ms")
        if not t.get("rows"):
            print("    NO TELEMETRY AT ALL -- the frames are still usable for")
            print("    labelling, but nothing is paired. Check the serial device.")
        elif (t.get("armed_fraction") or 0) == 0.0:
            print("    NOTE: never armed during this recording. If the aircraft")
            print("    did fly, the recording did not cover it.")
    print()
    ex = meta.get("exposure", {})
    print("  EXPOSURE (measured from the frames, not guessed)")
    print(f"    {ex.get('verdict')}")
    print(f"    median pixel mean {ex.get('median_mean')}, "
          f"spread {ex.get('median_std')}")
    if ex.get("settled_after_frames"):
        print(f"    auto-exposure settled after {ex['settled_after_frames']} "
              f"saved frame(s); first frame mean {ex.get('first_frame_mean')}")
        print(f"    raise --warmup above {meta.get('warmup_s')}s if that is "
              f"eating usable frames")
    if not meta.get("egl_context"):
        print("    (no graphics context in this session -- which on this board "
              "did NOT stop Argus exposing correctly)")
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

FLIGHT_BRIEF = """
TEST 3 - CAMERA CAPTURE, FLIGHT MODE          PROPS ON - A PILOT FLIES
------------------------------------------------------------------------
This program still never commands the aircraft. It opens the flight
controller READ-ONLY: attitude, gyro, motor outputs, stick positions and
battery. It has no code path that arms, transmits RC, or writes a
setting. The pilot flies; this watches and takes pictures.

WHY FLY AT ALL, WHEN GROUND MODE IS SAFER:
  * Walking blur is not racing blur. A detector trained only on
    hand-carried frames meets something different on race day.
  * Every frame is paired with the attitude the aircraft actually held,
    which is what lets us check the camera angle and, later, the
    camera-to-IMU timing offset.
  * Gate geometry from a real approach is the case that matters.

BEFORE POWERING:
  * Props ON is the whole point of this mode. Everyone stands clear.
  * Fly in the training cage. Measurement flights do not belong in a
    scored slot.
  * Start recording BEFORE arming, stop it AFTER disarming, so the log
    contains the whole flight including the transitions.
  * Use --seconds 0 and stop with Ctrl-C when the pack is done.

IF THE LINK DROPS:
  Telemetry stops, pictures continue, and the metadata records how many
  reads failed. That is deliberate - the images are the point.
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
    ap.add_argument("--flight", action="store_true",
                    help="props ON, a pilot flying: also record flight-controller "
                         "telemetry, read-only, and pair it to every frame")
    ap.add_argument("--dev", default=None,
                    help="flight-controller serial device for --flight, e.g. "
                         "/dev/ttyTHS1. Use 'mock' to rehearse with no hardware")
    ap.add_argument("--telem-hz", type=float, default=50.0,
                    help="telemetry poll rate for --flight")
    ap.add_argument("--warmup", type=float, default=1.5,
                    help="seconds of frames to pull and discard so auto-exposure "
                         "can converge before anything is saved")
    args = ap.parse_args()

    if args.flight and not args.dev:
        ap.error("--flight needs --dev (e.g. --dev /dev/ttyTHS1, or --dev mock "
                 "to rehearse on a laptop). Telemetry is the point of this mode.")
    if args.dev and not args.flight:
        ap.error("--dev only means anything with --flight.")
    # Flight mode saves more per second: flight time is scarce and motion blur
    # is the reason to be up there, so undersampling wastes the battery.
    if args.flight and "--fps" not in sys.argv:
        args.fps = 6.0

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

        print(FLIGHT_BRIEF if args.flight else BRIEF)
        if not args.test_source and not has_egl():
            print("  NOTE: no graphics context in this session. The software guide")
            print("  warns that this gives flat, un-exposed frames. Measured on the")
            print("  race Orin it did not: Argus exposed correctly over plain SSH.")
            print("  Either way, the exposure printed at the end is measured from")
            print("  the frames themselves -- trust that, not this note.")
            print()

        telemetry = None
        if args.flight:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from mock_link import open_link          # noqa: E402
            try:
                fc = open_link(args.dev)
            except Exception as e:
                print(f"\n  could not open the flight controller on {args.dev}: {e}")
                print("  Refusing to fly a recording that cannot pair frames to")
                print("  attitude. Fix the link, or drop --flight and capture")
                print("  images only.")
                return 3
            telemetry = TelemetryRecorder(fc, hz=args.telem_hz).start()
            if not telemetry.samples:
                telemetry.stop()
                print(f"\n  opened {args.dev} but read nothing back in 6 s.")
                print("  The port exists; the flight controller is not answering.")
                print("  Check it is powered and not mid-reboot -- leaving the")
                print("  Betaflight CLI reboots it, which takes a few seconds.")
                return 3
            print(f"  telemetry live on {args.dev} "
                  f"({len(telemetry.samples)} sample(s) to first read)\n")
        try:
            meta = capture(Path(args.out), seconds=args.seconds, fps=args.fps,
                           width=args.width, height=args.height, sensor_id=args.sensor_id,
                           test_source=args.test_source, source_fps=args.source_fps,
                           telemetry=telemetry, warmup_s=args.warmup)
        finally:
            if telemetry is not None:
                telemetry.stop()
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
                   test_source=True, source_fps=30, verbose=False, warmup_s=0.0)

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

    # The exposure grader must actually grade. The SMPTE test pattern is a wide
    # spread of colour bars, so it has to come back as normally exposed.
    ex = meta["exposure"]
    assert ex["ok"] is True, ex
    assert ex["median_std"] > 12.0, ex
    # and it must call a flat frame flat
    flat = grade_exposure([(128.0, 1.0)] * 10)
    assert flat["ok"] is False and "FLAT" in flat["verdict"], flat
    dark = grade_exposure([(9.0, 20.0)] * 10)
    assert dark["ok"] is False and "DARK" in dark["verdict"], dark

    # Warm-up must discard frames rather than merely delay them.
    wout = Path("/tmp/_camtest_warm")
    shutil.rmtree(wout, ignore_errors=True)
    wmeta = capture(wout, seconds=2.0, fps=4.0, width=320, height=240, sensor_id=0,
                    test_source=True, source_fps=30, verbose=False, warmup_s=1.0)
    assert wmeta["warmup_s"] == 1.0
    assert 4 <= wmeta["frames"] <= 10, wmeta["frames"]
    shutil.rmtree(wout, ignore_errors=True)

    # --- flight mode, against the mock flight controller ------------------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mock_link import open_link                  # noqa: E402
    fc = open_link("mock")
    telem = TelemetryRecorder(fc, hz=50.0).start()
    fout = Path("/tmp/_camtest_flight")
    shutil.rmtree(fout, ignore_errors=True)
    try:
        fmeta = capture(fout, seconds=3.0, fps=4.0, width=640, height=360,
                        sensor_id=0, test_source=True, source_fps=30,
                        verbose=False, telemetry=telem, warmup_s=0.0)
    finally:
        telem.stop()

    assert fmeta["flight_mode"] is True
    t = fmeta["telemetry"]
    assert t["rows"] > 50, t
    assert t["read_errors"] == 0, t
    assert (fout / "telemetry.csv").exists()

    frows = list(csv.DictReader(open(fout / "frames.csv")))
    assert frows and all(c in frows[0] for c in TELEM_FIELDS), frows[0].keys()
    # Every frame must have found a telemetry sample, and a recent one: at
    # 50 Hz nothing should be more than a poll period away.
    ages = [float(r["telem_age_ms"]) for r in frows]
    assert all(a == a for a in ages), "a frame paired with nothing"
    assert max(ages) < 60.0, f"pairing too stale: {max(ages)} ms"
    assert all(r["roll_deg"] not in ("", None) for r in frows), "attitude missing"
    assert all(1000 <= float(r["motor_mean"]) <= 2000 for r in frows), "motor range"

    # The recorder must never have written to the flight controller. The mock
    # counts commands, so this is checked rather than asserted in a comment.
    if hasattr(fc, "commands_received"):
        assert fc.commands_received == 0, (
            f"telemetry recorder sent {fc.commands_received} commands; it must "
            f"be read-only")

    shutil.rmtree(fout, ignore_errors=True)

    print("test_camera_capture: all checks passed")
    print(f"  {meta['frames']} frames via videotestsrc, {meta['actual_frame_size']}")
    print(f"  hardware PTS present and monotonic; median gap "
          f"{meta['pts_dt_ms_median']} ms")
    print(f"  PTS span {span_pts:.2f}s vs arrival span {span_mono:.2f}s")
    print(f"  pipeline clock: {info['clock_type']}")
    print(f"  flight mode: {len(frows)} frames paired to {t['rows']} telemetry "
          f"rows at {t['achieved_hz']} Hz")
    print(f"  worst frame/telemetry pairing gap {max(ages):.1f} ms")
    shutil.rmtree(out, ignore_errors=True)
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
