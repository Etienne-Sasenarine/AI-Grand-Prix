#!/usr/bin/env python3
"""
Per-frame capture timestamps, in a clock domain you can actually align with
the flight controller.

Why not OpenCV: cv2.VideoCapture hands you decoded pixels and throws the
buffer metadata away.  CAP_PROP_POS_MSEC on a live appsink is, at best, a
position query answered by the pipeline rather than the capture time of the
frame in your hand -- and there is no way to ask "when was THIS frame
exposed".  So this drives GStreamer directly through python3-gi and reads
GstBuffer.pts, which is the timestamp the capture element attached.

Where the number is taken, following it back up the chain:

  TKE TSC                 free-running SoC counter; the actual clock source
    -> VI / NVCSI         hardware latches the frame boundary
    -> RCE (camera R5)    writes start-of-frame and end-of-frame timestamps
                          into the capture status record
                          (camrtc-capture-messages.h, camrtc-trace.h: "TSC")
    -> VI kernel driver   set_timestamp() puts it on the vb2 buffer
                          (media/mc_common.h)
    -> v4l2_buffer.timestamp
        -> v4l2src        PTS
        -> Argus/libargus getSensorTimestamp() -> nvarguscamerasrc PTS
    -> nvvidconv, videoconvert, appsink -- all pass PTS through untouched

So the timestamp is taken in hardware at the frame boundary, before any of the
software in that list runs: CPU scheduling, ISP work and pipeline queueing move
the `latency` column, never the timestamp itself.

The sensor does NOT timestamp.  imx477 sends two lines of embedded metadata
(embedded_metadata_height = 2) carrying its own register state, but no clock
shared with the SoC -- which is why the boundary is timed at the receiver.

That receiver is on the SoC, so the timestamp is in the SoC's clock domain,
NOT the camera's: the sensor's oscillator decides when the frame happens, the
Tegra TSC decides what number describes it.  Three oscillators are in play and
only the middle one is ever read:

  sensor INCK   exposure and readout, free-running, never sampled
  Tegra TSC     every timestamp here, and CLOCK_MONOTONIC alongside it
  autopilot     MAVLink time_boot_ms, its own timebase again

Sensor and SoC crystals differ by tens of ppm, so a nominal 30.000 fps stream
measures as ~29.9997 fps in SoC time and accumulates -- a slow ramp in the `dt`
column, distinct from the random scatter of scheduling noise.

On THIS airframe the IMU is on the flight controller, so camera and IMU are in
different domains and no amount of precision here closes that gap by itself:
these timestamps are Tegra TSC, the IMU's are the autopilot's.  Fusing them
needs an estimated offset AND rate ratio between the two clocks, maintained
over the flight -- see the note at the bottom of this file.

Two cameras cannot be synchronised by timestamps either, however precise:
independent sensor oscillators drift apart, and that needs a framesync line.

CLOCK DOMAINS -- the part that bites.  GStreamer's default system clock and
V4L2 buffer timestamps are both CLOCK_MONOTONIC on Linux, so they are directly
comparable, and neither one is wall-clock time.  This script verifies that at
runtime rather than assuming it (--verify prints the evidence).  For MAVLink,
monotonic is what you want: ArduPilot/PX4 `time_boot_ms` and the usual
`time_usec` fields are boot-relative, not UTC.  The realtime column is offered
for logs a human reads, and is only as good as the board's clock -- a Jetson
with no RTC battery and no NTP boots in 1970.

Buffer PTS is running time (zero at pipeline start), so absolute monotonic is
`pipeline.base_time + buffer.pts`.

Usage:
    ./frame-timestamps.py                        # ISP path, print to stdout
    ./frame-timestamps.py --limit 300 --csv frames.csv
    ./frame-timestamps.py --show                 # also display the frames
    ./frame-timestamps.py --verify               # prove the clock domain
    ./frame-timestamps.py --test-source          # videotestsrc, no camera

ALIGNING WITH THE FLIGHT CONTROLLER

The IMU is on the FC, so its samples carry the autopilot's timebase and these
frames carry the Tegra TSC.  Ways to relate them, best first:

  1. A shared physical edge.  Feed one pulse (GPS PPS, or an FC-driven GPIO)
     into both: the Jetson timestamps it via nvpps in TSC/monotonic
     (uapi/linux/nvpps_ioctl.h), the FC timestamps it in its own clock, and
     each pulse gives one exact (tsc, fc) pair -- offset and drift both fall
     out, at microsecond scale.
  2. MAVLink TIMESYNC, continuously.  A round-trip estimate, so its accuracy
     is bounded by link asymmetry -- sub-millisecond over a fast serial link,
     worse over a shared or congested one.  Filter it (track the minimum-RTT
     samples); a single startup reading drifts out within minutes.
  3. Let the estimator solve for it.  VINS-Mono's `td` and Kalibr both model
     camera-IMU time offset online.  This absorbs a constant error but assumes
     it IS constant -- it will fight a drifting clock, not track it.

Note what is NOT available on this hardware: a frame strobe.  The 15-pin RPi
FPC on the A603 breaks out no XVS/strobe line, so the camera cannot tell the FC
when a frame happened.  Toggling a Jetson GPIO per frame in software is not a
substitute -- that timestamps your userspace, not the exposure, and re-adds
every millisecond of scheduling jitter this file exists to avoid.

VISUAL-INERTIAL USE -- do not assume this marks the start of exposure.  What
the hardware records is a frame boundary in the CSI data stream: SOF is when
the frame's first data arrives, i.e. when readout begins, which on a rolling
shutter sensor is when the FIRST ROW's exposure has already ended.  The
midpoint you want to pair with an IMU sample is therefore behind the timestamp,
not ahead of it, and the rows below the first are progressively later again.

Whether the kernel forwards SOF or EOF, and whether Argus shifts it further,
is not something to take on trust -- verify it on your hardware before relying
on the sign.  In practice the usual VIO toolchains (Kalibr, VINS) estimate the
camera-IMU time offset anyway, which absorbs a constant convention error, so
the thing that actually matters is that the offset stays CONSTANT: pin the
exposure (show-camera.py --exposure-time with equal min and max, --ae-lock) or
AE will move it every frame and no calibration will hold.
"""
import argparse
import csv
import sys
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

NS = 1_000_000_000


def build_pipeline(a):
    sink = "appsink name=sink emit-signals=false max-buffers=2 drop=true sync=false"
    if a.test_source:
        return (f"videotestsrc is-live=true ! "
                f"video/x-raw,format=BGR,width={a.width},height={a.height},"
                f"framerate={a.fps}/1 ! {sink}")
    if a.v4l2:
        return (f"v4l2src device={a.device} io-mode=mmap ! "
                f"video/x-raw,width={a.width},height={a.height},"
                f"framerate={a.fps}/1 ! "
                f"videoconvert ! video/x-raw,format=BGR ! {sink}")
    return (f"nvarguscamerasrc sensor-id={a.sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={a.width},height={a.height},"
            f"framerate={a.fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! {sink}")


def clock_domain(pipeline):
    """Which POSIX clock is the pipeline clock on?  Measured, not assumed.

    GStreamer's system clock is monotonic by default, but the clock-type is a
    property anyone can change, and getting this wrong silently shifts every
    timestamp by the machine's uptime -- a plausible-looking number that is
    wrong by days.
    """
    clk = pipeline.get_clock()
    if clk is None:
        return "unknown", 0
    gst_now = clk.get_time()
    mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    real = time.clock_gettime_ns(time.CLOCK_REALTIME)
    d_mono, d_real = abs(gst_now - mono), abs(gst_now - real)
    if min(d_mono, d_real) > NS:          # neither within a second: custom clock
        return "custom", gst_now - mono
    return ("monotonic", d_mono) if d_mono < d_real else ("realtime", d_real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--v4l2", action="store_true",
                    help="v4l2src instead of the ISP (UYVY modules, or raw)")
    ap.add_argument("--test-source", action="store_true",
                    help="videotestsrc — checks this script without a camera")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames")
    ap.add_argument("--csv", default=None, help="append rows to this file")
    ap.add_argument("--show", action="store_true", help="also display the frames")
    ap.add_argument("--verify", action="store_true",
                    help="print clock-domain evidence and exit")
    ap.add_argument("--quiet", action="store_true", help="csv only, no stdout rows")
    a = ap.parse_args()

    Gst.init(None)
    desc = build_pipeline(a)
    print(f"pipeline: {desc}\n", file=sys.stderr)
    pipeline = Gst.parse_launch(desc)
    sink = pipeline.get_by_name("sink")

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        sys.exit("pipeline failed to start — run the same pipeline under "
                 "gst-launch-1.0 to see the error")
    # PLAYING is async for a live source; wait for it so base_time is valid.
    if pipeline.get_state(5 * Gst.SECOND)[0] != Gst.StateChangeReturn.SUCCESS:
        pipeline.set_state(Gst.State.NULL)
        sys.exit("pipeline did not reach PLAYING within 5s — camera busy? "
                 "(only one client gets Argus)")

    base = pipeline.get_base_time()
    domain, delta = clock_domain(pipeline)
    print(f"clock   : {domain} (pipeline clock vs POSIX clock differs by "
          f"{delta / 1e6:.3f} ms)", file=sys.stderr)
    print(f"base    : {base} ns", file=sys.stderr)
    if domain not in ("monotonic", "unknown"):
        print("WARNING : pipeline clock is NOT monotonic — the mono_ns column "
              "is mislabelled for this run.", file=sys.stderr)

    # Sampled once: the offset drifts with NTP, so a per-frame conversion would
    # be no more correct, only more expensive.
    real_off = (time.clock_gettime_ns(time.CLOCK_REALTIME)
                - time.clock_gettime_ns(time.CLOCK_MONOTONIC))

    if a.verify:
        for i in range(3):
            s = sink.emit("try-pull-sample", 2 * Gst.SECOND)
            if s is None:
                sys.exit("no frames — is the camera producing?")
            b = s.get_buffer()
            mono = base + b.pts
            now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            print(f"  frame {i}: pts={b.pts} base+pts={mono} "
                  f"now_monotonic={now} latency={(now - mono) / 1e6:.2f} ms "
                  f"{'OK' if 0 <= now - mono < NS else 'SUSPECT'}")
        print("\nOK means base+pts lands just before 'now' on the monotonic "
              "clock — same domain, capture in the recent past.")
        pipeline.set_state(Gst.State.NULL)
        return

    writer = fh = None
    if a.csv:
        fh = open(a.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["frame", "pts_ns", "mono_ns", "realtime_ns",
                         "dt_ms", "latency_ms"])

    if a.show:
        import cv2
        import numpy as np

    n, prev = 0, None
    try:
        while True:
            sample = sink.emit("try-pull-sample", 2 * Gst.SECOND)
            if sample is None:
                print("timed out waiting for a frame", file=sys.stderr)
                break
            buf = sample.get_buffer()
            pts = buf.pts
            if pts == Gst.CLOCK_TIME_NONE:
                # Some elements leave PTS unset; a timestamp taken here would
                # be a CPU arrival time wearing a capture timestamp's name.
                print("frame has no PTS — this source does not timestamp",
                      file=sys.stderr)
                break
            mono = base + pts
            now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            dt_ms = (pts - prev) / 1e6 if prev is not None else 0.0
            lat_ms = (now - mono) / 1e6
            prev = pts

            if writer:
                writer.writerow([n, pts, mono, mono + real_off,
                                 f"{dt_ms:.3f}", f"{lat_ms:.3f}"])
            if not a.quiet:
                stamp = time.strftime("%H:%M:%S", time.localtime((mono + real_off) / NS))
                print(f"{n:6d}  pts {pts / 1e6:12.3f} ms  mono {mono}  "
                      f"{stamp}  dt {dt_ms:7.3f} ms  latency {lat_ms:6.2f} ms")

            if a.show:
                caps = sample.get_caps().get_structure(0)
                w, h = caps.get_value("width"), caps.get_value("height")
                ok, mi = buf.map(Gst.MapFlags.READ)
                if ok:
                    img = np.frombuffer(mi.data, np.uint8).reshape(h, w, 3).copy()
                    buf.unmap(mi)
                    cv2.putText(img, f"{n}  pts {pts / 1e6:.1f} ms  "
                                     f"lat {lat_ms:.1f} ms", (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("frames", img)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

            n += 1
            if a.limit and n >= a.limit:
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        if fh:
            fh.close()
            print(f"\nwrote {n} rows to {a.csv}", file=sys.stderr)
        if a.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
