#!/usr/bin/env python3
"""
Live camera view in a browser, with CAPTURE-ACCURATE timestamps.

Run ON THE JETSON; watch from any machine. Serves MJPEG over HTTP, so the
viewer needs nothing but a browser — nvarguscamerasrc needs an EGL display it
cannot get over SSH, and an RTP/UDP pipeline needs GStreamer on the receiving
end.

Timestamps come from the GstBuffer PTS, not from when Python saw the frame.
That matters: cv2.VideoCapture throws the PTS away, so an OpenCV-based viewer
can only stamp arrival time, which includes tens of milliseconds of pipeline
latency and jitter. For aligning frames with a flight controller or IMU that is
the whole error budget, so this reads the buffer directly via python3-gi.

    absolute capture time = pipeline.base_time + buffer.pts

both in the pipeline clock's domain. GstSystemClock uses CLOCK_MONOTONIC by
default, and the clock type is queried and reported at startup rather than
assumed.

Sources, auto-detected:
  * UYVY  (TEVS/TEVM — on-module ISP)  -> v4l2src; driver timestamps are the
                                          sensor's own, so PTS is exact
  * Bayer (imx219/imx477)              -> nvarguscamerasrc (Argus sets PTS from
                                          the sensor); software-debayer fallback

Usage:
    ./live-view.py [--port 8080] [--device /dev/video0]
                   [--width 1920] [--height 1080] [--fps 30] [--csv FILE]

Then open  http://<jetson-ip>:8080/
"""
import argparse
import atexit
import datetime
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except (ImportError, ValueError):
    sys.exit("python3-gi / gir1.2-gstreamer-1.0 missing:\n"
             "    sudo apt install -y python3-gi gir1.2-gstreamer-1.0")
try:
    import numpy as np
    import cv2
except ImportError:
    sys.exit("numpy/OpenCV missing — run ./provision.sh first")

Gst.init(None)
NS = 1_000_000_000


def list_formats(dev):
    try:
        return subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats-ext"],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def detect(dev):
    out = list_formats(dev)
    if "UYVY" in out:
        return "uyvy", "UYVY"
    m = re.search(r"'(RG10|RG12|BA10|GB10|BG10)'", out)
    return ("bayer", m.group(1)) if m else ("unknown", "")


def pipelines(kind, dev, w, h, fps):
    """Candidate pipelines, best first. Each ends in appsink name=sink."""
    tail = ("videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false")
    if kind == "uyvy":
        # do-timestamp is left at its default (false) ON PURPOSE: that keeps the
        # driver's own capture timestamp. Setting it true would restamp buffers
        # with pipeline arrival time — exactly the error this tool exists to avoid.
        return [(f"v4l2src device={dev} io-mode=mmap ! "
                 f"video/x-raw,format=UYVY,width={w},height={h},framerate={fps}/1 ! "
                 f"{tail}", "v4l2src UYVY")]
    if kind == "bayer":
        return [
            (f"nvarguscamerasrc ! "
             f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
             f"nvvidconv ! video/x-raw,format=BGRx ! {tail}",
             "nvarguscamerasrc (ISP)"),
            (f"v4l2src device={dev} io-mode=mmap ! "
             f"video/x-bayer,format=rggb,width={w},height={h},framerate={fps}/1 ! "
             f"bayer2rgb ! {tail}", "v4l2src + software debayer (no ISP)"),
        ]
    return []


class Camera:
    def __init__(self, dev, w, h, fps, csv=None):
        self.kind, self.fourcc = detect(dev)
        if self.kind == "unknown":
            raise SystemExit(f"no recognised format on {dev}\n"
                             f"{list_formats(dev) or '  (v4l2-ctl gave no output)'}")
        self.pipeline = self.sink = None
        for desc, label in pipelines(self.kind, dev, w, h, fps):
            p = Gst.parse_launch(desc)
            # set_state() returns a StateChangeReturn, NOT a tuple -- only
            # get_state() returns (ret, state, pending). Live sources answer
            # NO_PREROLL or ASYNC here rather than SUCCESS, so the only thing
            # worth rejecting is an outright FAILURE; try_pull_sample below is
            # what actually proves frames flow.
            if p.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                p.set_state(Gst.State.NULL)
                continue
            s = p.get_by_name("sink")
            if s.try_pull_sample(5 * Gst.SECOND) is not None:
                self.pipeline, self.sink, self.label = p, s, label
                break
            p.set_state(Gst.State.NULL)
        if self.pipeline is None:
            raise SystemExit(f"could not start a pipeline for {dev} "
                             f"({self.kind}/{self.fourcc})")

        # Report the clock domain rather than assuming it: GstSystemClock is
        # CLOCK_MONOTONIC by default, but a pipeline can be given another clock.
        clk = self.pipeline.get_clock()
        try:
            self.clock_type = clk.get_property("clock-type").value_nick
        except Exception:
            self.clock_type = "unknown"

        self.frame = None
        self.fps = 0.0
        self.frames = 0
        self.dropped = 0
        self.last_pts = None
        self.lock = threading.Lock()
        self.csv = open(csv, "w", buffering=1) if csv else None
        if self.csv:
            self.csv.write("frame,pts_ns,mono_s,utc\n")
        # Tear the pipeline down cleanly on exit. Without the NULL transition
        # Argus complains about outstanding client objects and can leave
        # nvargus-daemon in a state where the next run fails to open the camera.
        atexit.register(self.close)
        threading.Thread(target=self._loop, daemon=True).start()

    def close(self):
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        if self.csv:
            self.csv.close()
            self.csv = None

    @staticmethod
    def _label(img, y, text):
        for colour, thick in ((0, 0, 0), 4), ((0, 255, 0), 1):
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colour, thick, cv2.LINE_AA)

    def _loop(self):
        base = self.pipeline.get_base_time()
        # Monotonic->realtime offset, sampled once. Realtime can be stepped by
        # NTP; monotonic cannot, so the UTC column is "best effort" and the
        # monotonic column is the one to trust for correlation.
        offset = time.time() - time.clock_gettime(time.CLOCK_MONOTONIC)
        n, t0 = 0, time.monotonic()
        while True:
            sample = self.sink.try_pull_sample(Gst.SECOND)
            if sample is None:
                continue
            buf = sample.get_buffer()
            pts = buf.pts                       # running time, ns
            caps = sample.get_caps().get_structure(0)
            w, h = caps.get_value("width"), caps.get_value("height")

            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                img = np.ndarray((h, w, 3), dtype=np.uint8,
                                 buffer=info.data).copy()
            finally:
                buf.unmap(info)

            self.frames += 1
            n += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                self.fps, n, t0 = n / (now - t0), 0, now

            if pts == Gst.CLOCK_TIME_NONE:
                stamp, mono_s = "PTS unavailable", float("nan")
            else:
                mono_s = (base + pts) / NS      # absolute, pipeline clock domain
                stamp = datetime.datetime.fromtimestamp(
                    mono_s + offset, datetime.timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
                # A gap much larger than one frame interval means dropped frames
                # upstream — worth seeing on the image itself.
                if self.last_pts is not None:
                    gap = (pts - self.last_pts) / NS
                    if self.fps > 1 and gap > 1.8 / self.fps:
                        self.dropped += 1
                self.last_pts = pts

            self._label(img, 30, f"{stamp}   #{self.frames}"
                                 f"{'  drops ' + str(self.dropped) if self.dropped else ''}")
            self._label(img, 58, f"pts {pts/NS:12.6f}s   mono {mono_s:14.6f}s")
            self._label(img, 86, f"{self.label}   {w}x{h}   {self.fps:.1f} fps"
                                 f"   [{self.clock_type}]")

            if self.csv and pts != Gst.CLOCK_TIME_NONE:
                self.csv.write(f"{self.frames},{base+pts},{mono_s:.6f},{stamp}\n")

            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.frame = jpg.tobytes()

    def get(self):
        with self.lock:
            return self.frame


PAGE = b"""<!doctype html><meta charset=utf-8><title>Jetson camera</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui;text-align:center}
img{max-width:100%;height:auto;display:block;margin:0 auto}
p{padding:.6rem}</style>
<p>Live camera &mdash; MJPEG, GstBuffer PTS timestamps. Ctrl-C on the board to stop.</p>
<img src="/stream">"""


def make_handler(cam):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    f = cam.get()
                    if f is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(f)).encode() +
                                     b"\r\n\r\n" + f + b"\r\n")
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<jetson-ip>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--csv", help="also log frame,pts_ns,mono_s,utc to this file")
    a = ap.parse_args()

    cam = Camera(a.device, a.width, a.height, a.fps, a.csv)
    print(f"sensor   : {cam.kind} ({cam.fourcc})")
    print(f"pipeline : {cam.label}")
    print(f"clock    : {cam.clock_type}  (pts + base_time is in this domain)")
    if a.csv:
        print(f"csv      : {a.csv}")
    print(f"\n  open  http://{lan_ip()}:{a.port}/\n")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(cam))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
