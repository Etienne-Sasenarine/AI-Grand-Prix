#!/usr/bin/env python3
"""
Live camera view in a browser — run ON THE JETSON, watch from any machine.

Serves MJPEG over HTTP, so the viewer needs nothing but a browser. This exists
because the obvious alternatives don't work headless: nvarguscamerasrc needs an
EGL display it cannot get over SSH, and an RTP/UDP pipeline needs GStreamer set
up on the receiving end.

Auto-detects the sensor type and picks the right pipeline:
  * UYVY  (TEVS/TEVM — on-module ISP)  -> v4l2src, no Argus involved
  * Bayer (imx219/imx477)              -> nvarguscamerasrc, needs a display;
                                          falls back to raw v4l2 + software
                                          debayer if EGL is unavailable

Deliberately uses cv2.VideoCapture and carries NO timestamps: the capture-time
variant lives in live-view-pts.py, which reads GstBuffer.pts via python3-gi.
Keep this one simple — it is the fallback that has the fewest moving parts.

Usage:
    ./live-view.py [--port 8080] [--device /dev/video0]
                   [--width 1920] [--height 1080] [--fps 30]

Then open  http://<jetson-ip>:8080/  from your laptop.
"""
import argparse
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import cv2
except ImportError:
    sys.exit("OpenCV missing — run ./provision.sh first")


def list_formats(dev):
    try:
        return subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats-ext"],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def detect(dev):
    """Return ('uyvy'|'bayer'|'unknown', fourcc)."""
    out = list_formats(dev)
    if "UYVY" in out:
        return "uyvy", "UYVY"
    m = re.search(r"'(RG10|RG12|BA10|GB10|BG10)'", out)
    if m:
        return "bayer", m.group(1)
    return "unknown", ""


def pipelines(kind, dev, w, h, fps):
    """Candidate pipelines, best first."""
    if kind == "uyvy":
        # TEVS/TEVM already output processed YUV: no ISP, no EGL, no Argus.
        return [(f"v4l2src device={dev} io-mode=mmap ! "
                 f"video/x-raw,format=UYVY,width={w},height={h},framerate={fps}/1 ! "
                 f"videoconvert ! video/x-raw,format=BGR ! "
                 f"appsink drop=true max-buffers=1 sync=false", "v4l2src UYVY")]
    if kind == "bayer":
        return [
            # Argus gives a properly debayered, auto-exposed image, but needs EGL.
            (f"nvarguscamerasrc ! "
             f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
             f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
             f"video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false",
             "nvarguscamerasrc (ISP)"),
            # Fallback: raw Bayer, debayered on the CPU. Colour will be flat --
            # no AE/AWB -- but it proves the sensor and CSI path work headless.
            (f"v4l2src device={dev} io-mode=mmap ! "
             f"video/x-bayer,format=rggb,width={w},height={h},framerate={fps}/1 ! "
             f"bayer2rgb ! videoconvert ! video/x-raw,format=BGR ! "
             f"appsink drop=true max-buffers=1 sync=false",
             "v4l2src + software debayer (no ISP)"),
        ]
    return []


class Camera:
    def __init__(self, dev, w, h, fps):
        self.kind, self.fourcc = detect(dev)
        if self.kind == "unknown":
            raise SystemExit(f"no recognised format on {dev}\n"
                             f"{list_formats(dev) or '  (v4l2-ctl produced no output)'}")
        self.cap = None
        for pipe, label in pipelines(self.kind, dev, w, h, fps):
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self.cap, self.label = cap, label
                    break
            cap.release()
        if self.cap is None:
            raise SystemExit(
                f"could not open {dev} ({self.kind}/{self.fourcc}).\n"
                "If cv2 was installed from pip it has no GStreamer support — "
                "use the packaged OpenCV instead (./provision.sh).")
        self.frame = None
        self.fps = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        n, t0 = 0, time.time()
        while True:
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            n += 1
            dt = time.time() - t0
            if dt >= 1.0:
                self.fps, n, t0 = n / dt, 0, time.time()
            text = f"{self.label}   {img.shape[1]}x{img.shape[0]}   {self.fps:.1f} fps"
            # Outlined: stays readable over both bright sky and dark ground.
            for colour, thick in ((0, 0, 0), 4), ((0, 255, 0), 1):
                cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, colour, thick, cv2.LINE_AA)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.frame = jpg.tobytes()

    def get(self):
        with self.lock:
            return self.frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


PAGE = b"""<!doctype html><meta charset=utf-8><title>Jetson camera</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui;text-align:center}
img{max-width:100%;height:auto;display:block;margin:0 auto}
p{padding:.6rem}</style>
<p>Live camera &mdash; MJPEG. Ctrl-C on the board to stop.</p>
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
    a = ap.parse_args()

    cam = Camera(a.device, a.width, a.height, a.fps)
    print(f"sensor  : {cam.kind} ({cam.fourcc})")
    print(f"pipeline: {cam.label}")
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
