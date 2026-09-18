#!/usr/bin/env python3
"""
Live camera view WITH flight-controller IMU telemetry — run ON THE JETSON.

Serves one page carrying both halves of the signoff:

  * the camera stream (MJPEG, same pipeline logic as live-view.py), and
  * attitude / gyro / accel read over UART from a Betaflight FC via MSP.

Attitude is burned into the JPEG as a horizon indicator, so a screenshot is
evidence that camera and IMU were live *at the same instant*. The numeric panel
beside it is fed by a JSON endpoint, which keeps the digits crisp instead of
JPEG-blurred and lets it update faster than the video.

The two halves fail independently and say so: no FC still gives you video with
a red NO LINK panel, and a camera that will not open is a hard error. That is
deliberate — a signoff has to show you WHICH half is broken.

Betaflight speaks MSP, not MAVLink. For an ArduPilot/PX4 autopilot this is the
wrong tool; use pymavlink against ATTITUDE/RAW_IMU instead.

Usage:
    ./live-view-imu.py --msp /dev/ttyTHS1 [--baud 115200] [--imu-hz 30]
                       [--port 8080] [--device /dev/video0]
                       [--width 1920] [--height 1080] [--fps 30]
                       [--no-horizon]

Then open  http://<jetson-ip>:8080/  from your laptop.

If the UART is busy, free it first:   sudo ./msp/setup_jetson_uart.sh --apply
"""
import argparse
import json
import math
import os
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "msp"))


# ---------------------------------------------------------------- camera

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
    if m:
        return "bayer", m.group(1)
    return "unknown", ""


def pipelines(kind, dev, w, h, fps):
    if kind == "uyvy":
        return [(f"v4l2src device={dev} io-mode=mmap ! "
                 f"video/x-raw,format=UYVY,width={w},height={h},framerate={fps}/1 ! "
                 f"videoconvert ! video/x-raw,format=BGR ! "
                 f"appsink drop=true max-buffers=1 sync=false", "v4l2src UYVY")]
    if kind == "bayer":
        return [
            (f"nvarguscamerasrc ! "
             f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
             f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
             f"video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false",
             "nvarguscamerasrc (ISP)"),
            (f"v4l2src device={dev} io-mode=mmap ! "
             f"video/x-bayer,format=rggb,width={w},height={h},framerate={fps}/1 ! "
             f"bayer2rgb ! videoconvert ! video/x-raw,format=BGR ! "
             f"appsink drop=true max-buffers=1 sync=false",
             "v4l2src + software debayer (no ISP)"),
        ]
    return []


# ---------------------------------------------------------------- IMU over MSP

class IMULink:
    """Poll a Betaflight FC for attitude + raw IMU on a background thread.

    Owns the serial port exclusively for its lifetime. Never raises into the
    caller: every failure becomes a field in snapshot(), because losing the FC
    must not take the camera down with it.
    """

    STALE_AFTER = 1.0          # seconds without a good reply before we say so
    REOPEN_AFTER = 10.0        # ...and before we drop the port and start over

    def __init__(self, dev, baud=115200, hz=30):
        self.dev, self.baud, self.period = dev, baud, 1.0 / max(hz, 1)
        self.lock = threading.Lock()
        self.s = {
            "link": "connecting", "error": None, "dev": dev, "baud": baud,
            "fc": None, "api": None,
            "roll": None, "pitch": None, "yaw": None,
            "gyro": None, "accel": None, "mag": None,
            "gyro_dps": None, "accel_g": None,
            "voltage_v": None, "rssi": None, "armed": None,
            "hz": 0.0, "age": None, "replies": 0, "errors": 0, "crc_errors": 0,
        }
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _set(self, **kw):
        with self.lock:
            self.s.update(kw)

    def _loop(self):
        try:
            from msp import MSPLink, MSPError
        except Exception as e:
            self._set(link="error", error=f"cannot import msp: {e}")
            return

        while not self.stop.is_set():
            fc = None
            try:
                # Short per-request timeout: the default 0.5 s x 2 retries would
                # stall this loop for 1.5 s every time the FC misses a reply,
                # and a frozen panel next to live video reads as a Jetson bug.
                fc = MSPLink(self.dev, self.baud, timeout=0.15).open()
                variant = fc.fc_variant()
                api = ".".join(str(x) for x in fc.api_version())
                self._set(link="up", error=None, fc=variant, api=api)
                self._poll(fc)
            except Exception as e:
                self._set(link="down", error=f"{type(e).__name__}: {e}",
                          roll=None, pitch=None, yaw=None)
            finally:
                if fc is not None:
                    try:
                        fc.close()
                    except Exception:
                        pass
            if self.stop.wait(1.0):     # reconnect backoff
                break

    def _poll(self, fc):
        # MSPTimeout derives from TimeoutError, NOT MSPError -- catching only
        # MSPError sends every missed reply down the generic path, which closes
        # and reopens the port once a second. A timeout on an open port is
        # 'stale', not 'down'.
        from msp import MSPError, MSPTimeout
        n, t0, last_slow = 0, time.time(), 0.0
        while not self.stop.is_set():
            t = time.time()
            try:
                roll, pitch, yaw = fc.attitude()
                acc, gyro, mag = fc.raw_imu()
                upd = {
                    "roll": roll, "pitch": pitch, "yaw": yaw,
                    "accel": list(acc), "gyro": list(gyro), "mag": list(mag),
                    # Betaflight's MSP_RAW_IMU convention: accel scaled so
                    # 512 = 1 G, gyro already in deg/s. INAV and older builds
                    # differ, so the raw counts stay visible alongside.
                    "accel_g": [round(v / 512.0, 3) for v in acc],
                    "gyro_dps": [float(v) for v in gyro],
                    "link": "up", "error": None,
                }
                # Battery/arming are nice-to-have and cost a round trip each;
                # poll them at 1 Hz so they never slow the attitude stream.
                if t - last_slow > 1.0:
                    last_slow = t
                    try:
                        a = fc.analog()
                        upd["voltage_v"] = a["voltage_v"]
                        upd["rssi"] = a["rssi"]
                    except Exception:
                        pass
                    try:
                        upd["armed"] = fc.status().get("armed")
                    except Exception:
                        pass
                n += 1
                dt = t - t0
                if dt >= 1.0:
                    upd["hz"] = n / dt
                    n, t0 = 0, t
                with self.lock:
                    self.s.update(upd)
                    self.s["replies"] += 1
                    self.s["crc_errors"] = fc.crc_errors
                    self.s["last_ok"] = t
            except (MSPError, MSPTimeout) as e:
                with self.lock:
                    self.s["errors"] += 1
                    age = t - self.s.get("last_ok", 0)
                    if age > self.STALE_AFTER:
                        self.s.update(link="stale",
                                      error=f"{type(e).__name__}: {e}")
                if not fc.is_open:
                    return
                # Persistently silent: the FC may have been power-cycled and
                # need a fresh port, so fall back to the reconnect loop.
                if age > self.REOPEN_AFTER:
                    self._set(link="down", error=f"no reply for {age:.0f}s")
                    return
            except Exception as e:
                self._set(link="down", error=f"{type(e).__name__}: {e}")
                return
            slack = self.period - (time.time() - t)
            if slack > 0:
                time.sleep(slack)

    def snapshot(self):
        with self.lock:
            s = dict(self.s)
        last = s.pop("last_ok", None)
        s["age"] = round(time.time() - last, 2) if last else None
        # A link that stopped replying must not keep showing the last attitude
        # as if it were current -- that is the one failure mode a signoff
        # cannot afford to miss.
        if s["link"] == "up" and s["age"] is not None and s["age"] > self.STALE_AFTER:
            s["link"] = "stale"
        return s

    def close(self):
        self.stop.set()


# ---------------------------------------------------------------- overlay

GREEN, RED, AMBER, WHITE = (0, 255, 0), (0, 0, 255), (0, 190, 255), (255, 255, 255)


def outlined(img, text, org, scale, colour, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def draw_horizon(img, roll, pitch):
    """Attitude indicator. A bench reference, not a certified instrument."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    ppd = h / 90.0                       # 90 deg of pitch across the frame
    r = math.radians(roll)
    # Image y grows downward: a positive (right-wing-down) roll lifts the
    # right-hand end of the horizon.
    dx, dy = math.cos(r), -math.sin(r)
    nx, ny = math.sin(r), math.cos(r)    # unit normal, "down" in the aircraft frame
    ox, oy = cx + nx * pitch * ppd, cy + ny * pitch * ppd
    L = w * 0.30
    p1 = (int(ox - dx * L), int(oy - dy * L))
    p2 = (int(ox + dx * L), int(oy + dy * L))
    cv2.line(img, p1, p2, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.line(img, p1, p2, GREEN, 2, cv2.LINE_AA)
    # Fixed aircraft reference, so roll is readable against it.
    g = int(w * 0.05)
    for a, b in (((cx - g, cy), (cx - g // 3, cy)), ((cx + g // 3, cy), (cx + g, cy))):
        cv2.line(img, a, b, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(img, a, b, AMBER, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, AMBER, -1, cv2.LINE_AA)


def draw_imu(img, s, horizon=True):
    h, w = img.shape[:2]
    sc = w / 1920.0
    live = s["link"] == "up" and s["roll"] is not None
    if live and horizon:
        draw_horizon(img, s["roll"], s["pitch"])

    if live:
        colour = GREEN
        line = (f"FC {s['fc'] or '?'}   "
                f"R {s['roll']:+6.1f}  P {s['pitch']:+6.1f}  Y {s['yaw']:5.1f}   "
                f"{s['hz']:.0f} Hz")
    elif s["link"] == "stale":
        colour = AMBER
        line = f"FC LINK STALE  ({s['age']}s since last reply)"
    else:
        colour = RED
        line = f"NO FC LINK  {s['dev']}  {s['error'] or ''}"[:88]
    outlined(img, line, (12, int(72 * sc)), 0.8 * sc, colour, max(1, int(2 * sc)))

    if live and s["gyro_dps"]:
        gx, gy, gz = s["gyro_dps"]
        ax, ay, az = s["accel_g"]
        outlined(img, f"gyro {gx:+5.0f} {gy:+5.0f} {gz:+5.0f} dps   "
                      f"acc {ax:+5.2f} {ay:+5.2f} {az:+5.2f} g",
                 (12, int(104 * sc)), 0.6 * sc, WHITE, max(1, int(1 * sc)))


class Camera:
    def __init__(self, dev, w, h, fps, imu, horizon=True):
        self.imu, self.horizon = imu, horizon
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
            sc = img.shape[1] / 1920.0
            outlined(img, f"{self.label}   {img.shape[1]}x{img.shape[0]}   "
                          f"{self.fps:.1f} fps",
                     (12, int(34 * sc)), 0.8 * sc, GREEN, max(1, int(1 * sc)))
            if self.imu is not None:
                draw_imu(img, self.imu.snapshot(), self.horizon)
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


PAGE = """<!doctype html><meta charset=utf-8><title>Jetson camera + FC IMU</title>
<style>
body{margin:0;background:#111;color:#ddd;font:14px system-ui}
.wrap{display:flex;flex-wrap:wrap;gap:1rem;padding:1rem;align-items:flex-start}
.vid{flex:1 1 640px;min-width:320px}
img{width:100%;height:auto;display:block;border-radius:6px}
.pan{flex:0 0 300px;background:#1b1b1b;border-radius:6px;padding:1rem}
h2{margin:.2rem 0 .8rem;font-size:1rem;font-weight:600}
table{width:100%;border-collapse:collapse;font:13px ui-monospace,monospace}
td{padding:.22rem 0}td:last-child{text-align:right;color:#fff}
.k{color:#888}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:3px;font-weight:600}
.up{background:#14532d;color:#86efac}.stale{background:#713f12;color:#fde68a}
.down{background:#7f1d1d;color:#fca5a5}
.err{color:#fca5a5;font:12px ui-monospace,monospace;word-break:break-all;margin-top:.6rem}
.note{color:#777;font-size:12px;margin-top:.8rem;line-height:1.4}
</style>
<div class=wrap>
  <div class=vid><img src="/stream"></div>
  <div class=pan>
    <h2>Flight controller <span id=badge class="badge down">…</span></h2>
    <table id=t></table>
    <div class=err id=err></div>
    <div class=note>Accel/gyro use Betaflight's MSP_RAW_IMU scaling
      (512 = 1&nbsp;G, gyro in deg/s). Raw counts shown in brackets.</div>
  </div>
</div>
<script>
const f = (v,d=2) => v===null||v===undefined ? '--' : (+v).toFixed(d);
const v3 = (a,d=1) => a ? a.map(x=>f(x,d)).join('  ') : '--';
async function tick(){
  try{
    const s = await (await fetch('/imu.json',{cache:'no-store'})).json();
    const b = document.getElementById('badge');
    b.textContent = s.link.toUpperCase();
    b.className = 'badge ' + (s.link==='up'?'up':s.link==='stale'?'stale':'down');
    const rows = [
      ['firmware', s.fc ? s.fc+' (API '+s.api+')' : '--'],
      ['port', s.dev+' @ '+s.baud],
      ['roll', f(s.roll,1)+'&deg;'], ['pitch', f(s.pitch,1)+'&deg;'],
      ['yaw', f(s.yaw,1)+'&deg;'],
      ['gyro dps', v3(s.gyro_dps,0)], ['gyro raw', v3(s.gyro,0)],
      ['accel g', v3(s.accel_g,2)],   ['accel raw', v3(s.accel,0)],
      ['mag raw', v3(s.mag,0)],
      ['armed', s.armed===null?'--':(s.armed?'YES':'no')],
      ['battery', s.voltage_v===null?'--':f(s.voltage_v,2)+' V'],
      ['rssi', s.rssi===null?'--':s.rssi],
      ['rate', f(s.hz,1)+' Hz'], ['age', s.age===null?'--':s.age+' s'],
      ['replies', s.replies], ['errors', s.errors], ['crc errors', s.crc_errors],
    ];
    document.getElementById('t').innerHTML =
      rows.map(r=>'<tr><td class=k>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');
    document.getElementById('err').textContent = s.error || '';
  }catch(e){}
}
setInterval(tick, 200); tick();
</script>""".encode()


def make_handler(cam, imu):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(PAGE, "text/html; charset=utf-8")
                return
            if self.path.startswith("/imu.json"):
                s = imu.snapshot() if imu else {"link": "disabled", "error":
                                                "started without --msp"}
                self._send(json.dumps(s).encode(), "application/json")
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
    ap.add_argument("--msp", default="", help="FC serial port, e.g. /dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--imu-hz", type=int, default=30)
    ap.add_argument("--no-horizon", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    imu = None
    if a.msp:
        if not os.path.exists(a.msp):
            print(f"warning: {a.msp} does not exist — the panel will show NO LINK.")
            print("         free the UART with:  sudo ./msp/setup_jetson_uart.sh --apply")
        imu = IMULink(a.msp, a.baud, a.imu_hz)
    else:
        print("no --msp given: video only, no IMU panel.")

    cam = Camera(a.device, a.width, a.height, a.fps, imu, not a.no_horizon)
    print(f"sensor  : {cam.kind} ({cam.fourcc})")
    print(f"pipeline: {cam.label}")
    if imu:
        time.sleep(1.2)                       # let the first MSP exchange land
        s = imu.snapshot()
        print(f"fc      : {s['link']}" + (f" — {s['fc']} (API {s['api']})"
                                          if s["fc"] else f" — {s['error'] or ''}"))
    print(f"\n  open  http://{lan_ip()}:{a.port}/\n")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(cam, imu))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        cam.close()
        if imu:
            imu.close()


if __name__ == "__main__":
    main()
