#!/usr/bin/env python3
"""
Live view of the *raw sensor data* — no ISP, no Argus, no white balance unless
you ask for it.  Run ON THE JETSON, with a display.

show-camera.py hands the frame to Argus, which debayers, auto-exposes and
auto-white-balances it: you see a picture, not the sensor.  This shows what the
sensor actually puts on the CSI bus — the 10-bit Bayer mosaic, code values
intact — which is what you want when the question is "is the sensor producing
sane data", not "does this look nice".

Capture goes through `v4l2-ctl --stream-to=-` rather than GStreamer on purpose:
gstreamer's video/x-bayer is 8-bit only, so a 10-bit RG10 stream cannot be
carried by v4l2src at all.  v4l2-ctl writes frames to stdout with no headers
and no padding beyond the driver's stride, which numpy can reshape directly.

Usage:
    ./raw-view.py [--device /dev/video0] [--width 1920] [--height 1080]
                  [--pixelformat RG10] [--bits 10] [--bayer RGGB]
                  [--scale 1.0] [--gamma 2.2] [--gain 1.0]
                  [--sensor-mode N] [--ctrl gain=200 --ctrl exposure=20000]

There is no auto-exposure on this path -- that lives in Argus, which is exactly
what we are bypassing.  A dark or blank frame is the expected first result; set
exposure and gain yourself with --ctrl (names from `v4l2-ctl --list-ctrls`) and
use the OSD's min/max/mean to tell whether you are under- or over-exposed.

Views (number keys):
    1  mosaic     raw code values as greyscale, full resolution, no WB.
                  Zoom in and you should see the Bayer checkerboard.
    2  demosaic   2x2 quad average -> half resolution.  Exact, not
                  interpolated: each output pixel is one real Bayer quad.
    3  planes     the four planes (R, G1, G2, B) side by side.
    4  cv         OpenCV interpolating demosaic, full resolution.

Other keys:
    a  grey-world white balance on/off      h  histogram of raw code values
    g/G  gamma down/up                      -/+  display gain down/up
    c  cycle the OpenCV Bayer code (view 4) -- use it if colours look swapped
    s  save frame-NNN.raw (exact bytes) + frame-NNN.png (current view)
    p  print the raw 2x2 quad under the mouse to the terminal
    q / ESC  quit

The OSD reports code values, not pixels: min/max/mean and the count of clipped
samples.  A mean near zero means no light or no exposure; a max stuck at the
full-scale value means clipping; a max well below full scale on a bright scene
usually means the exposure or gain controls were never applied.
"""
import argparse
import subprocess
import sys
import threading
import time

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("OpenCV/numpy missing — run ./provision.sh first")

WINDOW = "Jetson raw"

# Position of (R, G1, G2, B) within each 2x2 quad as (row, col).  Same table as
# tools/raw2png.py -- keep them in step.
BAYER_PATTERNS = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "BGGR": ((1, 1), (0, 1), (1, 0), (0, 0)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (0, 0), (1, 1), (0, 1)),
}

# OpenCV's Bayer codes are named for a different corner of the quad than V4L2's
# fourccs, and which convention a given build follows is not worth arguing
# about from a chair.  View 4 cycles these with 'c'; the right one is the one
# where a red object comes out red.
CV_CODES = [("BayerBG2BGR", cv2.COLOR_BayerBG2BGR),
            ("BayerGB2BGR", cv2.COLOR_BayerGB2BGR),
            ("BayerRG2BGR", cv2.COLOR_BayerRG2BGR),
            ("BayerGR2BGR", cv2.COLOR_BayerGR2BGR)]


class RawStream:
    """v4l2-ctl streaming raw frames to stdout; keeps only the newest frame."""

    def __init__(self, dev, w, h, fmt, stride_bytes, ctrls):
        self.w, self.h = w, h
        # 10- and 12-bit Bayer are packed into 16-bit little-endian words, so a
        # line is stride_bytes and a frame is stride_bytes * height.
        self.stride = stride_bytes or w * 2
        self.frame_bytes = self.stride * h
        cmd = ["v4l2-ctl", "-d", dev]
        for c in ctrls:
            cmd += ["--set-ctrl", c]
        cmd += [f"--set-fmt-video=width={w},height={h},pixelformat={fmt}",
                "--stream-mmap", "--stream-to=-"]
        self.cmd = cmd
        try:
            self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            sys.exit("v4l2-ctl not found — run stage-rootfs-extras.sh before flashing")
        self.frame = None
        self.raw_bytes = None
        self.count = 0
        self.lock = threading.Lock()
        self.dead = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            buf = self.p.stdout.read(self.frame_bytes)
            if len(buf) < self.frame_bytes:
                self.dead = True
                return
            img = np.frombuffer(buf, dtype="<u2").reshape(self.h, self.stride // 2)
            with self.lock:
                # Trim the driver's line padding; copy so the buffer can be freed.
                self.frame = img[:, :self.w].copy()
                self.raw_bytes = buf
                self.count += 1

    def get(self):
        with self.lock:
            return self.frame, self.raw_bytes

    def stop(self):
        self.p.terminate()


def make_lut(mask, gain, gamma):
    x = np.minimum(1.0, np.arange(mask + 1) / mask * gain)
    return (np.power(x, 1.0 / gamma) * 255.0 + 0.5).astype(np.uint8)


def planes(raw, pattern):
    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = BAYER_PATTERNS[pattern]
    return (raw[ry::2, rx::2], raw[g1y::2, g1x::2],
            raw[g2y::2, g2x::2], raw[by::2, bx::2])


def grey_world(r, g1, g2, b):
    mr, mg, mb = float(r.mean()), (float(g1.mean()) + float(g2.mean())) / 2, float(b.mean())
    kr = mg / mr if mr > 1e-6 else 1.0
    kb = mg / mb if mb > 1e-6 else 1.0
    return kr, kb


def draw_histogram(img, raw, mask):
    """Histogram of raw code values, log scale, over the bottom of the frame."""
    h, w = img.shape[:2]
    # Every 4th pixel in each direction: 16x cheaper, same shape.
    hist = np.bincount((raw[::4, ::4]).ravel(), minlength=mask + 1)[:mask + 1]
    hist = np.log1p(hist.astype(np.float64))
    bins = cv2.resize(hist.reshape(1, -1), (min(w, 512), 1),
                      interpolation=cv2.INTER_AREA).ravel()
    if bins.max() <= 0:
        return
    hh = min(140, h // 3)
    bins = (bins / bins.max() * (hh - 8)).astype(np.int32)
    x0, y0 = 12, h - 12
    pts = np.array([[x0 + i, y0 - v] for i, v in enumerate(bins)], np.int32)
    cv2.rectangle(img, (x0 - 4, y0 - hh), (x0 + len(bins) + 4, y0 + 4), (0, 0, 0), -1)
    cv2.polylines(img, [pts], False, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"0..{mask}", (x0, y0 - hh + 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 200, 0), 1, cv2.LINE_AA)


def overlay(img, lines):
    y = 30
    for text in lines:
        for colour, thick in (((0, 0, 0), 4), ((0, 255, 0), 1)):
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colour, thick, cv2.LINE_AA)
        y += 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--pixelformat", default="RG10",
                    help="V4L2 fourcc as v4l2-ctl --list-formats reports it")
    ap.add_argument("--bits", type=int, default=10, help="valid bits (RG10 -> 10)")
    ap.add_argument("--bayer", default="RGGB", choices=sorted(BAYER_PATTERNS))
    ap.add_argument("--stride", type=int, default=0,
                    help="bytes per line if the driver pads (default width*2)")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="display scale; default fits the long edge to 1280")
    ap.add_argument("--gamma", type=float, default=2.2)
    ap.add_argument("--gain", type=float, default=1.0, help="display gain only")
    ap.add_argument("--view", type=int, default=2, choices=(1, 2, 3, 4))
    ap.add_argument("--sensor-mode", type=int, default=None,
                    help="tegra sensor_mode control (see --list-formats-ext order)")
    ap.add_argument("--ctrl", action="append", default=[], metavar="NAME=VALUE",
                    help="extra v4l2 control, repeatable; without Argus there is "
                         "no auto-exposure, so gain/exposure are yours to set")
    ap.add_argument("--no-bypass-ctrl", action="store_true",
                    help="do not send bypass_mode=0 (non-Tegra capture paths)")
    a = ap.parse_args()

    if a.width % 2 or a.height % 2:
        sys.exit("width and height must be even for a Bayer mosaic")

    mask = (1 << a.bits) - 1

    # The Tegra VI driver hands buffers straight to Argus unless bypass_mode is
    # cleared; leave it set and v4l2-ctl streams zero frames and says nothing
    # about why.  Controls are applied in the order given, so sensor_mode goes
    # first -- it changes which exposure/gain ranges are even legal.
    ctrls = []
    if a.sensor_mode is not None:
        ctrls.append(f"sensor_mode={a.sensor_mode}")
    if not a.no_bypass_ctrl:
        ctrls.append("bypass_mode=0")
    ctrls += a.ctrl

    stream = RawStream(a.device, a.width, a.height, a.pixelformat, a.stride, ctrls)
    scale = a.scale or min(1.0, 1280 / max(a.width, a.height))

    print(f"device  : {a.device}  {a.width}x{a.height} {a.pixelformat} "
          f"({a.bits}-bit, {a.bayer})")
    print(f"frame   : {stream.frame_bytes} bytes")
    print(f"capture : {' '.join(stream.cmd)}")
    print("keys    : 1 mosaic  2 demosaic  3 planes  4 cv | a awb  h hist  "
          "g/G gamma  -/+ gain  c code  s save  p probe  q quit")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(a.width * scale), int(a.height * scale))

    state = {"mx": 0, "my": 0, "probe": False}

    def on_mouse(event, x, y, flags, _):
        state["mx"], state["my"] = x, y
    cv2.setMouseCallback(WINDOW, on_mouse)

    view, awb, hist_on, gamma, gain, cv_code = a.view, True, False, a.gamma, a.gain, 0
    lut_cache = {}

    def lut(g):
        key = (round(g, 4), round(gamma, 4))
        if key not in lut_cache:
            if len(lut_cache) > 64:
                lut_cache.clear()
            lut_cache[key] = make_lut(mask, g, gamma)
        return lut_cache[key]

    fps, n, t0, shot, last = 0.0, 0, time.time(), 0, -1
    try:
        while True:
            frame, raw_bytes = stream.get()
            if frame is None:
                if stream.dead:
                    sys.exit(
                        f"v4l2-ctl produced no frames on {a.device}.\n"
                        f"  * check the format:  v4l2-ctl -d {a.device} --list-formats-ext\n"
                        f"  * check the controls: v4l2-ctl -d {a.device} --list-ctrls\n"
                        f"  * run the capture command above by hand to see its stderr\n"
                        f"  * nvargus-daemon holding the sensor will block this: "
                        f"sudo systemctl stop nvargus-daemon")
                time.sleep(0.02)
                continue

            if stream.count != last:
                last = stream.count
                n += 1
                dt = time.time() - t0
                if dt >= 1.0:
                    fps, n, t0 = n / dt, 0, time.time()

            # High bits above the valid range are undefined padding, not signal.
            raw = frame & mask
            r, g1, g2, b = planes(raw, a.bayer)
            kr, kb = grey_world(r, g1, g2, b) if awb else (1.0, 1.0)

            if view == 1:                       # raw mosaic, untouched
                img = cv2.cvtColor(lut(gain)[raw], cv2.COLOR_GRAY2BGR)
                label = "mosaic (raw code values, no WB)"
            elif view == 2:                     # exact 2x2 quad average
                g = (g1.astype(np.uint32) + g2) >> 1
                img = np.dstack((lut(gain * kb)[b],
                                 lut(gain)[g.astype(np.uint16)],
                                 lut(gain * kr)[r]))
                label = "demosaic 2x2 quad (half res)"
            elif view == 3:                     # the four planes, as captured
                tiles = []
                for name, p in (("R", r), ("G1", g1), ("G2", g2), ("B", b)):
                    t = cv2.cvtColor(lut(gain)[p], cv2.COLOR_GRAY2BGR)
                    cv2.putText(t, f"{name}  mean {p.mean():.0f}", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    tiles.append(t)
                img = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
                label = "planes R / G1 / G2 / B"
            else:                               # OpenCV interpolating demosaic
                name, code = CV_CODES[cv_code]
                bgr = cv2.demosaicing(raw << (16 - a.bits), code)
                bgr = (bgr >> (16 - a.bits)).astype(np.uint16)
                img = np.dstack((lut(gain * kb)[bgr[:, :, 0]],
                                 lut(gain)[bgr[:, :, 1]],
                                 lut(gain * kr)[bgr[:, :, 2]]))
                label = f"cv2.{name}"

            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            img = np.ascontiguousarray(img)

            # Map the cursor back to sensor coordinates so the value shown is a
            # real code value at a known pixel, not a scaled display sample.
            sx = int(min(a.width - 2, max(0, state["mx"] / scale)))
            sy = int(min(a.height - 2, max(0, state["my"] / scale)))
            if view == 2:
                sx, sy = min(a.width - 2, sx * 2), min(a.height - 2, sy * 2)
            quad = raw[sy & ~1:(sy & ~1) + 2, sx & ~1:(sx & ~1) + 2]

            clipped = int((raw >= mask).sum())
            if hist_on:
                draw_histogram(img, raw, mask)
            overlay(img, [
                f"{label}   {fps:.1f} fps   frame {stream.count}",
                f"raw {a.bits}-bit  min {int(raw.min())}  max {int(raw.max())}  "
                f"mean {raw.mean():.1f}  clipped {100.0 * clipped / raw.size:.2f}%",
                f"R {r.mean():.0f}  G {(g1.mean() + g2.mean()) / 2:.0f}  B {b.mean():.0f}"
                f"   awb {'on' if awb else 'off'} (kr {kr:.2f} kb {kb:.2f})"
                f"   gamma {gamma:.1f}  gain {gain:.2f}",
                f"@({sx},{sy}) quad {quad[0, 0]} {quad[0, 1]} / {quad[1, 0]} {quad[1, 1]}",
            ])
            cv2.imshow(WINDOW, img)

            if state["probe"]:
                state["probe"] = False
                print(f"({sx},{sy}) {a.bayer} quad: "
                      f"[{quad[0, 0]:5d} {quad[0, 1]:5d}] [{quad[1, 0]:5d} {quad[1, 1]:5d}]")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                view = key - ord("0")
            elif key == ord("a"):
                awb = not awb
            elif key == ord("h"):
                hist_on = not hist_on
            elif key == ord("g"):
                gamma = max(1.0, gamma - 0.2)
            elif key == ord("G"):
                gamma = min(4.0, gamma + 0.2)
            elif key in (ord("-"), ord("_")):
                gain = max(0.1, gain / 1.25)
            elif key in (ord("+"), ord("=")):
                gain = min(64.0, gain * 1.25)
            elif key == ord("c"):
                cv_code = (cv_code + 1) % len(CV_CODES)
                print(f"cv code: {CV_CODES[cv_code][0]}")
            elif key == ord("p"):
                state["probe"] = True
            elif key == ord("s"):
                base = f"frame-{shot:03d}"
                with open(base + ".raw", "wb") as f:
                    f.write(raw_bytes)
                cv2.imwrite(base + ".png", img)
                print(f"wrote {base}.raw ({len(raw_bytes)} bytes) and {base}.png  "
                      f"-> tools/raw2png.py {base}.raw out.png "
                      f"--width {a.width} --height {a.height} "
                      f"--bayer {a.bayer} --bits {a.bits}")
                shot += 1

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
