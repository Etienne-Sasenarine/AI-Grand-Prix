#!/usr/bin/env python3
"""
Live camera view in an OpenCV window — run ON THE JETSON, with a display.

The counterpart to live-view.py: same pipeline selection, but the frames go to
a local cv2 window instead of a browser. Needs a display (HDMI/DP, or an X
session); over plain SSH use live-view.py instead, and note that
nvarguscamerasrc will not start without EGL at all.

Auto-detects the sensor and picks the right pipeline:
  * Bayer (imx219/imx477)              -> nvarguscamerasrc: the Tegra ISP does
                                          the debayer, AE, AWB, denoise and
                                          edge enhancement.
  * UYVY  (TEVS/TEVM)                  -> v4l2src.  Not a downgrade: these
                                          modules carry their own ISP and emit
                                          processed YUV, and Argus only accepts
                                          Bayer, so there is nothing for the
                                          Tegra ISP to do.

The ISP is not optional here.  If Argus cannot start, the script says so and
stops rather than quietly showing a software-debayered image that looks like a
broken camera -- pass --allow-cpu-debayer if you want that fallback anyway.

Usage:
    ./show-camera.py [--device /dev/video0] [--width 1920] [--height 1080]
                     [--fps 30] [--sensor-id 0] [--sensor-mode N] [--flip 0]
                     [--wb auto] [--exposure-time 10000000 40000000]
                     [--gain 1 16] [--isp-gain 1 8] [--saturation 1.0]
                     [--ee-mode 1] [--tnr-mode 1] [--ev 0] [--ae-lock]
                     [--awb-lock] [--no-osd]

ISP settings are pipeline properties, so they are fixed when the stream starts;
OpenCV's VideoCapture gives no handle to change them live.  Restart with
different flags.  Defaults mean "let the ISP decide", which is usually right --
reach for --exposure-time and --gain only when AE is chasing a scene it cannot
win, e.g. a bright window behind the subject.

Keys:  q or ESC  quit        s  save a PNG to ./capture-NNN.png        f  fullscreen
"""
import argparse
import re
import subprocess
import sys
import time

try:
    import cv2
except ImportError:
    sys.exit("OpenCV missing — run ./provision.sh first")

WINDOW = "Jetson camera"


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


# nvarguscamerasrc property names, straight from `gst-inspect-1.0
# nvarguscamerasrc`. Ranges are strings of two numbers: exposure in
# nanoseconds, gains as multipliers.
def argus_props(a):
    props = [f"sensor-id={a.sensor_id}"]
    if a.sensor_mode is not None:
        props.append(f"sensor-mode={a.sensor_mode}")
    if a.wb != "auto":
        props.append(f"wbmode={WB_MODES[a.wb]}")
    if a.exposure_time:
        lo, hi = a.exposure_time
        props.append(f'exposuretimerange="{lo} {hi}"')
    if a.gain:
        lo, hi = a.gain
        props.append(f'gainrange="{lo} {hi}"')
    if a.isp_gain:
        lo, hi = a.isp_gain
        props.append(f'ispdigitalgainrange="{lo} {hi}"')
    if a.saturation is not None:
        props.append(f"saturation={a.saturation}")
    if a.ee_mode is not None:
        props.append(f"ee-mode={a.ee_mode}")
    if a.tnr_mode is not None:
        props.append(f"tnr-mode={a.tnr_mode}")
    if a.ev:
        props.append(f"exposurecompensation={a.ev}")
    if a.ae_lock:
        props.append("aelock=true")
    if a.awb_lock:
        props.append("awblock=true")
    return " ".join(props)


def pipelines(kind, dev, w, h, fps, a):
    """Candidate pipelines, best first."""
    sink = "appsink drop=true max-buffers=1 sync=false"
    if kind == "uyvy":
        # TEVS/TEVM already output processed YUV from their on-module ISP.
        return [(f"v4l2src device={dev} io-mode=mmap ! "
                 f"video/x-raw,format=UYVY,width={w},height={h},framerate={fps}/1 ! "
                 f"videoconvert ! video/x-raw,format=BGR ! {sink}",
                 "v4l2src UYVY (module ISP)")]
    if kind == "bayer":
        out = [
            # The Tegra ISP: debayer, AE, AWB, lens shading, denoise, EE.
            # nvvidconv does the NVMM -> system-memory copy on the VIC, so the
            # CPU only sees the finished BGRx.
            (f"nvarguscamerasrc {argus_props(a)} ! "
             f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
             f"nvvidconv flip-method={a.flip} ! video/x-raw,format=BGRx ! "
             f"videoconvert ! video/x-raw,format=BGR ! {sink}",
             "nvarguscamerasrc (Tegra ISP)"),
        ]
        if a.allow_cpu_debayer:
            # Explicitly asked for: raw Bayer debayered on the CPU. Flat colour,
            # no AE/AWB. Proves the sensor and CSI path, nothing more.
            out.append(
                (f"v4l2src device={dev} io-mode=mmap ! "
                 f"video/x-bayer,format=rggb,width={w},height={h},framerate={fps}/1 ! "
                 f"bayer2rgb ! videoconvert ! video/x-raw,format=BGR ! {sink}",
                 "v4l2src + CPU debayer (NO ISP)"))
        return out
    return []


WB_MODES = {"off": 0, "auto": 1, "incandescent": 2, "fluorescent": 3,
            "warm-fluorescent": 4, "daylight": 5, "cloudy": 6, "twilight": 7,
            "shade": 8, "manual": 9}


def open_camera(dev, w, h, fps, a):
    kind, fourcc = detect(dev)
    if kind == "unknown":
        sys.exit(f"no recognised format on {dev}\n"
                 f"{list_formats(dev) or '  (v4l2-ctl produced no output)'}")
    tried = []
    for pipe, label in pipelines(kind, dev, w, h, fps, a):
        tried.append(pipe)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap, kind, fourcc, label
        cap.release()

    hint = ["could not open {} ({}/{}) with the ISP.".format(dev, kind, fourcc), ""]
    if kind == "bayer":
        hint += [
            "Argus is what does the debayer/AE/AWB, so check it first:",
            "  * is the daemon up?   systemctl status nvargus-daemon",
            "  * another process holding the sensor? (this script twice, a",
            "    stale gst-launch, raw-view.py with bypass_mode=0) — only one",
            "    client gets the camera.  Stop it, then:",
            "      sudo systemctl restart nvargus-daemon",
            "  * does the mode exist?  the requested {}x{}@{} must be one the".format(w, h, fps),
            "    driver advertises: v4l2-ctl -d {} --list-formats-ext".format(dev),
            "  * daemon log:  journalctl -u nvargus-daemon -n 50",
            "",
            "To see whether the sensor produces anything at all, bypassing the",
            "ISP entirely, use ./raw-view.py — but that is a diagnostic, not this.",
            "",
        ]
    hint += [
        "If cv2 came from pip it has no GStreamer support at all — use the",
        "packaged OpenCV (./provision.sh).  Check with:",
        "  python3 -c 'import cv2; print(cv2.getBuildInformation())' | grep -i gstreamer",
        "",
        "pipeline tried:",
    ] + ["  " + p for p in tried]
    sys.exit("\n".join(hint))


def overlay(img, text):
    for colour, thick in (((0, 0, 0), 4), ((0, 255, 0), 1)):
        cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colour, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--sensor-mode", type=int, default=None,
                    help="Argus sensor-mode index; default lets Argus choose")
    ap.add_argument("--flip", type=int, default=0,
                    help="nvvidconv flip-method 0-7")
    ap.add_argument("--wb", default="auto", choices=sorted(WB_MODES),
                    help="ISP white balance mode (default auto)")
    ap.add_argument("--exposure-time", nargs=2, type=int, metavar=("MIN_NS", "MAX_NS"),
                    help="clamp ISP auto-exposure; equal values pin it")
    ap.add_argument("--gain", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="analogue gain range")
    ap.add_argument("--isp-gain", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="ISP digital gain range")
    ap.add_argument("--saturation", type=float, default=None, help="0.0-2.0")
    ap.add_argument("--ee-mode", type=int, default=None,
                    help="edge enhancement: 0 off, 1 fast, 2 high quality")
    ap.add_argument("--tnr-mode", type=int, default=None,
                    help="temporal noise reduction: 0 off, 1 fast, 2 high quality")
    ap.add_argument("--ev", type=float, default=0.0,
                    help="exposure compensation, -2.0 to 2.0")
    ap.add_argument("--ae-lock", action="store_true", help="freeze auto-exposure")
    ap.add_argument("--awb-lock", action="store_true", help="freeze auto white balance")
    ap.add_argument("--allow-cpu-debayer", action="store_true",
                    help="fall back to a software debayer if Argus fails "
                         "(no ISP: flat colour, no AE/AWB — diagnostic only)")
    ap.add_argument("--no-osd", action="store_true", help="hide the text overlay")
    a = ap.parse_args()

    cap, kind, fourcc, label = open_camera(a.device, a.width, a.height, a.fps, a)
    print(f"sensor  : {kind} ({fourcc})")
    print(f"pipeline: {label}")
    if kind == "bayer":
        print(f"isp     : nvarguscamerasrc {argus_props(a)}")
        if "NO ISP" in label:
            print("WARNING : Argus failed — this image is NOT ISP-processed.",
                  file=sys.stderr)
    print("keys    : q/ESC quit, s save PNG, f fullscreen")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(a.width, 1280), min(a.height, 720))

    fps, n, t0, shot, full = 0.0, 0, time.time(), 0, False
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                print("frame read failed — camera gone?", file=sys.stderr)
                break

            n += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps, n, t0 = n / dt, 0, time.time()

            if not a.no_osd:
                overlay(img, f"{label}   {img.shape[1]}x{img.shape[0]}   {fps:.1f} fps")
            cv2.imshow(WINDOW, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"capture-{shot:03d}.png"
                cv2.imwrite(name, img)
                print(f"wrote {name}")
                shot += 1
            if key == ord("f"):
                full = not full
                cv2.setWindowProperty(
                    WINDOW, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            # window closed with the X button
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
