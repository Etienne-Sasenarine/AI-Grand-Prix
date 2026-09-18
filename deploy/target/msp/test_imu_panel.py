#!/usr/bin/env python3
"""Drive live-view-imu.py's IMULink against a simulated Betaflight FC on a pty.

Needs no flight controller and no camera -- run it to prove the IMU half of the
signoff works before you trust a red panel on the bench.

Checks the two things a signoff depends on: that live attitude reaches the
snapshot, and that a dead link is REPORTED dead rather than freezing on the
last-known attitude.

cv2 is stubbed unconditionally, even where a real one exists: what is under
test is the overlay's arithmetic and its format strings against missing fields,
not OpenCV's rendering. Actual drawing is proven by running the live view.

Usage:  python3 msp/test_imu_panel.py
"""
import importlib.util, math, os, pty, struct, sys, threading, time

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(T, "msp"))
from msp import MSP_API_VERSION, MSP_FC_VARIANT, MSP_FC_VERSION, MSP_ATTITUDE, \
                MSP_RAW_IMU, MSP_ANALOG, MSP_STATUS, MSP_BOXIDS

# The host has no OpenCV. Stub it so the module imports: the overlay logic we
# care about here is the arithmetic and the format strings, not the rendering.
import types
_calls = []
cv2s = types.ModuleType("cv2")
cv2s.FONT_HERSHEY_SIMPLEX = 0; cv2s.LINE_AA = 16
cv2s.IMWRITE_JPEG_QUALITY = 1; cv2s.CAP_GSTREAMER = 1800
for _n in ("putText", "line", "circle", "imencode", "VideoCapture"):
    cv2s.__dict__[_n] = (lambda n: (lambda *a, **k: _calls.append(n)))(_n)
sys.modules.setdefault("cv2", cv2s)

spec = importlib.util.spec_from_file_location("lvimu", os.path.join(T, "live-view-imu.py"))
lv = importlib.util.module_from_spec(spec); spec.loader.exec_module(lv)

master, slave = pty.openpty()
alive = {"v": True}

def reply_v1(cmd, p=b""):
    body = bytes((len(p), cmd)) + p
    crc = 0
    for b in body: crc ^= b
    return b"$M>" + body + bytes((crc,))

def parse(buf):
    out = []
    while True:
        i = buf.find(b"$")
        if i < 0: del buf[:]; break
        if i: del buf[:i]
        if len(buf) < 5 or buf[2] != 0x3C or buf[1] != 0x4D: 
            if len(buf) >= 5: del buf[:1]; continue
            break
        size, cmd = buf[3], buf[4]; total = 6 + size
        if len(buf) < total: break
        out.append((cmd, bytes(buf[5:5+size]))); del buf[:total]
    return out

def fc():
    buf, t0 = bytearray(), time.time()
    while True:
        try: data = os.read(master, 4096)
        except OSError: return
        buf.extend(data)
        for cmd, _ in parse(buf):
            if not alive["v"]:
                continue                       # FC goes silent, port stays open
            t = time.time() - t0
            roll = int(round(300 * math.sin(t)))          # +/-30.0 deg
            if   cmd == MSP_API_VERSION: r = bytes([0,1,46])
            elif cmd == MSP_FC_VARIANT:  r = b"BTFL"
            elif cmd == MSP_FC_VERSION:  r = bytes([4,5,1])
            elif cmd == MSP_BOXIDS:      r = bytes([0,1,2,13])
            elif cmd == MSP_ATTITUDE:    r = struct.pack("<3h", roll, 45, 271)
            elif cmd == MSP_RAW_IMU:     r = struct.pack("<9h",0,0,512,1,-2,3,0,0,0)
            elif cmd == MSP_ANALOG:      r = struct.pack("<BHHhH",165,350,99,512,1652)
            elif cmd == MSP_STATUS:
                r = (struct.pack("<HHHIB",312,0,0b100011,0,0) + struct.pack("<H",21)
                     + struct.pack("<H",125) + bytes([0]) + bytes([26])
                     + struct.pack("<I",0) + bytes([0]))
            else: r = b""
            os.write(master, reply_v1(cmd, r))

threading.Thread(target=fc, daemon=True).start()

fails = []
def chk(cond, label, extra=""):
    print(("PASS " if cond else "FAIL ") + label + ("  " + str(extra) if extra else ""))
    if not cond: fails.append(label)

imu = lv.IMULink(os.ttyname(slave), 115200, hz=30)
time.sleep(2.0)
s = imu.snapshot()
chk(s["link"] == "up", "link comes up", s["link"] + " " + str(s["error"] or ""))
chk(s["fc"] == "BTFL", "firmware identified", s["fc"])
chk(s["api"] == "0.1.46", "API version decoded", s["api"])
chk(s["roll"] is not None and -30.1 <= s["roll"] <= 30.1, "roll in range", s["roll"])
chk(abs(s["pitch"] - 4.5) < 0.01, "pitch scaled /10", s["pitch"])
chk(abs(s["yaw"] - 271) < 0.01, "yaw in degrees", s["yaw"])
chk(s["accel_g"] == [0.0, 0.0, 1.0], "accel 512 counts -> 1 G", s["accel_g"])
chk(s["gyro_dps"] == [1.0, -2.0, 3.0], "gyro passthrough dps", s["gyro_dps"])
chk(abs(s["voltage_v"] - 16.52) < 0.01, "battery from 0.01V field", s["voltage_v"])
chk(s["hz"] > 10, "poll rate sane", round(s["hz"], 1))
chk(s["errors"] == 0, "no MSP errors", s["errors"])

r1 = imu.snapshot()["roll"]; time.sleep(0.6); r2 = imu.snapshot()["roll"]
chk(r1 != r2, "attitude is live, not cached", f"{r1} -> {r2}")

# Overlay must not throw on any link state, and must draw something.
frame = types.SimpleNamespace(shape=(720, 1280, 3))
_calls.clear(); lv.draw_imu(frame, imu.snapshot())
chk(_calls.count("line") >= 3, "horizon + reference marks drawn", _calls.count("line"))
chk(_calls.count("putText") >= 4, "attitude text drawn", _calls.count("putText"))
for st in ("down", "stale", "connecting", "error"):
    q = imu.snapshot(); q["link"] = st; q["age"] = 3.0
    _calls.clear()
    try:
        lv.draw_imu(frame, q); ok = _calls.count("putText") >= 2
    except Exception as e:
        ok = False; print("   raised:", type(e).__name__, e)
    chk(ok, f"overlay survives link={st}")
# The nastiest case: link says up but the fields are still None (first frame,
# before any reply has landed). A bare format string would blow up here.
q = imu.snapshot(); q.update(link="up", roll=None, pitch=None, yaw=None,
                             gyro_dps=None, accel_g=None)
try:
    _calls.clear(); lv.draw_imu(frame, q); ok = True
except Exception as e:
    ok = False; print("   raised:", type(e).__name__, e)
chk(ok, "overlay survives link=up with no data yet")

# The critical one: a silent FC must be reported, not frozen.
alive["v"] = False
time.sleep(2.5)
s = imu.snapshot()
chk(s["link"] in ("stale", "down"), "silent FC reported, not frozen", s["link"])
chk(s["age"] is None or s["age"] > 1.0, "age reflects staleness", s["age"])

alive["v"] = True
time.sleep(2.5)
chk(imu.snapshot()["link"] == "up", "recovers when the FC returns")

imu.close()
print("\n" + ("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
