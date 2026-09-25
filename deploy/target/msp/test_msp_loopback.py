#!/usr/bin/env python3
"""Loopback test for msp.py / msp_rc.py against a fake Betaflight FC on a PTY.

No hardware required. Run from anywhere: python3 src/test_msp_loopback.py
Exits non-zero on any failure.
"""
import os, pty, struct, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import serial  # noqa: F401
except ImportError as exc:
    sys.stderr.write(
        f"this test needs pyserial ({exc}).\n"
        "  sudo apt install python3-serial\n"
        "  or: pip3 install pyserial\n"
    )
    sys.exit(1)
from msp import (_Decoder, encode_v1, MSPLink, decode_arming_flags, decode_sensor_flags,
                 MSP_API_VERSION, MSP_FC_VARIANT, MSP_FC_VERSION, MSP_STATUS, MSP_ATTITUDE,
                 MSP_ANALOG, MSP_RC, MSP_BOXIDS, MSP_SET_RAW_RC, MSP_BOARD_INFO,
                 MSP_SET_ARMING_DISABLED, MSP_RAW_IMU, MSP_BUILD_INFO)
from msp_rc import RCTransmitter

master, slave = pty.openpty()
name = os.ttyname(slave)
S = {"rc": [1500]*8, "n": 0, "armed": False, "msp_lock": True}

def reply_v1(cmd, payload=b""):
    """FC->host MSPv1 frame ('$M>')."""
    body = bytes((len(payload), cmd)) + payload
    crc = 0
    for b in body: crc ^= b
    return b"$M>" + body + bytes((crc,))

def parse_requests(buf):
    """Parse host->FC MSPv1/v2 frames ('$M<' / '$X<'). Returns list of (cmd, payload)."""
    out = []
    while True:
        i = buf.find(b"$")
        if i < 0:
            del buf[:]; break
        if i: del buf[:i]
        if len(buf) < 3: break
        if buf[2] != 0x3C: del buf[:1]; continue
        if buf[1] == 0x4D:
            if len(buf) < 5: break
            size, cmd = buf[3], buf[4]; total = 6 + size
            if len(buf) < total: break
            payload = bytes(buf[5:5+size])
        elif buf[1] == 0x58:
            if len(buf) < 8: break
            _f, cmd, size = struct.unpack_from("<BHH", buf, 3); total = 9 + size
            if len(buf) < total: break
            payload = bytes(buf[8:8+size])
        else:
            del buf[:1]; continue
        del buf[:total]
        out.append((cmd, payload))
    return out

def fc():
    buf = bytearray()
    while True:
        try: data = os.read(master, 4096)
        except OSError: return
        buf.extend(data)
        for cmd, payload in parse_requests(buf):
            if   cmd == MSP_API_VERSION: r = bytes([0,1,46])
            elif cmd == MSP_FC_VARIANT:  r = b"BTFL"
            elif cmd == MSP_FC_VERSION:  r = bytes([4,5,1])
            elif cmd == MSP_BOARD_INFO:  r = b"S405"+bytes(10)
            elif cmd == MSP_BUILD_INFO:  r = b"Aug 26 2026" + b"23:35:00" + b"b463b1e"
            elif cmd == MSP_BOXIDS:      r = bytes([0,1,2,13])
            elif cmd == MSP_ATTITUDE:    r = struct.pack("<3h",-123,45,271)
            elif cmd == MSP_RAW_IMU:     r = struct.pack("<9h",1,2,4096,5,6,7,0,0,0)
            elif cmd == MSP_ANALOG:      r = struct.pack("<BHHhH",165,350,99,512,1652)
            elif cmd == MSP_RC:          r = struct.pack("<8H",*S["rc"])
            elif cmd == MSP_SET_ARMING_DISABLED:
                S["msp_lock"] = bool(payload[0]); r = b""
            elif cmd == MSP_STATUS:
                flags = (1<<16) if S["msp_lock"] else 0
                r = (struct.pack("<HHHIB",312,0,0b100011,1 if S["armed"] else 0,0)
                     + struct.pack("<H",21) + struct.pack("<H",125)
                     + bytes([0]) + bytes([26]) + struct.pack("<I",flags) + bytes([0]))
            elif cmd == MSP_SET_RAW_RC:
                n = len(payload)//2
                S["rc"] = list(struct.unpack(f"<{n}H", payload[:n*2]))[:8]
                S["n"] += 1; S["armed"] = S["rc"][4] > 1700; r = b""
            else: r = b""
            os.write(master, reply_v1(cmd, r))

threading.Thread(target=fc, daemon=True).start()

fails = []
def check(label, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + label + ("  " + str(extra) if extra else ""))
    if not cond: fails.append(label)

with MSPLink(name, 115200) as link:
    check("fc_variant", link.fc_variant() == "BTFL", link.fc_variant())
    check("fc_version", link.fc_version() == "4.5.1", link.fc_version())
    check("api_version", link.api_version() == (0,1,46))
    check("board_id", link.board_id() == "S405")
    d, c, rev = link.build_info()
    check("build_info", (d, c, rev) == ("Aug 26 2026", "23:35:00", "b463b1e"), (d, c, rev))
    check("arm_bit", link.arm_bit() == 0)
    r,p,y = link.attitude()
    check("attitude", (r,p,y) == (-12.3, 4.5, 271.0), (r,p,y))
    acc,gyro,mag = link.raw_imu()
    check("raw_imu", acc == (1,2,4096) and gyro == (5,6,7), (acc,gyro))
    a = link.analog()
    check("analog volts (0.01V field)", abs(a["voltage_v"]-16.52) < 1e-6, a["voltage_v"])
    check("analog amps", abs(a["current_a"]-5.12) < 1e-6, a["current_a"])
    st = link.status()
    check("sensors decode", decode_sensor_flags(st["sensor_flags"]) == ["GYRO","ACC","BARO"],
          decode_sensor_flags(st["sensor_flags"]))
    check("status load", st["system_load_pct"] == 21, st["system_load_pct"])
    check("arming reasons parsed", st["arming_disable_reasons"] == ["MSP"],
          st["arming_disable_reasons"])
    check("not armed initially", st["armed"] is False)

    # ---- RC transmitter ----
    tx = RCTransmitter(link, rate_hz=100, max_throttle=0.20, max_angle_cmd=0.5).start()
    tx.wait_until_streaming()
    tx.set_control(roll=1.0, pitch=-1.0, yaw=0.5, throttle=1.0)
    time.sleep(0.15)
    rc = link.rc_channels()
    check("roll clamped by max_angle_cmd", rc[0] == 1750, rc[0])
    check("pitch clamped", rc[1] == 1250, rc[1])
    check("throttle clamped by max_throttle", rc[2] == 1200, rc[2])
    check("yaw scaled", rc[3] == 1625, rc[3])
    check("AUX1 low while disarmed", rc[4] == 1000, rc[4])
    check("FC sees stream", S["n"] > 5, S["n"])

    # arm refused with throttle up
    try:
        tx.arm(); armed_ok = False
    except RuntimeError:
        armed_ok = True
    check("arm refused when throttle high", armed_ok)

    tx.set_control(throttle=0.0)
    time.sleep(0.05)
    link.set_arming_disabled(False)
    check("msp arming lock released", link.status()["arming_disable_reasons"] == [],
          link.status()["arming_disable_reasons"])
    tx.arm()
    time.sleep(0.15)
    check("FC armed via AUX1", link.status()["armed"] is True)
    check("AUX1 high", link.rc_channels()[4] == 1800)

    # ---- watchdog ----
    n_before = S["n"]
    time.sleep(0.45)   # exceed command_timeout_s (0.25) without calling set_control
    rc = link.rc_channels()
    check("watchdog neutralised sticks", rc[0] == 1500 and rc[2] == 1000, rc[:5])
    check("watchdog disarmed", rc[4] == 1000 and link.status()["armed"] is False, rc[4])
    check("stream continued during watchdog", S["n"] > n_before + 20, S["n"]-n_before)
    check("timeouts counted", tx.timeouts > 0, tx.timeouts)

    tx.close()
    check("rate ~100Hz", 80 <= tx.frames_sent / 1.0 or tx.frames_sent > 80, tx.frames_sent)
    check("no decoder crc errors", link._decoder.crc_errors == 0, link._decoder.crc_errors)

# ---- codec unit checks ----
from msp import encode_v2, crc8_dvb_s2
check("v1 encode golden", encode_v1(101) == b"$M<\x00\x65\x65", encode_v1(101))

def reply_v2(cmd, payload=b"", flag=0):
    body = struct.pack("<BHH", flag, cmd, len(payload)) + payload
    return b"$X>" + body + bytes((crc8_dvb_s2(body),))

d = _Decoder()
frames = d.feed(b"\x00\xffgarbage" + reply_v1(108, b"\x01\x02") + b"\x00" + reply_v2(200, b"\xaa"))
check("decoder resyncs past junk", len(frames) == 2, frames)
check("decoder v1 payload", frames[0] == (108, b"\x01\x02", True), frames[0])
check("decoder v2 payload", frames[1] == (200, b"\xaa", True), frames[1])
d2 = _Decoder()
bad = bytearray(reply_v1(108, b"\x01\x02")); bad[-1] ^= 0xFF
check("decoder rejects bad crc", d2.feed(bytes(bad)) == [] and d2.crc_errors == 1)
d3 = _Decoder()
check("decoder surfaces '!' as not-ok",
      d3.feed(b"$M!\x00\x65\x65") == [(101, b"", False)], d3.feed(b""))
check("arming flag decode unknown bit", decode_arming_flags(1 << 30) == ["BIT_30"])

print()
print(f"{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
