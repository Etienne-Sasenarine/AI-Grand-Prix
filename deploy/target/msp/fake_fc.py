"""Simulated Betaflight flight controller on a pty, with selectable fault modes.

Used by the test harnesses so the MSP and IMU code can be exercised with no
hardware attached. Modes: healthy, frozen, saturated, badscale, noaccel,
i2cerr, nouid -- each reproducing a real way an IMU fails while still talking.
"""
import math, os, pty, random, struct, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msp import (MSP_API_VERSION, MSP_FC_VARIANT, MSP_FC_VERSION, MSP_BOARD_INFO,
                 MSP_BUILD_INFO, MSP_BOXIDS, MSP_ATTITUDE, MSP_RAW_IMU, MSP_ANALOG,
                 MSP_STATUS, MSP_NAME, MSP_UID)

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
        if len(buf) < 5: break
        if buf[1] != 0x4D or buf[2] != 0x3C: del buf[:1]; continue
        size, cmd = buf[3], buf[4]; total = 6 + size
        if len(buf) < total: break
        out.append((cmd, bytes(buf[5:5+total]))); del buf[:total]
    return out

class FakeFC:
    def __init__(self, mode="healthy", uid="deadbeef0123456789abcdef"):
        self.mode, self.uid = mode, uid
        self.master, self.slave = pty.openpty()
        self.dev = os.ttyname(self.slave)
        self.rot = 0.0                      # driven externally for the motion test
        threading.Thread(target=self._loop, daemon=True).start()

    def _imu(self):
        m = self.mode
        if m == "frozen":                   # plausible constants, never changing
            return (0, 0, 512), (0, 0, 0)
        if m == "saturated":
            return (0, 0, 512), (1800, -1750, 1900)
        if m == "noaccel":
            return (0, 0, 0), (int(random.gauss(0, 1)) for _ in range(3))
        if m == "badscale":                 # accel alive but reading ~0.5 g
            n = lambda: random.gauss(0, 2)
            return (int(n()), int(n()), int(256 + n())), tuple(int(random.gauss(0, 1)) for _ in range(3))
        # healthy: level board, real MEMS dither, plus any commanded rotation
        r = math.radians(self.rot)
        n = lambda s: random.gauss(0, s)
        acc = (int(n(3)), int(512 * math.sin(r) + n(3)), int(512 * math.cos(r) + n(3)))
        gyro = tuple(int(n(0.8)) for _ in range(3))
        return acc, gyro

    def _att(self, acc):
        ax, ay, az = acc
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
        return int(roll * 10), int(pitch * 10), 271

    def _loop(self):
        buf = bytearray()
        while True:
            try: data = os.read(self.master, 4096)
            except OSError: return
            buf.extend(data)
            for cmd, _ in parse(buf):
                acc, gyro = self._imu(); acc = tuple(acc); gyro = tuple(gyro)
                if   cmd == MSP_API_VERSION: r = bytes([0, 1, 46])
                elif cmd == MSP_FC_VARIANT:  r = b"BTFL"
                elif cmd == MSP_FC_VERSION:  r = bytes([4, 5, 1])
                elif cmd == MSP_BOARD_INFO:  r = b"S405" + bytes(10)
                elif cmd == MSP_BUILD_INFO:  r = b"Aug 26 2026" + b"23:35:00" + b"b463b1e"
                elif cmd == MSP_NAME:        r = b"DCL-A603"
                elif cmd == MSP_BOXIDS:      r = bytes([0, 1, 2, 13])
                elif cmd == MSP_UID:
                    r = b"" if self.mode == "nouid" else struct.pack(
                        "<3I", *[int(self.uid[i:i+8], 16) for i in (0, 8, 16)])
                elif cmd == MSP_RAW_IMU:
                    r = struct.pack("<9h", *acc, *gyro, 0, 0, 0)
                elif cmd == MSP_ATTITUDE:
                    r = struct.pack("<3h", *self._att(acc))
                elif cmd == MSP_ANALOG:
                    r = struct.pack("<BHHhH", 165, 350, 99, 512, 1652)
                elif cmd == MSP_STATUS:
                    sensors = 0b100011 if self.mode != "noaccel" else 0b100010
                    i2c_err = 47 if self.mode == "i2cerr" else 0
                    r = (struct.pack("<HHHIB", 312, i2c_err, sensors, 0, 0)
                         + struct.pack("<H", 21) + struct.pack("<H", 125)
                         + bytes([0]) + bytes([26]) + struct.pack("<I", 0) + bytes([0]))
                else: r = b""
                os.write(self.master, reply_v1(cmd, r))
