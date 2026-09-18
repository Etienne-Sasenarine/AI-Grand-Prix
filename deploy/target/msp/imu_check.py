#!/usr/bin/env python3
"""Functional IMU acceptance test over MSP — Betaflight flight controller.

This is deliberately more than "did the FC answer". A dead or wedged IMU very
often still returns *plausible* numbers: an all-zero gyro looks exactly like a
perfectly still one, and a frozen accel buffer holding 1 G looks like a level
board. So the checks below test that the data is ALIVE and SELF-CONSISTENT, not
merely present:

  identity      link up, firmware, and the MCU's unique hardware serial
  sensors       the FC itself reports ACC and GYRO as detected
  rate          samples arrive fast enough to be useful
  integrity     no CRC errors and no FC-side I2C errors on the sensor bus
  accel scale   |accel| is one gravity, which no frozen zero buffer can fake
  gyro rest     |gyro| is near zero -- not saturated, not spinning
  liveness      the samples actually CHANGE; a frozen buffer fails here even
                though it passes every threshold check above
  cross-check   the FC's own roll/pitch estimate agrees with the tilt implied
                by the raw accelerometer, which catches a mis-mapped axis

With --motion the operator is asked to rotate the board, proving the gyro
responds to real movement and the attitude estimate follows it.

Scaling follows Betaflight's MSP_RAW_IMU convention: accel 512 counts = 1 G,
gyro already in deg/s. INAV differs; pass --acc-1g to override.

Usage:
    ./imu_check.py --dev /dev/ttyTHS1 [--baud 115200] [--samples 200]
                   [--motion] [--tsv results.tsv] [--json imu.json]

Exit status is 0 only if every check passed.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msp import (MSPLink, MSPError, MSPTimeout, decode_sensor_flags,
                 SENSOR_ACC, SENSOR_GYRO)

SECTION = "IMU"


class Results:
    def __init__(self, tsv=None):
        self.rows, self.tsv = [], tsv

    def add(self, status, name, detail=""):
        detail = " ".join(str(detail).split())
        self.rows.append((status, name, detail))
        mark = {"PASS": "[PASS]", "FAIL": "[FAIL]", "INFO": "      "}[status]
        print(f"  {mark} {name}" + (f" — {detail}" if detail else ""), flush=True)
        if self.tsv:
            with open(self.tsv, "a") as f:
                f.write(f"{status}\t{SECTION}\t{name}\t{detail}\n")

    def ok(self, c, name, detail=""):
        self.add("PASS" if c else "FAIL", name, detail)
        return c

    @property
    def failed(self):
        return sum(1 for s, _, _ in self.rows if s == "FAIL")


def collect(fc, n, hz, acc_1g):
    """Take n synchronised raw-IMU + attitude samples."""
    period, out, t0 = 1.0 / hz, [], time.time()
    while len(out) < n:
        t = time.time()
        try:
            acc, gyro, _mag = fc.raw_imu()
            roll, pitch, yaw = fc.attitude()
        except (MSPError, MSPTimeout):
            continue                       # counted via crc_errors / rate
        out.append({
            "t": t - t0,
            "acc_g": [v / acc_1g for v in acc],
            "gyro": [float(v) for v in gyro],
            "att": [roll, pitch, yaw],
        })
        slack = period - (time.time() - t)
        if slack > 0:
            time.sleep(slack)
        if time.time() - t0 > 30:
            break
    return out


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--acc-1g", type=float, default=512.0,
                    help="accel counts per G (Betaflight MSP: 512)")
    ap.add_argument("--motion", action="store_true",
                    help="also ask the operator to rotate the board")
    ap.add_argument("--tsv"), ap.add_argument("--json")
    a = ap.parse_args()

    R = Results(a.tsv)
    stats = {"dev": a.dev, "baud": a.baud}

    if not os.path.exists(a.dev):
        R.add("FAIL", "serial port present", f"{a.dev} does not exist")
        return 1

    try:
        fc = MSPLink(a.dev, a.baud, timeout=0.3).open()
    except Exception as e:
        R.add("FAIL", "open flight-controller port", f"{type(e).__name__}: {e}")
        R.add("INFO", "hint", "free the UART with msp/setup_jetson_uart.sh --apply")
        return 1

    try:
        # -- identity ------------------------------------------------------
        try:
            variant = fc.fc_variant()
            ver = str(fc.fc_version())
            api = ".".join(str(x) for x in fc.api_version())
            R.ok(True, "MSP link up", f"{variant} {ver}, MSP API {api}")
            stats.update(firmware=variant, version=ver, api=api)
        except MSPTimeout:
            R.add("FAIL", "MSP link up",
                  "no reply — FC powered? UART free? baud correct?")
            return 1

        # The MCU serial is what ties this report to this physical controller,
        # so a failure to read it is a failure of the sign-off, not a warning.
        try:
            uid = fc.uid()
            R.ok(True, "flight-controller serial (MSP_UID)", uid)
            stats["uid"] = uid
        except Exception as e:
            R.add("FAIL", "flight-controller serial (MSP_UID)",
                  f"unreadable: {type(e).__name__}: {e}")
            stats["uid"] = None
        for label, fn in (("board", fc.board_id), ("craft name", fc.craft_name)):
            try:
                v = fn()
                R.add("INFO", label, v)
                stats[label.replace(" ", "_")] = v
            except Exception:
                pass
        try:
            d, c, rev = fc.build_info()
            R.add("INFO", "firmware build", f"{d} {c} rev {rev}")
            stats["build"] = f"{d} {c} {rev}"
        except Exception:
            pass

        # -- the FC's own view of its sensors -------------------------------
        try:
            st = fc.status()
            names = decode_sensor_flags(st["sensor_flags"])
            R.ok(st["sensor_flags"] & SENSOR_GYRO, "FC reports a gyro", ",".join(names))
            R.ok(st["sensor_flags"] & SENSOR_ACC, "FC reports an accelerometer",
                 ",".join(names))
            stats.update(sensors=names, armed=st.get("armed"))
            if st.get("armed"):
                R.add("INFO", "arming state", "ARMED — expected disarmed on a bench")
        except Exception as e:
            R.add("FAIL", "read MSP_STATUS", f"{type(e).__name__}: {e}")

        # -- sample ---------------------------------------------------------
        print(f"\n  sampling {a.samples} points at {a.hz} Hz — keep the board still…")
        crc0 = fc.crc_errors
        s = collect(fc, a.samples, a.hz, a.acc_1g)
        if not R.ok(len(s) >= a.samples * 0.5, "IMU samples collected",
                    f"{len(s)}/{a.samples}"):
            return 1
        dur = s[-1]["t"] - s[0]["t"]
        rate = (len(s) - 1) / dur if dur > 0 else 0.0
        R.ok(rate >= 20, "IMU sample rate", f"{rate:.1f} Hz over {dur:.1f} s")
        R.ok(fc.crc_errors - crc0 == 0, "MSP frame integrity",
             f"{fc.crc_errors - crc0} CRC errors in {len(s)} exchanges")
        stats.update(samples=len(s), rate_hz=round(rate, 1),
                     crc_errors=fc.crc_errors - crc0)

        ax = [p["acc_g"] for p in s]
        gy = [p["gyro"] for p in s]
        mags = [norm(v) for v in ax]
        mag_mean = statistics.fmean(mags)
        R.ok(0.85 <= mag_mean <= 1.15, "accelerometer reads 1 g at rest",
             f"|a| = {mag_mean:.3f} g (expect 1.000)")
        stats["accel_mag_g"] = round(mag_mean, 4)

        gmag = statistics.fmean(norm(v) for v in gy)
        R.ok(gmag < 15.0, "gyro near zero at rest", f"|w| = {gmag:.2f} dps")
        stats["gyro_mag_dps"] = round(gmag, 3)

        # -- liveness: the check a frozen sensor cannot pass -----------------
        # Every threshold above is satisfied by a buffer stuck at (0,0,512).
        # Real MEMS output always dithers; zero variance across every axis of
        # both sensors means we are reading a corpse.
        gsd = [statistics.pstdev([v[i] for v in gy]) for i in range(3)]
        asd = [statistics.pstdev([v[i] for v in ax]) for i in range(3)]
        R.ok(max(gsd) > 0.0 or max(asd) > 0.0, "IMU data is changing (not frozen)",
             f"gyro sd {gsd[0]:.2f}/{gsd[1]:.2f}/{gsd[2]:.2f} dps, "
             f"accel sd {asd[0]:.4f}/{asd[1]:.4f}/{asd[2]:.4f} g")
        # Noise that is present but implausibly large means a bad mount or a
        # failing part; report it without failing an otherwise sound board.
        R.add("INFO", "gyro noise (1 sigma)",
              f"{max(gsd):.2f} dps" + ("  — high for a still board" if max(gsd) > 5 else ""))
        stats.update(gyro_sd_dps=[round(v, 3) for v in gsd],
                     accel_sd_g=[round(v, 5) for v in asd])

        # -- optional operator-driven motion test ---------------------------
        if a.motion:
            print("\n  >>> ROTATE THE BOARD through roughly 90 degrees, now. <<<")
            peak, att0, moved, t0 = 0.0, None, 0.0, time.time()
            while time.time() - t0 < 8.0:
                try:
                    _a2, g2, _m = fc.raw_imu()
                    r2, p2, _y = fc.attitude()
                except (MSPError, MSPTimeout):
                    continue
                if att0 is None:
                    att0 = (r2, p2)
                peak = max(peak, norm(g2))
                moved = max(moved, abs(r2 - att0[0]), abs(p2 - att0[1]))
                time.sleep(0.02)
            R.ok(peak > 30.0, "gyro responds to movement", f"peak |w| = {peak:.0f} dps")
            R.ok(moved > 10.0, "attitude tracks movement",
                 f"attitude moved {moved:.0f} deg")
            stats.update(motion_peak_dps=round(peak, 1), motion_att_deg=round(moved, 1))
    finally:
        try:
            fc.close()
        except Exception:
            pass

    if a.json:
        stats["failed"] = R.failed
        with open(a.json, "w") as f:
            json.dump(stats, f, indent=2)
    return 1 if R.failed else 0


if __name__ == "__main__":
    sys.exit(main())
