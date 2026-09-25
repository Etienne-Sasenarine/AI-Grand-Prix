#!/usr/bin/env python3
"""Run imu_check.py against a simulated FC in each fault mode."""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_fc import FakeFC

CHK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imu_check.py")
UID = "deadbeef0123456789abcdef"
fails = []

def run(mode):
    fc = FakeFC(mode, UID)
    time.sleep(0.2)
    p = subprocess.run([sys.executable, CHK, "--dev", fc.dev, "--samples", "60",
                        "--hz", "80"], capture_output=True, text=True, timeout=90)
    return p.returncode, p.stdout + p.stderr

def chk(cond, label, extra=""):
    print(("PASS " if cond else "FAIL ") + label + ("  " + str(extra) if extra else ""))
    if not cond: fails.append(label)

rc, out = run("healthy")
chk(rc == 0, "healthy FC passes overall", f"rc={rc}")
chk(UID in out, "serial number reported", UID if UID in out else "MISSING")
chk("[FAIL]" not in out, "no spurious failures on a good board",
    [l.strip() for l in out.splitlines() if "[FAIL]" in l])
for want in ("MSP link up", "accelerometer reads 1 g", "gyro near zero",
             "data is changing", "sample rate",
             "FC reports a gyro", "frame integrity"):
    chk(want in out, f"check present: {want}")

rc, out = run("frozen")
chk(rc != 0, "FROZEN imu is caught", f"rc={rc}")
chk("[FAIL] IMU data is changing (not frozen)" in out,
    "frozen caught by the liveness check specifically",
    [l.strip() for l in out.splitlines() if "[FAIL]" in l])
chk("[PASS] accelerometer reads 1 g at rest" in out,
    "...and it fools every threshold check, as expected")
chk("[PASS] gyro near zero at rest" in out, "...including the gyro threshold")

rc, out = run("saturated")
chk(rc != 0, "saturated gyro is caught", f"rc={rc}")
chk("[FAIL] gyro near zero at rest" in out, "saturated caught by rest check")

rc, out = run("badscale")
chk(rc != 0, "mis-scaled accel is caught", f"rc={rc}")
chk("[FAIL] accelerometer reads 1 g at rest" in out, "badscale caught by 1 g check")

rc, out = run("noaccel")
chk(rc != 0, "missing accelerometer is caught", f"rc={rc}")
chk("[FAIL] FC reports an accelerometer" in out, "caught via FC sensor flags")

rc, out = run("nouid")
chk(rc != 0, "unreadable serial fails the run", f"rc={rc}")
chk("[FAIL] flight-controller serial" in out, "caught at MSP_UID")

print("\n" + ("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
