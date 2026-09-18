#!/usr/bin/env python3
"""Read-only Betaflight hover-throttle logger for the Orin.

This program never sends MSP control, arm, disarm, or motor commands. The pilot
flies with the normal radio while it records FC telemetry and identifies stable
hover intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, "/home/dcl/target/msp")
from msp import MSPError, MSPLink, MSPTimeout


def stable(row):
    return (
        row["armed"]
        and abs(row["roll_deg"]) < 8.0
        and abs(row["pitch_deg"]) < 8.0
        and max(abs(row["gx_dps"]), abs(row["gy_dps"]), abs(row["gz_dps"])) < 20.0
        and abs(row["vario_m_s"]) <= 0.12
        # This airframe idles near 997 us. Do not mistake armed-on-ground time
        # for hover when the barometer reports a constant zero vario.
        and 1050 <= row["throttle_us"] <= 2000
    )


def stable_segments(rows, minimum_s=3.0, transient_gap_s=0.40):
    """Return stable segments while tolerating brief vibration/telemetry spikes."""
    segments, current = [], []
    for row in rows:
        if stable(row):
            if current and row["t_s"] - current[-1]["t_s"] > transient_gap_s:
                if current[-1]["t_s"] - current[0]["t_s"] >= minimum_s:
                    segments.append(current)
                current = []
            current.append(row)
        else:
            if current and row["t_s"] - current[-1]["t_s"] > transient_gap_s:
                if current[-1]["t_s"] - current[0]["t_s"] >= minimum_s:
                    segments.append(current)
                current = []
    if current and current[-1]["t_s"] - current[0]["t_s"] >= minimum_s:
        segments.append(current)
    return segments


def summarize(rows, throttle_index):
    segments = stable_segments(rows)
    trials = []
    for segment in segments:
        throttle = [r["throttle_us"] for r in segment]
        trials.append({
            "start_s": round(segment[0]["t_s"], 3),
            "duration_s": round(segment[-1]["t_s"] - segment[0]["t_s"], 3),
            "samples": len(segment),
            "median_throttle_us": statistics.median(throttle),
            "throttle_std_us": round(statistics.pstdev(throttle), 3),
            "mean_vario_m_s": round(statistics.mean(r["vario_m_s"] for r in segment), 4),
        })
    result = {
        "read_only": True,
        "samples": len(rows),
        "throttle_channel_one_based": throttle_index + 1,
        "stable_trials": trials,
        "accepted": len(trials) >= 3,
        "hover_throttle_us": None,
        "hover_thrust_normalized": None,
        "warning": None,
    }
    if len(trials) >= 3:
        value = statistics.median(t["median_throttle_us"] for t in trials)
        result["hover_throttle_us"] = value
        result["hover_thrust_normalized"] = (value - 1000.0) / 1000.0
        result["warning"] = (
            "Candidate assumes Betaflight RC throttle maps linearly from 1000..2000 us "
            "to 0..1. Verify this matches the AI_GP actuator path before applying it."
        )
    else:
        result["warning"] = "Need at least three separate stable hover periods of 3 seconds or longer."
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--throttle-channel", type=int, default=4,
                    help="one-based MSP_RC channel number; this FC reports throttle on channel 4")
    ap.add_argument("--output-dir", default="/home/dcl/target/msp/hover_logs")
    args = ap.parse_args()
    if not 1 <= args.throttle_channel <= 18:
        ap.error("--throttle-channel must be 1..18")
    if args.duration <= 0 or not 1 <= args.rate <= 50:
        ap.error("duration must be positive and rate must be 1..50 Hz")

    throttle_index = args.throttle_channel - 1
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"hover_{stamp}.csv")
    json_path = os.path.join(args.output_dir, f"hover_{stamp}.json")
    rows, timeouts = [], 0
    period = 1.0 / args.rate

    print("READ-ONLY logger: no arm, throttle, RC, or motor commands will be sent.", flush=True)
    print(f"Recording for up to {args.duration:.0f}s. Pilot owns the aircraft. Ctrl+C ends logging.", flush=True)
    with MSPLink(args.port, args.baud, timeout=0.25) as fc:
        identity = {"variant": fc.fc_variant(), "version": fc.fc_version(), "board": fc.board_id()}
        start, next_tick = time.monotonic(), time.monotonic()
        latest_status, latest_battery = fc.status(), fc.analog()
        try:
            while time.monotonic() - start < args.duration:
                now = time.monotonic()
                try:
                    rc = fc.rc_channels()
                    if throttle_index >= len(rc):
                        raise RuntimeError(f"FC returned only {len(rc)} RC channels")
                    roll, pitch, yaw = fc.attitude()
                    acc, gyro, _mag = fc.raw_imu()
                    altitude, vario = fc.altitude()
                    if len(rows) % max(1, int(args.rate / 5)) == 0:
                        latest_status = fc.status()
                    if len(rows) % max(1, int(args.rate)) == 0:
                        latest_battery = fc.analog()
                    row = {
                        "t_s": round(now - start, 6),
                        "armed": bool(latest_status.get("armed")),
                        "throttle_us": int(rc[throttle_index]),
                        "rc_channels": list(rc),
                        "roll_deg": float(roll), "pitch_deg": float(pitch), "yaw_deg": float(yaw),
                        "gx_dps": float(gyro[0]), "gy_dps": float(gyro[1]), "gz_dps": float(gyro[2]),
                        "ax_raw": int(acc[0]), "ay_raw": int(acc[1]), "az_raw": int(acc[2]),
                        "altitude_m": float(altitude), "vario_m_s": float(vario),
                        "battery_v": float(latest_battery["voltage_v"]),
                    }
                    rows.append(row)
                    if len(rows) % max(1, int(args.rate * 2)) == 0:
                        print(
                            f"t={row['t_s']:.0f}s armed={row['armed']} throttle={row['throttle_us']} "
                            f"tilt={row['roll_deg']:+.1f}/{row['pitch_deg']:+.1f} "
                            f"vario={row['vario_m_s']:+.2f}", flush=True
                        )
                except (MSPError, MSPTimeout):
                    timeouts += 1
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except KeyboardInterrupt:
            print("Logging stopped by operator.", flush=True)

    fields = [
        "t_s", "armed", "throttle_us", "rc_channels", "roll_deg", "pitch_deg", "yaw_deg",
        "gx_dps", "gy_dps", "gz_dps", "ax_raw", "ay_raw", "az_raw", "altitude_m",
        "vario_m_s", "battery_v",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["rc_channels"] = json.dumps(encoded["rc_channels"], separators=(",", ":"))
            writer.writerow(encoded)
    result = summarize(rows, throttle_index)
    result.update({"identity": identity, "timeouts": timeouts, "csv": csv_path})
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
