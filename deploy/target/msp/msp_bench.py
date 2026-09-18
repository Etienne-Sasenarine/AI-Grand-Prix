#!/usr/bin/env python3
"""Bench + flight entry point for the Orin NX -> Betaflight MSP link.

Subcommands, in the order you should run them the first time:

  info       identify the FC, sensors and arming blockers. Read-only, always safe.
  telemetry  stream attitude / IMU / battery at 20 Hz. Read-only, always safe.
  rc         stream RC frames and echo back what the FC reports via MSP_RC.
             Throttle stays at minimum and ARM is never raised. Safe with props on,
             but do it with the battery disconnected anyway.
  demo       ARM and run a scripted throttle/attitude profile. Requires --props-off
             and refuses to run without it. Motors WILL spin.

Examples
--------
    python3 msp_bench.py info --port /dev/ttyTHS1
    python3 msp_bench.py telemetry --port /dev/ttyTHS1
    python3 msp_bench.py rc --port /dev/ttyTHS1
    python3 msp_bench.py demo --port /dev/ttyTHS1 --props-off --max-throttle 0.15
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from msp import (
    MSPError,
    MSPLink,
    MSPTimeout,
    decode_sensor_flags,
)
from msp_rc import RCTransmitter

DEFAULT_PORT = "/dev/ttyTHS1"
DEFAULT_BAUD = 115200

_stop = False


def _install_signal_handlers():
    def handler(signum, frame):  # noqa: ARG001
        global _stop
        _stop = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def connect(args):
    link = MSPLink(args.port, args.baud, timeout=args.timeout, use_v2=args.msp_v2).open()
    try:
        variant = link.fc_variant()
        version = link.fc_version()
        proto, api_major, api_minor = link.api_version()
    except (MSPTimeout, MSPError) as exc:
        link.close()
        raise SystemExit(
            f"No MSP reply on {args.port} @ {args.baud}: {exc}\n"
            "Check: TX<->RX not swapped, GND shared, the UART has MSP enabled in the\n"
            "Betaflight Ports tab, and the Betaflight Configurator is not holding the port."
        )
    print(f"FC       : {variant} {version}  (MSP {proto}, API {api_major}.{api_minor})")
    return link


# --------------------------------------------------------------------------------------
def cmd_info(args):
    link = connect(args)
    try:
        try:
            print(f"Board    : {link.board_id()}")
            print(f"Craft    : {link.craft_name() or '(unnamed)'}")
        except (MSPTimeout, MSPError):
            pass
        try:
            date, clock, rev = link.build_info()
            print(f"Build    : {date} {clock}  git {rev}")
        except (MSPTimeout, MSPError, IndexError):
            pass

        st = link.status()
        print(f"Sensors  : {', '.join(decode_sensor_flags(st['sensor_flags'])) or 'none'}")
        print(f"Cycle    : {st['cycle_time_us']} us   load: {st['system_load_pct']}%")
        print(f"I2C errs : {st['i2c_errors']}")
        print(f"Armed    : {st['armed']}")
        reasons = st["arming_disable_reasons"]
        if reasons:
            print(f"Arming blocked by: {', '.join(reasons)}")
        else:
            print("Arming blocked by: nothing")

        batt = link.analog()
        print(f"Battery  : {batt['voltage_v']:.2f} V  {batt['current_a']:.2f} A  "
              f"{batt['mah_drawn']} mAh")
        print(f"RC in    : {link.rc_channels()}")
        print(f"ARM bit  : {link.arm_bit()} of flight_mode_flags")
    finally:
        link.close()
    return 0


def cmd_telemetry(args):
    link = connect(args)
    period = 1.0 / args.hz
    print(f"{'roll':>8}{'pitch':>8}{'yaw':>8} | {'gx':>7}{'gy':>7}{'gz':>7} | "
          f"{'volts':>7}{'amps':>7} | armed")
    try:
        while not _stop:
            t0 = time.monotonic()
            try:
                roll, pitch, yaw = link.attitude()
                _acc, gyro, _mag = link.raw_imu()
                batt = link.analog()
                st = link.status()
            except (MSPTimeout, MSPError) as exc:
                print(f"  telemetry error: {exc}", file=sys.stderr)
                continue
            print(f"{roll:8.1f}{pitch:8.1f}{yaw:8.1f} | "
                  f"{gyro[0]:7d}{gyro[1]:7d}{gyro[2]:7d} | "
                  f"{batt['voltage_v']:7.2f}{batt['current_a']:7.2f} | "
                  f"{'YES' if st['armed'] else 'no'}", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    finally:
        link.close()
    return 0


def cmd_rc(args):
    """Prove the control path end to end without ever arming."""
    link = connect(args)
    tx = RCTransmitter(link, rate_hz=args.rate, max_throttle=0.0).start()
    try:
        tx.wait_until_streaming()
        print("Streaming MSP_SET_RAW_RC with throttle at minimum and ARM low.")
        print("Sweeping roll only. Watch that the FC echoes the sweep back.\n")
        print(f"{'sent (AETR + AUX)':<44} {'FC reports (MSP_RC)'}")
        t_start = time.monotonic()
        while not _stop:
            phase = (time.monotonic() - t_start) % 4.0 / 4.0
            roll = 0.5 * (1.0 if phase < 0.5 else -1.0)
            tx.set_control(roll=roll, pitch=0.0, yaw=0.0, throttle=0.0)
            try:
                echo = link.rc_channels()[:8]
            except (MSPTimeout, MSPError) as exc:
                echo = [f"err: {exc}"]
            print(f"{str(tx.snapshot()):<44} {echo}", flush=True)
            time.sleep(0.1)
    finally:
        tx.close()
        link.close()
    return 0


def cmd_demo(args):
    if not args.props_off:
        raise SystemExit(
            "Refusing to run: 'demo' arms the FC and spins motors.\n"
            "Remove all propellers, verify they are off, then re-run with --props-off."
        )

    link = connect(args)
    tx = RCTransmitter(
        link,
        rate_hz=args.rate,
        max_throttle=args.max_throttle,
        max_angle_cmd=args.max_angle,
        command_timeout_s=0.25,
    ).start()
    try:
        tx.wait_until_streaming()

        # Betaflight sets ARMING_DISABLED_MSP as soon as an MSP client appears.
        # Release it explicitly, then confirm nothing else is blocking arming.
        link.set_arming_disabled(False)
        tx.set_control(throttle=0.0)
        time.sleep(0.5)

        st = link.status()
        blockers = [r for r in st["arming_disable_reasons"] if r != "ARM_SWITCH"]
        if blockers:
            raise SystemExit(f"Cannot arm, FC reports: {', '.join(blockers)}")

        print(f"Arming (max throttle clamped to {args.max_throttle:.0%})...")
        tx.arm()
        armed_at = time.monotonic()
        while time.monotonic() - armed_at < 2.0:
            if link.status()["armed"]:
                break
            time.sleep(0.1)
        else:
            raise SystemExit(
                "FC did not arm within 2 s. Check the ARM mode range on AUX1 "
                "(Betaflight Modes tab) and 'get rcmap'."
            )
        print("ARMED. Running profile; Ctrl-C aborts and disarms.")

        profile = [
            # (seconds, roll, pitch, yaw, throttle_fraction_of_max)
            (2.0, 0.0, 0.0, 0.0, 0.30),
            (2.0, 0.0, -0.15, 0.0, 0.50),
            (2.0, 0.15, 0.0, 0.0, 0.50),
            (2.0, 0.0, 0.0, 0.20, 0.40),
            (2.0, 0.0, 0.0, 0.0, 0.0),
        ]
        for duration, roll, pitch, yaw, thr in profile:
            if _stop:
                break
            end = time.monotonic() + duration
            print(f"  roll={roll:+.2f} pitch={pitch:+.2f} yaw={yaw:+.2f} "
                  f"throttle={thr * args.max_throttle:.0%} for {duration:.1f}s")
            while time.monotonic() < end and not _stop:
                tx.set_control(roll=roll, pitch=pitch, yaw=yaw,
                               throttle=thr * args.max_throttle)
                time.sleep(1.0 / args.rate)
    finally:
        print("Disarming...")
        tx.close()
        try:
            link.set_arming_disabled(True)
        except (MSPTimeout, MSPError):
            pass
        link.close()
        print("Link closed.")
    return 0


# --------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=0.5, help="MSP reply timeout (s)")
    parser.add_argument("--msp-v2", action="store_true", help="use MSPv2 framing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info").set_defaults(func=cmd_info)

    p_tel = sub.add_parser("telemetry")
    p_tel.add_argument("--hz", type=float, default=20.0)
    p_tel.set_defaults(func=cmd_telemetry)

    p_rc = sub.add_parser("rc")
    p_rc.add_argument("--rate", type=float, default=100.0)
    p_rc.set_defaults(func=cmd_rc)

    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--rate", type=float, default=100.0)
    p_demo.add_argument("--props-off", action="store_true",
                        help="required acknowledgement that propellers are removed")
    p_demo.add_argument("--max-throttle", type=float, default=0.15,
                        help="hard clamp, 0..1 (default 0.15)")
    p_demo.add_argument("--max-angle", type=float, default=0.3,
                        help="stick authority clamp, 0..1 (default 0.3)")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    _install_signal_handlers()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
