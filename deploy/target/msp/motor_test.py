#!/usr/bin/env python3
"""Bounded props-off bench test for this verified Betaflight 4.4.3 quad.

Runs on the Orin, using the supplied MSP transport. No RC arming or EEPROM writes.
Stop commands cannot protect against Orin power loss or a broken UART: disconnect
the flight battery if motors do not stop. An MSP arming block remains until reboot.
"""
import argparse
import json
import signal
import struct
import sys
import time

import os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
# MSP library location: prefer the in-repo copy under deploy/target/msp, fall
# back to the Jetson install, overridable via AIGP_MSP_DIR.
# (Was hardcoded to /home/dcl/target/msp.)
_here = _Path(__file__).resolve()
_msp_candidates = ([os.environ["AIGP_MSP_DIR"]] if os.environ.get("AIGP_MSP_DIR") else [])
_msp_candidates += [str(_here.parent), str(_here.parents[1] / "target" / "msp"), "/home/dcl/target/msp"]
sys.path.insert(0, next((c for c in _msp_candidates if _Path(c, "msp.py").exists()), _msp_candidates[-1]))
from msp import MSPLink

STOP = struct.pack('<8H', *([1000] * 8))


def request(fc, command, payload=b''):
    return fc.request(command, payload, timeout=0.2, retries=0)


def outputs(fc):
    data = request(fc, 104)
    if len(data) != 16:
        raise RuntimeError('Unexpected motor output response')
    return list(struct.unpack('<8H', data)[:4])


def rpms(fc):
    data = request(fc, 139)
    if len(data) != 53 or data[0] != 4:
        raise RuntimeError('Unexpected motor telemetry response')
    return [struct.unpack_from('<I', data, 1 + 13*i)[0] for i in range(4)]


def verify(fc):
    if fc.fc_variant() != 'BTFL' or fc.fc_version() != '4.4.3' or tuple(fc.api_version()) != (0, 1, 45):
        raise RuntimeError('Firmware differs from the verified configuration')
    cfg = request(fc, 131)
    advanced = request(fc, 90)
    features = struct.unpack('<I', request(fc, 36))[0]
    if len(cfg) != 10 or cfg[6] != 4 or cfg[8] != 1:
        raise RuntimeError('Expected four motors with DShot telemetry')
    if advanced[3] != 6 or features & (1 << 12):
        raise RuntimeError('Expected DShot300 with 3D disabled')
    if fc.status()['armed']:
        raise RuntimeError('FC is armed; refusing bench test')
    if outputs(fc) != [1000] * 4:
        raise RuntimeError('Motors are not at their stopped output')
    return {'firmware': 'BTFL 4.4.3', 'motor_count': 4, 'protocol': 'DShot300',
            'outputs': outputs(fc), 'rpm': rpms(fc), 'battery': fc.analog()}


def stop(fc):
    acknowledged = False
    for _ in range(3):
        try:
            request(fc, 214, STOP)
            acknowledged = True
        except Exception as exc:
            print('Stop attempt failed:', str(exc), file=sys.stderr, flush=True)
        time.sleep(0.05)
    if not acknowledged or outputs(fc) != [1000] * 4:
        raise RuntimeError('STOP NOT VERIFIED: disconnect the flight battery now')


def run(fc, selected, output=1050, seconds=1.0):
    result = []
    # This command blocks arming; it never enables arming or disables failsafe.
    request(fc, 99, b'\x01\x00')
    try:
        stop(fc)
        for motor in selected:
            if fc.status()['armed']:
                raise RuntimeError('Unexpected armed status')
            values = [1000] * 8
            values[motor - 1] = output
            print(f'Motor {motor}: {output} output for {seconds:.2f} seconds', flush=True)
            peak = 0
            request(fc, 214, struct.pack('<8H', *values))
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rpm = rpms(fc)
                peak = max(peak, rpm[motor - 1])
                if any(rpm[i] > 200 for i in range(4) if i != motor - 1):
                    raise RuntimeError('Unexpected RPM on another motor')
                time.sleep(0.08)
            stop(fc)
            time.sleep(1.0)
            after = rpms(fc)
            result.append({'motor': motor, 'peak_rpm': peak, 'rpm_after_stop': after})
            print(json.dumps(result[-1]), flush=True)
            if any(after):
                raise RuntimeError('RPM has not returned to zero; ending test')
            if peak == 0:
                raise RuntimeError('No rotation confirmed by telemetry; ending test')
    finally:
        # Ignore further termination signals while sending the final stop frames.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        signal.alarm(0)
        stop(fc)
        print('STOP VERIFIED: all four motor outputs are 1000', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--props-off', action='store_true')
    parser.add_argument('--motor', type=int, choices=range(1, 5))
    parser.add_argument('--output', type=int, default=1050,
                        help='external motor output; bench diagnostic limit is 1500')
    parser.add_argument('--seconds', type=float, default=1.0)
    args = parser.parse_args()
    if args.run and not args.props_off:
        parser.error('--run requires --props-off after physically removing all propellers')
    if not 1001 <= args.output <= 1500:
        parser.error('--output must be 1001..1500; unloaded full-throttle tests are refused')
    if not 0.10 <= args.seconds <= 2.0:
        parser.error('--seconds must be 0.10..2.0')
    def interrupted(signum, frame):
        raise RuntimeError(f'Test interrupted by signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, interrupted)
    with MSPLink('/dev/ttyTHS1', 115200, timeout=0.2) as fc:
        print(json.dumps(verify(fc)), flush=True)
        if args.run:
            signal.alarm(15)
            print(json.dumps(run(
                fc, [args.motor] if args.motor else [1, 2, 3, 4],
                output=args.output, seconds=args.seconds,
            )), flush=True)


if __name__ == '__main__':
    main()
