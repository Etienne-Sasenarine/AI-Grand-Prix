"""Per-motor RPM from the flight controller, best-effort.

Why best-effort
---------------
``dshot_bidir`` is **ON** on our airframe (measured 18 Sep), and another team
member's motor test read real RPM off it, so the data exists. What is not
established is which method name the organizers' ``msp.py`` exposes it under.
Rather than guess and have a flight test fail at the drone, this tries the
plausible routes in order and returns ``None`` if none work.

RPM matters more than its size suggests. With it:

    thrust-to-weight  =  (rpm_max / rpm_hover) ** 2
    thrust coeff k_f  =  m * g / (4 * rpm_hover ** 2)

Both fall out of ordinary flying, with no thrust stand and no accelerometer —
which matters because the accelerometer is unusable here for exactly this job:
vibration rectification biases it by about -0.46 * u^2 g, i.e. **-0.37 g at full
throttle**, a third of the signal, and biased hardest where the measurement is
taken.

Without RPM both tests still run; they fall back to motor command values and say
plainly that the answer now depends on an assumed thrust law.
"""

from __future__ import annotations

import struct

#: MSP_MOTOR_TELEMETRY. Betaflight replies with a count byte then, per motor,
#: uint32 rpm + uint16 invalid-packet-percent + temp/voltage/current/consumption.
MSP_MOTOR_TELEMETRY = 139
_ENTRY = struct.Struct("<IHHHHH")   # 14 bytes per motor


def _decode(payload: bytes, motors: int = 4):
    if not payload:
        return None
    n = payload[0]
    if n == 0 or len(payload) < 1 + _ENTRY.size:
        return None
    out = []
    for i in range(min(n, motors)):
        off = 1 + i * _ENTRY.size
        if off + _ENTRY.size > len(payload):
            break
        rpm, *_rest = _ENTRY.unpack_from(payload, off)
        out.append(int(rpm))
    return out or None


def read_rpm(fc, motors: int = 4):
    """Per-motor RPM as a list, or None if this link cannot provide it.

    Betaflight reports **eRPM in hundreds**; converting to mechanical RPM needs
    the pole count (14 on this airframe, measured). ``motor_poles`` wrong by any
    factor scales every reading by that same factor, silently — which is why it
    is recorded in ``SPEC.md`` rather than assumed.
    """
    # 1. A purpose-built method, if the library has one.
    for name in ("motor_telemetry", "motor_rpm", "rpm", "esc_telemetry"):
        fn = getattr(fc, name, None)
        if callable(fn):
            try:
                val = fn()
            except Exception:
                continue
            if isinstance(val, dict):
                val = val.get("rpm") or val.get("rpms")
            if val:
                try:
                    return [int(v) for v in val][:motors]
                except (TypeError, ValueError):
                    continue

    # 2. The generic request path the software guide documents.
    req = getattr(fc, "request", None)
    if callable(req):
        try:
            payload = req(MSP_MOTOR_TELEMETRY)
        except Exception:
            payload = None
        if payload is not None:
            if isinstance(payload, tuple):        # some libraries return (code, data)
                payload = payload[-1]
            if isinstance(payload, (bytes, bytearray)):
                return _decode(bytes(payload), motors)
    return None


def erpm_to_rpm(erpm: float, motor_poles: int = 14) -> float:
    """Betaflight eRPM (already in hundreds, and electrical) to mechanical RPM."""
    return float(erpm) * 100.0 / (motor_poles / 2.0)


def thrust_to_weight(rpm_hover: float, rpm_max: float) -> float:
    """Propeller thrust goes as rotor speed squared, so T/W is a pure ratio.

    No mass, no thrust stand, no accelerometer. This is the whole reason RPM is
    worth chasing.
    """
    if rpm_hover <= 0:
        return float("nan")
    return (rpm_max / rpm_hover) ** 2


def thrust_coefficient(mass_kg: float, rpm_hover: float, n_motors: int = 4,
                       g: float = 9.80665) -> float:
    """k_f such that thrust_per_rotor = k_f * rpm^2, from a hover.

    At hover total thrust equals weight, so one 20-second log and a kitchen
    scale give the coefficient the sim-to-real literature ranks second only to
    mass itself.
    """
    if rpm_hover <= 0 or n_motors <= 0:
        return float("nan")
    return mass_kg * g / (n_motors * rpm_hover ** 2)


def _self_test() -> None:
    # Decoder against a synthetic Betaflight reply: 4 motors at known eRPM.
    payload = bytes([4]) + b"".join(
        _ENTRY.pack(rpm, 0, 0, 0, 0, 0) for rpm in (180, 182, 179, 181))
    got = _decode(payload)
    assert got == [180, 182, 179, 181], got

    # Short/garbage payloads must degrade, never raise.
    assert _decode(b"") is None
    assert _decode(bytes([0])) is None
    assert _decode(bytes([4]) + b"\x00\x01") is None

    # eRPM -> RPM with 14 poles.
    rpm = erpm_to_rpm(180, motor_poles=14)
    assert abs(rpm - 180 * 100 / 7.0) < 1e-6, rpm

    # Physics helpers.
    assert abs(thrust_to_weight(1000.0, 2000.0) - 4.0) < 1e-9
    kf = thrust_coefficient(1.745, 1000.0)
    assert abs(4 * kf * 1000.0 ** 2 - 1.745 * 9.80665) < 1e-9

    # An object with no RPM support returns None rather than exploding.
    class Bare:
        pass
    assert read_rpm(Bare()) is None

    print("fc_rpm: all checks passed")
    print(f"  decoded 4 motors; 180 eRPM at 14 poles -> {rpm:.0f} mechanical RPM")
    print(f"  T/W from a 2x rotor-speed ratio: {thrust_to_weight(1000.0, 2000.0):.1f}")
    print(f"  k_f for 1.745 kg hovering at 1000 RPM: {kf:.3e} N/RPM^2")


if __name__ == "__main__":
    _self_test()
