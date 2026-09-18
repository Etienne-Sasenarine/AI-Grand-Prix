"""A stand-in for the organizers' ``MSPLink``, so the field tools can be built
and validated with no drone, no flight controller and no ``~/target/``.

Why this exists rather than a copy of their ``fake_fc.py``
---------------------------------------------------------
Their ``fake_fc.py`` simulates a Betaflight FC **on a pty**, at the wire level,
and it is the right thing to test against once ``~/target/`` is in hand. We do
not have it yet, and the one mirror we can see sits inside a repository with no
licence, so it is off limits.

This is deliberately *not* a second MSP implementation — the software guide says
"do not re-implement MSP framing", and nothing here frames a byte. It duck-types
the ``MSPLink`` **interface** documented in the Orin NX software guide, and
backs it with a small flight model.

That backing model is the point. It flies with **known** parameters, so an
analysis tool can be checked against a ground truth it was never told:
``analyze_hover`` must recover ``hover_throttle`` and ``analyze_rates`` must
recover ``tau_s`` and ``max_ang_accel`` from nothing but the log. A parser that
merely runs without crashing is not evidence of anything.

Swap it for the real thing with ``--dev /dev/ttyTHS1``; the tools take either.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

G = 9.80665


@dataclass
class MockConfig:
    """Ground truth. The analysis tools must recover these without being told."""

    #: Throttle that holds a hover, as a fraction of the 1000-2000 RC range.
    #: The whole point of the hover experiment is to find this number.
    hover_throttle: float = 0.31
    #: Closed-loop body-rate time constant, seconds. What a rate step measures.
    tau_s: float = 0.045
    #: Angular-acceleration ceiling, rad/s^2, per axis. The saturation point.
    max_ang_accel: tuple = (95.0, 85.0, 38.0)
    #: Rate at full stick deflection, rad/s.
    rate_limit: float = 3.2
    #: Pack voltage full and empty, and how far the throttle drifts across it.
    volt_full: float = 25.2
    volt_empty: float = 21.0
    pack_duration_s: float = 240.0
    #: Steady attitude offset at hover, radians. Centre-of-gravity and thrust
    #: axis misalignment show up here, and it adds to the camera tilt.
    trim_roll: float = math.radians(0.8)
    trim_pitch: float = math.radians(-2.1)
    #: Gyro noise, rad/s, and attitude noise, radians.
    gyro_noise: float = 0.02
    att_noise: float = math.radians(0.15)
    #: Motor command noise, in the 0-1 fraction.
    motor_noise: float = 0.004
    #: Raw gyro scaling. Betaflight reports raw counts; the conflict in SPEC.md
    #: is whether this is 16.4 counts/deg/s or already deg/s. The mock uses the
    #: counts convention so the units check has something to catch.
    gyro_counts_per_deg_s: float = 16.4
    acc_counts_per_g: float = 512.0
    seed: int = 0


class MockLink:
    """Duck-types ``msp.MSPLink``. See the module docstring."""

    def __init__(self, cfg: MockConfig | None = None) -> None:
        self.cfg = cfg or MockConfig()
        self._rng = random.Random(self.cfg.seed)
        self._t0 = time.monotonic()
        self._last = self._t0
        self._rates = [0.0, 0.0, 0.0]          # rad/s, body
        self._att = [self.cfg.trim_roll, self.cfg.trim_pitch, 0.0]
        self._throttle = 0.0                   # 0-1
        self._rate_cmd = [0.0, 0.0, 0.0]
        self._crc_errors = 0
        self._armed = False
        self.closed = False
        #: How many times anything asked this link to actuate the aircraft.
        #: Tools that claim to be read-only can then be *checked* rather than
        #: trusted -- see test_camera_capture.py, which runs beside a pilot with
        #: propellers on and must never command anything.
        self.commands_received = 0

    # ------------------------------------------------------------- driving it
    def command(self, throttle: float, rates_rad_s=(0.0, 0.0, 0.0)) -> None:
        """What a transmitter would be sending. Not part of the MSPLink API."""
        self.commands_received += 1
        self._throttle = float(min(max(throttle, 0.0), 1.0))
        self._rate_cmd = [float(r) for r in rates_rad_s]

    def _advance(self) -> None:
        now = time.monotonic()
        dt = now - self._last
        if dt <= 0:
            return
        self._last = now
        dt = min(dt, 0.1)
        c = self.cfg
        for i in range(3):
            err = self._rate_cmd[i] - self._rates[i]
            accel = err / max(c.tau_s, 1e-4)
            lim = c.max_ang_accel[i]
            accel = max(-lim, min(lim, accel))
            self._rates[i] += accel * dt
            self._att[i] += self._rates[i] * dt
        # Attitude relaxes toward trim in roll/pitch, as ANGLE mode would.
        self._att[0] += (c.trim_roll - self._att[0]) * min(1.0, dt * 3.0)
        self._att[1] += (c.trim_pitch - self._att[1]) * min(1.0, dt * 3.0)

    # ------------------------------------------------------- the MSPLink face
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def _sag(self) -> float:
        frac = min(1.0, self.elapsed / max(self.cfg.pack_duration_s, 1e-6))
        return self.cfg.volt_full + (self.cfg.volt_empty - self.cfg.volt_full) * frac

    def attitude(self):
        """(roll, pitch, yaw) in degrees, yaw 0-360 — as documented."""
        self._advance()
        n = self.cfg.att_noise
        r, p, y = (a + self._rng.gauss(0.0, n) for a in self._att)
        return (math.degrees(r), math.degrees(p), math.degrees(y) % 360.0)

    def raw_imu(self):
        """(acc, gyro, mag), each a 3-tuple, in **raw FC units**."""
        self._advance()
        c = self.cfg
        gyro = tuple(int(round((math.degrees(w) + self._rng.gauss(0.0, math.degrees(c.gyro_noise)))
                               * c.gyro_counts_per_deg_s)) for w in self._rates)
        # Thrust along body up, plus gravity, in g. A hovering drone reads 1 g.
        az = (self._throttle / max(c.hover_throttle, 1e-6))
        acc = (int(round(-math.sin(self._att[1]) * az * c.acc_counts_per_g)),
               int(round(math.sin(self._att[0]) * az * c.acc_counts_per_g)),
               int(round(az * c.acc_counts_per_g)))
        return acc, gyro, (0, 0, 0)

    def analog(self):
        v = self._sag()
        return {"voltage_v": round(v + self._rng.gauss(0.0, 0.02), 3),
                "current_a": round(18.0 + 40.0 * self._throttle, 2),
                "mah_drawn": int(1400 * min(1.0, self.elapsed / 240.0)),
                "rssi": 1000}

    def rc_channels(self):
        """The channels the FC is acting on, in 1000-2000 microseconds."""
        def us(x):
            return int(1000 + 1000 * min(max(x, 0.0), 1.0))
        mid = [500 + 500 * (r / max(self.cfg.rate_limit, 1e-6)) for r in self._rate_cmd]
        return [us(self._throttle)] + [int(1000 + m) for m in mid] + [1000] * 4

    def motors(self):
        """Per-motor output, 1000-2000. Sags with the pack, as a real one does.

        This is the channel the hover experiment reads, **not** the throttle
        stick: it is post-mix, post-curve and post-TPA, which is what the ESCs
        actually saw.
        """
        self._advance()
        c = self.cfg
        sag_comp = c.volt_full / max(self._sag(), 1e-6)
        base = self._throttle * sag_comp
        out = []
        for _ in range(4):
            v = base + self._rng.gauss(0.0, c.motor_noise)
            out.append(int(1000 + 1000 * min(max(v, 0.0), 1.0)))
        return out + [0, 0, 0, 0]

    def altitude(self):
        """(altitude_m, vario_m_s) -- a TUPLE, matching the real library.

        The mock reports a **dead barometer**, because that is what the real one
        does on this airframe: measured on 18 Sep, ``vario`` held exactly 0.000
        for all 2828 samples of a two-minute flight, and altitude fell to
        -6.5 m as the props spun up. Any stability gate built on vario passes
        on every frame and filters nothing -- which is how a log of the drone
        sitting on the ground was accepted as a hover.
        """
        return (0.0, 0.0)

    def status(self):
        return {"cycle_time_us": 125, "i2c_errors": 0, "flight_mode_flags": 1,
                "arming_flags": 0, "armed": self._armed}

    def fc_version(self):
        return "4.4.3"

    def board_id(self):
        return "SH74"

    def arm_state(self, armed: bool) -> None:
        """Test hook. Nothing here ever commands the aircraft."""
        self._armed = bool(armed)

    def api_version(self):
        return (1, 46, 0)

    def fc_variant(self):
        return "BTFL"

    def uid(self):
        return "MOCK-0000-0000"

    def crc_errors(self):
        return self._crc_errors

    def decode_arming_flags(self, *_a, **_k):
        return []

    def decode_sensor_flags(self, *_a, **_k):
        return ["ACC", "BARO", "GYRO"]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def open_link(dev: str | None, *, baud: int = 115200, cfg: MockConfig | None = None):
    """Return a real ``MSPLink`` when a device is named, otherwise the mock.

    Every tool in this directory takes ``--dev``; leaving it off runs against
    the mock, so the whole set is testable on a laptop.
    """
    if dev in (None, "", "mock"):
        return MockLink(cfg)
    import sys
    for p in ("/home/dcl/target/msp", "./target/msp", "../target/msp",
              "../../target/msp"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from msp import MSPLink  # noqa: E402  — only on the drone
    link = MSPLink(dev, baud)
    # MSPLink's constructor does NOT open the port: it leaves ``_ser`` as None
    # and every accessor then dies with "'NoneType' object has no attribute
    # 'write'". ``open()`` is what creates the serial port and starts the
    # background receive thread, and it returns self.
    #
    # This cost us a real debugging session on 18 Sep: the whole tool suite
    # passed against MockLink, which needs no opening, and then failed at first
    # contact with the flight controller. The mock hid it, so the mock is not
    # enough on its own.
    link.open()
    return link


def _self_test() -> None:
    cfg = MockConfig()
    fc = MockLink(cfg)
    assert fc.fc_variant() == "BTFL"
    r, p, y = fc.attitude()
    assert -10 < r < 10 and -10 < p < 10 and 0 <= y < 360, (r, p, y)

    fc.command(cfg.hover_throttle)
    time.sleep(0.05)
    m = fc.motors()
    assert len(m) == 8 and all(1000 <= v <= 2000 for v in m[:4]), m
    frac = (sum(m[:4]) / 4 - 1000) / 1000.0
    assert abs(frac - cfg.hover_throttle) < 0.02, (frac, cfg.hover_throttle)

    acc, gyro, _ = fc.raw_imu()
    assert abs(acc[2] / cfg.acc_counts_per_g - 1.0) < 0.1, acc

    fc.command(cfg.hover_throttle, (2.0, 0.0, 0.0))
    for _ in range(40):
        time.sleep(0.01)
        fc.attitude()
    _, gyro, _ = fc.raw_imu()
    roll_rate = gyro[0] / cfg.gyro_counts_per_deg_s
    assert roll_rate > 60.0, f"rate command should spin the roll axis up, got {roll_rate:.1f} deg/s"

    print("mock_link: all checks passed")
    print(f"  hover recovered from motors(): {frac:.3f} against a true {cfg.hover_throttle:.3f}")
    print(f"  roll rate after a 2.0 rad/s step: {roll_rate:.0f} deg/s")


if __name__ == "__main__":
    _self_test()
