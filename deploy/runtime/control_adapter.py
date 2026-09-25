"""Turn a policy action into Betaflight stick values, safely.

This is the seam between our trained policy and the organizers' MSP transmitter.
It does four things, in order:

  1. decode the policy's [-1, 1] action into physical units
  2. apply the safety envelope (clamps that only ever open after a clean flight)
  3. convert physical units to stick positions through Betaflight's own curves
  4. map to channel order and signs, and hand off to the transmitter

We do not implement MSP. The organizers ship ``~/target/msp/msp_rc.py`` with an
``RCTransmitter`` that streams ``MSP_SET_RAW_RC`` from a background thread and
already has a watchdog. Their software guide says plainly: "do not re-implement
MSP framing." This module calls theirs.

The most dangerous thing in this file
-------------------------------------
``AxisMap``. Isaac's body frame is forward-left-up; Betaflight's is
forward-right-down; and the channel order depends on the receiver map. A single
reversed sign flips the drone on takeoff, and it is the most common first-flight
failure in this field. The defaults below are the conventional ones and are
**UNVERIFIED**. Run ``verify_axis_map()`` on a bench, props off, one axis at a
time, with a second person watching, before any flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from betaflight_curves import RateCurve, ThrottleCurve, stick_to_pwm

# --- The action contract, mirroring isaac_drone_racer/dynamics/rate_control.py.
# If those constants change in training, they must change here too.
THRUST_MIN = 0.05
THRUST_HOVER = 0.255
THRUST_MAX = 0.90
RATE_LIMIT = 3.2  # rad/s at full deflection


def decode_action(action) -> tuple[float, np.ndarray]:
    """Policy action in [-1, 1]^4 to (thrust stick, body rates in rad/s, NED).

    Exactly mirrors ``decode_aigp_action`` in the training code: zero action is
    hover with no rotation; the thrust channel is piecewise so that the same
    action magnitude means different things climbing and sinking.
    """
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:4], -1.0, 1.0)
    x = float(a[0])
    up = THRUST_HOVER + x * (THRUST_MAX - THRUST_HOVER)
    down = THRUST_HOVER + x * (THRUST_HOVER - THRUST_MIN)
    thrust = float(np.clip(up if x >= 0.0 else down, THRUST_MIN, THRUST_MAX))
    return thrust, a[1:4] * RATE_LIMIT


@dataclass
class Envelope:
    """Deploy-time clamps. Start restrictive; open one step per clean flight.

    These are not tuning parameters. They are the difference between a mistake
    that costs a re-flight and one that costs an airframe.
    """

    max_rate_rad_s: float = 1.0          # target 3.2
    max_thrust_stick: float = THRUST_HOVER + 0.10   # target 0.90
    min_thrust_stick: float = THRUST_HOVER - 0.10   # target 0.05
    speed_governor: float = 0.30         # scales the policy's commanded-velocity input
    enabled: bool = True

    def apply(self, thrust: float, rates: np.ndarray) -> tuple[float, np.ndarray]:
        if not self.enabled:
            return thrust, rates
        t = float(np.clip(thrust, self.min_thrust_stick, self.max_thrust_stick))
        r = np.clip(rates, -self.max_rate_rad_s, self.max_rate_rad_s)
        return t, r

    def describe(self) -> str:
        return (f"rate<={self.max_rate_rad_s:.2f}rad/s thrust=[{self.min_thrust_stick:.3f},"
                f"{self.max_thrust_stick:.3f}] governor={self.speed_governor:.0%}")


@dataclass
class AxisMap:
    """Channel order and sign conventions. **UNVERIFIED — bench-test this.**

    ``order`` gives the index of roll, pitch, throttle and yaw within the RC
    channel array. The default is AETR, Betaflight's usual ``rcmap``. Read the
    real value from ``rcmap`` in a ``diff all`` dump.

    ``sign`` flips an axis.

    **The policy's convention, derived from first principles and asserted in
    ``dynamics._self_test``:** body NED has x forward, y right, z down. A pitch
    rate is a rotation about +y (the right wing); by the right-hand rule that
    carries +z (down) toward +x (forward), so the nose swings *up*. Therefore
    **positive pitch rate is nose up**, and positive roll rate is right-wing
    down. This is the standard aerospace convention.

    An earlier version of this file asserted the opposite for pitch and set
    ``sign_pitch = -1`` to compensate. That was wrong on both counts and would
    have inverted the pitch axis in flight.

    Betaflight's stick convention is believed to be the same (stick up = nose
    up), which is why all three signs now default to +1. **That belief is
    UNVERIFIED** — it is the single most dangerous unchecked assumption in this
    package, because a reversed sign flips the aircraft on takeoff. Run
    ``verify_axis_map`` on a bench with the propellers off, one axis at a time,
    with a second person watching, before anything spins.
    """

    order: tuple[int, int, int, int] = (0, 1, 2, 3)  # roll, pitch, throttle, yaw
    sign_roll: float = +1.0
    sign_pitch: float = +1.0
    sign_yaw: float = +1.0

    def channels(self, roll_pwm: int, pitch_pwm: int, throttle_pwm: int, yaw_pwm: int,
                 n_channels: int = 8, idle_pwm: int = 1500) -> list[int]:
        ch = [idle_pwm] * n_channels
        r, p, t, y = self.order
        ch[r], ch[p], ch[t], ch[y] = roll_pwm, pitch_pwm, throttle_pwm, yaw_pwm
        return ch


class Transmitter(Protocol):
    """The subset of the organizers' ``RCTransmitter`` this module uses.

    Their real signature is not in the written guide, so ``ControlAdapter``
    takes a ``send`` callable instead of calling a method directly. When the
    real API is in hand, adapt it in exactly one place: the ``send`` you pass
    to the constructor.
    """

    def set_control(self, *args, **kwargs) -> None: ...


@dataclass
class ControlAdapter:
    """Policy action in, channel values out, with the envelope applied."""

    rate_curve: RateCurve = field(default_factory=RateCurve)
    throttle_curve: ThrottleCurve = field(default_factory=ThrottleCurve)
    envelope: Envelope = field(default_factory=Envelope)
    axis_map: AxisMap = field(default_factory=AxisMap)
    send: Callable[[list[int]], None] | None = None
    #: The simulator's thrust stick is a fraction of *its* hover convention.
    #: If the real drone's measured hover stick differs, set it here and every
    #: thrust command is rescaled so that "hover" means hover.
    measured_hover_stick: float | None = None

    def action_to_channels(self, action) -> tuple[list[int], dict]:
        thrust_stick, rates = decode_action(action)
        thrust_stick, rates = self.envelope.apply(thrust_stick, rates)

        # Rescale if the real hover point differs from the trained one. The
        # policy asks for "1.4x hover"; deliver 1.4x the *real* hover.
        t = thrust_stick
        if self.measured_hover_stick is not None:
            t = float(np.clip(thrust_stick / THRUST_HOVER * self.measured_hover_stick, 0.0, 1.0))

        roll = self.rate_curve.stick_from_rate_rad_s(self.axis_map.sign_roll * rates[0])
        pitch = self.rate_curve.stick_from_rate_rad_s(self.axis_map.sign_pitch * rates[1])
        yaw = self.rate_curve.stick_from_rate_rad_s(self.axis_map.sign_yaw * rates[2])

        roll_pwm = stick_to_pwm(roll)
        pitch_pwm = stick_to_pwm(pitch)
        yaw_pwm = stick_to_pwm(yaw)
        thr_pwm = stick_to_pwm(t, bipolar=False)

        channels = self.axis_map.channels(roll_pwm, pitch_pwm, thr_pwm, yaw_pwm)
        debug = {
            "thrust_stick": thrust_stick,
            "thrust_stick_sent": t,
            "rates_rad_s": rates.tolist(),
            "rates_deg_s": np.degrees(rates).tolist(),
            "sticks": {"roll": roll, "pitch": pitch, "yaw": yaw, "throttle": t},
            "pwm": {"roll": roll_pwm, "pitch": pitch_pwm, "throttle": thr_pwm, "yaw": yaw_pwm},
        }
        return channels, debug

    def step(self, action) -> dict:
        channels, debug = self.action_to_channels(action)
        if self.send is not None:
            self.send(channels)
        debug["channels"] = channels
        return debug

    def idle_channels(self) -> list[int]:
        """Neutral sticks with throttle at minimum. The safe frame."""
        return self.axis_map.channels(1500, 1500, 1000, 1500)


def verify_axis_map(adapter: ControlAdapter) -> str:
    """Print the bench-test script for checking signs. Props off.

    Run each line, watch which way the airframe *tries* to go, and confirm it
    matches the expectation. Do this with a second person. It takes ten minutes
    and it is the cheapest insurance available.
    """
    checks = [
        ("roll right", [0.0, +0.5, 0.0, 0.0], "right side should drop / right motors speed up"),
        ("roll left", [0.0, -0.5, 0.0, 0.0], "left side should drop"),
        ("pitch nose-up", [0.0, 0.0, +0.5, 0.0], "nose should RISE / rear motors speed up"),
        ("pitch nose-down", [0.0, 0.0, -0.5, 0.0], "nose should DROP / front motors speed up"),
        ("yaw right", [0.0, 0.0, 0.0, +0.5], "nose should turn right"),
        ("yaw left", [0.0, 0.0, 0.0, -0.5], "nose should turn left"),
        ("climb", [+0.5, 0.0, 0.0, 0.0], "all motors speed up together"),
        ("sink", [-0.5, 0.0, 0.0, 0.0], "all motors slow together"),
    ]
    lines = ["Axis verification, PROPS OFF, one line at a time:", ""]
    for name, action, expect in checks:
        ch, dbg = adapter.action_to_channels(action)
        lines.append(f"  {name:18s} action={action}  ->  channels={ch}")
        lines.append(f"  {'':18s} expect: {expect}")
        lines.append(f"  {'':18s} sticks: roll {dbg['sticks']['roll']:+.3f} "
                     f"pitch {dbg['sticks']['pitch']:+.3f} yaw {dbg['sticks']['yaw']:+.3f} "
                     f"throttle {dbg['sticks']['throttle']:.3f}")
        lines.append("")
    return "\n".join(lines)


def _self_test() -> None:
    a = ControlAdapter(envelope=Envelope(enabled=False))

    # Zero action must be exactly hover with no rotation.
    thrust, rates = decode_action([0, 0, 0, 0])
    assert abs(thrust - THRUST_HOVER) < 1e-12, thrust
    assert np.allclose(rates, 0.0)
    ch, dbg = a.action_to_channels([0, 0, 0, 0])
    assert dbg["pwm"]["roll"] == 1500 and dbg["pwm"]["pitch"] == 1500 and dbg["pwm"]["yaw"] == 1500

    # Thrust decode is piecewise and hits both rails.
    assert abs(decode_action([1, 0, 0, 0])[0] - THRUST_MAX) < 1e-12
    assert abs(decode_action([-1, 0, 0, 0])[0] - THRUST_MIN) < 1e-12

    # Full rate command reaches the policy's limit.
    assert abs(decode_action([0, 1, 0, 0])[1][0] - RATE_LIMIT) < 1e-12

    # Monotonic: more roll command means more roll stick.
    prev = -1e9
    for c in np.linspace(-1, 1, 41):
        _, d = a.action_to_channels([0, c, 0, 0])
        assert d["pwm"]["roll"] >= prev - 1, (c, d["pwm"]["roll"])
        prev = d["pwm"]["roll"]

    # The envelope actually clamps.
    strict = ControlAdapter(envelope=Envelope(max_rate_rad_s=1.0,
                                              max_thrust_stick=0.30,
                                              min_thrust_stick=0.20))
    _, d = strict.action_to_channels([1, 1, 1, 1])
    assert abs(d["rates_rad_s"][0] - 1.0) < 1e-9, d["rates_rad_s"]
    assert abs(d["thrust_stick"] - 0.30) < 1e-9, d["thrust_stick"]

    # Positive pitch rate (nose up) must move the pitch channel above centre
    # with the default signs. If the bench test says otherwise, flip
    # AxisMap.sign_pitch -- do not "fix" it anywhere else.
    _, d = a.action_to_channels([0, 0, 1, 0])
    assert d["pwm"]["pitch"] > 1500, "positive pitch rate should raise the pitch channel"
    _, d = a.action_to_channels([0, 1, 0, 0])
    assert d["pwm"]["roll"] > 1500, "positive roll rate should raise the roll channel"
    _, d = a.action_to_channels([0, 0, 0, 1])
    assert d["pwm"]["yaw"] > 1500, "positive yaw rate should raise the yaw channel"

    # Hover rescaling: if the real drone hovers at 0.40 stick, a hover command
    # must send 0.40, and a "1.5x hover" command must send 0.60.
    resc = ControlAdapter(envelope=Envelope(enabled=False), measured_hover_stick=0.40)
    _, d = resc.action_to_channels([0, 0, 0, 0])
    assert abs(d["thrust_stick_sent"] - 0.40) < 1e-9, d["thrust_stick_sent"]
    half_up = (0.255 * 1.5 - THRUST_HOVER) / (THRUST_MAX - THRUST_HOVER)
    _, d = resc.action_to_channels([half_up, 0, 0, 0])
    assert abs(d["thrust_stick_sent"] - 0.60) < 1e-6, d["thrust_stick_sent"]

    # The idle frame is neutral sticks and minimum throttle.
    assert a.idle_channels()[:4] == [1500, 1500, 1000, 1500]

    # The send hook is called with the channel list.
    sent = []
    hooked = ControlAdapter(send=sent.append)
    hooked.step([0, 0, 0, 0])
    assert len(sent) == 1 and len(sent[0]) == 8

    print("control_adapter: all checks passed")
    print()
    print(verify_axis_map(ControlAdapter(envelope=Envelope(enabled=False))))


if __name__ == "__main__":
    _self_test()
