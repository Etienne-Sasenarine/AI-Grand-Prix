"""Convert between physical units and Betaflight stick values.

The problem
-----------
Our policy speaks physics: collective thrust, and body rotation rates in radians
per second. Betaflight speaks sticks: four numbers between 1000 and 2000, which
it then puts through its *own* non-linear curves to get a rate setpoint and a
throttle. To command 2.0 rad/s we have to work out which stick value Betaflight
will turn into 2.0 rad/s.

A linear approximation is wrong by tens of percent in the middle of the range,
which is where most flying happens.

Confidence, stated honestly
---------------------------
The *inverse* is solid: it is a bisection over a monotonic function, so it is
correct for whatever forward curve you give it.

The *forward* curves below are written from the Betaflight 4.x source as best
understood, and are marked UNVERIFIED. Two of them have alternative forms in
circulation. **Do not fly on the analytic model without checking it against the
real firmware.** Two ways to check, in increasing order of trust:

1. Betaflight Configurator draws the rate curve and prints the maximum rate for
   the current settings. Compare against ``RateCurve.max_rate_deg_s()``.
2. Command a stick value, read the resulting setpoint from a blackbox log, and
   feed the pairs to ``RateCurve.fit_from_samples()``. That replaces the
   analytic model with a measured one and makes the question moot.

Everything here is pure NumPy, so it runs on the Jetson as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PWM_MIN = 1000
PWM_MAX = 2000
PWM_MID = 1500

# ---------------------------------------------------------------------------
# Read from AI of Sauron's Betaflight dump of the same organizer-supplied drone
# (bilbo, 2026-09-16). Real settings on real hardware of the same model, but
# THEIR airframe -- replace from our own `diff all`.
#
#   set rates_type = BETAFLIGHT      <- confirms the formula family below
#   set roll_rc_rate = 55   set roll_srate = 75
#   set pitch_rc_rate = 55  set pitch_srate = 75
#   set yaw_rc_rate = 57    set yaw_srate = 70
#   set quickrates_rc_expo = OFF
#   set thr_mid = 54        set thr_expo = 68
# ---------------------------------------------------------------------------
DEFAULT_RC_RATE = 0.55
DEFAULT_SUPER_RATE = 0.75
DEFAULT_EXPO = 0.0
DEFAULT_YAW_RC_RATE = 0.57
DEFAULT_YAW_SUPER_RATE = 0.70
DEFAULT_THR_MID = 54  # 0-100
DEFAULT_THR_EXPO = 68  # 0-100


def default_yaw_curve() -> "RateCurve":
    """Yaw is geared differently from roll and pitch: 380 deg/s, not 440."""
    return RateCurve(rc_rate=DEFAULT_YAW_RC_RATE, super_rate=DEFAULT_YAW_SUPER_RATE)


def _bisect_inverse(fn, target: float, lo: float, hi: float, iters: int = 40) -> float:
    """Invert a monotonically increasing function by bisection.

    Correct for any monotonic ``fn``, which is why the accuracy of the forward
    model does not affect the correctness of this step. 40 iterations resolves
    the stick range to far below one PWM count.
    """
    f_lo, f_hi = fn(lo), fn(hi)
    if target <= f_lo:
        return lo
    if target >= f_hi:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class RateCurve:
    """Betaflight "BETAFLIGHT" rates: stick position to rotation rate.

    ``rc_rate``, ``super_rate`` and ``expo`` are the CLI values divided by 100,
    i.e. the numbers Configurator shows.
    """

    rc_rate: float = DEFAULT_RC_RATE
    super_rate: float = DEFAULT_SUPER_RATE
    expo: float = DEFAULT_EXPO
    #: Betaflight 4.x applies expo as ``x*|x|^3*e + x*(1-e)``. Older documentation
    #: and some third-party tools use the cubic ``x^3*e + x*(1-e)``. They differ
    #: only when expo is non-zero -- which is why expo defaults to 0 here.
    quartic_expo: bool = True
    measured: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    def rate_deg_s(self, stick: float) -> float:
        """Stick in [-1, 1] to rotation rate in degrees per second. UNVERIFIED."""
        if self.measured is not None:
            xs, ys = self.measured
            return float(np.interp(stick, xs, ys))

        x = float(np.clip(stick, -1.0, 1.0))
        ax = abs(x)
        if self.expo:
            shaped = ax**3 if self.quartic_expo else x * x  # multiplied by x below
            x = x * shaped * self.expo + x * (1.0 - self.expo) if self.quartic_expo \
                else x * x * x * self.expo + x * (1.0 - self.expo)
        rc_rate = self.rc_rate
        if rc_rate > 2.0:  # Betaflight's extended-rate kink
            rc_rate += 14.54 * (rc_rate - 2.0)
        rate = 200.0 * rc_rate * x
        if self.super_rate:
            factor = 1.0 / float(np.clip(1.0 - ax * self.super_rate, 0.01, 1.0))
            rate *= factor
        return rate

    def max_rate_deg_s(self) -> float:
        """Rate at full stick. Cross-check this against Betaflight Configurator."""
        return self.rate_deg_s(1.0)

    def stick_from_rate_deg_s(self, rate: float) -> float:
        """Rotation rate in degrees per second to stick in [-1, 1]."""
        sign = -1.0 if rate < 0 else 1.0
        target = abs(float(rate))
        x = _bisect_inverse(lambda s: self.rate_deg_s(s), target, 0.0, 1.0)
        return sign * x

    def stick_from_rate_rad_s(self, rate: float) -> float:
        return self.stick_from_rate_deg_s(np.degrees(rate))

    def fit_from_samples(self, sticks, rates_deg_s) -> "RateCurve":
        """Replace the analytic curve with a measured lookup table.

        ``sticks`` in [-1, 1], ``rates_deg_s`` the observed setpoint. Once this
        is set, the analytic model is unused and the UNVERIFIED caveat is gone.
        """
        xs = np.asarray(sticks, dtype=float)
        ys = np.asarray(rates_deg_s, dtype=float)
        order = np.argsort(xs)
        self.measured = (xs[order], ys[order])
        return self


@dataclass
class ThrottleCurve:
    """Betaflight throttle shaping: ``thr_mid`` / ``thr_expo``.

    Betaflight builds a 12-point lookup table and interpolates between the
    points. UNVERIFIED -- reproduced from the 4.x source. Check by commanding a
    stick value and reading back the motor output.
    """

    thr_mid: int = DEFAULT_THR_MID
    thr_expo: int = DEFAULT_THR_EXPO
    measured: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    _LOOKUP_LEN = 12

    def _table(self) -> np.ndarray:
        mid, expo = int(self.thr_mid), int(self.thr_expo)
        out = np.zeros(self._LOOKUP_LEN)
        for i in range(self._LOOKUP_LEN):
            tmp = 10 * i - mid
            y = 1
            if tmp > 0:
                y = 100 - mid
            elif tmp < 0:
                y = mid
            val = 10 * mid + tmp * (100 - expo + expo * (tmp * tmp) / (y * y)) / 10.0
            out[i] = PWM_MIN + (PWM_MAX - PWM_MIN) * val / 1000.0
        return out

    def throttle_from_stick(self, stick: float) -> float:
        """Stick in [0, 1] to the effective throttle in [0, 1]. UNVERIFIED."""
        if self.measured is not None:
            xs, ys = self.measured
            return float(np.interp(stick, xs, ys))
        s = float(np.clip(stick, 0.0, 1.0))
        table = self._table()
        # Entries 0..10 span 0..100 % throttle (entry i is throttle 10*i %).
        # Entry 11 exists only so that interpolation at exactly full stick has
        # an upper neighbour; it is above PWM_MAX and is never reached.
        span = self._LOOKUP_LEN - 2  # == 10
        pos = s * span
        i = int(np.clip(np.floor(pos), 0, span - 1))
        frac = pos - i
        pwm = table[i] + frac * (table[i + 1] - table[i])
        return (pwm - PWM_MIN) / (PWM_MAX - PWM_MIN)

    def stick_from_throttle(self, throttle: float) -> float:
        """Effective throttle in [0, 1] to the stick that produces it."""
        return _bisect_inverse(self.throttle_from_stick, float(np.clip(throttle, 0.0, 1.0)), 0.0, 1.0)

    def fit_from_samples(self, sticks, throttles) -> "ThrottleCurve":
        xs = np.asarray(sticks, dtype=float)
        ys = np.asarray(throttles, dtype=float)
        order = np.argsort(xs)
        self.measured = (xs[order], ys[order])
        return self


# ---------------------------------------------------------------------------
# Channel values
# ---------------------------------------------------------------------------

def stick_to_pwm(stick: float, *, bipolar: bool = True) -> int:
    """Stick to a 1000-2000 channel value.

    ``bipolar`` for roll/pitch/yaw (centre 1500); ``bipolar=False`` for
    throttle, where 0 maps to 1000.
    """
    if bipolar:
        v = PWM_MID + float(np.clip(stick, -1.0, 1.0)) * (PWM_MAX - PWM_MID)
    else:
        v = PWM_MIN + float(np.clip(stick, 0.0, 1.0)) * (PWM_MAX - PWM_MIN)
    return int(round(float(np.clip(v, PWM_MIN, PWM_MAX))))


def pwm_to_stick(pwm: int, *, bipolar: bool = True) -> float:
    if bipolar:
        return float(np.clip((pwm - PWM_MID) / (PWM_MAX - PWM_MID), -1.0, 1.0))
    return float(np.clip((pwm - PWM_MIN) / (PWM_MAX - PWM_MIN), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    r = RateCurve()

    # Anchor: with rc_rate 0.55 and super_rate 0.75, full stick should be about
    # 440 deg/s. This is the number to cross-check in Configurator.
    assert abs(r.max_rate_deg_s() - 440.0) < 1.0, r.max_rate_deg_s()

    # Monotonic, which is what makes the inverse valid.
    xs = np.linspace(-1, 1, 401)
    ys = np.array([r.rate_deg_s(x) for x in xs])
    assert np.all(np.diff(ys) > -1e-9), "rate curve is not monotonic"

    # Round trip: rate -> stick -> rate.
    for target in [0.0, 5.0, 25.0, 90.0, 183.0, 300.0, 439.0]:
        s = r.stick_from_rate_deg_s(target)
        assert abs(r.rate_deg_s(s) - target) < 0.05, (target, s, r.rate_deg_s(s))
        s_neg = r.stick_from_rate_deg_s(-target)
        assert abs(r.rate_deg_s(s_neg) + target) < 0.05

    # Saturation is clamped, not wrapped.
    assert r.stick_from_rate_deg_s(10_000.0) == 1.0
    assert r.stick_from_rate_deg_s(-10_000.0) == -1.0

    # The policy's envelope is +-3.2 rad/s = 183 deg/s. Note how far up the
    # stick that actually is: the super-rate curve is deliberately flat in the
    # middle, so 42 % of the maximum rate costs 74 % of the stick.
    s = r.stick_from_rate_rad_s(3.2)
    assert 0.70 < s < 0.80, s

    # The linear guess -- rate / max_rate -- is off by more than 40 %. This is
    # exactly the error this module exists to prevent.
    linear = 3.2 / np.radians(r.max_rate_deg_s())
    assert abs(linear - s) / s > 0.40, "expected the linear guess to be badly off"

    # Non-zero expo stays monotonic in both expo conventions.
    for quartic in (True, False):
        re = RateCurve(expo=0.3, quartic_expo=quartic)
        ys = np.array([re.rate_deg_s(x) for x in np.linspace(-1, 1, 201)])
        assert np.all(np.diff(ys) > -1e-9), f"expo curve not monotonic (quartic={quartic})"
        assert abs(re.rate_deg_s(1.0) - r.max_rate_deg_s()) < 1e-6, "expo must not move full stick"

    # Measured curves override the analytic one.
    r2 = RateCurve().fit_from_samples([-1, -0.5, 0, 0.5, 1], [-400, -100, 0, 100, 400])
    assert abs(r2.rate_deg_s(0.5) - 100) < 1e-6
    assert abs(r2.stick_from_rate_deg_s(100) - 0.5) < 1e-3

    t = ThrottleCurve()
    ts = np.array([t.throttle_from_stick(x) for x in np.linspace(0, 1, 201)])
    assert np.all(np.diff(ts) > -1e-9), "throttle curve is not monotonic"
    assert abs(t.throttle_from_stick(0.0)) < 1e-6
    assert abs(t.throttle_from_stick(1.0) - 1.0) < 1e-6
    for target in [0.05, 0.25, 0.4, 0.55, 0.8]:
        ts_stick = t.stick_from_throttle(target)
        assert abs(t.throttle_from_stick(ts_stick) - target) < 1e-3, (target, ts_stick)

    assert stick_to_pwm(0.0) == 1500
    assert stick_to_pwm(1.0) == 2000
    assert stick_to_pwm(-1.0) == 1000
    assert stick_to_pwm(0.0, bipolar=False) == 1000
    assert stick_to_pwm(1.0, bipolar=False) == 2000
    for pwm in (1000, 1250, 1500, 1750, 2000):
        assert stick_to_pwm(pwm_to_stick(pwm)) == pwm

    yaw = default_yaw_curve()
    assert abs(yaw.max_rate_deg_s() - 380.0) < 1.0, yaw.max_rate_deg_s()
    assert yaw.max_rate_deg_s() < r.max_rate_deg_s(), "yaw should be geared lower"

    print("betaflight_curves: all checks passed")
    print(f"  yaw full stick    = {yaw.max_rate_deg_s():.1f} deg/s (roll/pitch {r.max_rate_deg_s():.1f})")
    print(f"  full stick        = {r.max_rate_deg_s():.1f} deg/s")
    rate_stick = r.stick_from_rate_rad_s(3.2)
    print(f"  policy limit 3.2 rad/s = {np.degrees(3.2):.1f} deg/s -> stick {rate_stick:.4f} "
          f"(PWM {stick_to_pwm(rate_stick)})")
    print(f"  a linear guess would give stick {linear:.4f} -- off by "
          f"{100 * abs(linear - rate_stick) / rate_stick:.0f} %, "
          f"i.e. {stick_to_pwm(rate_stick) - stick_to_pwm(linear)} PWM counts")
    print(f"  stick 0.50 -> {r.rate_deg_s(0.5):.1f} deg/s (linear guess would say "
          f"{0.5 * r.max_rate_deg_s():.1f})")
    print(f"  hover throttle 0.255 stick -> effective {t.throttle_from_stick(0.255):.3f}")


if __name__ == "__main__":
    _self_test()
