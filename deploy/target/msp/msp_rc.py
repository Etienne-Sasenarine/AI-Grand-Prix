#!/usr/bin/env python3
"""Fixed-rate RC transmitter for Betaflight over MSP_SET_RAW_RC.

Betaflight treats MSP RC exactly like a radio link: it needs a *continuous* stream of
frames, and it triggers failsafe when that stream stops. Everything in here exists to
make sure the stream is steady while the script is healthy and stops cleanly when it is
not.

Safety behaviour, in order of precedence:
  1. Any unhandled exception in the send loop  -> stream stops -> FC failsafe.
  2. No ``set_control()`` call within ``command_timeout_s`` -> sticks neutral, throttle
     to minimum, AUX1 released (disarm). The stream keeps running so the FC stays under
     our control rather than dropping into its own failsafe.
  3. ``max_throttle`` hard-clamps throttle regardless of what the caller asks for.
  4. ``close()`` disarms, sends idle frames, then stops.

Channel order is Betaflight's default ``rcmap = AETR1234``:
    0 Roll   1 Pitch   2 Throttle   3 Yaw   4 AUX1(arm)   5 AUX2   6 AUX3   7 AUX4
Run ``get rcmap`` in the Betaflight CLI to confirm before you fly.
"""

from __future__ import annotations

import threading
import time

from msp import MSPLink

# Standard RC pulse widths in microseconds.
RC_MIN = 1000
RC_MID = 1500
RC_MAX = 2000

# AUX switch levels. Betaflight's default ARM range is 1700-2100.
AUX_LOW = 1000
AUX_HIGH = 1800

IDX_ROLL, IDX_PITCH, IDX_THROTTLE, IDX_YAW, IDX_AUX1, IDX_AUX2, IDX_AUX3, IDX_AUX4 = range(8)


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def norm_to_us(value, span=500):
    """[-1, 1] -> [1000, 2000] microseconds."""
    return int(round(RC_MID + clamp(value, -1.0, 1.0) * span))


def throttle_to_us(value, low=RC_MIN, high=RC_MAX):
    """[0, 1] -> [low, high] microseconds."""
    return int(round(low + clamp(value, 0.0, 1.0) * (high - low)))


class RCTransmitter:
    """Streams MSP_SET_RAW_RC at a fixed rate from a background thread.

    Typical use::

        with MSPLink("/dev/ttyTHS1", 115200) as link:
            tx = RCTransmitter(link, rate_hz=100, max_throttle=0.35).start()
            tx.wait_until_streaming()
            tx.arm()
            while flying:
                tx.set_control(roll=0.0, pitch=-0.1, yaw=0.0, throttle=0.30)
                time.sleep(0.01)
            tx.close()
    """

    def __init__(
        self,
        link: MSPLink,
        rate_hz=100.0,
        channels=8,
        max_throttle=1.0,
        max_angle_cmd=1.0,
        command_timeout_s=0.25,
        arm_channel=IDX_AUX1,
    ):
        if channels < 5:
            raise ValueError("need at least 5 channels so AUX1 can carry ARM")
        self.link = link
        self.rate_hz = float(rate_hz)
        self.period = 1.0 / float(rate_hz)
        self.channels = channels
        self.max_throttle = clamp(max_throttle, 0.0, 1.0)
        self.max_angle_cmd = clamp(max_angle_cmd, 0.0, 1.0)
        self.command_timeout_s = command_timeout_s
        self.arm_channel = arm_channel

        self._lock = threading.Lock()
        self._rc = self._idle_frame()
        self._armed_request = False
        self._last_command_at = 0.0
        self._thread = None
        self._running = False
        self._started_evt = threading.Event()

        self.frames_sent = 0
        self.timeouts = 0
        self.last_error = None

    # -- frame helpers -----------------------------------------------------------------
    def _idle_frame(self):
        rc = [RC_MID] * self.channels
        rc[IDX_THROTTLE] = RC_MIN
        for i in range(IDX_AUX1, self.channels):
            rc[i] = AUX_LOW
        return rc

    # -- lifecycle ---------------------------------------------------------------------
    def start(self):
        if self._thread is not None:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="msp-rc-tx", daemon=True)
        self._thread.start()
        return self

    def wait_until_streaming(self, timeout=2.0):
        """Block until the first frame has gone out (or the loop died)."""
        if not self._started_evt.wait(timeout):
            raise TimeoutError("RC stream did not start")
        if self.last_error is not None:
            raise self.last_error
        return self

    def close(self, disarm_hold_s=0.3):
        """Disarm, hold the idle frame briefly so the FC sees it, then stop the stream."""
        if self._thread is None:
            return
        self.disarm()
        deadline = time.monotonic() + disarm_hold_s
        while time.monotonic() < deadline:
            time.sleep(self.period)
        self._running = False
        self._thread.join(timeout=1.0)
        self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # -- commands ----------------------------------------------------------------------
    def set_control(self, roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0, aux=None):
        """Set the current setpoint. Call this at least every ``command_timeout_s``.

        roll/pitch/yaw: -1..1 (right / forward-nose-down / clockwise, Betaflight sign
        convention). throttle: 0..1, scaled into 1000..2000us and clamped by
        ``max_throttle``. ``aux`` is an optional dict {2: 1800, 3: 1000} of AUX index
        (1-based, AUX1 == 1) to raw microseconds; AUX1 stays owned by the arm state.
        """
        span = int(500 * self.max_angle_cmd)
        with self._lock:
            rc = list(self._rc)
            rc[IDX_ROLL] = norm_to_us(roll, span)
            rc[IDX_PITCH] = norm_to_us(pitch, span)
            rc[IDX_YAW] = norm_to_us(yaw, span)
            rc[IDX_THROTTLE] = throttle_to_us(
                clamp(throttle, 0.0, self.max_throttle)
            )
            if aux:
                for aux_index, value in aux.items():
                    ch = IDX_AUX1 + int(aux_index) - 1
                    if ch == self.arm_channel:
                        continue  # AUX1 belongs to arm()/disarm()
                    if IDX_AUX1 <= ch < self.channels:
                        rc[ch] = int(clamp(value, RC_MIN, RC_MAX))
            self._rc = rc
            self._last_command_at = time.monotonic()

    def arm(self):
        """Raise the ARM switch. Refuses unless throttle is already at minimum."""
        with self._lock:
            if self._rc[IDX_THROTTLE] > RC_MIN + 20:
                raise RuntimeError("refusing to arm: throttle is not at minimum")
            self._armed_request = True
            self._last_command_at = time.monotonic()

    def disarm(self):
        """Drop the ARM switch and cut throttle immediately."""
        with self._lock:
            self._armed_request = False
            self._rc[IDX_THROTTLE] = RC_MIN

    @property
    def arm_requested(self):
        return self._armed_request

    def snapshot(self):
        with self._lock:
            return list(self._rc)

    # -- the stream --------------------------------------------------------------------
    def _loop(self):
        next_tick = time.monotonic()
        try:
            while self._running:
                now = time.monotonic()

                with self._lock:
                    stale = (
                        self._last_command_at
                        and now - self._last_command_at > self.command_timeout_s
                    )
                    if stale:
                        # Lost the controller. Neutralise but keep the link alive so the
                        # FC stays disarmed under our command instead of failsafing.
                        self._rc = self._idle_frame()
                        self._armed_request = False
                        self.timeouts += 1
                    rc = list(self._rc)
                    rc[self.arm_channel] = AUX_HIGH if self._armed_request else AUX_LOW

                self.link.set_raw_rc(rc)
                self.frames_sent += 1
                self._started_evt.set()

                next_tick += self.period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()  # we fell behind; resync, do not burst
        except BaseException as exc:  # noqa: BLE001 - surfaced via wait_until_streaming
            self.last_error = exc
            self._started_evt.set()
            raise
        finally:
            self._running = False
