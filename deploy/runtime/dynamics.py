"""A quadrotor plant, so the stack can be flown closed-loop.

Everything tested so far has been open-loop: a fixed path that ignores what the
controller asks for. That is enough to check plumbing and perception, but it
cannot answer "does this actually fly". This module closes the loop.

What it models
--------------
The same plant the policy was trained against, plus the things reality adds:

  * **Collective thrust along body-up.** The autopilot's throttle stick maps to
    thrust through the hover point, exactly as ``rate_control.stick_to_newtons``
    does: ``a = (stick / hover_stick) * g``. Note the mass cancels — see the
    note below, it is the single most misunderstood thing about this plant.
  * **A first-order body-rate loop.** Betaflight closes the rate loop internally
    at kilohertz; from the outside it behaves like a lag with a time constant of
    a few tens of milliseconds, saturated by the airframe's maximum angular
    acceleration. Both are measurable from a blackbox step response, which is
    why they are parameters here rather than buried constants.
  * **Linear drag**, **thrust scale error**, **command latency**, and a floor.

What it does not model: rotor aerodynamics near gate frames, ground effect,
battery sag within a run, or prop wash. Those are Isaac's territory, and some of
them nobody models.

.. note::
   **Mass cancels.** Thrust is ``(stick/hover) * m * g`` and acceleration is
   force over mass, so the mass divides straight back out. Changing the drone's
   mass alone changes nothing about how it flies here. What does not cancel is
   the *thrust-to-weight ceiling* (how much stick is available above hover),
   rotational inertia, and drag. This is why "set the mass to 1745 g and
   retrain" is not a fix, and why doing it carelessly — updating inertia without
   re-deriving the rate loop — makes the simulated aircraft 13x too sluggish.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import geometry as G

G_ACCEL = 9.80665

THRUST_MIN = 0.05
THRUST_HOVER = 0.255
THRUST_MAX = 0.90
RATE_LIMIT = 3.2  # rad/s at full stick deflection


@dataclass
class PlantConfig:
    """Physical parameters. Every one of these is measurable on the bench."""

    #: Throttle stick that holds a hover. The simulator's 0.255 implies a
    #: thrust-to-weight of 3.5; measure it on the real aircraft.
    hover_stick: float = THRUST_HOVER
    #: Ratio of delivered to commanded thrust. 1.0 means the model is exact.
    thrust_scale: float = 1.0
    #: Closed-loop body-rate response, seconds. From a blackbox step response.
    rate_tau_s: float = 0.045
    #: Airframe angular-acceleration ceiling, rad/s^2, per axis.
    max_ang_accel: tuple[float, float, float] = (78.0, 78.0, 26.0)
    #: Linear drag, 1/s.
    drag_k: float = 0.35
    #: Command latency in control steps, i.e. how stale the drone's actions are.
    latency_steps: int = 0
    #: Below this height the aircraft has hit the floor.
    floor_z: float = 0.0
    gravity: float = G_ACCEL


@dataclass
class Quadrotor:
    """Rigid-body quadrotor flown by collective thrust and body-rate commands.

    World frame is x east, y north, z up. Attitude is held as a wxyz quaternion
    in body FLU, matching the rest of the package.
    """

    cfg: PlantConfig = field(default_factory=PlantConfig)

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    #: Body rates in NED (roll, pitch, yaw), rad/s. What the gyro reports.
    rates: np.ndarray = field(default_factory=lambda: np.zeros(3))

    t: float = 0.0
    crashed: bool = False
    crash_reason: str = ""
    _cmd_queue: deque = field(default_factory=deque, repr=False)

    # ------------------------------------------------------------------ setup
    def reset(self, pos, yaw: float = 0.0) -> None:
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.vel = np.zeros(3)
        self.quat = G.quat_from_euler(0.0, 0.0, yaw)
        self.rates = np.zeros(3)
        self.t = 0.0
        self.crashed = False
        self.crash_reason = ""
        self._cmd_queue.clear()

    # ------------------------------------------------------------- properties
    @property
    def roll_pitch(self) -> tuple[float, float]:
        return G.roll_pitch_from_quat(self.quat)

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @property
    def body_up_world(self) -> np.ndarray:
        """The thrust direction (body FLU +z) in world coordinates.

        Taken straight from the quaternion rather than rebuilt from Euler
        angles. Going via Euler invites a convention mismatch — the reported
        roll and pitch use a gravity-referenced AHRS convention to match the
        training code, which is *not* the same as a ZYX Euler decomposition.
        """
        w, x, y, z = self.quat
        return np.array([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ])

    @property
    def velocity_body_ned(self) -> np.ndarray:
        """Velocity in the body NED frame, which is what the policy reads."""
        v_flu = G.quat_rotate_inverse(self.quat, self.vel.reshape(1, 3))[0]
        return G.flu_to_ned(v_flu)

    def specific_thrust(self, thrust_stick: float) -> float:
        """Commanded thrust as an acceleration, m/s^2. Mass cancels."""
        return (thrust_stick / self.cfg.hover_stick) * self.cfg.gravity

    # ------------------------------------------------------------------- step
    def step(self, thrust_stick: float, rates_cmd_ned, dt: float, *, substeps: int = 2) -> None:
        """Advance by ``dt`` under a thrust stick and body-rate command.

        ``rates_cmd_ned`` is (roll, pitch, yaw) rate in rad/s, the same
        convention the policy emits.
        """
        if self.crashed:
            return

        # Latency: the aircraft acts on a command issued some steps ago.
        self._cmd_queue.append((float(thrust_stick), np.asarray(rates_cmd_ned, dtype=np.float64).copy()))
        if len(self._cmd_queue) > self.cfg.latency_steps:
            thrust_stick, rates_cmd = self._cmd_queue.popleft()
        else:
            thrust_stick, rates_cmd = self.cfg.hover_stick, np.zeros(3)

        h = dt / max(1, substeps)
        limits = np.asarray(self.cfg.max_ang_accel, dtype=np.float64)

        for _ in range(max(1, substeps)):
            # --- rate loop: first-order lag, saturated by airframe authority
            err = np.asarray(rates_cmd, dtype=np.float64) - self.rates
            ang_accel = np.clip(err / max(self.cfg.rate_tau_s, 1e-4), -limits, limits)
            self.rates = self.rates + ang_accel * h

            # --- attitude integration. Rates are NED; the quaternion is FLU.
            omega_flu = G.flu_to_ned(self.rates)  # involution
            self.quat = _integrate_quat(self.quat, omega_flu, h)

            # --- translation
            accel = self.body_up_world * (self.specific_thrust(thrust_stick) * self.cfg.thrust_scale)
            accel[2] -= self.cfg.gravity
            accel -= self.cfg.drag_k * self.vel
            self.vel = self.vel + accel * h
            self.pos = self.pos + self.vel * h
            self.t += h

            if self.pos[2] <= self.cfg.floor_z:
                self.pos[2] = self.cfg.floor_z
                if self.vel[2] < -1.5:
                    self.crashed = True
                    self.crash_reason = "ground impact"
                self.vel[2] = max(self.vel[2], 0.0)
                break


def _integrate_quat(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Integrate a wxyz quaternion under a body angular rate."""
    w = np.asarray(omega_body, dtype=np.float64)
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        return q
    axis = w / n
    half = 0.5 * n * dt
    dq = np.array([math.cos(half), *(axis * math.sin(half))])
    out = _quat_mul(q, dq)
    return out / max(np.linalg.norm(out), 1e-12)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _self_test() -> None:
    # Hover: at the hover stick, a level aircraft should not accelerate.
    q = Quadrotor()
    q.reset([0.0, 0.0, 2.0])
    for _ in range(120):
        q.step(THRUST_HOVER, [0, 0, 0], 1 / 60.0)
    assert abs(q.pos[2] - 2.0) < 1e-6, q.pos
    assert np.allclose(q.vel, 0.0, atol=1e-6), q.vel

    # Mass cancels: hover holds regardless of what mass you imagine.
    # (There is no mass in this plant at all, which is the point.)

    # More thrust climbs, less sinks.
    q.reset([0.0, 0.0, 5.0])
    for _ in range(60):
        q.step(THRUST_HOVER * 1.2, [0, 0, 0], 1 / 60.0)
    assert q.pos[2] > 5.05, q.pos
    q.reset([0.0, 0.0, 5.0])
    for _ in range(60):
        q.step(THRUST_HOVER * 0.8, [0, 0, 0], 1 / 60.0)
    assert q.pos[2] < 4.95, q.pos

    # A yaw-rate command turns the aircraft at about that rate.
    q.reset([0.0, 0.0, 2.0], yaw=0.0)
    for _ in range(60):
        q.step(THRUST_HOVER, [0, 0, 1.0], 1 / 60.0)
    assert 0.80 < abs(q.yaw) < 1.10, q.yaw

    # Rate response reaches ~63 % of a step within one time constant.
    q.reset([0.0, 0.0, 2.0])
    tau = q.cfg.rate_tau_s
    n = max(1, int(round(tau / (1 / 240.0))))
    for _ in range(n):
        q.step(THRUST_HOVER, [1.0, 0, 0], 1 / 240.0, substeps=1)
    assert 0.5 < q.rates[0] < 0.8, q.rates

    # Angular acceleration is capped by the airframe.
    q.reset([0.0, 0.0, 2.0])
    q.cfg.max_ang_accel = (10.0, 10.0, 10.0)
    q.step(THRUST_HOVER, [50.0, 0, 0], 1 / 60.0, substeps=1)
    assert q.rates[0] <= 10.0 / 60.0 + 1e-9, q.rates

    # Pitch sign, established from first principles and then asserted:
    # body NED has x forward, y right, z down. A pitch rate is a rotation about
    # +y (the right wing). By the right-hand rule that carries +z (down) toward
    # +x (forward), so the nose (+x) swings toward -z, i.e. UP. Positive pitch
    # rate is therefore NOSE UP, and nose up must push the aircraft BACKWARD.
    q = Quadrotor()
    q.reset([0.0, 0.0, 20.0], yaw=0.0)
    for _ in range(30):
        q.step(THRUST_HOVER, [0, 0.8, 0], 1 / 60.0)
    for _ in range(60):
        q.step(THRUST_HOVER * 1.3, [0, 0, 0], 1 / 60.0)
    assert q.vel[0] < -0.5, f"nose-up pitch must accelerate backwards, got {q.vel}"
    up_back = q.body_up_world[0]

    # ...and the opposite command must fly forward.
    q.reset([0.0, 0.0, 20.0], yaw=0.0)
    for _ in range(30):
        q.step(THRUST_HOVER, [0, -0.8, 0], 1 / 60.0)
    for _ in range(60):
        q.step(THRUST_HOVER * 1.3, [0, 0, 0], 1 / 60.0)
    assert q.vel[0] > 0.5, f"nose-down pitch must accelerate forwards, got {q.vel}"
    assert q.body_up_world[0] > 0 > up_back, "thrust vector must lean the way we fly"

    # The reported pitch must agree in sign with the physics: leaning forward
    # (flying +x) is a nose-DOWN attitude, which the AHRS convention reports as
    # negative pitch.
    _, pitch_fwd = q.roll_pitch
    assert pitch_fwd < 0, f"flying forward should report negative pitch, got {pitch_fwd}"

    # Roll sign: positive roll rate is right-wing-down, which pushes right.
    # In world terms with yaw=0 (nose along +x), right is -y.
    q.reset([0.0, 0.0, 20.0], yaw=0.0)
    for _ in range(30):
        q.step(THRUST_HOVER, [0.8, 0, 0], 1 / 60.0)
    for _ in range(60):
        q.step(THRUST_HOVER * 1.3, [0, 0, 0], 1 / 60.0)
    assert q.vel[1] < -0.5, f"positive roll rate must accelerate to the right (-y), got {q.vel}"
    roll_r, _ = q.roll_pitch
    assert roll_r > 0, f"right-wing-down should report positive roll, got {roll_r}"

    # Drag limits speed: held lean reaches a terminal velocity.
    q = Quadrotor(cfg=PlantConfig(drag_k=0.5))
    q.reset([0.0, 0.0, 10.0], yaw=0.0)
    for _ in range(20):
        q.step(THRUST_HOVER, [0, 0.9, 0], 1 / 60.0)
    speeds = []
    for _ in range(600):
        q.step(THRUST_HOVER * 1.25, [0, 0, 0], 1 / 60.0)
        speeds.append(float(np.linalg.norm(q.vel[:2])))
    assert speeds[-1] < 40.0 and abs(speeds[-1] - speeds[-60]) < 0.5, speeds[-1]

    # Latency delays the response by the configured number of steps.
    q = Quadrotor(cfg=PlantConfig(latency_steps=5))
    q.reset([0.0, 0.0, 5.0])
    for i in range(4):
        q.step(THRUST_HOVER, [2.0, 0, 0], 1 / 60.0)
    assert abs(q.rates[0]) < 1e-9, "command should not have taken effect yet"
    for i in range(4):
        q.step(THRUST_HOVER, [2.0, 0, 0], 1 / 60.0)
    assert q.rates[0] > 0.1, "command should have taken effect by now"

    # Hitting the ground hard is a crash; settling gently is not.
    q = Quadrotor()
    q.reset([0.0, 0.0, 3.0])
    for _ in range(600):
        q.step(THRUST_MIN, [0, 0, 0], 1 / 60.0)
        if q.crashed:
            break
    assert q.crashed and q.crash_reason == "ground impact"

    print("dynamics: all checks passed")


if __name__ == "__main__":
    _self_test()
