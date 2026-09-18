"""Where are we? A small filter that carries position between gate sightings.

We have no VIO: one forward camera, an IMU at 50 Hz over a polled serial link,
and no barometer worth using (measured elsewhere: altitude falls ~6 m within 6 s
of the props spinning up). So position has to come from gate sightings, and
something has to bridge the gaps between them.

Measured on this course, which is what makes the approach viable:

  * a gate is in view **75-79 %** of frames at a 12 m detector range
  * the worst gap with nothing in view is **1.5 s at 5 m/s**, 0.8 s at 12 m/s
  * dead reckoning at 10 % velocity error drifts **0.8-1.0 m** across that gap
  * the nearest two gates on the course are **7.54 m** apart

So worst-case uncertainty is around a metre against a 7.5 m separation. That is
why a light filter is enough here and full VIO is not needed — gates arrive
often enough that drift never gets going.

Design
------
The awkward part of a full EKF here is that the dynamics are non-linear in
attitude. But attitude is *measured*, not estimated — the flight controller
gives it to us. So treat it as a control input and the state stays linear:

    state  = [position (3), velocity (3), thrust scale (1)]
    input  = specific thrust along the measured body-z, plus gravity and drag

``thrust_scale`` is the ratio between commanded and actual thrust. Estimating it
is what makes the barometer unnecessary: vision tells us our height, and the
filter works backwards to how much thrust actually produces a newton.

Yaw is handled separately, as a slowly-varying bias on the flight controller's
own yaw, corrected by the *bearing* residual of gate fixes. Gyro yaw drifts;
gate sightings pin it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import geometry as G

G_ACCEL = 9.80665


@dataclass
class FilterConfig:
    #: Process noise: how much we distrust the dead reckoning, per second.
    accel_noise: float = 2.0          # m/s^2, covers thrust and drag error
    thrust_scale_noise: float = 0.02  # per sqrt(s), battery sag drifts slowly
    # Gyro heading drift. This was originally 0.01, which made the filter's
    # yaw gain about 0.005 -- it effectively never corrected heading, and a
    # 3 deg/s drift test blew position error out to 11.65 m. Heading has no
    # absolute reference except the gates, so it must be free to move.
    yaw_bias_noise: float = 0.02      # rad/sqrt(s)
    #: Assumed noise on the bearing residual of a gate fix, in radians.
    yaw_meas_var: float = 0.01
    yaw_var_max: float = 0.30
    #: Measurement noise on a gate fix. From the PnP study: 0.09 m at 8 m with
    #: 2 px of corner noise, 0.29 m at 5 px, growing with range.
    fix_noise_base: float = 0.10      # m
    fix_noise_per_m: float = 0.03     # m of error per m of range
    #: Reject a fix this far from the prediction.
    #:
    #: This was 3.0 m, justified by "the nearest gates are 7.54 m apart". That
    #: spacing excluded the double gate, whose two openings are the same
    #: structure **2.70 m apart** — so the threshold was wider than the
    #: ambiguity it existed to resolve, and a fix taken against the wrong
    #: opening sailed through it. Measured: the estimate stepped 2.70 m at the
    #: double gate, after which the aircraft flew over the lower opening
    #: believing it was lined up on it.
    #:
    #: A threshold has to sit below the closest confusable pair (2.70 m) and
    #: above the honest fix error (0.2–0.5 m at racing ranges). 1.5 m is the
    #: middle of that. The sigma-scaled term below still lets it open up when
    #: the filter genuinely does not know where it is.
    gate_reject_m: float = 1.5
    drag_k: float = 0.30              # linear drag, 1/s. Measure and replace.
    #: After this long with no accepted fix the estimate is not trustworthy.
    #: Long enough to cover a deliberate manoeuvre with nothing in view -- the
    #: turnaround between the double gate's two openings takes several seconds
    #: with the structure behind the aircraft, and measured drift across such a
    #: gap is 1-2 m against a 7.54 m gate separation. A much shorter limit was
    #: tried while chasing a runaway that turned out to be a mirrored
    #: acceleration in the prediction; with that fixed, the short limit only
    #: made the aircraft stop and hover for want of a sighting it was never
    #: going to get.
    max_coast_s: float = 2.5
    #: The floor. A drone cannot be underground, and saying so is free.
    #: Without this the estimator happily integrates downwards while the
    #: aircraft is still sitting on the pad during the throttle ramp, and by
    #: the time it lifts off the estimate is tens of metres out.
    floor_z: float = 0.0
    #: Do not adapt the thrust scale below this height. On the pad the ground
    #: pushes back, and that force is not in the model, so the filter explains
    #: "full thrust but no motion" by deciding the motors are weak. Measured it
    #: learning 0.59 against a true 1.0 during a throttle ramp, which then made
    #: the aircraft fly its climb on a 40 %-wrong thrust model and hit the
    #: ground. Weight-on-wheels is the standard answer; height is our proxy.
    thrust_learn_min_height: float = 0.40


@dataclass
class PoseFilter:
    """Position, velocity and thrust scale, corrected by gate fixes."""

    cfg: FilterConfig = field(default_factory=FilterConfig)

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    thrust_scale: float = 1.0
    yaw_bias: float = 0.0

    p: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.05]))
    yaw_var: float = 0.05

    t: float = 0.0
    _yaw_ref: float = 0.0
    last_fix_t: float = -1e9
    n_fixes: int = 0
    n_rejected: int = 0
    initialised: bool = False

    # ------------------------------------------------------------------ setup
    def initialise(self, pos, vel=(0.0, 0.0, 0.0), *, pos_sigma: float = 0.10) -> None:
        """Start from a known pose.

        The race begins in a fixed box at a known position, which is what
        removes the need for global relocalisation: we never have to work out
        where we are from scratch, only keep track.
        """
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.vel = np.asarray(vel, dtype=np.float64).copy()
        self.p = np.diag([pos_sigma**2] * 3 + [0.10] * 3 + [0.02])
        self.initialised = True
        self.last_fix_t = self.t

    @property
    def coasting_s(self) -> float:
        return self.t - self.last_fix_t

    @property
    def confident(self) -> bool:
        return self.initialised and self.coasting_s <= self.cfg.max_coast_s

    @property
    def airborne(self) -> bool:
        """Are we clear of the ground? Gates the thrust-scale estimate."""
        return bool(self.pos[2] > self.cfg.floor_z + self.cfg.thrust_learn_min_height)

    @property
    def position_sigma(self) -> float:
        return float(np.sqrt(max(np.trace(self.p[:3, :3]) / 3.0, 0.0)))

    # ---------------------------------------------------------------- predict
    def predict(self, dt: float, *, roll: float, pitch: float, yaw: float,
                specific_thrust: float) -> None:
        """Dead-reckon one step.

        ``specific_thrust`` is the commanded thrust divided by mass, in m/s^2 —
        i.e. what the autopilot is *asking* for. ``thrust_scale`` corrects it to
        what the aircraft actually delivers.
        """
        if dt <= 0:
            return
        self.t += dt
        if not self.initialised:
            return

        # Thrust acts along body "up", through the rotors. The attitude arrives
        # in the flight controller's gravity-referenced convention, so it must
        # be converted with the shared helper. Building this inline from a Euler
        # formula mirrors the horizontal components — AHRS pitch is the negation
        # of the FLU Euler pitch — which looks fine vertically while pushing the
        # predicted acceleration the wrong way across the ground.
        self._yaw_ref = yaw
        yaw_eff = yaw + self.yaw_bias
        up = G.body_up_from_ahrs(roll, pitch, yaw_eff)
        accel = up * (specific_thrust * self.thrust_scale)
        accel[2] -= G_ACCEL
        accel -= self.cfg.drag_k * self.vel

        self.pos = self.pos + self.vel * dt + 0.5 * accel * dt * dt
        self.vel = self.vel + accel * dt

        # Ground constraint. Hard physical knowledge, so apply it hard.
        if self.pos[2] < self.cfg.floor_z:
            self.pos[2] = self.cfg.floor_z
            self.vel[2] = max(self.vel[2], 0.0)

        # Covariance: F for [pos, vel, thrust_scale].
        f = np.eye(7)
        f[0:3, 3:6] = np.eye(3) * dt
        f[3:6, 3:6] = np.eye(3) * (1.0 - self.cfg.drag_k * dt)
        f[3:6, 6] = up * specific_thrust * dt
        q = np.zeros((7, 7))
        q[3:6, 3:6] = np.eye(3) * (self.cfg.accel_noise * dt) ** 2
        q[6, 6] = ((self.cfg.thrust_scale_noise * math.sqrt(dt)) ** 2
                   if self.airborne else 0.0)
        self.p = f @ self.p @ f.T + q
        self.yaw_var = min(self.yaw_var + (self.cfg.yaw_bias_noise * math.sqrt(dt)) ** 2,
                           self.cfg.yaw_var_max)

    # ----------------------------------------------------------------- update
    def update_gate_fix(self, fix_pos, *, range_m: float, gate=None) -> bool:
        """Fold in a position fix from a gate sighting. Returns False if rejected."""
        if not self.initialised:
            self.initialise(fix_pos)
            return True

        z = np.asarray(fix_pos, dtype=np.float64)
        innov = z - self.pos
        # A rejection gate that never opens is a trap: once the filter drifts
        # past it, every fix looks like an outlier and it can never recover.
        # So the gate scales with our own uncertainty, and after a long coast
        # without any accepted fix we trust vision over dead reckoning and
        # re-initialise outright.
        reject_at = max(self.cfg.gate_reject_m, 3.0 * self.position_sigma)
        if self.coasting_s > self.cfg.max_coast_s:
            self.initialise(z, self.vel, pos_sigma=0.5)
            self.n_fixes += 1
            return True
        if float(np.linalg.norm(innov)) > reject_at:
            self.n_rejected += 1
            return False

        sigma = self.cfg.fix_noise_base + self.cfg.fix_noise_per_m * float(range_m)
        r = np.eye(3) * sigma**2
        h = np.zeros((3, 7))
        h[:, :3] = np.eye(3)
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        dx = k @ innov
        self.pos = self.pos + dx[:3]
        self.vel = self.vel + dx[3:6]
        if self.airborne:
            self.thrust_scale = float(np.clip(self.thrust_scale + dx[6], 0.5, 2.0))
        self.p = (np.eye(7) - k @ h) @ self.p
        self.last_fix_t = self.t
        self.n_fixes += 1

        return True

    def update_yaw_from_bearing(self, gate, bearing_body_rad: float) -> None:
        """Correct heading from where a gate actually appears, in the body frame.

        This is the measurement that genuinely observes yaw. The earlier
        version compared bearings in the *world* frame, which measures the
        rotation of our position error about the gate -- a quantity that has
        nothing to do with heading, and which made the filter worse when it was
        given any authority.

        ``bearing_body_rad`` is the horizontal angle to the gate **measured
        from our own heading in a level frame** — build it with
        ``geometry.level_bearing_from_ahrs``, never with a bare ``atan2`` on
        body-frame components. A body-frame angle is measured in a frame that
        is rolled and pitched, so comparing it with a world bearing reports
        heading error that is not there, signed with the bank angle. On a
        course that turns mostly one way the filter integrates that into a
        steadily growing bias; measured, it reached +21 degrees by the seventh
        gate, which moved the predicted gate 163 px in the image and cost the
        tracker its lock.
        """
        if not self.initialised:
            return
        to_gate = np.asarray(gate.centre, dtype=np.float64)[:2] - self.pos[:2]
        if float(np.linalg.norm(to_gate)) < 1.0:
            return
        predicted_world = math.atan2(to_gate[1], to_gate[0])
        # Observed world bearing = our heading + where it sits in the body frame.
        observed_world = self._yaw_ref + self.yaw_bias + bearing_body_rad
        d = (observed_world - predicted_world + math.pi) % (2 * math.pi) - math.pi
        gain = self.yaw_var / (self.yaw_var + self.cfg.yaw_meas_var)
        self.yaw_bias -= gain * d
        self.yaw_var *= (1.0 - gain)
    def summary(self) -> str:
        return (f"pos ({self.pos[0]:.2f}, {self.pos[1]:.2f}, {self.pos[2]:.2f}) "
                f"sigma {self.position_sigma:.2f} m, thrust scale {self.thrust_scale:.3f}, "
                f"{self.n_fixes} fixes, {self.n_rejected} rejected, "
                f"{'confident' if self.confident else 'LOST'}")
