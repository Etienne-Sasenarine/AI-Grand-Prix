"""A guidance controller that flies the course from the estimated state.

This is two things at once:

1. **Plan B** — the slow, hand-written fallback. If the trained policy is not
   ready, or is not trusted on the day, this flies the course. It will not win,
   but scoring is most gates first and time only breaks ties, so a slow run that
   finishes beats a fast one that stops at gate four.
2. **A test harness for everything else.** It lets the whole perception and
   control chain be flown closed-loop without depending on a good policy, which
   is what makes it possible to say "the system works" as opposed to "the
   plumbing type-checks".

It consumes only what the real aircraft has: the filter's position and velocity
estimate, the flight controller's attitude, and the tracker's current target.
No privileged simulator state.

Structure is a standard cascade, the same shape every multirotor autopilot uses:

    where we want to be   ->  desired velocity
    desired velocity      ->  desired acceleration
    desired acceleration  ->  desired tilt and thrust
    desired tilt          ->  body rate command

The last step is the one worth care: thrust always points along body "up", so a
desired acceleration completely determines the attitude you must hold. Steering
becomes "rotate body-up onto the vector I want", which is a rotation between two
unit vectors — an axis and an angle, no Euler angles and no singularities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import geometry as G

G_ACCEL = 9.80665


@dataclass
class GuidanceConfig:
    """Speeds and gains. The speeds are the Plan B trade-off; the gains are not."""

    cruise_speed: float = 4.0          # m/s between gates
    gate_speed: float = 2.5            # m/s through the opening
    approach_distance: float = 3.5     # m: minimum staging distance
    #: The camera cannot look far below its own axis, so a gate well below us
    #: is invisible until we back off. The staging point is therefore pushed
    #: out until the look-down angle to the gate fits inside the cone. Without
    #: this the aircraft parks above the double gate's lower opening, cannot
    #: see it, stops for want of a fix, and hovers there until it times out --
    #: which is exactly what closed-loop testing showed it doing.
    max_look_down_rad: float = 0.35    # ~20 degrees below the camera axis
    max_approach_distance: float = 12.0
    capture_radius: float = 1.5        # m: "close enough" to the staging point
    #: Position gain: how hard to chase the waypoint. Higher is twitchier.
    kp_pos: float = 1.2
    #: Velocity gain: how hard to correct a velocity error.
    kp_vel: float = 2.2
    #: Attitude gain: desired tilt error to body rate.
    kp_att: float = 6.0
    #: Yaw gain: heading error to yaw rate.
    kp_yaw: float = 2.5
    max_tilt_rad: float = 0.60         # ~34 degrees
    max_rate_rad_s: float = 3.2
    max_accel: float = 6.0             # m/s^2 horizontal command ceiling
    hover_stick: float = 0.255
    min_stick: float = 0.05
    max_stick: float = 0.90
    #: Fly this far past a gate before switching to the next one, so we do not
    #: turn early and clip the frame on the way out.
    exit_distance: float = 1.0


@dataclass
class Guidance:
    """Position-and-velocity guidance to the current target gate."""

    cfg: GuidanceConfig = field(default_factory=GuidanceConfig)
    _staged: bool = False
    _last_gate: object = None

    def reset(self) -> None:
        self._staged = False
        self._last_gate = None

    # ----------------------------------------------------------------- targets
    def waypoint(self, gate: G.Gate, pos: np.ndarray) -> tuple[np.ndarray, float]:
        """Where to fly next, and how fast, for this gate.

        Every gate is approached in two stages: first a staging point on the
        correct side, then the opening itself. Without that, a gate whose
        through-axis points back the way you came — gate 6 on this course, the
        split-S — gets approached from behind and flown through backwards.
        """
        if gate is not self._last_gate:
            self._last_gate = gate
            self._staged = False

        through = gate.through
        centre = gate.centre
        ahead = float(np.dot(np.asarray(pos) - centre, through))

        # Back off far enough that the gate is not below the camera's cone.
        dz = float(np.asarray(pos)[2] - centre[2])
        need = abs(dz) / max(math.tan(self.cfg.max_look_down_rad), 1e-3) if dz > 1.0 else 0.0
        stage_dist = float(np.clip(max(self.cfg.approach_distance, need),
                                   self.cfg.approach_distance,
                                   self.cfg.max_approach_distance))
        stage = centre - through * stage_dist
        if not self._staged:
            close = float(np.linalg.norm(stage - np.asarray(pos))) < self.cfg.capture_radius
            if ahead < -0.5 and close:
                self._staged = True
            else:
                return stage, self.cfg.cruise_speed
        # Aim through the gate, not at it, so we keep flying while crossing.
        return centre + through * self.cfg.exit_distance, self.cfg.gate_speed

    def passed(self, gate: G.Gate, pos: np.ndarray) -> bool:
        return self._staged and float(np.dot(np.asarray(pos) - gate.centre, gate.through)) > 0.2

    # -------------------------------------------------------------- the cascade
    def command(self, *, pos, vel, quat, target: G.Gate | None,
                look_at: np.ndarray | None = None,
                thrust_scale: float = 1.0) -> tuple[float, np.ndarray, dict]:
        """Return (thrust stick, body rate command in NED, diagnostics).

        ``thrust_scale`` is the filter's estimate of how much thrust the
        aircraft actually delivers per unit of commanded thrust. Dividing by it
        is what turns that estimate into something useful: without it the
        filter learns the aircraft is 15 % down on thrust, writes the number
        down, and the controller carries on commanding 15 % too little anyway.
        ``hover_stick`` is the least trustworthy constant in this package -- it
        is a guess until someone hovers the real aircraft and reads the
        throttle -- so closing this loop is what makes being wrong survivable.
        """
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)

        if target is None:
            want_pos, want_speed = pos, 0.0
        else:
            want_pos, want_speed = self.waypoint(target, pos)

        # --- position -> desired velocity
        to_wp = want_pos - pos
        dist = float(np.linalg.norm(to_wp))
        if dist < 1e-6:
            vel_des = np.zeros(3)
        else:
            speed = min(want_speed, self.cfg.kp_pos * dist)
            vel_des = to_wp / dist * speed

        # --- velocity -> desired acceleration
        accel_des = self.cfg.kp_vel * (vel_des - vel)
        horiz = accel_des[:2]
        h_mag = float(np.linalg.norm(horiz))
        if h_mag > self.cfg.max_accel:
            accel_des[:2] = horiz / h_mag * self.cfg.max_accel
        accel_des[2] = float(np.clip(accel_des[2], -0.7 * G_ACCEL, 1.2 * G_ACCEL))

        # --- acceleration -> thrust vector. Thrust must also hold us up.
        thrust_vec = accel_des + np.array([0.0, 0.0, G_ACCEL])
        mag = float(np.linalg.norm(thrust_vec))
        if mag < 1e-6:
            thrust_vec, mag = np.array([0.0, 0.0, G_ACCEL]), G_ACCEL
        up_des = thrust_vec / mag

        # Limit the tilt: beyond this the aircraft trades lift for translation
        # faster than it can afford, and a guidance controller should never ask.
        cos_tilt = float(np.clip(up_des[2], -1.0, 1.0))
        tilt = math.acos(cos_tilt)
        if tilt > self.cfg.max_tilt_rad:
            flat = up_des[:2]
            n = float(np.linalg.norm(flat))
            if n > 1e-9:
                s = math.sin(self.cfg.max_tilt_rad)
                up_des = np.array([flat[0] / n * s, flat[1] / n * s,
                                   math.cos(self.cfg.max_tilt_rad)])
            mag = G_ACCEL / max(math.cos(self.cfg.max_tilt_rad), 1e-3)

        # --- thrust magnitude -> stick. Mass cancels, so this is a pure ratio.
        scale = float(np.clip(thrust_scale, 0.5, 2.0))
        stick = float(np.clip(self.cfg.hover_stick * mag / G_ACCEL / scale,
                              self.cfg.min_stick, self.cfg.max_stick))

        # --- attitude -> body rates
        up_now = _body_up(quat)
        axis_world = np.cross(up_now, up_des)
        sin_a = float(np.linalg.norm(axis_world))
        cos_a = float(np.dot(up_now, up_des))
        angle = math.atan2(sin_a, cos_a)
        if sin_a > 1e-9:
            axis_world = axis_world / sin_a
        else:
            axis_world = np.zeros(3)
        rot_world = axis_world * angle * self.cfg.kp_att

        # Desired heading: look where we are going, or at the gate.
        yaw_now = _yaw(quat)
        if look_at is not None:
            aim = np.asarray(look_at, dtype=np.float64)[:2] - pos[:2]
        elif target is not None:
            aim = target.centre[:2] - pos[:2]
        else:
            aim = np.zeros(2)
        if float(np.linalg.norm(aim)) > 0.5:
            yaw_des = math.atan2(aim[1], aim[0])
            yaw_err = (yaw_des - yaw_now + math.pi) % (2 * math.pi) - math.pi
        else:
            yaw_err = 0.0
        rot_world = rot_world + np.array([0.0, 0.0, self.cfg.kp_yaw * yaw_err])

        # World rotation vector into body FLU, then to the NED convention the
        # policy and the flight controller both use.
        rot_flu = G.quat_rotate_inverse(quat, rot_world.reshape(1, 3))[0]
        rates_ned = G.flu_to_ned(rot_flu)
        rates_ned = np.clip(rates_ned, -self.cfg.max_rate_rad_s, self.cfg.max_rate_rad_s)

        return stick, rates_ned, {
            "waypoint": want_pos, "want_speed": want_speed, "dist": dist,
            "staged": self._staged, "tilt": tilt, "yaw_err": yaw_err,
            "speed": float(np.linalg.norm(vel)),
        }


def _body_up(quat) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x),
                     1.0 - 2.0 * (x * x + y * y)])


def _yaw(quat) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _self_test() -> None:
    import dynamics as D

    # Hovering in place: the controller should hold height and not run away.
    g = Guidance()
    plant = D.Quadrotor()
    plant.reset([0.0, 0.0, 2.0])
    for _ in range(300):
        stick, rates, _ = g.command(pos=plant.pos, vel=plant.vel, quat=plant.quat, target=None)
        plant.step(stick, rates, 1 / 60.0)
    assert abs(plant.pos[2] - 2.0) < 0.35, plant.pos
    assert float(np.linalg.norm(plant.pos[:2])) < 0.5, plant.pos
    assert not plant.crashed

    # Flying to a gate: it should reach and pass through the opening.
    gate = G.official_gates()[0]
    g = Guidance()
    plant = D.Quadrotor()
    start = gate.centre - gate.through * 9.0
    plant.reset(start, yaw=math.atan2(gate.through[1], gate.through[0]))
    passed = False
    for _ in range(1500):
        stick, rates, _ = g.command(pos=plant.pos, vel=plant.vel, quat=plant.quat, target=gate)
        plant.step(stick, rates, 1 / 60.0)
        if g.passed(gate, plant.pos):
            passed = True
            break
    assert passed, f"never reached the gate, ended at {plant.pos}"
    assert not plant.crashed, plant.crash_reason
    side, vert = G.lateral_offsets(gate, plant.pos)
    assert abs(side) < 0.75 and abs(vert) < 0.75, f"missed the opening: {side:.2f}, {vert:.2f}"

    # And it should have gone through the right way round.
    assert float(np.dot(plant.vel, gate.through)) > 0.5, "crossed backwards"

    print("controller: all checks passed")
    print(f"  reached gate 1 from 9 m in {plant.t:.1f} s, "
          f"offset {side:+.2f} m lateral, {vert:+.2f} m vertical")


if __name__ == "__main__":
    _self_test()
