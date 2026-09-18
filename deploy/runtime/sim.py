"""A small synthetic flight, for testing the parts that cannot be flown yet.

This is deliberately *not* a physics simulator — Isaac already does that. It
flies a kinematic path through the real course and produces the things the
onboard code consumes: camera detections with realistic defects, attitude, and
gyro. That is enough to answer the questions that matter here:

  * does the gate tracker count the right number of crossings, in order?
  * does it survive dropouts, noise, false positives and a neighbouring gate
    appearing in shot?
  * does the observation pipeline produce sane values along a whole run?

The detector model is the important part. The simulator's projection is
perfect, so the defects are added deliberately:

  * ``pixel_noise``     Gaussian error per corner, growing with range
  * ``dropout``         probability a whole detection is missed
  * ``corner_dropout``  probability an individual corner is missed
  * ``max_range_m``     beyond this the gate is not detected at all
  * ``false_positives``  spurious detections per second (pylons, signage)
  * ``latency_frames``  detections arrive this many frames late
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import geometry as G
from gate_tracker import Detection


@dataclass
class DetectorModel:
    pixel_noise_px: float = 1.5
    noise_growth_per_m: float = 0.12
    dropout: float = 0.05
    corner_dropout: float = 0.03
    max_range_m: float = 14.0
    min_range_m: float = 0.6
    false_positives_per_s: float = 0.0
    latency_frames: int = 0
    behind_gate_visible: bool = True

    def observe(self, gates, pos, quat, camera, rng, dt) -> list[Detection]:
        out: list[Detection] = []
        for g in gates:
            d = float(np.linalg.norm(g.centre - np.asarray(pos)))
            if d > self.max_range_m or d < self.min_range_m:
                continue
            if not self.behind_gate_visible and G.signed_distance_through(g, pos) > 0.0:
                continue
            uv, vis = camera.project(g.corners_world(), pos, quat)
            if not vis.any():
                continue
            if rng.random() < self.dropout:
                continue
            sigma = self.pixel_noise_px + self.noise_growth_per_m * d
            uv = uv + rng.normal(0.0, sigma, size=uv.shape)
            vis = vis & (rng.random(vis.shape) > self.corner_dropout)
            if not vis.any():
                continue
            out.append(Detection(uv=uv, visible=vis, confidence=1.0))

        n_fp = rng.poisson(self.false_positives_per_s * dt)
        for _ in range(int(n_fp)):
            centre = rng.uniform([0, 0], [camera.frame_w, camera.frame_h])
            size = rng.uniform(20, 120)
            uv = centre + rng.uniform(-size, size, size=(8, 2))
            out.append(Detection(uv=uv, visible=np.ones(8, dtype=bool), confidence=0.5))
        return out


@dataclass
class FlightModel:
    """Flies the sequence by steering toward a point short of each gate.

    First-order lag on velocity so the path is smooth and the attitude is
    plausible, rather than a series of straight lines.
    """

    speed: float = 4.0
    lag_s: float = 0.35
    #: How far before the opening the staging point sits.
    stage_distance_m: float = 4.0
    #: How close to the staging point counts as staged.
    stage_radius_m: float = 2.5
    lateral_error_m: float = 0.0
    vertical_error_m: float = 0.0
    seed: int = 0

    def run(self, sequence, dt: float = 1.0 / 60.0, max_time: float = 400.0):
        """Yield (t, pos, quat, vel_world, gyro_ned, target_index) per step.

        Each gate is flown as two waypoints: first a staging point on the
        *approach* side, then the opening itself. Without that, a gate whose
        through-axis points back the way you came — gate 6 is the obvious one,
        the split-S — gets approached from behind and "passed" backwards. The
        real aircraft has the same problem and needs the same answer.
        """
        rng = np.random.default_rng(self.seed)
        offsets = [
            (rng.normal(0.0, self.lateral_error_m), rng.normal(0.0, self.vertical_error_m))
            for _ in sequence
        ]

        first = sequence[0]
        pos = first.centre - first.through * 7.0
        pos[2] = first.height
        vel = np.zeros(3)
        prev_vel = np.zeros(3)
        yaw = math.atan2(first.through[1], first.through[0])
        prev_yaw = yaw
        t = 0.0
        i = 0
        staged = False

        while i < len(sequence) and t < max_time:
            g = sequence[i]
            side, vert = offsets[i]
            opening = g.centre + g.right * side + np.array([0.0, 0.0, vert])
            through_d = G.signed_distance_through(g, pos)

            if not staged:
                stage = opening - g.through * self.stage_distance_m
                to_stage = stage - pos
                # Staged once we are behind the gate plane and reasonably close
                # to the approach point.
                if through_d < -0.5 and float(np.linalg.norm(to_stage)) < self.stage_radius_m:
                    staged = True
                    aim = opening + g.through * 1.0
                else:
                    aim = stage
            else:
                aim = opening + g.through * 1.0

            to = aim - pos
            dist = float(np.linalg.norm(to))
            if dist < 1e-6:
                i += 1
                staged = False
                continue
            want = to / dist * self.speed
            vel += (want - vel) * min(1.0, dt / self.lag_s)
            pos = pos + vel * dt

            if np.linalg.norm(vel[:2]) > 0.2:
                yaw = math.atan2(vel[1], vel[0])
            yaw_rate = _wrap(yaw - prev_yaw) / dt
            prev_yaw = yaw

            # Attitude from the acceleration the aircraft is actually
            # producing, not a heuristic. A quadrotor's thrust acts along body
            # "up", so body up must point along (acceleration + gravity). An
            # attitude inconsistent with the acceleration makes it impossible to
            # dead-reckon from attitude and thrust -- which is exactly how the
            # first version of this file invalidated the pose-filter test.
            accel = (vel - prev_vel) / dt
            prev_vel = vel.copy()
            required = accel + np.array([0.0, 0.0, 9.80665])
            n = float(np.linalg.norm(required))
            if n < 1e-6:
                required, n = np.array([0.0, 0.0, 1.0]), 1.0
            up = required / n
            cy_, sy_ = math.cos(yaw), math.sin(yaw)
            # Invert up = [cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr]
            roll = math.asin(float(np.clip(up[0] * sy_ - up[1] * cy_, -1.0, 1.0)))
            pitch = math.atan2(up[0] * cy_ + up[1] * sy_, up[2])
            quat = G.quat_from_euler(roll, pitch, yaw)
            gyro_ned = np.array([0.0, 0.0, yaw_rate])

            yield t, pos.copy(), quat, vel.copy(), gyro_ned, i

            if staged and G.signed_distance_through(g, pos) > 0.15:
                i += 1
                staged = False
            t += dt


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class SimResult:
    steps: int = 0
    true_crossings: int = 0
    counted: int = 0
    correct_order: bool = True
    wrong_gate: int = 0
    max_index_error: int = 0
    lag_index: int = 0
    lost_frames: int = 0
    confident_frames: int = 0
    reassociations: int = 0
    events: list = field(default_factory=list)


def run_tracker_sim(tracker, *, detector=None, flight=None, camera=None,
                    dt=1.0 / 60.0, seed=0, give_pose=True, pose_noise_m=0.0,
                    verbose=False) -> SimResult:
    """Fly the sequence and score the tracker against ground truth."""
    detector = detector or DetectorModel()
    flight = flight or FlightModel()
    camera = camera or tracker.camera
    rng = np.random.default_rng(seed)
    gates = tracker.sequence
    res = SimResult()

    pending: list[list[Detection]] = []

    for t, pos, quat, vel, gyro, true_i in flight.run(gates, dt=dt):
        res.steps += 1
        res.true_crossings = max(res.true_crossings, true_i)

        # Only gates that could plausibly be in shot, plus the neighbours, so
        # the tracker has to discriminate rather than being handed one answer.
        window = gates[max(0, true_i - 1): true_i + 3]
        dets = detector.observe(window, pos, quat, camera, rng, dt)

        pending.append(dets)
        if len(pending) > detector.latency_frames:
            dets_now = pending.pop(0)
        else:
            dets_now = []

        if give_pose:
            p = pos + rng.normal(0.0, pose_noise_m, size=3) if pose_noise_m else pos
            ev = tracker.update(dt, dets_now, p, quat)
        else:
            ev = tracker.update(dt, dets_now, None, None)

        if tracker.state.name == "LOST":
            res.lost_frames += 1
        if tracker.confident:
            res.confident_frames += 1
        if ev is not None:
            res.events.append((t, ev, true_i))
            res.counted += 1
            # "Wrong" means the tracker's own bookkeeping is inconsistent: the
            # k-th declared crossing must be sequence[k]. Comparing against the
            # flight model's index at the instant the event fires measures
            # *detection lag*, not correctness -- the vision fallback
            # legitimately fires a fraction of a second after the crossing.
            k = res.counted - 1
            if k < len(gates) and (ev.index != k or ev.gate_number != gates[k].number):
                res.wrong_gate += 1
            res.lag_index = max(res.lag_index, abs(ev.index - true_i))
            res.max_index_error = res.lag_index
            if verbose:
                print(f"    t={t:6.2f} counted gate {ev.gate_number}"
                      f"{' (upper)' if ev.is_double_upper else ''} by {ev.by}"
                      f"  true index {true_i}, tracker index {ev.index}")

    res.reassociations = len(tracker.reassociations)
    res.correct_order = all(
        e.index == k for k, (_, e, _) in enumerate(res.events)
    )
    return res
