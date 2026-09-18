"""The flight loop: the state machine that actually runs a race.

This is where every other module meets. Once per control step it:

    read attitude and gyro from the flight controller
      -> read detections from the camera
      -> solve a relative pose for the target gate  (pnp)
      -> update where we think we are            (pose_filter)
      -> update which gate is next               (gate_tracker)
      -> ask the controller or the policy what to do
      -> clamp it to the safety envelope         (control_adapter)
      -> send it                                 (the organizers' transmitter)
      -> log everything

Design decisions that came from measurements, not taste
-------------------------------------------------------
* **Takeoff does not try to detect liftoff.** Gyro-spread liftoff detection was
  measured failing in both directions on this airframe — firing while still on
  the pad at one threshold, and never firing at another while the aircraft
  "lifted and ran away". So takeoff is an open ramp to a known height, flown on
  the filter, which starts from the known start box.
* **Losing the gate count means holding, not guessing.** A frozen gate index
  scored 1.8 gates per crash in simulation; a wrong one scored 1.0. So when the
  tracker is not confident we hold position rather than race on a guess.
* **The watchdog is separate from the control loop.** The failure it guards
  against is the control loop stopping, so it cannot live inside it. The
  organizers' ``RCTransmitter`` already provides this; ``Watchdog`` here mirrors
  it for offline testing and for anything that does not go through them.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

import geometry as G
from control_adapter import ControlAdapter, Envelope
from controller import Guidance, GuidanceConfig
from gate_tracker import GateTracker, TrackState
from pnp import camera_position_world_from_attitude, solve_gate_pose
from pose_filter import PoseFilter


class Phase(Enum):
    IDLE = "idle"
    TAKEOFF = "takeoff"
    RACING = "racing"
    HOLD = "hold"
    LANDING = "landing"
    FINISHED = "finished"
    ABORT = "abort"


@dataclass
class LoopConfig:
    control_hz: float = 60.0
    takeoff_height: float = 1.35        # the gate centre height
    takeoff_ramp_s: float = 1.5
    takeoff_timeout_s: float = 10.0
    #: Climb rate during takeoff, m/s. Deliberately gentle: the handover to
    #: racing has to happen with the vertical speed already arrested, or the
    #: aircraft sails over the first gate and loses sight of it.
    takeoff_climb_rate: float = 0.8
    #: Proportional and integral gains on climb-rate error, as a fraction of
    #: hover stick. The integral term is the important one: it is an online
    #: estimate of the hover stick itself.
    #:
    #: Without it the takeoff law could only ever ask for ``hover_stick`` times
    #: 1.24, so an aircraft 15 % down on thrust never left the pad -- and it
    #: could not learn its thrust scale either, because that estimate is gated
    #: on being airborne. Measured: a 15 % thrust error produced a run that sat
    #: on the ground for 47 seconds and scored zero. ``hover_stick`` is a guess
    #: until someone hovers the real aircraft, so takeoff must not assume it.
    takeoff_climb_kp: float = 0.30
    takeoff_climb_ki: float = 0.15
    #: How far the integrator may move the stick, up and down, from hover.
    takeoff_trim_limit: float = 0.30
    #: Hand over only when both of these are satisfied.
    takeoff_height_tol: float = 0.12
    takeoff_speed_tol: float = 0.25
    #: How long the tracker may be unconfident before we stop racing and hold.
    hold_after_lost_s: float = 0.15
    #: Give up holding and land after this long with no recovery.
    hold_timeout_s: float = 10.0
    run_timeout_s: float = 240.0
    #: Boundaries. Outside these the run is aborted.
    max_height: float = 6.0
    min_height: float = 0.25
    margin_m: float = 4.0               # beyond the track edges
    max_speed: float = 25.0
    #: Reject a PnP solution whose reprojection error is worse than this.
    max_reprojection_px: float = 25.0
    #: Position uncertainty above which we stop *counting* crossings, though we
    #: keep associating. See GateTracker.update.
    max_counting_sigma: float = 1.0


class Watchdog:
    """Fails safe if the control loop stops feeding it.

    Runs on its own timer thread, because the thing it protects against is the
    control loop not running. Betaflight's MSP override holds the last stick
    values indefinitely with no timeout of its own, so if our process dies at
    speed the aircraft keeps flying that command into a wall.
    """

    def __init__(self, timeout_s: float = 0.1, on_timeout=None) -> None:
        self.timeout_s = float(timeout_s)
        self.on_timeout = on_timeout
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.tripped = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()

    def pet(self) -> None:
        with self._lock:
            self._last = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.timeout_s / 4.0):
            with self._lock:
                late = time.monotonic() - self._last
            if late > self.timeout_s and not self.tripped:
                self.tripped = True
                if self.on_timeout is not None:
                    self.on_timeout()


@dataclass
class FlightLoop:
    """One race, start to finish."""

    camera: G.Camera = field(default_factory=G.Camera.real)
    sequence: list = field(default_factory=lambda: G.race_sequence(laps=2))
    cfg: LoopConfig = field(default_factory=LoopConfig)
    guidance: Guidance = field(default_factory=Guidance)
    adapter: ControlAdapter = field(default_factory=ControlAdapter)
    filt: PoseFilter = field(default_factory=PoseFilter)
    tracker: GateTracker | None = None
    #: Optional trained policy. When present it flies; otherwise guidance does.
    policy: object = None
    obs_buffer: object = None

    phase: Phase = Phase.IDLE
    t: float = 0.0
    start_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    start_yaw: float = 0.0
    phase_t: float = 0.0
    hold_since: float | None = None
    abort_reason: str = ""
    #: Takeoff throttle trim, learned on the way up. See _takeoff.
    _climb_trim: float = 0.0
    log: list = field(default_factory=list)

    # Last reported attitude, refreshed every step.
    _roll: float = 0.0
    _pitch: float = 0.0
    _yaw: float = 0.0
    _gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        if self.tracker is None:
            self.tracker = GateTracker(sequence=self.sequence, camera=self.camera)

    # ------------------------------------------------------------------ set-up
    def arm(self, start_pos, start_yaw: float = 0.0) -> None:
        """Begin, from the known start box. This is what removes the need for
        global relocalisation: we never work out where we are from scratch."""
        self.start_pos = np.asarray(start_pos, dtype=np.float64).copy()
        self.start_yaw = float(start_yaw)
        self.filt.initialise(self.start_pos, (0.0, 0.0, 0.0))
        self.guidance.reset()
        self.phase = Phase.TAKEOFF
        self.phase_t = 0.0
        self.t = 0.0
        self.hold_since = None
        self.abort_reason = ""
        self._climb_trim = 0.0

    def _to(self, phase: Phase) -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_t = 0.0

    # -------------------------------------------------------------------- step
    def step(self, dt: float, *, roll: float, pitch: float, yaw: float,
             gyro_ned=(0.0, 0.0, 0.0), detections=None,
             specific_thrust_cmd: float = 0.0):
        """Advance one control step and return (thrust stick, rates NED, info).

        ``specific_thrust_cmd`` is the thrust we asked for on the *previous*
        step, expressed as an acceleration (m/s^2). The filter needs it to dead
        reckon; without it the prediction is just ballistic.
        """
        self.t += dt
        self.phase_t += dt
        self._roll, self._pitch, self._yaw = float(roll), float(pitch), float(yaw)
        self._gyro = np.asarray(gyro_ned, dtype=np.float64)

        # --- estimate where we are ------------------------------------------
        self.filt.predict(dt, roll=roll, pitch=pitch, yaw=yaw,
                          specific_thrust=specific_thrust_cmd)
        quat = G.quat_from_ahrs(roll, pitch, yaw + self.filt.yaw_bias)
        fix_used = None

        # Association happens exactly once, inside the tracker, using the
        # filter's *predicted* pose. Doing it again here with a second rule was
        # measured matching a detection of the gate behind us to the gate
        # ahead, which put a 9.8 m error straight into the filter.
        # Always give the tracker a pose, even a stale one. Confidence gates
        # whether we *race*, not whether we can *associate*: a rough pose still
        # tells you which gate is roughly where, and withholding it was
        # measured preventing the tracker from ever re-acquiring after a single
        # missed pass. The acceptance window widens on its own as the estimate
        # ages, which is the right way to express the growing uncertainty.
        pose_for_tracker = self.filt.pos if self.filt.initialised else None
        self.tracker.update(dt, detections, pose_for_tracker,
                            quat if self.filt.initialised else None,
                            pose_trusted=self.filt.confident
                            and self.filt.position_sigma < self.cfg.max_counting_sigma)

        target = self.tracker.expected
        det = self.tracker.accepted
        if det is not None and target is not None:
            gate_pose = solve_gate_pose(det.uv, det.visible, self.camera,
                                        roll=roll, pitch=pitch)
            if gate_pose is not None and gate_pose.reprojection_px <= self.cfg.max_reprojection_px:
                fix = camera_position_world_from_attitude(gate_pose, target, quat, self.camera)
                ok = self.filt.update_gate_fix(fix, range_m=gate_pose.distance, gate=target)
                # Whether the fix was consistent is the tracker's best evidence
                # that it is looking at the gate it thinks it is.
                self.tracker.note_fix(ok, range_m=gate_pose.distance)
                if ok:
                    fix_used = fix
                    # Where the gate sits relative to the body, from the PnP
                    # translation (well conditioned, unlike the PnP rotation).
                    r_cb = G.camera_body_rotation(self.camera.tilt_up_deg)
                    p_flu = (G.flu_to_ned(r_cb.T @ gate_pose.t_cam)
                             + np.asarray(self.camera.offset_body_flu))
                    # Must be tilt-compensated before it can be compared with a
                    # world bearing from the map. See level_bearing_from_ahrs:
                    # the raw body-frame angle reports phantom heading error
                    # proportional to bank, and this course banks mostly one
                    # way, so the yaw filter integrated it to +21 degrees.
                    self.filt.update_yaw_from_bearing(
                        target, G.level_bearing_from_ahrs(p_flu, roll, pitch))

        # --- safety ----------------------------------------------------------
        abort = self._check_bounds()
        if abort:
            self._to(Phase.ABORT)
            self.abort_reason = abort

        # --- decide ----------------------------------------------------------
        if self.phase is Phase.TAKEOFF:
            stick, rates = self._takeoff(dt)
        elif self.phase is Phase.RACING:
            stick, rates = self._race()
        elif self.phase is Phase.HOLD:
            stick, rates = self._hold()
        elif self.phase in (Phase.LANDING, Phase.ABORT):
            stick, rates = self._land()
        else:
            stick, rates = self.adapter.envelope.min_thrust_stick, np.zeros(3)

        info = {
            "t": self.t, "phase": self.phase.value,
            "pos": self.filt.pos.copy(), "sigma": self.filt.position_sigma,
            "gate_index": self.tracker.index,
            "tracker": self.tracker.state.value,
            "confident": self.filt.confident,
            "tracker_confident": self.tracker.confident,
            "thrust_scale": self.filt.thrust_scale,
            "fix": fix_used,
            "stick": stick, "rates": np.asarray(rates).copy(),
        }
        self.log.append(info)
        return stick, np.asarray(rates), info

    # ------------------------------------------------------------------ phases
    def _takeoff(self, dt: float):
        """Open ramp, then a controlled climb. Deliberately does not detect liftoff.

        Two stages. First a slow throttle ramp, so the aircraft leaves the
        ground gently even though the thrust scale is not yet known. Then a
        climb at a commanded *rate* to the target height.

        The handover to racing waits for the vertical speed to be arrested, not
        just for the height to be reached. Handing over mid-climb was measured
        putting the aircraft 2.3 m above the first gate, which took the gate out
        of the camera's view and starved the filter of fixes.
        """
        cfg = self.cfg
        target_z = float(self.start_pos[2]) + cfg.takeoff_height

        if self.phase_t < cfg.takeoff_ramp_s:
            frac = self.phase_t / cfg.takeoff_ramp_s
            floor = self.adapter.envelope.min_thrust_stick
            top = self.guidance.cfg.hover_stick * 1.05
            return float(floor + frac * (top - floor)), np.zeros(3)

        # Climb-rate control: ask for a gentle rate, taper it near the target.
        # The integral term trims away whatever ``hover_stick`` got wrong, so
        # this works without knowing the aircraft's real hover throttle.
        err_z = target_z - float(self.filt.pos[2])
        want_vz = float(np.clip(1.5 * err_z, -cfg.takeoff_climb_rate, cfg.takeoff_climb_rate))
        vz_err = want_vz - float(self.filt.vel[2])
        hover = self.guidance.cfg.hover_stick
        # Drive the trim from *height* progress, not from the estimated climb
        # rate. The climb rate is dead-reckoned through the thrust model, which
        # is the very thing that is wrong when this matters: with 15 % less
        # thrust than modelled, the filter believed it was climbing at
        # 1.7 m/s^2 while the aircraft sat on the pad, so the climb-rate error
        # stayed near zero and the integrator never wound up. Height is pinned
        # by gate fixes and owes the thrust model nothing.
        self._climb_trim = float(np.clip(
            self._climb_trim + cfg.takeoff_climb_ki * hover * err_z * dt,
            -cfg.takeoff_trim_limit, cfg.takeoff_trim_limit))
        stick = float(np.clip(hover * (1.0 + cfg.takeoff_climb_kp * vz_err) + self._climb_trim,
                              self.guidance.cfg.min_stick, self.guidance.cfg.max_stick))

        # Hold heading on the first gate while climbing, so it is in view the
        # moment we start racing.
        gate = self.tracker.expected
        _, rates, _ = self.guidance.command(
            pos=self.filt.pos, vel=self.filt.vel,
            quat=G.quat_from_ahrs(*self._attitude()),
            target=None, look_at=gate.centre if gate is not None else None,
            thrust_scale=self.filt.thrust_scale)

        settled = (abs(err_z) < cfg.takeoff_height_tol
                   and abs(float(self.filt.vel[2])) < cfg.takeoff_speed_tol)
        if settled or self.phase_t > cfg.takeoff_timeout_s:
            self._to(Phase.RACING)
        return stick, rates

    def _race(self):
        if self.tracker.expected is None:
            self._to(Phase.FINISHED)
            return self.guidance.cfg.hover_stick, np.zeros(3)

        # What matters for racing is whether we know *where we are*, not
        # whether a gate happens to be in view right now. After a confirmed
        # crossing the index is known, and with a good position estimate and
        # the surveyed map the next gate is a lookup — we can fly toward it
        # blind across a turn. Requiring a live sighting made the aircraft stop
        # and hold every time a gate went out of frame, which on this course is
        # after every corner, and it then timed out and landed.
        confident = self.filt.confident
        if not confident:
            if self.hold_since is None:
                self.hold_since = self.t
            if self.t - self.hold_since > self.cfg.hold_after_lost_s:
                self._to(Phase.HOLD)
                return self._hold()
        else:
            self.hold_since = None

        quat = G.quat_from_ahrs(*self._attitude())
        stick, rates, _ = self.guidance.command(
            pos=self.filt.pos, vel=self.filt.vel, quat=quat,
            target=self.tracker.expected, thrust_scale=self.filt.thrust_scale)
        return stick, rates

    def _hold(self):
        """Stop and wait for the count to recover. Better than guessing."""
        if self.filt.confident:
            self._to(Phase.RACING)
            self.hold_since = None
            return self._race()
        if self.phase_t > self.cfg.hold_timeout_s:
            self._to(Phase.LANDING)
            self.abort_reason = "lost the gate count for too long"
        quat = G.quat_from_ahrs(*self._attitude())
        stick, rates, _ = self.guidance.command(
            pos=self.filt.pos, vel=self.filt.vel, quat=quat, target=None,
            thrust_scale=self.filt.thrust_scale)
        return stick, rates

    def _land(self):
        """Controlled descent. Not a motor cut — that is the pilot's switch."""
        sink = 0.6
        err = -sink - float(self.filt.vel[2])
        hover = self.guidance.cfg.hover_stick / max(self.filt.thrust_scale, 0.5)
        stick = float(np.clip(hover + 0.05 * err,
                              self.guidance.cfg.min_stick, self.guidance.cfg.max_stick))
        if self.filt.pos[2] <= 0.15:
            self._to(Phase.FINISHED)
            return self.guidance.cfg.min_stick, np.zeros(3)
        return stick, np.zeros(3)

    # ------------------------------------------------------------------ helpers
    def _attitude(self) -> tuple[float, float, float]:
        """The attitude to steer from: measured roll and pitch, corrected yaw.

        Yaw carries the filter's estimated bias because the flight controller's
        heading drifts and the gates are the only absolute reference for it.
        """
        return (self._roll, self._pitch, self._yaw + self.filt.yaw_bias)

    def _check_bounds(self) -> str:
        if self.phase in (Phase.ABORT, Phase.FINISHED, Phase.LANDING):
            return ""
        p = self.filt.pos
        if self.t > self.cfg.run_timeout_s:
            return "run timeout"
        if p[2] > self.cfg.max_height:
            return "above the height ceiling"
        if float(np.linalg.norm(self.filt.vel)) > self.cfg.max_speed:
            return "over the speed limit"
        m = self.cfg.margin_m
        if not (-m <= p[0] <= G.TRACK_WIDTH_M + m and -m <= p[1] <= G.TRACK_LENGTH_M + m):
            return "outside the track boundary"
        return ""

    @property
    def crossings(self) -> int:
        """Plane crossings declared, clean or not."""
        return len(self.tracker.passes)

    @property
    def scoring_crossings(self) -> int:
        """Crossings that actually went through the opening.

        The competition scores gates *passed*. A crossing more than 0.75 m off
        centre does not clear a 1.5 m opening -- it hits the frame -- so it is
        not a pass, and counting it would flatter the result.

        Corroboration is the second requirement, and it exists because the
        first one is self-assessed: "did I go through the opening" is answered
        from our own position estimate, which is exactly what is wrong when it
        is wrong. A crossing we never ranged close to is not something to put
        in a score.
        """
        return sum(1 for e in self.tracker.passes if e.clean and e.corroborated)

    def summary(self) -> str:
        return (f"{self.phase.value} at t={self.t:.1f}s, {self.crossings} crossings, "
                f"{self.tracker.summary()}"
                + (f", ABORT: {self.abort_reason}" if self.abort_reason else ""))
