"""Work out which gate is next, from the camera alone.

Why this exists
---------------
The policy's observation includes an 18-slot one-hot saying "this is the gate
you are flying at". In the simulator that was handed over for free. On the real
course nothing publishes it, and **we may not put markers on the gates**, so
identity has to come from geometry and sequence.

The hard part is not seeing a gate. It is knowing *which* gate you are seeing,
and noticing the moment you have gone through it — because gate 9 is a stacked
double gate that **counts twice per lap**, and one physical structure carrying
two different crossing numbers is exactly where a counter loses its place.

How it works
------------
1. The race sequence is fixed and known, so we always have an *expectation*.
2. Each frame, predict where the expected gate should appear, from the current
   pose estimate and the surveyed map. Accept a detection only if it lands near
   that prediction — this is what stops a neighbouring gate being mistaken for
   the target.
3. Declare a pass when the drone crosses the gate's plane going forwards, while
   inside the opening. With no pose, fall back to the visual signature of a
   pass: the gate grows, fills the view, then leaves downward or sideways.
4. A short lockout after each pass stops one crossing counting twice.
5. If the expected gate is not seen for a while, widen the search, then hold.
   **Never silently re-label** — every re-association is recorded.

Confidence in a count is as important as the count. ``state`` reports whether
the tracker is locked on, coasting, or lost, and the flight loop should slow
down or hold rather than fly blind on a guessed index.

A measured limit: a pose is not optional
----------------------------------------
On this course, **34 % of frames have two or more gates in view at once**
(measured over a simulated two-lap flight). Without a pose estimate there is no
principled way to tell which one is the target, and the vision-only fallback
was measured drifting **13 crossings ahead of reality** while still declaring a
plausible-looking 22 passes. That is the worst kind of failure: confident and
wrong.

So the vision-only path is a *degraded* mode, not an alternative. It keeps a
best-effort count for the log, but ``confident`` is False throughout, and the
flight loop must treat that as a reason to hold rather than to keep racing on a
guessed gate index. The normal path has a pose: the detector gives eight
corners, the gate geometry is known, so PnP gives gate-relative pose, and the
surveyed map turns that into a world pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

import geometry as G


class TrackState(Enum):
    LOCKED = "locked"      # seeing the expected gate where it should be
    COASTING = "coasting"  # not seeing it, but recently did; dead-reckoning
    SEARCHING = "searching"  # lost it for a while; acceptance widened
    LOST = "lost"          # no confident association; the loop should hold
    FINISHED = "finished"  # the whole sequence is done


@dataclass
class Detection:
    """One gate seen by the detector.

    ``uv`` is (8, 2) pixels in the trained corner order; ``visible`` is the
    per-corner mask. ``confidence`` is the detector's own score.
    """

    uv: np.ndarray
    visible: np.ndarray
    confidence: float = 1.0

    def centroid(self) -> tuple[float, float] | None:
        m = np.asarray(self.visible, dtype=bool)
        if not m.any():
            return None
        pts = np.asarray(self.uv, dtype=np.float64)[m]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def span(self) -> float | None:
        """Apparent size in pixels. Prefers the outer ring, which encodes range."""
        m = np.asarray(self.visible, dtype=bool)
        outer = m.copy()
        outer[4:] = False
        use = outer if outer.sum() >= 2 else m
        if use.sum() < 2:
            return None
        pts = np.asarray(self.uv, dtype=np.float64)[use]
        return float(math.hypot(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))))


@dataclass
class TrackerConfig:
    #: Accept a detection whose centroid is within this many pixels of the
    #: predicted centroid. Grows while coasting, shrinks when locked.
    accept_radius_px: float = 140.0
    accept_radius_max_px: float = 420.0
    #: How fast the acceptance window opens while the gate is unseen.
    widen_px_per_s: float = 260.0
    #: Seconds without the expected gate before we stop trusting dead reckoning.
    coast_timeout_s: float = 0.8
    search_timeout_s: float = 3.0
    #: After a pass, ignore pass conditions for this long.
    lockout_s: float = 0.35
    #: A pass needs the drone inside this much of the opening half-width.
    pass_half_width_m: float = 0.75
    #: A plane crossing further out than this sideways is a miss, not a pass.
    miss_half_width_m: float = 1.35
    #: Vision-only fallback: the gate must have grown past this apparent size.
    #: For a 2.7 m frame at fx=320 this is about 3.3 m away.
    commit_span_px: float = 260.0
    #: ...and then vanish for this many consecutive frames. One dropped frame
    #: must never look like a pass -- with a 5 % dropout rate that would fire
    #: constantly.
    commit_missing_frames: int = 4
    #: ...or collapse to this fraction of its peak size. Flying through a gate
    #: takes it from filling the view to being behind you, so apparent size
    #: falls off a cliff. This is the signal that works when the detector can
    #: still see the gate from behind, where waiting for it to vanish never
    #: fires at all.
    commit_shrink_ratio: float = 0.55
    #: A gate may only be committed to while it is *growing*. Without this, the
    #: gate you have just flown through is still the largest thing in view, so
    #: the tracker commits to it again and counts it twice -- which is exactly
    #: what the first simulation of the no-pose path did, ending up 13
    #: crossings ahead of reality. Slope is measured over ``growth_window_s``.
    growth_window_s: float = 0.25
    min_growth_px_per_s: float = 40.0
    #: Give up on a commit that never resolves.
    commit_timeout_s: float = 1.2
    min_confidence: float = 0.25
    #: When the expected gate has been unseen this long, also try the gates
    #: *after* it. Missing a single crossing must not be terminal: without this
    #: the tracker stays locked on a gate that is already behind the aircraft,
    #: every new detection is matched to the wrong gate and rejected, and the
    #: position estimate starves and diverges. Measured: one missed pass turned
    #: a healthy 0.2 m estimate into 11 m of error in under two seconds.
    reacquire_after_s: float = 0.6
    #: ...or after this many consecutive rejected position fixes. Time-since-
    #: seen is a weak trigger on its own: a detection of the *next* gate often
    #: lands near the stale prediction of the current one, so the tracker keeps
    #: believing it can see a gate it has already flown past. What actually
    #: gives it away is that the resulting metric fix is nowhere near the
    #: estimate, which is the association failing its own consistency check.
    reacquire_after_rejects: int = 15
    #: How far ahead to look when re-acquiring. The sequence only moves
    #: forwards, so we never look back.
    reacquire_lookahead: int = 3
    #: A crossing is corroborated if the gate was ranged this close by PnP at
    #: some point during the approach. Apparent size gives range independently
    #: of the position estimate, so the two can disagree -- and when they do,
    #: the crossing is the one to doubt. Measured today: fixes stop coming
    #: about 1.4 m out as the corners leave the frame, so the closest honest
    #: range is 1.4-1.8 m; 2.5 m clears that with margin while being far too
    #: close for a gate we never actually approached.
    corroborate_range_m: float = 2.5
    #: A later gate must beat the expected one by this factor to steal the lock.
    reacquire_margin: float = 0.6
    #: A detection may only be claimed by the expected gate if no *other* nearby
    #: gate explains it this much better. Without this the acceptance window is
    #: the only test, and the window has to widen while a gate is unseen -- so
    #: the longer we fail to find the gate we want, the more willing we become
    #: to accept a different one. Two measured failures, both from that:
    #:
    #:  * the double gate. Its two openings are the same structure 2.70 m
    #:    apart vertically, which at 3.5 m range is 328 px -- inside the
    #:    widened 420 px window. Hunting for the lower opening, the tracker
    #:    locked onto the upper one, and the fix moved the estimate 2.70 m.
    #:  * gate 7 and gate 8, 7.54 m apart and both in shot after the turn.
    #:
    #: Being nearest to a prediction is not enough. The gate must also be the
    #: best available explanation of what was actually seen.
    claim_margin: float = 0.7


@dataclass
class PassEvent:
    index: int              # index into the race sequence
    gate_number: int        # the organizer's gate number
    is_double_upper: bool
    t: float
    by: str                 # "plane" or "vision"
    lateral_m: float | None
    vertical_m: float | None
    clean: bool             # inside the opening, i.e. a valid crossing
    #: Did an *independent* witness agree we were really there? The plane
    #: crossing above is judged from our own position estimate, so a
    #: drifted estimate declares crossings that never happened -- measured:
    #: eight declared where six occurred, each one reported "clean" and
    #: under 0.15 m off centre, while the estimate was 8.5 m wrong.
    #: Apparent size is the second witness: to pass through a gate you must
    #: first have been right up against it, and how big it looks owes
    #: nothing to where we think we are.
    corroborated: bool = True


@dataclass
class GateTracker:
    """Tracks the expected crossing and counts passes."""

    sequence: list[G.Gate] = field(default_factory=lambda: G.race_sequence(laps=2))
    camera: G.Camera = field(default_factory=G.Camera.training)
    cfg: TrackerConfig = field(default_factory=TrackerConfig)

    index: int = 0
    state: TrackState = TrackState.SEARCHING
    t: float = 0.0
    passes: list[PassEvent] = field(default_factory=list)
    reassociations: list[tuple[float, str]] = field(default_factory=list)

    _last_seen_t: float = -1e9
    _last_pass_t: float = -1e9
    _prev_through: float | None = None
    #: Where we were on the previous frame. Needed to interpolate the
    #: crossing point: a control step at racing speed covers 0.27 m at
    #: 16 m/s, so measuring the offset at the frame *after* the plane is a
    #: quarter of a metre wrong against a 0.75 m half-opening.
    _prev_pos: object = None
    _committed_t: float | None = None
    _committed_span: float | None = None
    _missing_frames: int = 0
    #: The detection associated with the expected gate this frame, or None.
    #: Public because the flight loop uses it for the position fix: association
    #: must happen exactly once, here, or two different rules can disagree.
    accepted: Detection | None = None
    #: Consecutive position fixes rejected as inconsistent, fed back by the
    #: flight loop. Association is judged in metres, not pixels.
    fix_rejects: int = 0
    _span_history: list = field(default_factory=list)
    _had_pose: bool = False
    #: Closest PnP range to the current target, fed back by the flight loop.
    #: inf means "nobody is telling us", in which case corroboration is not
    #: attempted -- the tracker is being driven by something that does not
    #: run PnP, and inventing a verdict would be worse than not having one.
    _min_fix_range: float = float("inf")
    _ranges_reported: bool = False

    # ---------------------------------------------------------------- helpers
    @property
    def expected(self) -> G.Gate | None:
        if self.index >= len(self.sequence):
            return None
        return self.sequence[self.index]

    @property
    def gate_index_for_policy(self) -> int:
        """The 0-based slot for the observation's one-hot.

        The policy was trained with an 18-slot context. We feed the crossing
        index within the lap, which is what varies around a lap.
        """
        if not self.sequence:
            return 0
        per_lap = len(self.sequence) // max(1, self._laps())
        if per_lap <= 0:
            return 0
        return min(17, self.index % per_lap)

    def _laps(self) -> int:
        numbers = [g.number for g in self.sequence]
        return max(1, numbers.count(1))

    @property
    def confident(self) -> bool:
        """Is the count trustworthy enough to fly on?

        Requires a pose. See the module docstring: without one, two or more
        gates are in view a third of the time and the count silently drifts.
        """
        return self._had_pose and self.state in (TrackState.LOCKED, TrackState.COASTING)

    @property
    def degraded(self) -> bool:
        """True when counting without a pose. The loop should hold, not race."""
        return not self._had_pose

    def accept_radius(self) -> float:
        dt = max(0.0, self.t - self._last_seen_t)
        r = self.cfg.accept_radius_px + self.cfg.widen_px_per_s * dt
        return min(r, self.cfg.accept_radius_max_px)

    def predicted_view(self, pose_pos_w, pose_quat_wxyz):
        """Where the expected gate should appear. (uv, visible) or None."""
        g = self.expected
        if g is None or pose_pos_w is None:
            return None
        uv, vis = self.camera.project(g.corners_world(), pose_pos_w, pose_quat_wxyz)
        return uv, vis

    # ------------------------------------------------------------------ update
    def update(self, dt: float, detections=None, pose_pos_w=None, pose_quat_wxyz=None,
               pose_trusted: bool = True):
        """Advance one frame.

        ``detections`` is a list of ``Detection``. ``pose_pos_w`` /
        ``pose_quat_wxyz`` are the current world pose estimate, or None if the
        estimator has nothing — in which case the vision-only path is used.

        ``pose_trusted`` says whether the estimate is good enough to *count* a
        crossing from. Association and counting have different standards on
        purpose: associating from a rough pose is useful and safe, because a
        wrong association simply gets rejected downstream. Counting from a
        rough pose is neither — a crossing declared from a drifted estimate is
        indistinguishable from a real one afterwards, and it corrupts the gate
        index for the rest of the run. Measured: allowing it produced eight
        declared crossings where six had happened.
        """
        self.t += float(dt)
        self._had_pose = pose_pos_w is not None
        if self.expected is None:
            self.state = TrackState.FINISHED
            return None

        self.accepted = self._associate(detections, pose_pos_w, pose_quat_wxyz)
        if self.accepted is not None:
            self._last_seen_t = self.t
            self.state = TrackState.LOCKED
        else:
            since = self.t - self._last_seen_t
            if since <= self.cfg.coast_timeout_s:
                self.state = TrackState.COASTING
            elif since <= self.cfg.search_timeout_s:
                self.state = TrackState.SEARCHING
            else:
                self.state = TrackState.LOST

        # A pose is strictly better evidence than apparent size, so when the
        # estimator has one we use it alone. Running both would let the vision
        # heuristic race ahead of the truth, which is exactly what it did the
        # first time this was simulated.
        if pose_pos_w is not None and pose_trusted:
            event = self._check_plane_crossing(pose_pos_w)
        elif pose_pos_w is None:
            event = self._check_vision_pass()
        else:
            # Rough pose: keep tracking the plane so the crossing is not missed
            # when trust returns, but do not declare anything on it.
            self._prev_through = G.signed_distance_through(self.expected, pose_pos_w)
            self._prev_pos = np.asarray(pose_pos_w, dtype=np.float64).copy()
            event = None

        if event is not None:
            self.passes.append(event)
            self.index += 1
            self._last_pass_t = self.t
            self._prev_through = None
            self._prev_pos = None
            self._committed_t = None
            self._committed_span = None
            self._missing_frames = 0
            self._span_history.clear()
            self._min_fix_range = float("inf")
            if self.expected is None:
                self.state = TrackState.FINISHED
        return event

    # ------------------------------------------------------------- association
    def _predicted_centroid(self, j, pose_pos_w, pose_quat_wxyz):
        """Where sequence entry ``j`` should appear, or None if it is not in shot."""
        if j < 0 or j >= len(self.sequence):
            return None
        uv, vis = self.camera.project(self.sequence[j].corners_world(),
                                      pose_pos_w, pose_quat_wxyz)
        vis = np.asarray(vis, dtype=bool)
        if not vis.any():
            return None
        pts = np.asarray(uv)[vis]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def _neighbour_predictions(self, pose_pos_w, pose_quat_wxyz) -> dict:
        """Predicted centroids for the gates around the expected one.

        The gate behind matters as much as the gates ahead: right after a
        crossing it is the largest thing in view, and at the double gate it is
        the other opening of the same structure.
        """
        lo = self.index - 1
        hi = self.index + self.cfg.reacquire_lookahead
        out = {}
        for j in range(lo, hi + 1):
            c = self._predicted_centroid(j, pose_pos_w, pose_quat_wxyz)
            if c is not None:
                out[j] = c
        return out

    def _claimable(self, det, j, preds) -> bool:
        """Is entry ``j`` this detection's own best explanation?

        See ``TrackerConfig.claim_margin``. A detection that some other nearby
        gate explains far better is that other gate, whatever the acceptance
        window happens to allow.
        """
        c = det.centroid()
        if c is None or j not in preds:
            return False
        mine = math.hypot(c[0] - preds[j][0], c[1] - preds[j][1])
        for k, p in preds.items():
            if k == j:
                continue
            if math.hypot(c[0] - p[0], c[1] - p[1]) < self.cfg.claim_margin * mine:
                return False
        return True

    def _match(self, detections, j, preds, radius):
        """Best claimable detection for sequence entry ``j``, and how far off it was."""
        predicted = preds.get(j)
        if predicted is None:
            return None, float("inf")
        best, best_d = None, float("inf")
        for d in detections:
            c = d.centroid()
            if c is None:
                continue
            dist = math.hypot(c[0] - predicted[0], c[1] - predicted[1])
            if dist < best_d and self._claimable(d, j, preds):
                best, best_d = d, dist
        if best is None or best_d > radius:
            return None, best_d
        return best, best_d

    def _associate(self, detections, pose_pos_w, pose_quat_wxyz):
        if not detections:
            return None
        usable = [d for d in detections
                  if d.confidence >= self.cfg.min_confidence
                  and np.asarray(d.visible, dtype=bool).any()]
        if not usable:
            return None

        if pose_pos_w is None:
            # No pose: with one candidate take it; with several prefer
            # confidence then size, and record that we had to guess.
            if len(usable) == 1:
                return usable[0]
            best = max(usable, key=lambda d: (round(d.confidence, 2), d.span() or 0.0))
            self.reassociations.append((self.t, "no pose: chose by confidence then size"))
            return best

        radius = self.accept_radius()
        preds = self._neighbour_predictions(pose_pos_w, pose_quat_wxyz)
        best, best_d = self._match(usable, self.index, preds, radius)

        # Recovery: if the expected gate has been missing for a while, the most
        # likely explanation is that we already flew through it and the pass
        # went unconfirmed. Look forward — never back, the sequence is
        # monotonic — and let a clearly better match take the lock.
        unseen_for = self.t - self._last_seen_t
        # Re-acquiring means concluding we missed a crossing. **You cannot have
        # missed a gate you have not yet reached.** So while the aircraft is
        # still on the approach side of the expected gate's plane, refuse to
        # skip it, however much better some later gate looks in the image.
        #
        # Without this the double gate is unflyable. After the upper opening
        # the aircraft is past the structure and 2.7 m above the lower one,
        # which a 20 deg up-tilted camera cannot see -- so no detection is
        # claimable for it and ``best_d`` is infinite. The margin test below is
        # ``cand_d < 0.6 * best_d``, and every finite number beats 0.6 * inf,
        # so any visible gate stole the lock. Measured: the tracker skipped the
        # lower opening one second after the upper one, on every seed.
        #
        # A rejected *fix* still triggers regardless: that is positive metric
        # evidence that we are not where we think we are, which is a different
        # claim from "I cannot see it".
        ahead_of_us = False
        if pose_pos_w is not None and self.expected is not None:
            ahead_of_us = G.signed_distance_through(self.expected, pose_pos_w) < 0.0
        if ((unseen_for > self.cfg.reacquire_after_s and not ahead_of_us)
                or self.fix_rejects >= self.cfg.reacquire_after_rejects):
            for ahead in range(1, self.cfg.reacquire_lookahead + 1):
                j = self.index + ahead
                if j >= len(self.sequence):
                    break
                cand, cand_d = self._match(usable, j, preds, radius)
                if cand is not None and cand_d < self.cfg.reacquire_margin * best_d:
                    skipped = ahead
                    self.reassociations.append(
                        (self.t, f"re-acquired {skipped} gate(s) ahead: gate "
                                 f"{self.sequence[j].number} matched at {cand_d:.0f} px "
                                 f"vs {best_d:.0f} px for the expected one"))
                    self.index = j
                    self.fix_rejects = 0
                    self._prev_through = None
                    self._prev_pos = None
                    self._committed_t = None
                    self._committed_span = None
                    self._span_history.clear()
                    self._min_fix_range = float("inf")
                    return cand
        if best is not None and best_d > self.cfg.accept_radius_px:
            self.reassociations.append(
                (self.t, f"accepted at {best_d:.0f} px, outside the nominal window"))
        return best

    # ---------------------------------------------------------- pass detection
    def _check_plane_crossing(self, pose_pos_w):
        g = self.expected
        if g is None:
            return None
        pos = np.asarray(pose_pos_w, dtype=np.float64)
        through = G.signed_distance_through(g, pos)
        prev = self._prev_through
        prev_pos = self._prev_pos
        self._prev_through = through
        self._prev_pos = pos.copy()

        if prev is None or self.t - self._last_pass_t < self.cfg.lockout_s:
            return None
        if not (prev < 0.0 <= through):
            return None

        # Measure the offsets **at the plane**, by interpolating between the
        # frame before the crossing and the frame after it. Measuring at the
        # later frame instead -- which this did until it was checked -- is wrong
        # by however far the aircraft travelled in one control step: 0.27 m at
        # 16 m/s, against a half-opening of 0.75 m. That is a third of the
        # margin, and it decides whether a pass counts as clean or as a miss.
        denom = through - prev
        alpha = 0.0 if abs(denom) < 1e-9 else (-prev / denom)
        at_plane = pos if prev_pos is None else prev_pos + alpha * (pos - prev_pos)
        side, vert = G.lateral_offsets(g, at_plane)
        clean = abs(side) <= self.cfg.pass_half_width_m and abs(vert) <= self.cfg.pass_half_width_m
        missed = abs(side) > self.cfg.miss_half_width_m or abs(vert) > self.cfg.miss_half_width_m
        if missed:
            # Went past the plane well outside the frame: not a crossing at all.
            self.reassociations.append(
                (self.t, f"plane crossed {side:+.2f} m / {vert:+.2f} m out - not counted"))
            return None
        corroborated = (not self._ranges_reported
                        or self._min_fix_range <= self.cfg.corroborate_range_m)
        if not corroborated:
            self.reassociations.append(
                (self.t, f"gate {g.number} plane crossed but never ranged closer than "
                         f"{self._min_fix_range:.1f} m - counted, not corroborated"))
        return PassEvent(self.index, g.number, g.is_double_upper, self.t, "plane",
                         side, vert, clean, corroborated)

    def _check_vision_pass(self):
        """Fallback when there is no pose: the gate grew large, then vanished.

        Deliberately conservative. A single dropped detection must not look
        like a pass, so the gate has to be missing for several consecutive
        frames after having filled a good part of the view.
        """
        g = self.expected
        if g is None or self.t - self._last_pass_t < self.cfg.lockout_s:
            return None

        span = self.accepted.span() if self.accepted is not None else None

        if span is not None:
            self._missing_frames = 0
            self._span_history.append((self.t, span))
            cutoff = self.t - self.cfg.growth_window_s
            while len(self._span_history) > 2 and self._span_history[0][0] < cutoff:
                self._span_history.pop(0)
            growing = self._is_growing()
            if span >= self.cfg.commit_span_px and (growing or self._committed_t is not None):
                # Commit, and keep the largest size seen so far.
                if self._committed_t is None:
                    self._committed_t = self.t
                self._committed_span = max(self._committed_span or 0.0, span)
                return None
            if (self._committed_span is not None
                    and span <= self.cfg.commit_shrink_ratio * self._committed_span):
                # It filled the view and has now collapsed: we went through it.
                return PassEvent(self.index, g.number, g.is_double_upper, self.t,
                                 "vision", None, None, True, True)
            return None

        if self._committed_t is None:
            return None

        self._missing_frames += 1
        if self.t - self._committed_t > self.cfg.commit_timeout_s:
            # It went away without us passing through it -- an aborted approach.
            self.reassociations.append(
                (self.t, f"gate {g.number} committed then lost without a pass"))
            self._committed_t = None
            self._committed_span = None
            self._missing_frames = 0
            return None

        if self._missing_frames >= self.cfg.commit_missing_frames:
            return PassEvent(self.index, g.number, g.is_double_upper, self.t, "vision",
                             None, None, True, True)
        return None

    def _is_growing(self) -> bool:
        """Is the accepted gate getting bigger? A least-squares slope on span."""
        if len(self._span_history) < 3:
            return False
        ts = np.array([p[0] for p in self._span_history])
        ss = np.array([p[1] for p in self._span_history])
        ts = ts - ts[0]
        denom = float(((ts - ts.mean()) ** 2).sum())
        if denom < 1e-12:
            return False
        slope = float(((ts - ts.mean()) * (ss - ss.mean())).sum() / denom)
        return slope >= self.cfg.min_growth_px_per_s

    # -------------------------------------------------------------- reporting
    def note_fix(self, accepted: bool, range_m: float | None = None) -> None:
        """Tell the tracker how the position fix it enabled turned out.

        ``range_m`` is PnP's range to the gate. It is recorded because it is the
        one measurement of "how close am I really" that does not come from the
        position estimate, which makes it the only thing able to contradict it.
        """
        self.fix_rejects = 0 if accepted else self.fix_rejects + 1
        if range_m is not None:
            self._ranges_reported = True
            if accepted:
                self._min_fix_range = min(self._min_fix_range, float(range_m))

    def summary(self) -> str:
        n = len(self.passes)
        clean = sum(1 for p in self.passes if p.clean and p.corroborated)
        return (f"{n} crossings ({clean} clean+corroborated), "
                f"at index {self.index}/{len(self.sequence)}, "
                f"{self.state.value}, {len(self.reassociations)} re-associations")
