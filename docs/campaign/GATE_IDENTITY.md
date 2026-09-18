# Knowing which gate is which, without markers

**The constraint:** 10 identical gates, flown in a fixed order, two laps, gate 9 counted twice per
lap = 22 crossings. We may not mark the gates. The policy needs an 18-slot one-hot saying which gate
is next, and in the simulator that came free.

**Measured on our course:** 34 % of frames have two or more gates in view at once. So "look at the
gate and know which it is" is not available from appearance. Ever.

---

## 1. The reframe that makes this tractable

Identity is **not a perception problem**. It is a **localization** problem.

If you know where you are, identity is a table lookup — project every map gate and see which one you
are looking at. If you do not know where you are, no amount of staring at an identical square will
tell you. Every serious system in the literature reaches this same conclusion, and none of them
tries to recognise gates individually.

- **Swift** ([Kaufmann et al., *Nature* 2023](https://www.nature.com/articles/s41586-023-06419-4))
  detects corners of *all* visible gates, estimates a relative pose for each with IPPE
  (infinitesimal plane-based pose estimation — planar PnP), and then "each gate observation is
  assigned to the **nearest gate in the known trajectory layout**". Association happens in **metric
  pose space**, not image space. The result fuses with VIO through a Kalman filter, and the gate
  fixes are what stop VIO drifting.
- **AlphaPilot** ([Foehn et al., RSS 2020](https://www.roboticsproceedings.org/rss16/p081.pdf))
  uses the same shape: gate corner detection → PnP → association against the known map → filter.
- **Dual pose-graph semantic localization**
  ([arXiv 2604.15168](https://arxiv.org/html/2604.15168)) associates with the **Hungarian algorithm
  on Euclidean distance in the global frame**, plus an orientation check that explicitly handles
  *reverse* observations (seeing a gate from behind). Known gate positions enter as priors on
  landmark nodes. Its interesting trick is a **dual graph**: detections accumulate in a temporary
  graph and only commit to the main graph once they are consistent — deferring commitment instead of
  associating greedily. Reported 56–74 % ATE reduction over VIO alone, and up to 4.2 m of drift
  corrected per lap.
- Lightweight lines of work ([Li & de Croon](https://arxiv.org/pdf/1905.10110),
  [Li et al. 2018](https://arxiv.org/abs/1809.05958)) skip global identity altogether and fly
  gate-relative, accepting that they cannot plan beyond the gate in view.

**So: get a pose, and identity is free.** The rest of this document is about getting a pose cheaply,
given that we have no VIO.

---

## 2. Our course makes this much easier than it sounds

| Quantity | Value | Why it matters |
|---|---|---|
| Minimum separation between any two gates | **7.54 m** | The association decision has an enormous margin |
| Leg lengths between consecutive crossings | 7.5 – 14.1 m | Distinctive enough to sanity-check progress |
| The double-gate hop (9 upper → 9 lower) | **2.70 m** | Uniquely short. Nothing else is close — a free, unambiguous signature for the one place a counter is most likely to desynchronise |
| Confusable gate pairs (within 1 m and 15°) | 12 of 990 | And most differ by 0.3–0.9 m, resolvable with a decent range estimate |

**Range accuracy from a single gate** (2.7 m outer frame, real camera fx ≈ 425 at 640×360), against
that 7.54 m minimum separation:

| Distance | Apparent size | 5 px corner noise | 10 px corner noise | Margin at 10 px |
|---|---|---|---|---|
| 5 m | 230 px | 0.11 m | 0.22 m | 34× |
| 10 m | 115 px | 0.44 m | 0.87 m | **9×** |
| 12 m | 96 px | 0.63 m | 1.25 m | 6× |
| 15 m | 76 px | 0.98 m | 1.96 m | 4× |

Bearing is far better still: 1 px at 10 m is 2.4 cm.

**Conclusion: once we run PnP, association is not a close call.** Even with a sloppy detector at the
edge of its range we have a 4–6× margin, and in the working range 9–34×. This is the single most
important number in this document, because it means we do **not** need anything exotic — no
constellation matching, no place recognition, no learned descriptors.

---

## 3. What I recommend, in layers

Cheapest first. Each layer is useful alone and strengthens the next.

### Layer 0 — Start from a known pose, so global localization is never needed

The race starts in a fixed box at a known position and heading, before gate 1. **That breaks the
chicken-and-egg entirely.** We never have to solve "where am I on this course" from scratch; we only
have to *keep* knowing, and the gates keep correcting us. Global relocalization becomes an
emergency path, not the normal one.

Cost: nothing. Just record the start pose and initialise the filter with it.

### Layer 1 — Associate in metric space, not image space (this is the main change)

Current tracker: predicts where the expected gate should appear and accepts detections near that
*pixel* location. That works, but the threshold is in pixels and its physical meaning changes with
range.

Better, and what Swift does: run planar PnP on each detection to get a **relative pose**, convert to
a world-position hypothesis, and pick the map gate it is nearest to. The threshold is then in
**metres**, against a known 7.54 m minimum separation.

Two details worth getting right:
- **Planar PnP is two-fold ambiguous.** Four coplanar points admit a mirrored solution. Resolve it
  with gravity — we know roll and pitch from the IMU, and gates are vertical, so the flipped
  solution is geometrically impossible. Use all eight corners (outer *and* inner ring) and reject on
  reprojection error as a second check.
- **Handle reverse observations.** A gate seen from behind has its normal pointing at us. The
  pose-graph paper calls this out explicitly. Sign the association by the gate's through-axis.

### Layer 2 — A small EKF over position, velocity and yaw, corrected by gate fixes

We have no VIO, so something has to carry us between gate sightings. We already have the pieces:
`ekf/commanded_accel.py` and `BodyVelocityIntegrator` in the flight client integrate *commanded*
acceleration from attitude and thrust, which is exactly the dead-reckoning Plan B was built on.

- **Predict** from attitude + thrust + a drag estimate.
- **Correct** from each accepted gate PnP fix.
- Gates arrive every ~2 s at racing speed, so drift has very little time to accumulate — this is why
  we can get away without VIO, where Swift could not.

Also estimate a **thrust bias** in the same filter. That is what makes the barometer unnecessary,
which matters because the barometer is unusable with props spinning.

### Layer 3 — Sequence prior, monotonic index, and a leg-length consistency check

The index can only increase, and by one. So the failure modes are exactly two: **double-count** and
**miss**. Both are detectable from geometry:

- Distance travelled since the last crossing should match the map's leg (7.5–14.1 m, or 2.70 m for
  the double hop). A double-count shows an implausibly short leg; a miss shows a suspiciously long
  one.
- The 2.70 m double-gate hop is unique, so the one place most likely to desynchronise is also the
  one place with the most distinctive signature. Use it.

Nobody in the literature bothers with this because they have VIO and do not need it. We should,
because we do not.

### Layer 4 — Keep a few hypotheses instead of committing greedily

Not a full particle filter — just **three to five weighted hypotheses** over the crossing index
(k−1, k, k+1), reweighted each frame by how well each explains the detections. Commit when one
dominates for a sustained period. This is the dual-graph paper's "defer commitment" idea at a
fraction of the cost, and it turns a hard, irreversible misassociation into a soft one that
evidence can undo.

Cheap: five hypotheses × one projection each is nothing against a 16 ms budget when the whole
current pipeline uses 9 %.

### Layer 5 — Make the policy tolerant of being told the wrong gate

This is the layer **nobody else has**, because it only exists if you are training a policy — and we
are retraining anyway.

- **Randomise the context channel during training.** Feed the wrong one-hot occasionally, or a
  smeared distribution over neighbouring gates. The policy then degrades gracefully under a
  mislabel instead of switching into a completely wrong mode.
- **Train with the one-hot dropped out** some fraction of the time, so the policy can fly on the
  visible gate alone when the tracker reports low confidence.

Both cost nothing but training time, and they convert our worst failure — confident misidentification
— from catastrophic to merely slow.

### Layer 6 — Consider replacing the one-hot with continuous geometry

A discrete 18-slot one-hot is a brittle encoding: index 7 and index 8 are as different to the network
as index 7 and index 2, so an off-by-one is a mode switch rather than a small error.

The repo already supports the alternative: the **lookahead** variant (`GATE_LOOKAHEAD`,
`ISAAC_LOOKAHEAD = 7`) feeds *the next gate's pose in the current gate's frame* — continuous relative
geometry instead of an identity label. An error in that encoding is a small numerical error, not a
different one-hot.

This is a training-time decision and worth testing in the retrain, but it is a bigger change than
Layer 5 and should not block anything.

---

## 4. What this changes in what we have built

| Component | Change |
|---|---|
| `gate_tracker.py` | Associate on metric distance after PnP rather than pixel distance. Keep the pixel gate as a cheap pre-filter |
| New: `pnp.py` | Planar PnP on eight corners, gravity-disambiguated, with reprojection-error rejection |
| New: `pose_filter.py` | Small EKF: position, velocity, yaw, thrust bias. Predict from commanded acceleration, correct from gate fixes. Initialise from the known start pose |
| `gate_tracker.py` | Add the leg-length consistency check and the 2.70 m double-gate signature |
| `gate_tracker.py` | Three-hypothesis index tracking, committing on sustained dominance |
| Training | Randomise and drop out the context channel. Optionally test the lookahead encoding |

The degraded-mode behaviour already built stays exactly as it is: **without a pose, `confident` is
False and the loop holds.** Everything above is about making sure we have a pose almost always.

---

## 5. What I explicitly rejected, and why

- **Constellation / geometric-hash matching** (star-tracker style, matching observed gate pairs
  against map pairs). Elegant, and it would give global relocalization with no prior. But we do not
  need global relocalization — we start from a known pose — and only 12 of 990 gate pairs are even
  mildly confusable, so it solves a problem we do not have. Keep it in the back pocket for recovery
  after a total loss of track.
- **Learned place recognition / visual descriptors.** The gates are deliberately identical and the
  hall is visually repetitive. This is the hardest possible setting for appearance-based methods,
  and it needs training data we do not have.
- **Appearance-based gate distinction** (subtle manufacturing differences, scuffs, background
  context). Fragile, unverifiable before the event, and it would silently degrade if a gate were
  moved or replaced after a knock.
- **Full VIO.** The right answer in general, and what Swift and the pose-graph work rely on. But it
  is a large piece of work, the Jetson has one camera and a 25 W budget, and gate fixes every ~2 s
  make the drift problem small enough to handle with dead reckoning. Revisit only if the EKF proves
  insufficient.

---

## Sources

- Kaufmann et al., "Champion-level drone racing using deep reinforcement learning", *Nature* 2023 — <https://www.nature.com/articles/s41586-023-06419-4>
- Foehn et al., "AlphaPilot: Autonomous Drone Racing", RSS 2020 — <https://www.roboticsproceedings.org/rss16/p081.pdf>
- "Dual Pose-Graph Semantic Localization for Vision-Based Autonomous Drone Racing" — <https://arxiv.org/html/2604.15168>
- "Drift-Corrected Monocular VIO and Perception-Aware Planning for Autonomous Drone Racing" — <https://arxiv.org/pdf/2512.20475>
- Li et al., "Visual Model-predictive Localization for Computationally Efficient Autonomous Racing of a 72-gram Drone" — <https://arxiv.org/pdf/1905.10110>
- Li et al., "Autonomous drone race: A computationally efficient vision-based navigation and control strategy" — <https://arxiv.org/abs/1809.05958>
