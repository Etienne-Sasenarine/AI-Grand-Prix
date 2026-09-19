"""Make ANGLE -- self-levelling -- what a human pilot gets by default.

    python3 setup_angle_mode.py --dev /dev/ttyTHS1 --beginner --dry-run   # look first
    python3 setup_angle_mode.py --dev /dev/ttyTHS1 --beginner --yes       # THE ONE COMMAND

    python3 setup_angle_mode.py --dev /dev/ttyTHS1 --show       # what is assigned
    python3 setup_angle_mode.py --dev /dev/ttyTHS1 --watch      # which channel a switch drives

``--beginner`` in one paragraph
-------------------------------
**A person flying always gets ANGLE. The Jetson flying always gets ACRO. One
switch decides which, and it is the one that already hands the drone to the
Jetson.** Betaflight has a mode called MSP override -- it is how the Jetson is
given the sticks -- and on this airframe it is switched on by one channel going
above 1700. ``--beginner`` reads that off the flight controller and puts ANGLE
on **the same channel, for every position where override is off**. Nothing has
to be known about which physical switch is which, and nothing can be left in the
wrong position: override off means a human is flying and the aircraft
self-levels; override on means the policy is flying, and ANGLE drops away so its
rotation-rate commands mean what it was trained to expect. When the pilot takes
the aircraft back, it is self-levelling again the instant they do. It also sets
the beginner tilt limit (25 degrees unless ``--tilt-limit`` says otherwise).

The label on the transmitter does not matter to any of this, which is as well:
the switch marked ACRO / ANGLE does not select ANGLE, and flipping it to "ANGLE"
has made the drone die (reported 18 Sep). Whatever it drives, **the position that
engages MSP override gives the sticks to the Jetson -- never go there unless the
autonomy program is running**, or roll, pitch and yaw go dead in the pilot's
hands.

The problem
-----------
The Futaba T16IZ has a switch labelled **ACRO / ANGLE**. Flipping it does
nothing, because Betaflight has no ANGLE assignment to listen with: reading the
mode ranges off the aircraft on 18 Sep gave boxIds 0, 40, 41, 43 and 50, and
**ANGLE is boxId 1**, which is not among them. ANGLE is available on this
firmware -- ``box_ids()`` lists it -- it has simply never been wired to a switch.

The consequence is that the team has only ever flown a 1745 g 8-inch racing quad
in full manual, with no self-levelling, which is about the hardest way there is
to learn.

ANGLE is the default, ACRO is the exception
-------------------------------------------
Nobody on the team is a pilot, so the safe state has to be the one you get
without doing anything. ``--assign --acro-at <us>`` therefore switches ANGLE on
**everywhere except the ACRO end of the switch**: the resting position, the
middle of a three-position switch, and a channel that is not being sent at all
(Betaflight reads that as 1500) all give self-levelling. ACRO -- full manual --
needs a deliberate flip, all the way to the end.

``--acro-at`` is the number ``--watch`` shows with the switch *on ACRO*. Which
end that is depends on how the transmitter is set up, which is why this asks
instead of guessing: guessing wrong would make ACRO the default, silently.

**The one time ACRO matters: the scored run.** The trained policy commands
rotation *rates*, which is what ACRO means, so for an autonomous run the switch
must be on ACRO. The Jetson cannot do that for you. MSP override only takes the
four stick channels; every switch stays with the pilot's radio, including this
one. If the pilot takes the aircraft back mid-run, flipping to ANGLE at the same
moment as releasing the override hands them a self-levelling aircraft.

The tilt limit, and why it goes on the 8-inch profile
-----------------------------------------------------
``--tilt-limit 25`` sets how far ANGLE mode lets the aircraft lean at full stick
(Betaflight's ``level_limit``). Ours ships at **55 degrees**, which is a racing
number; published beginner guidance is 25-35. It only exists in the
self-levelling modes, so **it changes nothing about ACRO** and nothing about
what the policy can do.

It is written to the **active** PID profile, and that is deliberate. This board
carries one PID profile per airframe size -- read off our own dump on 18 Sep:
0 is the vendor's 10-inch tune, 1 the 5-inch, **2 the 8-inch (ours, active)**,
3 "8-inch Fiber". A "beginner profile" kept on a spare slot would mean flying an
8-inch aircraft on somebody else's PID gains, which is a worse idea than any
tilt limit is a good one. So the tune stays and only the limit moves.

Rates and the throttle curve are deliberately **not** touched. They live in the
rate profile the policy's stick conversion is built around (``thr_mid`` 54,
``thr_expo`` 68, rates 55/75), and in ANGLE mode the roll and pitch rates do not
apply anyway -- the stick sets a lean angle, not a rotation speed.

Why this goes over MSP and not the CLI
--------------------------------------
``bf_cli.py`` can do this, and it is how Betaflight configuration is usually
scripted. But entering the CLI over the MSP UART **wedges this flight
controller** -- measured twice out of two on 18 Sep, silent at every baud rate
afterwards, recovered only by pulling the flight battery.

``MSP_SET_MODE_RANGE`` and ``MSP_SET_PID_ADVANCED`` do the same jobs as ordinary
MSP messages, which is what Betaflight Configurator itself uses. No CLI, no
reboot, no wedge.

How the tilt-limit byte is found
--------------------------------
``MSP_PID_ADVANCED`` is a run of bytes whose layout shifts between firmware
versions, and a remembered offset is exactly the kind of thing that is wrong by
one. So this does not remember it. We know three values of ``level_limit`` from
our own dump -- 30, 23 and 55 on profiles 0, 1 and 2 -- and ``MSP_SELECT_SETTING``
is measured to switch PID profiles safely. The tool reads the reply under the
profiles it is *not* going to change, finds the one byte that carries their
known limits, and refuses to write unless exactly one does. It then changes that
byte alone, reads everything back, and saves only if nothing else moved.

What it will not do
-------------------
* It will not touch a slot belonging to another mode. It finds a free one, and
  if there is none it says so and stops.
* It will not move an existing ANGLE assignment unless told to (``--reassign``).
* It will not write while the aircraft is armed.
* It verifies by reading back, and does not save what it could not verify.
* ``--show`` and ``--watch`` never write anything.

**None of the writing paths has run on the real flight controller yet.** They
pass against a mock shaped like it. First use is a bench job, propellers off.

Afterwards
----------
**Bench test with propellers off before flying it**, exactly as the Betaflight
setup guide says: with ANGLE selected and the throttle up a little, tilt the
aircraft and check the low motor spins up and stays up until you level it again.
Then flip to ACRO and check that it stops doing that.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_link import open_link  # noqa: E402

MSP_MODE_RANGES = 34
MSP_SET_MODE_RANGE = 35
MSP_PID_ADVANCED = 94
MSP_SET_PID_ADVANCED = 95
MSP_SELECT_SETTING = 210
MSP_EEPROM_WRITE = 250

#: Betaflight's permanent id for the self-levelling mode.
BOX_ANGLE = 1
BOX_ARM = 0
#: MSP override: the mode that hands the stick channels to the Jetson.
BOX_MSP_OVERRIDE = 50
#: What --beginner sets the tilt limit to when not told otherwise.
BEGINNER_TILT_DEG = 25

#: Betaflight stores range edges as steps of 25 us from 900 us.
STEP_US = 25
STEP_BASE = 900

#: The legacy range: ANGLE only in the upper half of the travel, which makes
#: **ACRO the default**. Kept so ``--low/--high`` can still ask for it on
#: purpose; nothing reaches for it on its own any more.
DEFAULT_LOW_US = 1600
DEFAULT_HIGH_US = 2100

#: Where ANGLE stops and ACRO starts, either side of centre. 200 us clear of
#: 1500 so the middle of a three-position switch is unambiguously ANGLE, and
#: about 300 us short of where our switches actually sit (1094 and 1992 in the
#: 18 Sep log), so a slightly lazy switch still lands on the right side.
ACRO_EDGE_HIGH_US = 1700
ACRO_EDGE_LOW_US = 1300
#: ``--acro-at`` must be at least this far past the edge, or the switch does
#: not reach ACRO reliably and the tool says so rather than assigning it.
ACRO_MARGIN_US = 100

#: ``level_limit`` per PID profile, from our own full dump of 18 Sep
#: (logs/orin/bf_dump_before_beginner_20260918.txt lines 1226, 1331, 1436).
#: These are the anchors the tilt-limit byte is found with. They are vendor
#: values on profiles we never write, so they stay true after this tool has run.
KNOWN_LEVEL_LIMITS = {0: 30, 1: 23, 2: 55}
#: A tilt limit outside this is a typo, not a choice.
TILT_MIN_DEG, TILT_MAX_DEG = 10, 90


def us_to_step(us: float) -> int:
    return int(round((us - STEP_BASE) / STEP_US))


def step_to_us(step: int) -> int:
    return STEP_BASE + step * STEP_US


def angle_range_for(acro_at_us: int) -> tuple[int, int]:
    """The ANGLE range that leaves only the ACRO end of the switch in ACRO.

    ``acro_at_us`` is what the channel reads with the switch on ACRO. Everything
    on the other side of the edge -- the rest position, the middle, an unsent
    channel at 1500 -- is ANGLE.
    """
    if acro_at_us >= ACRO_EDGE_HIGH_US + ACRO_MARGIN_US:
        return STEP_BASE, ACRO_EDGE_HIGH_US
    if acro_at_us <= ACRO_EDGE_LOW_US - ACRO_MARGIN_US:
        return ACRO_EDGE_LOW_US, 2100
    raise ValueError(
        f"--acro-at {acro_at_us} is too close to the middle of the travel to be "
        f"the ACRO end of a switch. It has to be below "
        f"{ACRO_EDGE_LOW_US - ACRO_MARGIN_US} or above "
        f"{ACRO_EDGE_HIGH_US + ACRO_MARGIN_US}. Run --watch, leave the switch on "
        f"ACRO, and use the 'now' value it prints.")


def read_ranges(fc) -> list:
    """[(index, boxId, auxChannel, startUs, endUs)], including empty slots."""
    p = fc.request(MSP_MODE_RANGES)
    out = []
    for i in range(len(p) // 4):
        pid, aux, a, b = p[i * 4], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]
        out.append((i, pid, aux, step_to_us(a), step_to_us(b)))
    return out


def is_empty(entry) -> bool:
    """An unused slot has a zero-width range. boxId 0 alone is not enough --
    that is ARM's id, and ARM is a real assignment."""
    _i, _pid, _aux, lo, hi = entry
    return us_to_step(lo) == 0 and us_to_step(hi) == 0


def describe(entry) -> str:
    i, pid, aux, lo, hi = entry
    if is_empty(entry):
        return f"  slot {i:2d}  (free)"
    name = {BOX_ARM: "ARM", BOX_ANGLE: "ANGLE", 2: "HORIZON",
            BOX_MSP_OVERRIDE: "MSP OVERRIDE"}.get(pid, f"boxId {pid}")
    return (f"  slot {i:2d}  {name:<12s} aux channel {aux} "
            f"(RC ch {aux + 5})  {lo}-{hi} us")


def find_angle(ranges) -> tuple | None:
    for e in ranges:
        if e[1] == BOX_ANGLE and not is_empty(e):
            return e
    return None


def find_override(ranges) -> tuple | None:
    for e in ranges:
        if e[1] == BOX_MSP_OVERRIDE and not is_empty(e):
            return e
    return None


def angle_range_beside_override(override) -> tuple[int, int]:
    """Every position of the override channel where override is OFF.

    Betaflight treats a range as ``low <= value < high``, so ANGLE ending
    exactly where override begins leaves no gap and no overlap.
    """
    _i, _pid, _aux, lo, hi = override
    if hi >= 2000 and lo >= 1300:
        return STEP_BASE, lo           # override at the top: ANGLE is everything below
    if lo <= 1000 and hi <= 1700:
        return hi, 2100                # override at the bottom: ANGLE is everything above
    raise ValueError(
        f"MSP override is on {lo}-{hi} us, which is neither the top nor the "
        f"bottom of the travel, so there is no single range that means 'override "
        f"off'. Use --assign with --acro-at instead.")


def angle_active_now(fc) -> bool | None:
    """Is the flight controller in ANGLE right now? None if it cannot say."""
    try:
        ids = list(fc.box_ids())
        flags = int(fc.status()["flight_mode_flags"])
        return bool((flags >> ids.index(BOX_ANGLE)) & 1)
    except Exception:
        return None


def default_mode(entry) -> str:
    """What a pilot gets without touching the switch: 'ANGLE' or 'ACRO'.

    Judged at 1500 us, which is both the middle of the travel and what
    Betaflight substitutes for a channel that is not being sent.
    """
    if entry is None:
        return "ACRO"
    _i, _pid, _aux, lo, hi = entry
    return "ANGLE" if lo <= 1500 < hi else "ACRO"


def watch(fc, seconds: float = 60.0) -> dict:
    """Report which RC channels move, and where each one is *now*. Read-only."""
    lo, hi, now, n = {}, {}, {}, 0
    t0 = time.monotonic()
    print(f"  watching for {seconds:.0f}s -- flip the ACRO/ANGLE switch a few times,")
    print("  and LEAVE IT ON ACRO before the time runs out")
    while time.monotonic() - t0 < seconds:
        try:
            rc = fc.rc_channels()
        except Exception:
            time.sleep(0.1)
            continue
        n += 1
        for i, v in enumerate(rc):
            lo[i] = min(lo.get(i, v), v)
            hi[i] = max(hi.get(i, v), v)
            now[i] = v
        time.sleep(0.05)
    moved = {i: (lo[i], hi[i]) for i in lo if hi[i] - lo[i] > 200}
    return {"samples": n, "moved": moved, "lo": lo, "hi": hi, "now": now}


def assign(fc, *, aux_channel: int, low_us: int, high_us: int,
           dry_run: bool, reassign: bool = False) -> int:
    ranges = read_ranges(fc)

    existing = find_angle(ranges)
    if existing and (existing[2], existing[3], existing[4]) == (aux_channel, low_us, high_us):
        print(f"  ANGLE is ALREADY assigned exactly like that: {describe(existing).strip()}")
        print("  Nothing to do.")
        return 0
    if existing and not reassign:
        print(f"  ANGLE is already assigned, differently: {describe(existing).strip()}")
        print(f"  That makes {default_mode(existing)} the default. To replace it with "
              f"what you asked for, add --reassign.")
        return 8

    if existing:
        # Its own slot, so rewriting it disturbs nothing else.
        slot = existing[0]
    else:
        # Refuse to write to a slot that is doing something.
        free = [e for e in ranges if is_empty(e)]
        if not free:
            print("  every mode slot is occupied; not overwriting one. "
                  "Free a slot in Configurator first.")
            return 4
        slot = free[0][0]

    # And refuse to share a channel with another mode, which would turn two
    # things on at once.
    clash = [e for e in ranges
             if not is_empty(e) and e[0] != slot and e[2] == aux_channel
             and not (high_us <= e[3] or low_us >= e[4])]
    if clash:
        print(f"  aux channel {aux_channel} already carries "
              f"{describe(clash[0]).strip()}")
        print("  Overlapping ranges would switch both on together. Pick another "
              "channel, or a non-overlapping range.")
        return 5

    verb = "move ANGLE in" if existing else f"write ANGLE (boxId {BOX_ANGLE}) into"
    print(f"  will {verb} slot {slot}, aux channel {aux_channel} "
          f"(RC ch {aux_channel + 5}), {low_us}-{high_us} us")
    would_be = default_mode((slot, BOX_ANGLE, aux_channel, low_us, high_us))
    print(f"  -> with the switch untouched, the pilot gets {would_be}")
    if dry_run:
        print("  --dry-run: nothing written.")
        return 0

    payload = bytes([slot, BOX_ANGLE, aux_channel,
                     us_to_step(low_us), us_to_step(high_us)])
    fc.request(MSP_SET_MODE_RANGE, payload)

    # Read back BEFORE saving: if the board did not take it, nothing is
    # persisted and there is nothing to undo.
    check = find_angle(read_ranges(fc))
    if not check or (check[2], check[3], check[4]) != (aux_channel, low_us, high_us):
        print("  the flight controller did not accept the assignment; "
              "nothing was saved. Power-cycle it to be sure nothing lingers.")
        return 6
    print(f"  accepted: {describe(check).strip()}")

    fc.request(MSP_EEPROM_WRITE)
    time.sleep(0.5)
    final = find_angle(read_ranges(fc))
    if not final:
        print("  saved, but it is gone on re-read. Not trusting that; "
              "check in Configurator.")
        return 7
    print("  saved to EEPROM and verified.")
    return 0


# ------------------------------------------------------------------ tilt limit
def _select_pid_profile(fc, index: int) -> None:
    fc.request(MSP_SELECT_SETTING, bytes([index]))
    time.sleep(0.05)
    got = fc.status().get("pid_profile")
    if got != index:
        raise RuntimeError(f"asked for PID profile {index}, the flight controller "
                           f"reports {got}")


def find_level_limit_offset(fc, target_profile: int) -> tuple[int, int]:
    """Locate the tilt-limit byte in MSP_PID_ADVANCED. Returns (offset, current).

    Reads the reply under every known profile *except* the one about to be
    changed, and looks for the byte that carries their known limits. Always
    leaves the flight controller back on ``target_profile``.
    """
    anchors = {p: v for p, v in KNOWN_LEVEL_LIMITS.items() if p != target_profile}
    seen = {}
    try:
        for prof in anchors:
            _select_pid_profile(fc, prof)
            seen[prof] = bytes(fc.request(MSP_PID_ADVANCED))
    finally:
        _select_pid_profile(fc, target_profile)
    target = bytes(fc.request(MSP_PID_ADVANCED))

    n = min(len(target), *(len(b) for b in seen.values()))
    hits = [i for i in range(n) if all(seen[p][i] == v for p, v in anchors.items())]
    # A byte that matches the anchors but holds nonsense on our profile is a
    # coincidence, not a tilt limit.
    hits = [i for i in hits if TILT_MIN_DEG <= target[i] <= TILT_MAX_DEG]
    if len(hits) != 1:
        raise RuntimeError(
            f"cannot identify the tilt-limit byte: {len(hits)} candidates "
            f"(offsets {hits}) carry {anchors} on the anchor profiles. Either a "
            f"vendor level_limit was changed or this firmware packs the reply "
            f"differently. Not writing blind.")
    return hits[0], target[hits[0]]


def set_tilt_limit(fc, degrees: int, *, dry_run: bool) -> int:
    if not TILT_MIN_DEG <= degrees <= TILT_MAX_DEG:
        print(f"  a tilt limit of {degrees} deg is outside {TILT_MIN_DEG}-{TILT_MAX_DEG}; "
              "refusing.")
        return 9
    profile = fc.status().get("pid_profile")
    if profile is None:
        print("  the flight controller did not report its PID profile; refusing.")
        return 10
    print(f"  active PID profile: {profile}"
          + ("  (the 8-inch tune)" if profile == 2 else
             "  -- NOT profile 2, the 8-inch tune. Check that is intended."))
    try:
        offset, current = find_level_limit_offset(fc, profile)
    except RuntimeError as e:
        print(f"  {e}")
        return 11
    print(f"  tilt limit found at byte {offset} of MSP_PID_ADVANCED: currently {current} deg")
    if current == degrees:
        print("  already set. Nothing to do.")
        return 0
    print(f"  will change it {current} -> {degrees} deg on PID profile {profile}. "
          "ACRO is unaffected.")
    if dry_run:
        print("  --dry-run: nothing written.")
        return 0

    before = bytes(fc.request(MSP_PID_ADVANCED))
    want = bytearray(before)
    want[offset] = degrees
    fc.request(MSP_SET_PID_ADVANCED, bytes(want))
    after = bytes(fc.request(MSP_PID_ADVANCED))
    if after != bytes(want):
        moved = [i for i in range(min(len(after), len(want))) if after[i] != want[i]]
        print(f"  the read-back does not match: bytes {moved} differ from what was "
              "sent. NOT saving.")
        print("  Nothing is persisted. Power-cycle the flight controller to drop "
              "the unsaved change before flying.")
        return 12
    if fc.status().get("pid_profile") != profile:
        print("  the active PID profile changed underneath us. NOT saving.")
        return 13

    fc.request(MSP_EEPROM_WRITE)
    time.sleep(0.5)
    final = bytes(fc.request(MSP_PID_ADVANCED))
    if final[offset] != degrees:
        print("  saved, but the limit reads back differently. Not trusting that.")
        return 14
    print(f"  saved to EEPROM and verified: ANGLE mode now leans to {degrees} deg at most.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Make ANGLE the pilot's default, over MSP")
    ap.add_argument("--dev", default=None, help="/dev/ttyTHS1; omit for the mock")
    ap.add_argument("--show", action="store_true", help="print current assignments")
    ap.add_argument("--watch", action="store_true", help="find which channel a switch drives")
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to watch")
    ap.add_argument("--beginner", action="store_true",
                    help="THE ONE COMMAND: ANGLE whenever MSP override is off (a "
                         "human is flying), ACRO when it is on (the Jetson is), "
                         "plus the beginner tilt limit")
    ap.add_argument("--assign", action="store_true", help="assign ANGLE by hand")
    ap.add_argument("--channel", type=int, default=None,
                    help="RC channel the switch drives (5-16), as --watch reports it")
    ap.add_argument("--acro-at", type=int, default=None, metavar="US",
                    help="what that channel reads with the switch ON ACRO. ANGLE "
                         "is then on everywhere else, so ANGLE is the default")
    ap.add_argument("--low", type=int, default=None,
                    help="explicit range instead of --acro-at (advanced)")
    ap.add_argument("--high", type=int, default=None)
    ap.add_argument("--reassign", action="store_true",
                    help="replace an existing ANGLE assignment")
    ap.add_argument("--tilt-limit", type=int, default=None, metavar="DEG",
                    help="ANGLE-mode tilt limit on the active PID profile; "
                         "25 is the beginner figure, the airframe ships 55")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm a write")
    args = ap.parse_args()

    if args.beginner and args.assign:
        ap.error("--beginner does the assignment itself; drop --assign")
    if args.beginner and args.tilt_limit is None:
        args.tilt_limit = BEGINNER_TILT_DEG
    writing = args.assign or args.beginner or args.tilt_limit is not None
    if not (args.show or args.watch or writing):
        ap.error("pick one of --beginner, --show, --watch, --assign or --tilt-limit")
    low = high = None
    if args.assign:
        if args.channel is None:
            ap.error("--assign needs --channel (run --watch first to find it)")
        if not 5 <= args.channel <= 16:
            ap.error("--channel is an RC channel 5-16; 1-4 are the sticks")
        explicit = args.low is not None or args.high is not None
        if explicit and args.acro_at is not None:
            ap.error("give --acro-at, or --low with --high, not both")
        if explicit:
            if args.low is None or args.high is None:
                ap.error("--low and --high go together")
            low, high = args.low, args.high
        elif args.acro_at is not None:
            try:
                low, high = angle_range_for(args.acro_at)
            except ValueError as e:
                ap.error(str(e))
        else:
            ap.error("--assign needs --acro-at US: what the channel reads with the "
                     "switch ON ACRO (--watch prints it as 'now'). ANGLE then "
                     "becomes the default and ACRO the deliberate flip.")
    if writing and not args.yes and not args.dry_run:
        ap.error("that writes to the flight controller, so it also needs --yes "
                 "(or --dry-run to see what it would do)")

    fc = open_link(args.dev)
    try:
        if writing or args.show:
            st = fc.status()
            if st.get("armed", False):
                print("  the aircraft is ARMED. Refusing to touch it.")
                return 2

        if args.show or args.assign or args.beginner:
            print()
            print("CURRENT MODE ASSIGNMENTS")
            print("=" * 68)
            ranges = read_ranges(fc)
            for e in ranges:
                if not is_empty(e):
                    print(describe(e))
            nfree = sum(1 for e in ranges if is_empty(e))
            print(f"  ({nfree} free slots)")
            a = find_angle(ranges)
            print()
            if a:
                print(f"  ANGLE: assigned -> {describe(a).strip()}")
                print(f"  DEFAULT (switch untouched): {default_mode(a)}")
            else:
                print("  ANGLE: NOT ASSIGNED. No switch selects it, whatever its label")
                print("  says, and the aircraft is ALWAYS in ACRO.")
            now = angle_active_now(fc)
            if now is not None:
                print(f"  RIGHT NOW the flight controller reports: "
                      f"{'ANGLE (self-levelling)' if now else 'ACRO (no self-levelling)'}")
            print("=" * 68)

        if args.watch:
            res = watch(fc, args.seconds)
            print()
            print(f"  {res['samples']} samples")
            if not res["moved"]:
                print("  Nothing moved. Either the transmitter is off or not")
                print("  bound -- check the flight controller is not reporting")
                print("  RX_FAILSAFE before blaming the switch.")
                return 3
            print("  channels that moved:")
            for i, (a, b) in sorted(res["moved"].items()):
                ch = i + 1
                where = "stick" if ch <= 4 else f"aux {ch - 5}"
                print(f"    RC ch {ch:2d} ({where}):  {a} .. {b} us   now {res['now'][i]}")
            print()
            print("  If you left the switch on ACRO, 'now' is the ACRO value. Use it:")
            print(f"    python3 {Path(__file__).name} --dev {args.dev} "
                  f"--assign --channel <N> --acro-at <now> --yes")

        code = 0
        if args.beginner:
            ov = find_override(read_ranges(fc))
            if ov is None:
                print("  MSP OVERRIDE is not assigned on this flight controller, so there")
                print("  is no 'Jetson is flying' switch to hang ANGLE off. Use --watch and")
                print("  --assign --acro-at instead.")
                return 15
            try:
                low, high = angle_range_beside_override(ov)
            except ValueError as e:
                print(f"  {e}")
                return 16
            print()
            print(f"  MSP override (Jetson flies) : RC ch {ov[2] + 5}, {ov[3]}-{ov[4]} us")
            print(f"  ANGLE (a human flies)       : RC ch {ov[2] + 5}, {low}-{high} us  "
                  "<- every other position")
            # An ANGLE left over from the older way of doing this is ours to move.
            code = assign(fc, aux_channel=ov[2], low_us=low, high_us=high,
                          dry_run=args.dry_run, reassign=True)
        elif args.assign:
            code = assign(fc, aux_channel=args.channel - 5, low_us=low, high_us=high,
                          dry_run=args.dry_run, reassign=args.reassign)
        if code == 0 and args.tilt_limit is not None:
            print()
            print("ANGLE TILT LIMIT")
            print("=" * 68)
            code = set_tilt_limit(fc, args.tilt_limit, dry_run=args.dry_run)
            print("=" * 68)
        if code == 0 and (args.assign or args.beginner) and not args.dry_run:
            now = angle_active_now(fc)
            if now is not None:
                print()
                print(f"  The flight controller now reports: "
                      f"{'ANGLE -- self-levelling is ON' if now else 'ACRO -- self-levelling is OFF'}")
                if not now:
                    print("  If the override switch is off, that is wrong. Do not fly; run --show.")
        if code == 0 and writing and not args.dry_run:
            print()
            print("  BENCH TEST BEFORE FLYING, PROPELLERS OFF:")
            print("   1. Leave the switch alone. That should now be ANGLE.")
            print("   2. Throttle up slightly, tilt the aircraft over.")
            print("   3. The low motor should spin up and stay up until it")
            print("      is level again. If it does not, ANGLE is not active.")
            print("   4. Flip to ACRO and tilt again: the motors should NOT fight")
            print("      to level it. Flip back before anybody flies.")
        return code
    finally:
        try:
            fc.close()
        except Exception:
            pass


def _self_test() -> int:
    """Assign ANGLE on the mock, which starts shaped like the real aircraft."""
    fc = open_link("mock")

    ranges = read_ranges(fc)
    assert find_angle(ranges) is None, "the mock should start with ANGLE unassigned"
    assert default_mode(None) == "ACRO", "no assignment means always ACRO"
    assigned_before = [e for e in ranges if not is_empty(e)]
    assert len(assigned_before) == 5, assigned_before
    assert any(e[1] == BOX_ARM for e in assigned_before), "ARM should be assigned"

    # Step conversion must round-trip at the values we actually use.
    for us in (900, 1000, 1300, 1600, 1700, 2100):
        assert step_to_us(us_to_step(us)) == us, us

    # --- the point of this tool: ANGLE is what you get by default -----------
    # The two switch ends measured on the aircraft on 18 Sep were 1094 and 1992.
    for acro_at in (1992, 2000, 1811):
        lo, hi = angle_range_for(acro_at)
        assert (lo, hi) == (900, 1700), (acro_at, lo, hi)
        assert not lo <= acro_at < hi, "the ACRO position must be outside ANGLE"
        for resting in (1000, 1094, 1500):
            assert lo <= resting < hi, f"{resting} should be ANGLE with ACRO at {acro_at}"
    for acro_at in (1094, 1000, 988):
        lo, hi = angle_range_for(acro_at)
        assert (lo, hi) == (1300, 2100), (acro_at, lo, hi)
        assert not lo <= acro_at < hi
        for resting in (1500, 1992, 2000):
            assert lo <= resting < hi
    # A value near the middle cannot be "the ACRO end" of anything.
    for bad in (1500, 1350, 1650, 1750, 1250):
        try:
            angle_range_for(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"--acro-at {bad} should have been refused")

    # Dry run writes nothing.
    lo, hi = angle_range_for(1992)
    assign(fc, aux_channel=1, low_us=lo, high_us=hi, dry_run=True)
    assert find_angle(read_ranges(fc)) is None, "dry run must not write"
    assert fc.eeprom_writes == 0

    # Refuses a channel already carrying a mode with an overlapping range.
    code = assign(fc, aux_channel=2, low_us=1600, high_us=2100, dry_run=False)
    assert code == 5, f"expected a clash refusal on aux 2, got {code}"
    code = assign(fc, aux_channel=2, low_us=lo, high_us=hi, dry_run=False)
    assert code == 5, "900-1700 overlaps the 1600-2100 mode on aux 2 as well"
    assert find_angle(read_ranges(fc)) is None
    assert fc.eeprom_writes == 0, "a refused assignment must not save"

    # An older run of this tool left ANGLE in the upper half: ACRO by default.
    code = assign(fc, aux_channel=1, low_us=DEFAULT_LOW_US, high_us=DEFAULT_HIGH_US,
                  dry_run=False)
    assert code == 0, code
    a = find_angle(read_ranges(fc))
    assert a is not None and a[2] == 1 and a[3] == 1600 and a[4] == 2100, a
    assert default_mode(a) == "ACRO", "the legacy range leaves ACRO as the default"
    assert fc.eeprom_writes == 1, fc.eeprom_writes

    # Asking for something different must not silently move it...
    code = assign(fc, aux_channel=1, low_us=lo, high_us=hi, dry_run=False)
    assert code == 8 and fc.eeprom_writes == 1, (code, fc.eeprom_writes)
    assert find_angle(read_ranges(fc))[3:] == (1600, 2100)
    # ...but --reassign moves it, in its own slot, and now ANGLE is the default.
    slot_before = find_angle(read_ranges(fc))[0]
    code = assign(fc, aux_channel=1, low_us=lo, high_us=hi, dry_run=False, reassign=True)
    assert code == 0, code
    a = find_angle(read_ranges(fc))
    assert a[0] == slot_before and (a[2], a[3], a[4]) == (1, 900, 1700), a
    assert default_mode(a) == "ANGLE"
    assert fc.eeprom_writes == 2

    # Nothing else moved.
    after = [e for e in read_ranges(fc) if not is_empty(e)]
    assert len(after) == 6, after
    for e in assigned_before:
        assert e in after, f"assignment {e} was disturbed"

    # Running it again is a no-op rather than a second slot.
    code = assign(fc, aux_channel=1, low_us=lo, high_us=hi, dry_run=False)
    assert code == 0 and fc.eeprom_writes == 2, "second run should not rewrite"
    assert len([e for e in read_ranges(fc) if e[1] == BOX_ANGLE and not is_empty(e)]) == 1
    fc.close()

    # --- --beginner: ANGLE exactly where MSP override is off ------------------
    b = open_link("mock")
    ov = find_override(read_ranges(b))
    assert ov is not None and (ov[2], ov[3], ov[4]) == (4, 1700, 2100), ov
    blo, bhi = angle_range_beside_override(ov)
    assert (blo, bhi) == (900, 1700), (blo, bhi)
    assert angle_active_now(b) is False, "nothing assigned yet: ACRO"
    code = assign(b, aux_channel=ov[2], low_us=blo, high_us=bhi, dry_run=False, reassign=True)
    assert code == 0, f"sharing the override channel without overlapping must be allowed, got {code}"
    assert find_override(read_ranges(b)) == ov, "the override assignment must be untouched"
    # The two switch positions seen on that channel in the 18 Sep flight log,
    # and an unsent channel: a human is flying, so ANGLE.
    for us in (1094, 1520, 1500):
        b.aux = {9: us}
        assert angle_active_now(b) is True, f"ch9 at {us}: override off, must be ANGLE"
    # Override engaged: the Jetson is flying, so ANGLE must be gone.
    for us in (1700, 1992, 2012):
        b.aux = {9: us}
        assert angle_active_now(b) is False, f"ch9 at {us}: override on, must be ACRO"
    b.close()
    # Override parked somewhere odd: refuse rather than invent a range.
    for odd in ((0, 50, 4, 1300, 1700), (0, 50, 4, 1100, 1500)):
        try:
            angle_range_beside_override(odd)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{odd} should have been refused")
    assert angle_range_beside_override((0, 50, 4, 900, 1300)) == (1300, 2100)

    # --- tilt limit ----------------------------------------------------------
    # Found, not assumed: move the byte and the tool must still find it.
    for where in (17, 16, 22):
        t = open_link("mock")
        t.level_limit_offset = where
        off, cur = find_level_limit_offset(t, 2)
        assert (off, cur) == (where, 55), (where, off, cur)
        assert t.pid_profile == 2, "must leave the flight controller on its own profile"
        t.close()

    t = open_link("mock")
    assert set_tilt_limit(t, 25, dry_run=True) == 0 and t.eeprom_writes == 0
    assert t.request(MSP_PID_ADVANCED)[17] == 55, "dry run must not write"
    untouched = {p: bytes(t._pid_advanced_payloads()[p]) for p in (0, 1, 3)}
    before = bytes(t.request(MSP_PID_ADVANCED))
    assert set_tilt_limit(t, 25, dry_run=False) == 0
    after_b = bytes(t.request(MSP_PID_ADVANCED))
    assert after_b[17] == 25 and t.eeprom_writes == 1 and t.pid_profile == 2
    assert [i for i in range(len(before)) if before[i] != after_b[i]] == [17], \
        "exactly one byte may change"
    for p, b in untouched.items():
        assert bytes(t._pid_advanced_payloads()[p]) == b, f"profile {p} was disturbed"
    # Still findable afterwards, because the anchors are profiles we never write.
    assert find_level_limit_offset(t, 2) == (17, 25)
    assert set_tilt_limit(t, 25, dry_run=False) == 0 and t.eeprom_writes == 1, "no-op re-run"
    assert set_tilt_limit(t, 5, dry_run=False) == 9, "a 5 degree limit is a typo"
    t.close()

    # A write that disturbs another byte must not be saved.
    t = open_link("mock")
    t.pid_advanced_corrupts = True
    assert set_tilt_limit(t, 25, dry_run=False) == 12 and t.eeprom_writes == 0
    t.close()

    # If a vendor limit has been changed the anchors no longer identify the
    # byte, and the tool must refuse rather than guess.
    t = open_link("mock")
    t._pid_advanced_payloads()[1][17] = 40
    assert set_tilt_limit(t, 25, dry_run=False) == 11 and t.eeprom_writes == 0
    assert t.pid_profile == 2
    t.close()

    # Armed: MSP_SELECT_SETTING is ignored, the profile check catches it.
    t = open_link("mock")
    t.arm_state(True)
    try:
        find_level_limit_offset(t, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("profile selection while armed must be noticed")
    t.close()

    print("setup_angle_mode: all checks passed")
    print("  mock starts shaped like the real aircraft: 5 modes, ANGLE absent -> always ACRO")
    print("  --acro-at 1992 -> ANGLE 900-1700; --acro-at 1094 -> ANGLE 1300-2100;")
    print("    the rest position, the middle and an unsent channel are all ANGLE")
    print("  refused to share aux 2 with an existing mode, and refused a mid-travel --acro-at")
    print("  an old upper-half assignment (ACRO default) is only moved with --reassign")
    print("  left all five existing assignments untouched; re-running is a no-op")
    print("  --beginner: ANGLE on the override channel at 900-1700; ch9 at 1094/1520 -> ANGLE,")
    print("    ch9 at 1700/1992 (Jetson flying) -> ACRO; override assignment untouched")
    print("  tilt limit: byte found at 3 different offsets, 55 -> 25 changed exactly one byte,")
    print("    other PID profiles untouched, a corrupting write and a moved anchor both refused")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
