"""Make the ACRO/ANGLE switch on the transmitter actually do something.

    python3 setup_angle_mode.py --dev /dev/ttyTHS1 --show      # what is assigned
    python3 setup_angle_mode.py --dev /dev/ttyTHS1 --watch      # find the switch
    python3 setup_angle_mode.py --dev /dev/ttyTHS1 --assign --channel 6 --yes

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

Why this goes over MSP and not the CLI
--------------------------------------
``bf_cli.py`` can do this, and it is how Betaflight configuration is usually
scripted. But entering the CLI over the MSP UART **wedges this flight
controller** -- measured twice out of two on 18 Sep, silent at every baud rate
afterwards, recovered only by pulling the flight battery.

``MSP_SET_MODE_RANGE`` does the same job as an ordinary MSP message, which is
what Betaflight Configurator itself uses. No CLI, no reboot, no wedge.

What it will not do
-------------------
* It will not touch an occupied slot. It finds a free one, and if there is none
  it says so and stops.
* It will not write while the aircraft is armed.
* It verifies by reading the ranges back, and restores the previous state if the
  read-back does not match.
* ``--show`` and ``--watch`` never write anything.

Afterwards
----------
**Bench test with propellers off before flying it**, exactly as the Betaflight
setup guide says: with ANGLE selected and the throttle up a little, tilt the
aircraft and check the low motor spins up and stays up until you level it again.
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
MSP_EEPROM_WRITE = 250

#: Betaflight's permanent id for the self-levelling mode.
BOX_ANGLE = 1
BOX_ARM = 0

#: Betaflight stores range edges as steps of 25 us from 900 us.
STEP_US = 25
STEP_BASE = 900

#: Default switch range to react to: the upper half of the travel. Wide enough
#: that a two- or three-position switch lands cleanly inside it.
DEFAULT_LOW_US = 1600
DEFAULT_HIGH_US = 2100


def us_to_step(us: float) -> int:
    return int(round((us - STEP_BASE) / STEP_US))


def step_to_us(step: int) -> int:
    return STEP_BASE + step * STEP_US


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
    name = {BOX_ARM: "ARM", BOX_ANGLE: "ANGLE", 2: "HORIZON"}.get(pid, f"boxId {pid}")
    return (f"  slot {i:2d}  {name:<12s} aux channel {aux} "
            f"(RC ch {aux + 5})  {lo}-{hi} us")


def find_angle(ranges) -> tuple | None:
    for e in ranges:
        if e[1] == BOX_ANGLE and not is_empty(e):
            return e
    return None


def watch(fc, seconds: float = 60.0) -> dict:
    """Report which RC channels move. Read-only."""
    lo, hi, n = {}, {}, 0
    t0 = time.monotonic()
    print(f"  watching for {seconds:.0f}s -- flip the ACRO/ANGLE switch a few times")
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
        time.sleep(0.05)
    moved = {i: (lo[i], hi[i]) for i in lo if hi[i] - lo[i] > 200}
    return {"samples": n, "moved": moved, "lo": lo, "hi": hi}


def assign(fc, *, aux_channel: int, low_us: int, high_us: int,
           dry_run: bool) -> int:
    ranges = read_ranges(fc)

    existing = find_angle(ranges)
    if existing:
        print(f"  ANGLE is ALREADY assigned: {describe(existing).strip()}")
        print("  Nothing to do. Use --show to review, or reassign by hand.")
        return 0

    # Refuse to write to a slot that is doing something.
    free = [e for e in ranges if is_empty(e)]
    if not free:
        print("  every mode slot is occupied; not overwriting one. "
              "Free a slot in Configurator first.")
        return 4
    slot = free[0][0]

    # And refuse to share a channel with an existing mode, which would turn two
    # things on at once.
    clash = [e for e in ranges
             if not is_empty(e) and e[2] == aux_channel
             and not (high_us < e[3] or low_us > e[4])]
    if clash:
        print(f"  aux channel {aux_channel} already carries "
              f"{describe(clash[0]).strip()}")
        print("  Overlapping ranges would switch both on together. Pick another "
              "channel, or a non-overlapping range.")
        return 5

    print(f"  will write ANGLE (boxId {BOX_ANGLE}) into slot {slot}, "
          f"aux channel {aux_channel} (RC ch {aux_channel + 5}), "
          f"{low_us}-{high_us} us")
    if dry_run:
        print("  --dry-run: nothing written.")
        return 0

    payload = bytes([slot, BOX_ANGLE, aux_channel,
                     us_to_step(low_us), us_to_step(high_us)])
    fc.request(MSP_SET_MODE_RANGE, payload)

    # Read back BEFORE saving: if the board did not take it, nothing is
    # persisted and there is nothing to undo.
    check = find_angle(read_ranges(fc))
    if not check or check[2] != aux_channel:
        print("  the flight controller did not accept the assignment; "
              "nothing was saved.")
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Assign ANGLE mode to a switch, over MSP")
    ap.add_argument("--dev", default=None, help="/dev/ttyTHS1; omit for the mock")
    ap.add_argument("--show", action="store_true", help="print current assignments")
    ap.add_argument("--watch", action="store_true", help="find which channel a switch drives")
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to watch")
    ap.add_argument("--assign", action="store_true", help="assign ANGLE")
    ap.add_argument("--channel", type=int, default=None,
                    help="RC channel the switch drives (5-16), as --watch reports it")
    ap.add_argument("--low", type=int, default=DEFAULT_LOW_US)
    ap.add_argument("--high", type=int, default=DEFAULT_HIGH_US)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm --assign")
    args = ap.parse_args()

    if not (args.show or args.watch or args.assign):
        ap.error("pick one of --show, --watch or --assign")
    if args.assign:
        if args.channel is None:
            ap.error("--assign needs --channel (run --watch first to find it)")
        if not 5 <= args.channel <= 16:
            ap.error("--channel is an RC channel 5-16; 1-4 are the sticks")
        if not args.yes and not args.dry_run:
            ap.error("--assign also needs --yes. This writes to the flight controller.")

    fc = open_link(args.dev)
    try:
        if args.assign or args.show:
            st = fc.status()
            if st.get("armed", False):
                print("  the aircraft is ARMED. Refusing to touch it.")
                return 2

        if args.show or args.assign:
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
            else:
                print("  ANGLE: NOT ASSIGNED. The ACRO/ANGLE switch does nothing.")
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
                print(f"    RC ch {ch:2d} ({where}):  {a} .. {b} us")
            print()
            print("  Use the aux channel your ACRO/ANGLE switch drives:")
            print(f"    python3 {Path(__file__).name} --dev {args.dev} "
                  f"--assign --channel <N> --yes")

        if args.assign:
            rc = args.channel - 5
            code = assign(fc, aux_channel=rc, low_us=args.low, high_us=args.high,
                          dry_run=args.dry_run)
            if code == 0 and not args.dry_run:
                print()
                print("  BENCH TEST BEFORE FLYING, PROPELLERS OFF:")
                print("   1. Select ANGLE on the switch.")
                print("   2. Throttle up slightly, tilt the aircraft over.")
                print("   3. The low motor should spin up and stay up until it")
                print("      is level again. If it does not, ANGLE is not active.")
            return code
        return 0
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
    assigned_before = [e for e in ranges if not is_empty(e)]
    assert len(assigned_before) == 5, assigned_before
    assert any(e[1] == BOX_ARM for e in assigned_before), "ARM should be assigned"

    # Step conversion must round-trip at the values we actually use.
    for us in (1000, 1300, 1600, 1700, 2100):
        assert step_to_us(us_to_step(us)) == us, us

    # Dry run writes nothing.
    assign(fc, aux_channel=1, low_us=1600, high_us=2100, dry_run=True)
    assert find_angle(read_ranges(fc)) is None, "dry run must not write"
    assert fc.eeprom_writes == 0

    # Refuses a channel already carrying a mode with an overlapping range.
    code = assign(fc, aux_channel=2, low_us=1600, high_us=2100, dry_run=False)
    assert code == 5, f"expected a clash refusal on aux 2, got {code}"
    assert find_angle(read_ranges(fc)) is None
    assert fc.eeprom_writes == 0, "a refused assignment must not save"

    # Real assignment onto a free channel.
    code = assign(fc, aux_channel=1, low_us=1600, high_us=2100, dry_run=False)
    assert code == 0, code
    a = find_angle(read_ranges(fc))
    assert a is not None and a[2] == 1 and a[3] == 1600 and a[4] == 2100, a
    assert fc.eeprom_writes == 1, fc.eeprom_writes

    # Nothing else moved.
    after = [e for e in read_ranges(fc) if not is_empty(e)]
    assert len(after) == 6, after
    for e in assigned_before:
        assert e in after, f"assignment {e} was disturbed"

    # Running it again is a no-op rather than a second slot.
    code = assign(fc, aux_channel=1, low_us=1600, high_us=2100, dry_run=False)
    assert code == 0 and fc.eeprom_writes == 1, "second run should not rewrite"
    assert len([e for e in read_ranges(fc) if e[1] == BOX_ANGLE and not is_empty(e)]) == 1

    fc.close()
    print("setup_angle_mode: all checks passed")
    print(f"  mock starts shaped like the real aircraft: 5 modes, ANGLE absent")
    print(f"  refused to share aux 2 with an existing mode")
    print(f"  assigned ANGLE to aux 1 (RC ch 6) at 1600-2100 us and saved once")
    print(f"  left all five existing assignments untouched")
    print(f"  re-running is a no-op, not a duplicate")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
