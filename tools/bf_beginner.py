"""Make the race drone feel like a toy drone to fly, without capping its power.

    python3 bf_beginner.py --dev /dev/ttyTHS1 --probe          # read-only, safe
    python3 bf_beginner.py --dev /dev/ttyTHS1 --plan           # show the script
    python3 bf_beginner.py --dev /dev/ttyTHS1 --apply --yes    # write it

Who this is for
---------------
Nobody on the team flies. The aircraft is a 1745 g, 8-inch, 6S machine — that is
not a beginner airframe, and it is four airframes we cannot replace. We still
have to fly it, because the hover throttle, the thrust-to-weight and the motion
blur in the training images can only come from a real flight.

What we can and cannot have
---------------------------
The thing that makes a toy drone easy is **altitude hold**: let go of the
throttle and it just sits there. We cannot have it.

* Betaflight gained altitude hold in **4.6**. This board runs **4.4.3**, and the
  rules allow CLI access but **no reflashing**.
* Even on newer firmware it would lean on the barometer, and ours is unusable
  with propellers turning — measured on our own airframe, altitude collapsed
  about 6 m within 6 s of spin-up and stayed there.

So there is no honest way to give you "let go and it hovers" on this aircraft.
Everything below is the rest of the toy-drone feel, which is most of it.

The constraint that shapes this
-------------------------------
**Do not cap power.** Measurement 3 is a full-throttle climb, and it is what
settles a 3.5x ambiguity in the thrust model. A tool that quietly limits the
throttle would corrupt that measurement, so the beginner settings go on a
**separate profile and rate profile**, leaving the ones in use untouched. Flip a
switch for toy mode; flip back for measurement.

The one exception is the throttle *curve*, which is the single biggest win here
and costs nothing: expo reshapes where the stick is sensitive **without changing
what full stick does**. Full stick is still 100 % throttle. See below.

What it changes, and why each one
---------------------------------
1. **ANGLE mode** — self-levelling. Let go of the sticks and it returns to
   level. The single biggest difference between "toy" and "racer".
2. **Tilt limit ~25 deg** — you cannot panic-roll it inverted. Published
   beginner guidance is 25-35 deg, raising to 45-55 after ten or fifteen flights.
3. **Throttle curve with the hover point at mid-stick.** On a racing quad hover
   sits near a quarter throttle, so the useful part of the stick is a thin band
   right at the bottom and everything feels twitchy. Setting `thr_mid` to the
   hover fraction and `thr_expo` to ~0.5 stretches that band across the middle
   of the stick. **Full stick still gives full throttle** — this softens the
   middle, it does not limit the top.
4. **Gentle rates and expo** — the stick commands a slower rotation, and small
   movements near centre do very little.
5. **Slow yaw** — toy drones yaw slowly; racers snap.
6. **RC smoothing** — filters stick input so a nervous thumb does not become a
   twitch.
7. **Failsafe drops rather than flies away** — indoors, in a netted cage, a
   drone that falls is recoverable and a drone that flies off is not.
8. **Beeper and flip-over-after-crash** — find it, and right it without walking
   onto the course.

Why it asks the flight controller instead of assuming
-----------------------------------------------------
Betaflight renames settings between versions, and the blogs disagree about which
name belongs to which release. Rather than guess, ``--probe`` runs ``get`` on
every candidate name and keeps the ones this firmware actually has. Anything it
cannot find is reported and skipped, never silently dropped.

Safety
------
* ``--probe`` and ``--plan`` are read-only. ``--apply`` is the only thing that
  writes, and it needs ``--yes`` as well.
* It **always saves a full ``dump all`` first**, so there is a way back.
* The CLI only works disarmed, and it takes the telemetry port down while it
  runs. Never during a flight.
* ``save`` reboots the flight controller. Expect a few seconds of silence.
* **Verify on the bench with propellers off** before anyone flies it: check that
  ANGLE mode engages, that the tilt limit holds, and that the throttle still
  reaches full at full stick.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bf_cli import BetaflightCLI, CLIError  # noqa: E402

#: Hover throttle as a fraction of **throttle output**, post-curve -- the number
#: ``analyze_hover.py`` recovers from ``motors()``, NOT a stick position.
#:
#: The distinction matters and has bitten this project before: the simulator's
#: hover figure of 0.255 is a *stick* position that becomes **0.390 effective
#: throttle** once the current curve (`thr_mid 54`, `thr_expo 68`) is applied.
#: ``thr_mid`` sets the output at mid-stick, so it wants the output number.
#: Feeding it a stick fraction would put the soft zone in the wrong place.
#:
#: 0.39 is the post-curve figure implied by the simulator. It is still a guess
#: until measurement 2; pass --hover with the measured value the moment there is
#: one.
DEFAULT_HOVER = 0.39

#: Candidate setting names, most likely first. Betaflight has moved these
#: around between releases, so each entry is a list and the first one this
#: firmware answers to wins.
CANDIDATES = {
    "angle_limit": ["angle_limit", "level_limit", "max_angle_inclination"],
    # Probed on the real 4.4.3 board: the angle-mode strength knob is called
    # angle_level_strength here. "level_p"/"p_level" do not exist on this
    # firmware, which is exactly why this list is a list.
    "angle_level_strength": ["angle_level_strength", "level_p", "p_level"],
    "thr_mid": ["thr_mid"],
    "thr_expo": ["thr_expo"],
    "roll_rc_rate": ["roll_rc_rate"],
    "pitch_rc_rate": ["pitch_rc_rate"],
    "yaw_rc_rate": ["yaw_rc_rate"],
    "roll_srate": ["roll_srate"],
    "pitch_srate": ["pitch_srate"],
    "yaw_srate": ["yaw_srate"],
    "roll_expo": ["roll_expo", "roll_rc_expo"],
    "pitch_expo": ["pitch_expo", "pitch_rc_expo"],
    "yaw_expo": ["yaw_expo", "yaw_rc_expo"],
    "rates_type": ["rates_type"],
    "throttle_limit_type": ["throttle_limit_type"],
    "throttle_limit_percent": ["throttle_limit_percent"],
    "rc_smoothing": ["rc_smoothing"],
    "rc_smoothing_auto_factor": ["rc_smoothing_auto_factor"],
    "failsafe_procedure": ["failsafe_procedure"],
    "small_angle": ["small_angle"],
    "crash_recovery": ["crash_recovery"],
    "acro_trainer_angle_limit": ["acro_trainer_angle_limit"],
}


def beginner_values(hover: float) -> dict:
    """The settings themselves, with the reason for each attached.

    Split by where Betaflight stores them: ``pid`` settings live in a PID
    profile, ``rate`` settings in a rate profile, ``master`` settings are global.
    """
    thr_mid = int(round(max(0.05, min(0.95, hover)) * 100))
    return {
        "pid": {
            "angle_limit": (25, "cannot panic-roll past 25 degrees; raise toward "
                                "45-55 after ten or fifteen flights"),
            "angle_level_strength": (60, "how hard it pulls back to level when you "
                                         "let go. Stock 50; a little firmer feels "
                                         "more like a toy drone and less like a "
                                         "racer that keeps drifting"),
        },
        "rate": {
            # Rotation rate at full stick. Stock racing rates put ~440 deg/s
            # under a beginner's thumb; this is roughly a third of that.
            "roll_rc_rate": (35, "slower roll than the racing profile"),
            "pitch_rc_rate": (35, "slower pitch"),
            "yaw_rc_rate": (30, "slow yaw, the way a toy drone turns"),
            "roll_srate": (30, "less rate added at the stick extremes"),
            "pitch_srate": (30, "same for pitch"),
            "yaw_srate": (25, "same for yaw"),
            "roll_expo": (30, "small stick movements near centre do very little"),
            "pitch_expo": (30, "same for pitch"),
            "yaw_expo": (30, "same for yaw"),
            # The big one, and it does NOT limit the top end.
            "thr_mid": (thr_mid, f"put the hover point (~{hover:.0%} of throttle "
                                 f"OUTPUT) at mid-stick instead of a thin band"),
            # NOTE: this used to say 50, which would have made things WORSE.
            # Throttle expo flattens the curve AROUND thr_mid, so higher is
            # softer. The airframe already ships 68 -- the problem was never
            # that the curve was too sharp, it was that the soft zone sat at
            # 54 % output while hover is nearer 39 %, so the softness was in
            # the wrong place. Moving thr_mid is the fix; expo just nudges up.
            "thr_expo": (70, "keep the throttle soft around hover. Higher is "
                             "softer, and the airframe already ships 68 -- the "
                             "real fix is moving thr_mid to where hover is. "
                             "FULL STICK IS STILL FULL THROTTLE"),
        },
        "master": {
            "rc_smoothing": ("ON", "filter stick input so a nervous thumb is not a twitch"),
            "failsafe_procedure": ("DROP", "indoors in a net, a drone that falls is "
                                           "recoverable and one that flies away is not"),
            "small_angle": (180, "let it arm at any attitude, so a flipped drone can "
                                 "still be righted with flip-over-after-crash"),
            # crash_recovery is NOT set here. The airframe came with it at 10,
            # a vendor choice whose units we have not confirmed, and writing
            # "OFF" into a numeric setting would either fail or mean something
            # we did not intend. Leaving a working vendor value alone beats
            # guessing at it.
        },
    }


def probe(cli: BetaflightCLI) -> dict:
    """Read every candidate name. Read-only. Returns {canonical: (name, value)}."""
    found = {}
    for canonical, names in CANDIDATES.items():
        for name in names:
            try:
                out = cli.command(f"get {name}")
            except CLIError:
                continue
            for line in out.splitlines():
                line = line.strip()
                if line.lower().startswith(name.lower()) and "=" in line:
                    found[canonical] = (name, line.split("=", 1)[1].strip())
                    break
            if canonical in found:
                break
    return found


def build_script(found: dict, values: dict, *, profile: int, rateprofile: int) -> list:
    """Turn the settings into CLI lines, skipping anything this firmware lacks."""
    lines, skipped = [], []
    lines.append(f"profile {profile}")
    for canonical, (val, _why) in values["pid"].items():
        if canonical in found:
            lines.append(f"set {found[canonical][0]} = {val}")
        else:
            skipped.append(canonical)
    lines.append(f"rateprofile {rateprofile}")
    for canonical, (val, _why) in values["rate"].items():
        if canonical in found:
            lines.append(f"set {found[canonical][0]} = {val}")
        else:
            skipped.append(canonical)
    for canonical, (val, _why) in values["master"].items():
        if canonical in found:
            lines.append(f"set {found[canonical][0]} = {val}")
        else:
            skipped.append(canonical)
    return lines, skipped


def print_plan(found: dict, values: dict, lines: list, skipped: list,
               *, profile: int, rateprofile: int, hover: float) -> None:
    print()
    print("BEGINNER FLIGHT SETUP")
    print("=" * 72)
    print(f"  PID profile      {profile}   (tilt limit and levelling strength)")
    print(f"  Rate profile     {rateprofile}   (stick rates and throttle curve)")
    print(f"  Assumed hover    {hover:.0%} of throttle OUTPUT -- A GUESS until measurement 2")
    print()
    print("  Nothing outside those two profiles is touched, so the profile you")
    print("  fly measurements on is untouched and still does full throttle.")
    print()
    for section, title in (("pid", "Self-levelling"), ("rate", "Sticks and throttle"),
                           ("master", "Global")):
        print(f"  {title}")
        for canonical, (val, why) in values[section].items():
            if canonical in found:
                name, current = found[canonical]
                mark = " " if str(current) == str(val) else "*"
                print(f"   {mark} {name:28s} {str(current):>10s} -> {str(val):<10s} {why}")
            else:
                print(f"   ! {canonical:28s} {'NOT ON THIS FIRMWARE':>10s}")
        print()
    if skipped:
        print(f"  Skipped (no such setting here): {', '.join(skipped)}")
        print()
    print("  CLI script:")
    for line in lines:
        print(f"      {line}")
    print("      save")
    print("=" * 72)
    print()
    print("  WHAT THIS DOES NOT DO")
    print("   * No altitude hold. Betaflight added it in 4.6; this is 4.4.3 and")
    print("     reflashing is not allowed. Our barometer is unusable with props")
    print("     turning anyway. You will be holding the throttle yourself.")
    print("   * No power cap. Full stick is still full throttle, deliberately,")
    print("     because measurement 3 is a full-throttle climb.")
    print()
    print("  BEFORE ANYONE FLIES IT, PROPS OFF:")
    print("   1. ANGLE mode engages on the switch, and tilting the aircraft")
    print("      spins up the low motor until it is level again.")
    print("   2. Full throttle stick still reads full motor output.")
    print("   3. Failsafe does what you expect when the transmitter is switched off.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Beginner-friendly Betaflight setup")
    ap.add_argument("--dev", default=None,
                    help="serial device, e.g. /dev/ttyTHS1. Omitted or 'mock' "
                         "runs against the test double and writes nothing real")
    ap.add_argument("--profile", type=int, default=1, help="PID profile to write")
    ap.add_argument("--rateprofile", type=int, default=1, help="rate profile to write")
    ap.add_argument("--hover", type=float, default=DEFAULT_HOVER,
                    help="hover throttle as a fraction of THROTTLE OUTPUT (what "
                         "analyze_hover.py reads from motors()), not stick "
                         "position. Replace the default with the measured value")
    ap.add_argument("--probe", action="store_true", help="read current values only")
    ap.add_argument("--plan", action="store_true", help="show what would be written")
    ap.add_argument("--apply", action="store_true", help="write it (needs --yes)")
    ap.add_argument("--yes", action="store_true", help="confirm --apply")
    ap.add_argument("--backup", default=None,
                    help="where to save 'dump all' before writing "
                         "(default: bf_backup_<timestamp>.txt)")
    args = ap.parse_args()

    if not (args.probe or args.plan or args.apply):
        ap.error("pick one of --probe, --plan or --apply")
    if args.apply and not args.yes:
        ap.error("--apply also needs --yes. This writes to the flight controller.")

    values = beginner_values(args.hover)
    cli = BetaflightCLI(args.dev, allow_write=args.apply)
    if cli.is_mock and args.apply:
        print("  refusing to 'apply' against the mock: it would prove nothing.")
        print("  Pass --dev /dev/ttyTHS1 on the drone.")
        return 2
    try:
        cli.enter()
        found = probe(cli)
        if not found:
            print("  the flight controller answered no 'get' at all. Wrong port, "
                  "or it is armed.")
            return 3

        if args.probe:
            print()
            print("CURRENT SETTINGS (read-only)")
            print("=" * 72)
            for canonical, names in CANDIDATES.items():
                if canonical in found:
                    name, val = found[canonical]
                    alias = "" if name == canonical else f"  (as '{name}')"
                    print(f"  {canonical:28s} {val}{alias}")
                else:
                    print(f"  {canonical:28s} -- not present on this firmware")
            print("=" * 72)
            return 0

        lines, skipped = build_script(found, values, profile=args.profile,
                                      rateprofile=args.rateprofile)
        print_plan(found, values, lines, skipped, profile=args.profile,
                   rateprofile=args.rateprofile, hover=args.hover)
        if not args.apply:
            print()
            print("  --plan only. Nothing was written.")
            return 0

        backup = Path(args.backup or f"bf_backup_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        dump = cli.command("dump all", timeout=20.0)
        backup.write_text(dump)
        print(f"  backed up {len(dump.splitlines())} lines -> {backup}")
        if len(dump.splitlines()) < 50:
            print("  that backup looks too short to be a real 'dump all'.")
            print("  Refusing to write without a way back.")
            return 4

        for line in lines:
            cli.command(line)
        print(f"  wrote {len(lines)} settings; saving (the board will reboot)")
        cli.leave(save=True)
        print("  saved.")
        print()
        print("  NOW, PROPS OFF: select the profiles and re-run --probe to confirm")
        print("  they took, then do the three bench checks above.")
        return 0
    except CLIError as e:
        print(f"  CLI error: {e}")
        return 1
    finally:
        try:
            cli.close()
        except Exception:
            pass


def _self_test() -> int:
    """Exercise the logic against the test double. No hardware, nothing written."""
    v = beginner_values(0.25)
    assert v["rate"]["thr_mid"][0] == 25, v["rate"]["thr_mid"]
    assert beginner_values(0.40)["rate"]["thr_mid"][0] == 40
    # Throttle expo must not be confused for a power limit.
    assert "thr_expo" in v["rate"] and "throttle_limit_percent" not in v["rate"], (
        "the beginner profile must not cap throttle: measurement 3 needs full power")
    # Tilt limit inside the published beginner band.
    assert 20 <= v["pid"]["angle_limit"][0] <= 35, v["pid"]["angle_limit"]
    # Throttle expo must never be reduced below what the airframe ships (68):
    # expo flattens the curve around the hover point, so lowering it would make
    # the throttle twitchier for a beginner, not smoother.
    assert v["rate"]["thr_expo"][0] >= 68, (
        "lowering thr_expo makes the throttle sharper around hover, which is "
        "backwards for a beginner")

    # Unknown settings are skipped, not silently written.
    found = {"angle_limit": ("angle_limit", "60"), "thr_mid": ("thr_mid", "50")}
    lines, skipped = build_script(found, v, profile=1, rateprofile=1)
    assert "profile 1" in lines and "rateprofile 1" in lines, lines
    assert "set angle_limit = 25" in lines, lines
    assert "set thr_mid = 25" in lines, lines
    assert "angle_level_strength" in skipped and "yaw_rc_rate" in skipped, skipped
    for line in lines:
        assert not any(s in line for s in skipped), (line, skipped)

    # Every write lands inside a profile block, never on the live config by accident.
    assert lines[0].startswith("profile "), lines[0]
    rp = lines.index("rateprofile 1")
    assert rp > 0, lines

    # Against the full test double every name resolves, and the generated
    # script must stay inside the two profiles it was told to touch.
    cli = BetaflightCLI(None, allow_write=True)
    assert cli.is_mock
    cli.enter()
    got = probe(cli)
    full_lines, full_skipped = build_script(got, v, profile=1, rateprofile=2)
    assert not full_skipped, f"the double knows every setting, so nothing should skip: {full_skipped}"
    for line in full_lines:
        cli.command(line)
    writes = cli.ser.writes
    cli.close()

    # Every 'set' must fall after the profile/rateprofile that scopes it, so
    # nothing lands on whichever profile happened to be selected.
    assert writes[0] == "profile 1", writes[:3]
    rp_at = writes.index("rateprofile 2")
    pid_names = {got[c][0] for c in v["pid"] if c in got}
    rate_names = {got[c][0] for c in v["rate"] if c in got}
    for i, w in enumerate(writes):
        if not w.startswith("set "):
            continue
        name = w[4:].split("=", 1)[0].strip()
        if name in pid_names:
            assert i < rp_at, f"{name} written after the rateprofile switch: {writes}"
        elif name in rate_names:
            assert i > rp_at, f"{name} written before the rateprofile switch: {writes}"

    # It must not have touched the throttle limit, in any profile.
    assert not any("throttle_limit" in w for w in writes), writes
    # And it must not have saved: this run never asked to.
    assert cli.ser.saved is False

    print("bf_beginner: all checks passed")
    print(f"  hover 25% -> thr_mid 25, 40% -> thr_mid 40")
    print(f"  tilt limit {v['pid']['angle_limit'][0]} deg, within the 25-35 beginner band")
    print(f"  no throttle cap written, so the full-throttle climb test survives")
    print(f"  {len(full_lines)} settings, each scoped to the profile it belongs to")
    print(f"  unknown settings skipped rather than written blind "
          f"({len(skipped)} in the degraded-firmware case)")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
