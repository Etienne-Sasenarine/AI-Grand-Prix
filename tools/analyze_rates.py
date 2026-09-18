"""Turn a rate-step log into the simulator's rate-loop parameters.

    python3 analyze_rates.py rates.csv

This is what replaces measuring rotational inertia — which is worth stating
plainly, because the team document tells you to do the opposite.

The simulator's rotational dynamics are

    omega_dot = (kp/I)(omega_des - omega) - (kd/I) omega,  clamped at +-M/I

**Only ratios appear.** The absolute inertia is not identifiable from flight and
is not needed. It is also not settable: the drone's USD sets
``physics:diagonalInertia = (0,0,0)``, so PhysX auto-computes inertia from the
collision hulls, and the ``ixx/iyy/izz`` that ``PHYSICAL_TUNING.md`` Phase 4
instructs you to edit in the URDF **never reach the simulator at all.**

So measure the two numbers that do matter, per axis:

  * **tau** -- the closed-loop time constant, from the rise of the gyro toward
    a commanded step. Sets ``kp`` for any chosen ``I`` via ``kp = I/tau``.
  * **omega_dot_max** -- the angular-acceleration ceiling, from the steepest
    part of the rise. Sets ``moment_limit = I * omega_dot_max``.

Pick any plausible ``I``, derive both from it, and the simulated aircraft is
dynamically identical to the real one regardless of what ``I`` you picked. That
is the whole argument for not owning a bifilar pendulum.

A caution about sample rate
---------------------------
A 40 ms time constant sampled at 50 Hz is two points on the rise. ``tau`` from
MSP telemetry is therefore indicative, not precise, and this tool says so in its
output. ``omega_dot_max`` and the steady-state rate survive the low rate much
better -- the ramp lasts ~100 ms and the plateau lasts as long as you hold the
stick. **For a trustworthy tau, use a blackbox log.** This tool reads either.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

GYRO_COUNTS_PER_DEG_S = 16.4
AXES = ("x", "y", "z")
AXIS_NAMES = {"x": "roll", "y": "pitch", "z": "yaw"}


def load(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(row, key, default=float("nan")) -> float:
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _series(rows, axis: str, gyro_scale: float):
    t, w = [], []
    for r in rows:
        tt = _f(r, "t")
        g = _f(r, f"gyro_{axis}_raw")
        if tt != tt or g != g:
            continue
        t.append(tt)
        w.append(math.radians(g / gyro_scale))
    return t, w


def _smooth(vals, times, *, span_s: float):
    """Centred moving average over a fixed time span. Keeps edges usable."""
    out = []
    n = len(vals)
    for i in range(n):
        lo = hi = i
        while lo > 0 and times[i] - times[lo - 1] <= span_s / 2:
            lo -= 1
        while hi < n - 1 and times[hi + 1] - times[i] <= span_s / 2:
            hi += 1
        out.append(sum(vals[lo:hi + 1]) / (hi - lo + 1))
    return out


def _percentile(vals, q: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def find_steps(t, w, *, min_rate=0.8, min_gap=0.4):
    """Locate step events: where |rate| rises through a threshold and holds."""
    events, last = [], -1e9
    for i in range(1, len(w)):
        if abs(w[i]) >= min_rate > abs(w[i - 1]) and t[i] - last >= min_gap:
            events.append(i)
            last = t[i]
    return events


def fit_step(t, w, i0, *, window=0.6, lookback=0.3):
    """Measure rise time, tau and peak angular acceleration for one step.

    Two traps, both of which produced nonsense before they were fixed:

    1. ``i0`` is where the rate crossed the *detection* threshold, which is well
       up the rise already -- the 10 % point is behind it. Fitting forward from
       ``i0`` puts the 10 % crossing outside the window and the 10-90 rise comes
       out **negative**. So walk backwards to where the axis was still.
    2. The step is *held* for a while and then released. A fixed forward window
       wider than the hold includes the fall, which drags the plateau estimate
       toward zero; the 10 % and 90 % levels then sit inside the noise and the
       rise collapses to a couple of samples. Measured tau came out at 0.002 s
       against a true 0.052 s. So take the plateau from the top of the step, and
       cut the segment where it starts to fall.
    """
    t_end = t[i0] + window
    fwd = [i for i in range(i0, len(t)) if t[i] <= t_end]
    if len(fwd) < 3:
        return None
    sign = 1.0 if w[fwd[len(fwd) // 3]] >= 0 else -1.0
    vals = [sign * w[i] for i in fwd]
    peak = max(vals)
    if peak <= 1e-6:
        return None
    # Plateau from the top of the step only, so the release cannot pull it down.
    top = [v for v in vals if v >= 0.85 * peak]
    plateau = statistics.median(top) if top else peak

    # Walk back to the onset: the last sample before the rise began.
    start = i0
    while (start > 0 and t[i0] - t[start] < lookback
           and sign * w[start] > 0.03 * plateau):
        start -= 1

    # Walk forward only until the step is released.
    stop = i0
    seen_top = False
    for i in range(i0, len(t)):
        if t[i] > t_end:
            break
        v = sign * w[i]
        if v >= 0.9 * plateau:
            seen_top = True
        if seen_top and v < 0.5 * plateau:
            break
        stop = i
    if stop - start < 3:
        return None

    seg_t = [t[i] for i in range(start, stop + 1)]
    seg = [sign * w[i] for i in range(start, stop + 1)]

    # Angular acceleration over the rise. Differencing a noisy rate and taking
    # the maximum is biased high -- at 500 Hz, 0.02 rad/s of gyro noise looks
    # like 10 rad/s^2 of acceleration on its own, and max() finds the worst
    # sample every time. So smooth first, then take a high percentile rather
    # than the extreme.
    sm = _smooth(seg, seg_t, span_s=0.008)
    accs = []
    for a in range(len(sm) - 1):
        dt = seg_t[a + 1] - seg_t[a]
        if dt > 1e-6 and sm[a] < 0.95 * plateau:      # the rise only
            accs.append((sm[a + 1] - sm[a]) / dt)
    accs = [x for x in accs if x > 0]
    peak_acc = _percentile(accs, 0.90) if accs else float("nan")

    # Did this step actually hit the airframe's ceiling? Only then does the
    # acceleration measure a *limit* rather than the response to a small
    # demand. Without the distinction, a gentle step yields a "moment_limit"
    # that is really just kp times the step size.
    #
    # The discriminator is the *shape of the start of the rise*, not how much
    # of it is flat. An unsaturated first-order response has its maximum
    # acceleration at t=0 and decays from there immediately. A saturated one
    # ramps at a constant rate until the demand falls below the ceiling. So
    # compare the first fifth of the rise against the second fifth: flat means
    # clamped, already decaying means not.
    #
    # ("What fraction of the rise is flat" fails: even a firmly saturated axis
    # spends only about a third of its rise clamped, because the exponential
    # tail afterwards is the longer part.)
    saturated = False
    if len(accs) >= 10:
        fifth = max(2, len(accs) // 5)
        first = statistics.median(accs[:fifth])
        second = statistics.median(accs[fifth:2 * fifth])
        if first > 1e-9:
            saturated = (second / first) > 0.80

    def cross(level):
        for a in range(len(seg) - 1):
            if seg[a] < level <= seg[a + 1]:
                span = seg[a + 1] - seg[a]
                f = 0.0 if abs(span) < 1e-12 else (level - seg[a]) / span
                return seg_t[a] + f * (seg_t[a + 1] - seg_t[a])
        return None

    t10, t90 = cross(0.1 * plateau), cross(0.9 * plateau)
    rise = (t90 - t10) if (t10 is not None and t90 is not None and t90 > t10) else float("nan")
    # First order: the 10-90 rise is 2.197 tau. Also take the 63 % crossing.
    tau_rise = rise / 2.197 if rise == rise else float("nan")
    t63 = cross(0.632 * plateau)
    tau_63 = (t63 - seg_t[0]) if t63 is not None else float("nan")

    return {"plateau": plateau, "peak_acc": peak_acc, "rise_10_90": rise,
            "tau_from_rise": tau_rise, "tau_from_63": tau_63, "saturated": saturated,
            "samples_in_rise": sum(1 for x in seg if 0.1 * plateau < x < 0.9 * plateau)}


def analyse(rows, *, gyro_scale: float = GYRO_COUNTS_PER_DEG_S) -> dict:
    out = {"axes": {}, "rate_hz": float("nan")}
    t_all = [_f(r, "t") for r in rows]
    dts = [b - a for a, b in zip(t_all, t_all[1:]) if b > a]
    if dts and sum(dts) > 0:
        out["rate_hz"] = len(dts) / sum(dts)

    for axis in AXES:
        t, w = _series(rows, axis, gyro_scale)
        if len(t) < 10:
            continue
        fits = [f for f in (fit_step(t, w, i) for i in find_steps(t, w)) if f]
        if not fits:
            continue
        out["axes"][axis] = {
            "n_steps": len(fits),
            "plateau_rad_s": statistics.median([f["plateau"] for f in fits]),
            "peak_ang_accel": statistics.median([f["peak_acc"] for f in fits]),
            "tau_s": statistics.median([f["tau_from_rise"] for f in fits
                                        if f["tau_from_rise"] == f["tau_from_rise"]] or [float("nan")]),
            "tau_63_s": statistics.median([f["tau_from_63"] for f in fits
                                           if f["tau_from_63"] == f["tau_from_63"]] or [float("nan")]),
            "samples_in_rise": statistics.median([f["samples_in_rise"] for f in fits]),
            "saturated": sum(1 for f in fits if f["saturated"]) > len(fits) / 2,
        }
    return out


def report(a: dict, *, inertia=(2.75e-3, 3.16e-3, 5.2e-3)) -> None:
    print()
    print("Rate step analysis")
    print("=" * 74)
    print(f"  telemetry rate {a['rate_hz']:.1f} Hz")
    print()
    print(f"  {'axis':>6} {'steps':>6} {'plateau':>11} {'tau':>9} {'peak accel':>12} {'pts in rise':>12}")
    for axis in AXES:
        d = a["axes"].get(axis)
        if not d:
            print(f"  {AXIS_NAMES[axis]:>6}      -  no steps found")
            continue
        print(f"  {AXIS_NAMES[axis]:>6} {d['n_steps']:>6} "
              f"{d['plateau_rad_s']:>8.2f}r/s {d['tau_s']:>8.3f}s "
              f"{d['peak_ang_accel']:>9.1f}r/s2 {d['samples_in_rise']:>12.0f}")

    thin = [AXIS_NAMES[ax] for ax, d in a["axes"].items() if d["samples_in_rise"] < 4]
    if thin:
        print()
        print(f"  WARNING: only a few samples inside the rise on {', '.join(thin)}.")
        print("  tau from this log is indicative only. The plateau and the peak")
        print("  acceleration are still sound. Use a blackbox log for a real tau.")

    print()
    print("  Simulator settings. Inertia is NOT identifiable and NOT settable")
    print("  (PhysX auto-computes it); pick any plausible I and derive from it:")
    print()
    print(f"  {'axis':>6} {'I assumed':>12} {'rate_kp':>10} {'moment_limit':>14}")
    for i, axis in enumerate(AXES):
        d = a["axes"].get(axis)
        if not d or d["tau_s"] != d["tau_s"] or d["tau_s"] <= 0:
            continue
        inert = inertia[i]
        kp = inert / d["tau_s"]
        m = inert * d["peak_ang_accel"]
        print(f"  {AXIS_NAMES[axis]:>6} {inert:>12.2e} {kp:>10.4f} {m:>14.4f}")
    print()
    print("  Scale rate_kp, rate_kd and moment_limit by the SAME factor as any")
    print("  inertia change. Changing inertia alone makes the aircraft sluggish")
    print("  in exact proportion, which no policy can recover from.")
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--gyro-scale", type=float, default=GYRO_COUNTS_PER_DEG_S)
    args = ap.parse_args()
    report(analyse(load(args.log), gyro_scale=args.gyro_scale))
    return 0


def _self_test() -> int:
    """Fly synthetic rate steps with a known tau and ceiling, and recover them."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mock_link import MockConfig, MockLink
    from msp_logger import record

    # Yaw has the lowest ceiling, so a full-stick step saturates it while roll
    # and pitch stay in the linear regime. That asymmetry is the test.
    cfg = MockConfig(tau_s=0.052, max_ang_accel=(88.0, 80.0, 30.0))
    fc = MockLink(cfg)

    def fly(amp, path):
        plan = [(ax, amp) for ax in range(3) for _ in range(3)]
        def driver(link, t):
            slot = int(t / 0.9)
            if slot >= len(plan):
                link.command(cfg.hover_throttle)
                return
            ax, a = plan[slot]
            rates = [0.0, 0.0, 0.0]
            if (t - slot * 0.9) < 0.4:
                rates[ax] = a
            link.command(cfg.hover_throttle, rates)
        record(fc, path, seconds=len(plan) * 0.9 + 0.5, hz=500.0,
               verbose=False, driver=driver)
        return analyse(load(path))

    # --- moderate step: measures tau, should NOT saturate any axis ----------
    mod_path = Path("/tmp/_rates_mod.csv")
    mod = fly(2.0, mod_path)
    assert set(mod["axes"]) == {"x", "y", "z"}, f"missed an axis: {list(mod['axes'])}"
    for ax in AXES:
        d = mod["axes"][ax]
        assert d["n_steps"] >= 2, (ax, d)
        err = abs(d["tau_s"] - cfg.tau_s)
        assert err < 0.015, f"{ax}: tau {d['tau_s']:.4f} vs true {cfg.tau_s:.4f}"

    # --- full-stick step: should saturate yaw (lowest ceiling) only ---------
    big_path = Path("/tmp/_rates_big.csv")
    big = fly(3.2, big_path)
    demanded = 3.2 / cfg.tau_s          # 61.5 rad/s^2 initial demand
    assert demanded > cfg.max_ang_accel[2], "test is not set up to saturate yaw"
    assert demanded < cfg.max_ang_accel[0], "test should leave roll unsaturated"

    yaw = big["axes"]["z"]
    assert yaw["saturated"], "a full-stick yaw step should be detected as saturated"
    acc_err = abs(yaw["peak_ang_accel"] - cfg.max_ang_accel[2]) / cfg.max_ang_accel[2]
    assert acc_err < 0.20, (f"saturated yaw acceleration {yaw['peak_ang_accel']:.1f} "
                            f"should be near the {cfg.max_ang_accel[2]:.0f} ceiling")
    assert not big["axes"]["x"]["saturated"], "roll should not saturate at this step size"

    print("analyze_rates: all checks passed")
    print("  moderate step (2.0 rad/s) -- measures tau:")
    for ax in AXES:
        d = mod["axes"][ax]
        print(f"    {AXIS_NAMES[ax]:>5}: tau {d['tau_s']:.4f} s vs true {cfg.tau_s:.4f} s, "
              f"saturated={d['saturated']}")
    print("  full-stick step (3.2 rad/s) -- finds the ceiling:")
    for i, ax in enumerate(AXES):
        d = big["axes"][ax]
        print(f"    {AXIS_NAMES[ax]:>5}: peak accel {d['peak_ang_accel']:>5.1f} rad/s2 "
              f"(true ceiling {cfg.max_ang_accel[i]:>4.0f}), saturated={d['saturated']}")
    print("  -> only the saturated axis reports its true ceiling, which is why")
    print("     the field procedure steps 1.0, 2.0 and 3.2 rad/s rather than one size.")
    for f in (mod_path, big_path):
        f.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
