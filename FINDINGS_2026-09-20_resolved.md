# What the first smoke run found, and what it turned out to be

Companion to [`FINDINGS_2026-09-20.md`](FINDINGS_2026-09-20.md), which records
the 10-pass/4-fail run as it was read at the time. This is the resolution.
Two of those four were what they looked like. The other two were not, and
chasing them turned up two further bugs that nothing was looking for.

Final state: **15 of 15 checks pass on an RTX 6000 Ada**, verified.

---

## 1. The aircraft weighed five times too much — confirmed, fixed

`MassPropertiesCfg(mass=1.745)` reads as a statement about the aircraft and
acts as a statement about every rigid body in it. Five links, so 8.725 kg,
against thrust sized for 1.745. Thrust-to-weight 0.2.

Every piece was individually correct — the contract said 1.745, the thrust
curve was built for 1.745, the asset config said 1.745. Only the *product* of
the config and the asset's link count was wrong, and nothing looked at that
product.

**Fixed** by removing `mass_props` from the spawn entirely and distributing
mass at startup in `events.set_body_mass`, where the links can be told apart.
The airframe takes whatever the props leave, so the total is arithmetic rather
than three numbers maintained by hand: revising the 5 g prop estimate moves
mass between links but can never change what the aircraft weighs.

1.745 kg is all-up competition weight **with props fitted**, confirmed with the
person who weighed it. The props therefore come out of that number, not on top
of it — the other reading gives 1.765 kg and is the plausible way to "fix" this
wrongly later, so a test pins it.

An unmatched link raises rather than silently keeping its spawn mass. That is
the shape the original bug had: a link nobody was thinking about, carrying mass
nobody intended.

## 2. The inertia products — confirmed, but not for the reason assumed

The products were real. The assumption that `set_inertias` would clear them
was wrong, and that mattered more than the original finding.

**PhysX does not store a 3x3 inertia tensor.** It stores three principal
moments plus the rotation taking body axes to principal axes — and that
rotation lives in the *COM pose*, not in the tensor. `set_inertias` writes the
moments and leaves the rotation exactly where it was. Writing zeros into the
products does nothing at all.

The read-back gave itself away:

```
ixx  0.004054  vs 0.0040 written   +1.35%
iyy  0.005500  vs 0.0055 written    exact
izz  0.007446  vs 0.0075 written   -0.72%
```

Middle axis exact, outer two moved in opposite directions. That is what a
rotation about y does and essentially nothing else does. Solving it:

```
(Izz - Ixx)·sin t·cos t = 4.32e-4     ->  t = 7.1 deg
Ixx·cos²t + Izz·sin²t   = 0.004054    ->  matches
Ixx·sin²t + Izz·cos²t   = 0.007446    ->  matches
```

The COM quaternion then read `(-0.000775, 0.062293, -0.003859, 0.99805)` — a
rotation about y of **7.16 degrees**, against 7.1 predicted from the inertia
alone. Clearing it makes the read-back exactly `(0.004, 0.0055, 0.0075)` with
products at `0.00e+00`.

It is a USD conversion artifact: the URDF has no products of inertia and
neither does the airframe being modelled. Not harmless — a tilt about y
couples roll into yaw on every input. The four prop links have identity here
and read back clean, which is what confirmed the mechanism was this one link's
authoring rather than a limit of the API.

> **The quaternion is xyzw.** Isaac Lab's own math utilities are wxyz. Writing
> wxyz identity here stores a 180-degree rotation about x, which maps a
> diagonal tensor to itself — so every inertia assertion still passes while the
> body is upside down in its own mass frame. A test asserts the real part is in
> the last slot, because that is the only thing that catches it.

The test that was supposed to cover the products seeded its fixture with zeros,
so it could not distinguish "cleared" from "never touched". It now seeds the
measured tilt.

## 3. The inertia tolerance — a bad check, not a bad plant

Absolute `1e-6` against `4e-3` quantities making a float32 round trip. It
failed on values correct to six significant figures and said nothing useful
when it did. Now relative, and the products are compared against the smallest
principal moment so the bar scales with the airframe rather than with float32.

## 4. The `0.000 g` env — not a bug at all, and it was hiding one

`reset()` zeroes `_applied_thrust_n` (`actions.py:261`), which is correct: a
fresh episode has issued no command. The smoke test read that buffer *after*
stepping, so any env that terminated mid-step scored 0 g. The plant was fine.

The check now excludes envs that ended and fails if too few survived — and
immediately reported `18/64 envs still flying`, which is how the next one was
found.

## 5. `flyaway` was killing 48 of 64 envs on step one — new, and the worst of them

At zero speed, before the policy had done anything.

`_resample_command` picks the next gate and teleports the drone to it, but
only `_update_command` publishes the gate *pose* — and Isaac Lab's step loop
runs `command_manager.compute()` **after** the termination and reward managers.
So for exactly one step after a reset, every consumer of the command saw a pose
belonging to the previous episode. On a freshly built env it was still the
zeros the buffer is initialised to, while the drone had been scattered along a
22 m course: distances of 13–27 m against a 20 m `flyaway` threshold.

**Fixed** by extracting `_refresh_gate_poses()` and calling it from both paths.
Step-one terminations went from 48 to 1.

This one is pre-existing, independent of the mass bug, and invisible by
design — the only symptom is episodes quietly ending early. It would have
poisoned any run started from a fresh env.

## 6. One env still spawns inside a gate

The surviving termination: 185 N of contact at x = 12.5, 1 in 64. Left alone
deliberately. It is a real but separate problem, and it is worth knowing
whether it is one bad gate or bad luck before changing where the aircraft is
allowed to appear.

---

## Two things about the tooling that cost more than the bugs

### The smoke test was reporting success while failing

`SimulationApp.close()` hard-exits. It runs in the `finally` block, so
`sys.exit(status)` never executes and the shell sees 0 whether every check
passed or none did. The docstring claimed the exit status was CI-safe. It was
not: a runner trusting it would green-light training on a broken plant, which
is the one thing the script exists to prevent.

The same hard exit discards Python's stdio buffer. Redirected to a file, a run
that completed fine produced *no output at all* — 60 lines of C++ MDL warnings
(native, unbuffered), then nothing, then `WRAPPER_RC=0`. That reads exactly
like the startup segfault we already had a runbook entry for, and cost a round
of chasing the fd limit when the run had already finished and its results were
sitting in a buffer nobody flushed.

Now: flush per check, and an authoritative `SMOKE_RESULT=` line on stdout.
Runners grep that. `$?` is not evidence.

### `ulimit -n` inside the script is not enough under tmux

The tmux **server** outlives the session that started it and hands its own
limits to every pane. Started once from an ordinary 1024-fd SSH session, it
caps the run no matter what the script asks for — and the symptom is
indistinguishable from the original segfault.

Raise the limit before the first `tmux` invocation, or `tmux kill-server`
first. `scripts/pod/train_launch.sh` checks both and refuses to start.

---

## What this says about the plant, going in

Reward weights were tuned against an aircraft that could not fly, in episodes
that ended on step one. Both are now fixed. The balance those weights were
chosen under no longer exists, so the first curves will not resemble anything
previous — that is the fixes working, not new breakage. Weights have not been
touched.
