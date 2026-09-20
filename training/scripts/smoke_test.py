"""Does the plant we just described actually load and behave?

Every plant change in this repo was written on a machine with no Isaac install:
the inertia is pushed straight into PhysX through a tensor API that was only
ever exercised against a fake view, the thrust curve changes what every
environment does, and the per-episode plant sampling changes reset. None of it
has been run.

This is the cheap check before an expensive one. A few hundred steps on one
environment, headless, asserting the things that would otherwise fail silently
or three hours into a rented run:

  1. the environment builds and steps without raising
  2. PhysX reports the mass and inertia we believe we set
  3. the thrust curve is pinned to 1 g at each env's own hover point
  4. the randomised plant actually varies across resets, and stays in range
  5. nothing produces NaN or non-finite observations

Run it before any training run, and again on any new box:

    python scripts/smoke_test.py --headless

Exit status is 0 only if every check passes, so it works in CI or a setup
script. ``--num_envs`` defaults low because the point is to fail fast.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Drone-Racer-v0")
parser.add_argument("--num_envs", type=int, default=64,
                    help="enough envs to see the plant spread; small enough to start fast")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--resets", type=int, default=3,
                    help="how many forced resets to sample the plant over")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Everything below needs the app running first.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: E402,F401
from contract import observation, plant, verify  # noqa: E402


class Checks:
    """Collect results so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def record(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), name, detail))
        return bool(ok)

    def report(self) -> int:
        width = max(len(name) for _, name, _ in self.rows)
        print("\n" + "=" * (width + 30))
        print("SMOKE TEST")
        print("=" * (width + 30))
        for ok, name, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<{width}}  {detail}")
        failed = [name for ok, name, _ in self.rows if not ok]
        print("=" * (width + 30))
        if failed:
            print(f"{len(failed)} of {len(self.rows)} checks FAILED: {', '.join(failed)}")
            print("Do not start a training run until these pass.")
            return 1
        print(f"all {len(self.rows)} checks passed")
        return 0


def main() -> int:
    checks = Checks()
    print(f"contract: {verify.short_hash('v1')}  ({plant.MASS_KG} kg airframe)")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    checks.record(True, "environment builds", args.task)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    view = robot.root_physx_view

    # --- 2. does PhysX agree with contract/plant.py? ------------------------
    masses = view.get_masses()
    total_mass = float(masses[0].sum())
    checks.record(
        abs(total_mass - plant.MASS_KG) < 1e-3,
        "PhysX mass matches the contract",
        f"{total_mass:.4f} kg vs {plant.MASS_KG}",
    )

    # Inertia is written by a startup event through a tensor API that has never
    # run here. This is the single most likely thing in this file to fail.
    body_id = robot.find_bodies("body")[0][0]
    inertia = view.get_inertias()[0, body_id]
    diag = (float(inertia[0]), float(inertia[4]), float(inertia[8]))
    expected = plant.INERTIA_DIAG
    close = all(abs(a - b) < 1e-6 for a, b in zip(diag, expected))
    checks.record(
        close,
        "PhysX inertia matches the estimate",
        f"{tuple(round(d, 5) for d in diag)} vs {expected}",
    )
    off_diagonal = [float(inertia[i]) for i in (1, 2, 3, 5, 6, 7)]
    checks.record(
        all(abs(v) < 1e-9 for v in off_diagonal),
        "inertia products are zero",
        f"max |off-diagonal| = {max(abs(v) for v in off_diagonal):.2e}",
    )

    # --- 3 & 4. the randomised plant ---------------------------------------
    action_term = unwrapped.action_manager.get_term("control_action")
    seen_hover: list[float] = []
    seen_twr: list[float] = []

    obs, _ = env.reset()
    for _ in range(args.resets):
        seen_hover += action_term._plant_hover.tolist()
        seen_twr += action_term._plant_twr.tolist()
        env.reset()

    lo_h, hi_h = plant.PLANT_HOVER_RANGE
    lo_t, hi_t = plant.PLANT_TWR_RANGE
    checks.record(
        all(lo_h - 1e-6 <= h <= hi_h + 1e-6 for h in seen_hover),
        "sampled hover points stay in range",
        f"{min(seen_hover):.3f}..{max(seen_hover):.3f} of {plant.PLANT_HOVER_RANGE}",
    )
    checks.record(
        all(lo_t - 1e-4 <= t <= hi_t + 1e-4 for t in seen_twr),
        "sampled thrust-to-weight stays in range",
        f"{min(seen_twr):.2f}..{max(seen_twr):.2f} of "
        f"({lo_t:.2f}, {hi_t:.2f})",
    )
    # A plant that does not vary is the failure this whole session was about.
    hover_spread = max(seen_hover) - min(seen_hover)
    checks.record(
        hover_spread > 0.01,
        "the plant actually varies between envs",
        f"hover spread {hover_spread:.4f}",
    )
    quad = action_term._plant_quad_share
    checks.record(
        bool(((quad >= -1e-6) & (quad <= 1.0 + 1e-6)).all()),
        "thrust curve stays convex",
        f"quad_share {float(quad.min()):.3f}..{float(quad.max()):.3f}",
    )

    # --- zero action must hover, on every sampled plant ---------------------
    # Stepped through the normal path rather than by calling the action manager
    # directly, so this exercises what training exercises.
    env.reset()
    zero = torch.zeros(unwrapped.num_envs, 4, device=unwrapped.device)
    env.step(zero)
    applied = action_term._applied_thrust_n
    weight = plant.MASS_KG * plant.G
    ratio = applied / weight
    # Each env is pinned to 1 g at *its own* hover point, so handing it the
    # client's constant should over- or under-thrust in a bounded way.
    checks.record(
        bool(((ratio > 0.5) & (ratio < 1.6)).all()),
        "zero action gives a plausible collective",
        f"{float(ratio.min()):.3f}..{float(ratio.max()):.3f} g",
    )

    # --- latency: is the plant delaying anything at all? -------------------
    if action_term.cfg.action_delay_steps_range is not None:
        delays = action_term._action_delay
        lo_d, hi_d = action_term.cfg.action_delay_steps_range
        checks.record(
            bool(((delays >= lo_d) & (delays <= hi_d)).all()),
            "action delays stay in range",
            f"{int(delays.min())}..{int(delays.max())} steps of {(lo_d, hi_d)}",
        )
        checks.record(
            len(delays.unique()) > 1 or unwrapped.num_envs == 1,
            "action delay varies between envs",
            f"{len(delays.unique())} distinct values",
        )
    if action_term.cfg.rate_tau_s_range is not None:
        alpha = action_term._rate_alpha
        checks.record(
            bool(((alpha > 0.0) & (alpha < 1.0)).all()),
            "rate filter leaves real lag",
            f"alpha {float(alpha.min()):.3f}..{float(alpha.max()):.3f}",
        )

    # --- observation: is it the width the contract promises? ---------------
    obs, _ = env.reset()
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    width = int(policy_obs.shape[-1])
    expected_v1 = observation.observation_dim("v1")
    expected_v2 = observation.observation_dim("v2")
    checks.record(
        width in (expected_v1, expected_v2),
        "observation width matches the contract",
        f"{width} (v1={expected_v1}, v2={expected_v2})",
    )
    if width == expected_v2:
        print(f"  note: running observation v2, hash {verify.short_hash('v2')}")
    elif width == expected_v1:
        print(f"  note: running observation v1, hash {verify.short_hash('v1')}")

    # --- 5. step it --------------------------------------------------------
    obs, _ = env.reset()
    finite = True
    nan_step = -1
    for step in range(args.steps):
        action = torch.zeros(unwrapped.num_envs, 4, device=unwrapped.device)
        obs, reward, terminated, truncated, _ = env.step(action)
        tensor = obs["policy"] if isinstance(obs, dict) else obs
        if not bool(torch.isfinite(tensor).all()) or not bool(torch.isfinite(reward).all()):
            finite, nan_step = False, step
            break
    checks.record(
        finite,
        f"{args.steps} steps produce finite observations",
        "ok" if finite else f"non-finite at step {nan_step}",
    )

    env.close()
    return checks.report()


if __name__ == "__main__":
    try:
        status = main()
    finally:
        simulation_app.close()
    sys.exit(status)
