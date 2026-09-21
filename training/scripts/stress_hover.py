"""Does the hover policy hold up on an aircraft it did not train on?

Station keeping is the cage deployment test, so this is the closest thing to a
pre-flight check that costs nothing to fail. It flies a trained policy against
a plant deliberately outside the training distribution and reports whether it
still parks.

**One condition per process.** Building a second environment in the same
interpreter does not give a second clean measurement -- Isaac keeps working on
a stage that still holds the first, and the numbers come out of a dirty card.
``scripts/pod/stress_sweep.sh`` drives the conditions in a loop.

    python scripts/stress_hover.py --headless --checkpoint <path> --condition nominal
    python scripts/stress_hover.py --headless --checkpoint <path> --condition mass=1.20

Conditions, all chosen to be *plausible* rather than extreme -- the point is
to predict the cage test, not to find the breaking point of an arbitrary
number:

    nominal          the training distribution, as a control
    mass=<k>         airframe mass scaled by k (a heavier build, a bigger pack)
    hover=<s>        every env pinned to hover stick s, outside the trained band
    delay=<n>        action delay pinned to n policy steps, beyond the trained 0-2
    tau=<s>          rate-filter time constant pinned to s, beyond the trained band
    wind=<n>         constant lateral force of n newtons
    rate=<hz>        policy control rate changed from the trained 60 Hz

Conditions may be combined with commas -- ``rate=40,delay=0`` runs at 40 Hz
with the delay line pinned off, which isolates the control rate from the
latency that a slower loop also brings.

What it reports, and why these:

    survived         fraction of episodes reaching time-out rather than crashing
    settled          fraction of steps both near the hold point and slow
    median_err       typical distance from the hold point, metres
    p95_err          the tail -- a policy can look fine on the median and still
                     wander badly a few percent of the time, and in a cage the
                     tail is what hits the netting
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Drone-Hover-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--condition", default="nominal")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seconds", type=float, default=15.0,
                    help="flown per episode; converted to steps at the policy rate")
parser.add_argument("--settle_seconds", type=float, default=3.0,
                    help="discarded before scoring; the policy is still arriving")
parser.add_argument("--ml_framework", default="torch")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import tasks  # noqa: E402,F401
from tasks.drone_hover.drone_hover_env_cfg import STANDOFF_M  # noqa: E402
from tasks.drone_hover.mdp.rewards import hold_point  # noqa: E402


def apply_condition(env_cfg, condition: str) -> str:
    """Mutate the config for one or more comma-separated conditions."""
    parts = [p for p in condition.split(",") if p]
    if len(parts) > 1:
        return "; ".join(apply_one(env_cfg, p) for p in parts)
    return apply_one(env_cfg, condition)


def apply_one(env_cfg, condition: str) -> str:
    """Mutate the config for one condition. Returns a human description."""
    if condition == "nominal":
        return "training distribution (control)"

    name, _, raw = condition.partition("=")
    if not raw:
        raise SystemExit(f"condition '{condition}' needs a value, e.g. mass=1.2")
    value = float(raw)
    action = env_cfg.actions.control_action

    if name == "mass":
        # Scale what the plant weighs while leaving the action decode alone, so
        # the policy is still commanding thrust for a 1.745 kg aircraft and the
        # aircraft is not one. That mismatch is the realistic failure: a build
        # with a heavier pack, not a mis-typed constant.
        from tasks.drone_racer.mdp import events as ev
        env_cfg.events.body_mass.params["total_mass_kg"] = ev.AIRFRAME_MASS_KG * value
        action.mass_kg = ev.AIRFRAME_MASS_KG * value
        return f"airframe mass x{value:g} ({ev.AIRFRAME_MASS_KG * value:.3f} kg)"

    if name == "hover":
        action.plant_hover_range = (value, value)
        return f"hover point pinned to {value:g} stick"

    if name == "delay":
        action.action_delay_steps_range = (int(value), int(value))
        return f"action delay pinned to {int(value)} steps ({int(value) * 1000 / 60:.0f} ms)"

    if name == "tau":
        action.rate_tau_s_range = (value, value)
        return f"rate filter tau pinned to {value:g} s"

    if name == "wind":
        # A steady lateral push, which is what a cage fan or an open doorway
        # actually does. The interval push event already exists for gusts; this
        # reuses it with a constant magnitude rather than a symmetric range, so
        # the policy has to hold a trim rather than reject a disturbance.
        env_cfg.events.push_robot.params["force_range"] = (value, value)
        env_cfg.events.push_robot.interval_range_s = (0.0, 0.0)
        return f"steady lateral force {value:g} N"

    if name == "rate":
        # Only the *policy* rate moves. The rate controller and the plant stay
        # on the 120 Hz physics tick, which is the right model of the real
        # aircraft: Betaflight's inner loop does not slow down because our
        # companion computer does. Three things change at once, and that is
        # the point -- it is what actually happens on the Jetson:
        #   * each action is held 1.5x longer at 40 Hz,
        #   * the 6-frame observation history spans 150 ms instead of 100,
        #   * the delay line, counted in policy steps, grows in milliseconds.
        # Pin ``delay`` alongside this to separate the last one from the rest.
        physics_hz = 1.0 / float(env_cfg.sim.dt)
        decimation = physics_hz / value
        if abs(decimation - round(decimation)) > 1e-6:
            raise SystemExit(
                f"rate={value:g} Hz needs a whole decimation of the "
                f"{physics_hz:g} Hz physics tick; got {decimation:.4f}"
            )
        env_cfg.decimation = int(round(decimation))
        env_cfg.sim.render_interval = env_cfg.decimation
        return f"policy rate {value:g} Hz (decimation {env_cfg.decimation}, trained at 60)"

    raise SystemExit(f"unknown condition '{name}'")


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    description = apply_condition(env_cfg, args.condition)

    # After the condition, because ``rate`` moves it. Scoring the same number
    # of *steps* at two rates would compare two different durations.
    policy_hz = 1.0 / (float(env_cfg.sim.dt) * int(env_cfg.decimation))
    steps = int(round(args.seconds * policy_hz))
    settle_steps = int(round(args.settle_seconds * policy_hz))

    print(f"\ncheckpoint: {args.checkpoint}", flush=True)
    print(f"condition: {args.condition}", flush=True)
    print(f"           {description}", flush=True)
    print(f"           {policy_hz:.1f} Hz, {steps} steps "
          f"({args.seconds:g} s), scoring after {settle_steps}", flush=True)

    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    experiment_cfg = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    wrapped = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
    runner = Runner(wrapped, experiment_cfg)
    runner.agent.load(args.checkpoint)
    runner.agent.set_running_mode("eval")

    obs, _ = wrapped.reset()

    # Per-env accumulators. Scored only after the settling window, because a
    # policy starting one jitter away from the hold point is still arriving and
    # that arrival is not what this measures.
    errors: list[torch.Tensor] = []
    settled_count = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
    scored_steps = 0
    ever_terminated = torch.zeros(unwrapped.num_envs, dtype=torch.bool,
                                  device=unwrapped.device)

    robot = unwrapped.scene["robot"]

    with torch.inference_mode():
        for step in range(steps):
            actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
            obs, _, terminated, truncated, _ = wrapped.step(actions)
            ever_terminated |= terminated.squeeze(-1).bool()

            if step < settle_steps:
                continue

            target = hold_point(unwrapped, "target", standoff_m=STANDOFF_M)
            err = torch.norm(robot.data.root_pos_w - target, dim=1)
            speed = torch.norm(robot.data.root_lin_vel_w, dim=1)
            errors.append(err.clone())
            settled_count += ((err < 0.30) & (speed < 0.30)).float()
            scored_steps += 1

    all_err = torch.cat(errors)
    survived = 1.0 - float(ever_terminated.float().mean())
    settled = float((settled_count / max(scored_steps, 1)).mean())

    print(f"\n  survived     {survived:6.1%}", flush=True)
    print(f"  settled      {settled:6.1%}", flush=True)
    print(f"  median_err   {float(all_err.median()):6.3f} m", flush=True)
    print(f"  p95_err      {float(all_err.quantile(0.95)):6.3f} m", flush=True)
    print(f"  max_err      {float(all_err.max()):6.3f} m", flush=True)

    # One machine-readable line, because Isaac's hard exit makes the status
    # code worthless and a driver script needs something to grep.
    print(f"STRESS checkpoint={args.checkpoint} "
          f"condition={args.condition} hz={policy_hz:.1f} survived={survived:.4f} "
          f"settled={settled:.4f} median_err={float(all_err.median()):.4f} "
          f"p95_err={float(all_err.quantile(0.95)):.4f}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        status = main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    sys.exit(status)
