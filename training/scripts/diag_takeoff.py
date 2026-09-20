"""Does the aircraft actually take off from the competition pad?

Spawning in the right place is not the same as flying from it, and only the
first of those has been checked. This forces *every* env to the pad, rolls a
checkpoint out, and asks the three questions that decide whether the start
works at all:

  1. Does it survive the ground?  The collision termination fires at 0.01 N and
     the spawn is 0.12-0.50 m up. If the airframe clips the floor the episode
     ends before the policy has acted, and 35% of training would be spent on
     episodes that are over at step one.

  2. Does it climb?  A policy that has only ever begun at gate height has no
     reason to know that thrust must exceed hover before anything else happens.

  3. Can it see G1?  From the pad the gate is 7.8 m away and 1.2 m up, about
     9 degrees of elevation. The camera is pitched 20 deg up with a 45 deg
     vertical field, so the window is roughly -2.6 to +42.6 deg. That should
     contain it -- but "should" is what put the drone 47 cm underground earlier
     today, so it gets measured from the observation the policy actually reads.

    python scripts/diag_takeoff.py --headless --checkpoint <path>
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Drone-Racer-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--early_steps", type=int, default=30,
                    help="a death inside this many steps is a spawn failure, "
                         "not a flying failure")
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
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import tasks  # noqa: E402,F401
from contract import observation as obs_contract  # noqa: E402

VIS = slice(16, 24)


def main() -> int:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # Every episode is the real start. This is a test of one thing, so the
    # other start types are switched off rather than averaged in.
    cfg.commands.target.race_start_fraction = 1.0
    cfg.commands.target.floor_start_fraction = 0.0
    pad = cfg.commands.target.race_start_xy
    print(f"\nALL {args.num_envs} envs forced to the pad {pad}, "
          f"targeting Isaac gate {cfg.commands.target.race_start_idx} (official G1)",
          flush=True)

    env = gym.make(args.task, cfg=cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    origins = unwrapped.scene.env_origins
    cmd = unwrapped.command_manager.get_term("target")

    xc = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
    xc["trainer"]["close_environment_at_exit"] = False
    xc["agent"]["experiment"]["write_interval"] = 0
    xc["agent"]["experiment"]["checkpoint_interval"] = 0
    wrapped = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
    runner = Runner(wrapped, xc)
    runner.agent.load(args.checkpoint)
    runner.agent.set_running_mode("eval")

    obs, _ = wrapped.reset()
    n = args.num_envs
    dev = obs.device

    # Where the gate is, and whether it is in frame, at the instant of spawn.
    frame_dim = obs.shape[-1] // obs_contract.HISTORY
    vis0 = obs.view(n, obs_contract.HISTORY, frame_dim)[..., VIS]
    # The history is seeded with the spawn frame, so any frame answers this;
    # take the maximum over frames so a zero-filled tail cannot mask a hit.
    seen_any = (vis0.amax(dim=1) > 0.5).float()
    print(f"\n  gate keypoints visible at spawn: "
          f"{float(seen_any.sum(-1).mean()):.2f} of 8 corners, "
          f"{float((seen_any.sum(-1) > 0).float().mean()):.1%} of envs see the gate",
          flush=True)

    z0 = (robot.data.root_pos_w[:, 2] - origins[:, 2]).clone()
    print(f"  spawn height: min={float(z0.min()):.3f} "
          f"median={float(z0.median()):.3f} max={float(z0.max()):.3f}", flush=True)

    alive = torch.ones(n, dtype=torch.bool, device=dev)
    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    t_reach = torch.full((n,), -1, dtype=torch.long, device=dev)
    died_at = torch.full((n,), -1, dtype=torch.long, device=dev)
    max_z = z0.clone()
    profile: dict[int, float] = {}

    with torch.inference_mode():
        for t in range(args.steps):
            # Snapshot before stepping. After step() the terminated envs have
            # already been respawned and their position is the next episode's.
            z = (robot.data.root_pos_w[:, 2] - origins[:, 2]).clone()
            max_z = torch.where(alive, torch.maximum(max_z, z), max_z)
            if t in (0, 15, 30, 60, 120, 240):
                live = z[alive]
                profile[t] = float(live.median()) if live.numel() else float("nan")

            a = runner.agent.act(obs, timestep=0, timesteps=0)[0]
            obs, _, term, trunc, _ = wrapped.step(a)

            gp = cmd.gate_passed.bool().view(-1)
            done = (term | trunc).bool().view(-1)

            newly = alive & gp & ~reached
            reached |= newly
            t_reach = torch.where(newly, torch.full_like(t_reach, t), t_reach)

            dying = alive & done & ~gp
            died_at = torch.where(dying, torch.full_like(died_at, t), died_at)
            # The first episode is over once it ends or once G1 is passed.
            alive &= ~(done | gp)

    early = int(((died_at >= 0) & (died_at < args.early_steps)).sum())
    died = int((died_at >= 0).sum())
    got = int(reached.sum())

    print(f"\n  altitude profile (median z of still-flying envs)", flush=True)
    for t, v in profile.items():
        print(f"    step {t:>3}  ({t/60:.2f} s)   z = {v:.3f} m", flush=True)
    print(f"    peak z reached, median over envs: "
          f"{float(max_z.median()):.3f} m", flush=True)

    print(f"\n  RESULTS over {n} pad starts", flush=True)
    print(f"    died within {args.early_steps} steps "
          f"({args.early_steps/60:.2f} s):  {early:4d}  {early/n:6.1%}"
          f"{'   <-- SPAWN IS BROKEN' if early > n * 0.05 else ''}", flush=True)
    print(f"    died before reaching G1:          {died - got if died > got else died:4d}",
          flush=True)
    print(f"    PASSED G1:                        {got:4d}  {got/n:6.1%}", flush=True)
    if got:
        tr = t_reach[reached].float()
        print(f"    time to G1: median {float(tr.median())/60:.2f} s  "
              f"min {float(tr.min())/60:.2f} s  max {float(tr.max())/60:.2f} s",
              flush=True)
    print(f"\nTAKEOFF g1_pass_rate={got/n:.4f} early_death_rate={early/n:.4f}",
          flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        status = main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    sys.exit(status)
