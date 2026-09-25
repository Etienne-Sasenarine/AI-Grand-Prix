"""Where does the aircraft actually appear at reset?

Every start-distribution setting we have is a config field that nothing
verifies end-to-end. ``race_start_fraction`` says 20% of episodes begin on the
competition pad; the only way to know that happened is to read the aircraft's
position out of the simulation immediately after a reset and look.

Read *before* stepping. Reset has already placed the aircraft; one step and the
terminated envs have been respawned underneath the numbers.

    python scripts/diag/diag_spawn.py --headless --num_envs 256
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Drone-Racer-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--resets", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: E402,F401


def main() -> int:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cmd_cfg = cfg.commands.target
    pad = getattr(cmd_cfg, "race_start_xy", None)
    frac = float(getattr(cmd_cfg, "race_start_fraction", 0.0) or 0.0)
    floor = float(getattr(cmd_cfg, "floor_start_fraction", 0.0) or 0.0)
    zr = getattr(cmd_cfg, "spawn_z_range", None)
    print(f"\nCONFIG  race_start_xy={pad}  race_start_fraction={frac}"
          f"  floor_start_fraction={floor}  spawn_z_range={zr}", flush=True)

    env = gym.make(args.task, cfg=cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    origins = unwrapped.scene.env_origins

    all_xy = []
    all_z = []
    for _ in range(args.resets):
        unwrapped.reset()
        # Local (track-frame) position: world minus this env's origin.
        pos = (robot.data.root_pos_w - origins).clone()
        all_xy.append(pos[:, :2])
        all_z.append(pos[:, 2])

    xy = torch.cat(all_xy)
    z = torch.cat(all_z)
    n = xy.shape[0]

    print(f"\n{n} spawns observed over {args.resets} resets\n", flush=True)
    print(f"  z   min={z.min():.3f}  max={z.max():.3f}  mean={z.mean():.3f}", flush=True)
    # Underground is not a small error. The reset offset runs along the gate
    # normal, which on a tilted gate is not horizontal, so a sampled height
    # can arrive with a sin(tilt) term on top of it.
    under = int((z < 0.0).sum())
    print(f"    BELOW FLOOR (z<0):   {under:5d}  {under/n:6.1%}"
          f"{'   <-- BAD' if under else ''}", flush=True)
    if zr is not None:
        lo, hi = float(zr[0]), float(zr[1])
        inb = int(((z >= lo - 1e-3) & (z <= hi + 1e-3)).sum())
        print(f"    inside spawn_z_range [{lo}, {hi}]:  {inb:5d}  {inb/n:6.1%}",
              flush=True)
    for lo, hi in [(0.0, 0.6), (0.6, 1.0), (1.0, 1.6), (1.6, 3.0), (3.0, 99.0)]:
        k = int(((z >= lo) & (z < hi)).sum())
        print(f"    z in [{lo:>4.1f},{hi:>4.1f})  {k:5d}  {k/n:6.1%}", flush=True)

    if pad is not None:
        px, py = float(pad[0]), float(pad[1])
        d = torch.norm(xy - torch.tensor([px, py], device=xy.device), dim=-1)
        for r in (1.5, 3.0, 6.0):
            k = int((d < r).sum())
            print(f"\n  within {r:.1f} m of the pad ({px}, {py}):  {k:5d}  {k/n:6.1%}",
                  flush=True)
        print(f"  closest spawn to the pad: {float(d.min()):.2f} m", flush=True)
        print(f"\n  EXPECTED if race_start works: ~{frac:.0%} within 1.5 m", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        status = main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    sys.exit(status)
