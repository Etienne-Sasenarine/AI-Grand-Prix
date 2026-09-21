"""Record one station-keeping flight: floor start, climb, hold.

DroneHoverEnvCfg_PLAY subclasses the RACING play config and overrides only
rewards, terminations and episode length -- it does not carry hover's spawn.
Recorded as it ships, the video shows a racing run-in with a hover reward,
which is not the thing anyone wants to look at. The spawn geometry is
therefore copied from the hover TRAINING config, and the racing start fields
are cleared explicitly, because inheritance carries a parent's future changes
and hover has been bitten by exactly that twice.

Speed is baked into the frame count rather than declared in the header: a
container stamped with an unusual fps is played at whatever rate the player
feels like, and the speed-up silently does not happen.

    python scripts/record_hover.py --headless --enable_cameras --checkpoint <path>
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Drone-Hover-Play-v0")
parser.add_argument("--train_task", default="Isaac-Drone-Hover-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=900, help="60 steps = 1 s of flight")
parser.add_argument("--out_fps", type=int, default=30)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--out", default="/workspace/videos/hover")
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


def main() -> int:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    train = parse_env_cfg(args.train_task, device=args.device, num_envs=1)
    t, c = train.commands.target, cfg.commands.target

    for field in ("spawn_z_range", "spawn_scatter_m", "start_at_run_in",
                  "start_run_in_m"):
        setattr(c, field, getattr(t, field))
    # Racing's start must never reach hover; PLAY inherits it from the racing
    # play config, which is not the same object the training clear touches.
    c.race_start_xy = None
    c.race_start_fraction = 0.0
    c.floor_start_fraction = 0.0

    if c.spawn_z_range is None:
        raise SystemExit("hover spawn_z_range is None; this would record a "
                         "racing run-in, not a floor start")
    print(f"\nhover: spawn z {c.spawn_z_range}, scatter {c.spawn_scatter_m}, "
          f"run-in {c.start_run_in_m} m", flush=True)

    os.makedirs(args.out, exist_ok=True)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")

    xc = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
    xc["trainer"]["close_environment_at_exit"] = False
    xc["agent"]["experiment"]["write_interval"] = 0
    xc["agent"]["experiment"]["checkpoint_interval"] = 0
    wrapped = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
    runner = Runner(wrapped, xc)
    runner.agent.load(args.checkpoint)
    runner.agent.set_running_mode("eval")

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    origins = unwrapped.scene.env_origins

    import imageio.v2 as imageio  # noqa: PLC0415

    skip = max(1, int(round(60.0 / args.out_fps * args.speed)))
    path = os.path.join(args.out, "hover.mp4")
    writer = imageio.get_writer(path, fps=args.out_fps, macro_block_size=None)
    print(f"  {args.steps} steps, every {skip} rendered, {args.out_fps} fps "
          f"-> {args.steps/60:.1f} s of flight in "
          f"{args.steps/skip/args.out_fps:.1f} s of video", flush=True)

    obs, _ = wrapped.reset()
    z0 = float(robot.data.root_pos_w[0, 2] - origins[0, 2])
    peak = z0
    heights = []
    ended = -1
    try:
        with torch.inference_mode():
            for step in range(args.steps):
                z = float(robot.data.root_pos_w[0, 2] - origins[0, 2])
                peak = max(peak, z)
                if step % 60 == 0:
                    heights.append((step, z))
                if step % skip == 0:
                    writer.append_data(env.render())
                a = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                obs, _, term, trunc, _ = wrapped.step(a)
                if bool((term | trunc).any()):
                    ended = step
                    break
    finally:
        writer.close()

    env.close()
    print(f"\n  spawn z {z0:.2f} m, peak {peak:.2f} m", flush=True)
    print("  height each second:", flush=True)
    for s, h in heights:
        print(f"    t={s/60:4.1f}s  z={h:5.2f} m", flush=True)
    print(f"  {'ended at step %d' % ended if ended >= 0 else 'flew the whole clip'}",
          flush=True)
    print(f"RECORDED {os.path.getsize(path)/1e6:.1f} MB -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    sys.exit(status)
