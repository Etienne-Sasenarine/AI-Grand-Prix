"""Record the real race start: on the ground at the pad, then up and through G1.

Every episode is a pad start (race_start_fraction forced to 1.0), and each one
is written to its own file. Deliberately small: one env, one camera, and only
every Nth frame rendered. The training chains own this GPU and this is a
spectator.

Four things this file is careful about, all of them because the first version
got them wrong and produced a confident, wrong answer:

  * skrl's IsaacLabWrapper.reset() resets ONCE. Every later call returns a
    cached observation and never touches the simulation. Calling it in a loop
    recorded one continuous flight while reporting six separate starts.
  * Speed-up is baked into the frame count, not declared in the header. A file
    stamped 180 fps played back at 60 in QuickTime, so the 3x never happened.
  * The video is finalised by writer.close() and env.close().
    SimulationApp.close() hard-exits, so anything unflushed is gone -- an
    earlier recording had every frame and no moov atom, which no player opens.
  * DroneRacerEnvCfg_PLAY subclasses ManagerBasedRLEnvCfg, NOT
    DroneRacerEnvCfg, so it inherits none of the racing start settings and
    race_start_xy is None. The geometry is copied from the training config
    below, and the script refuses to run if it is still missing.

Isaac's exit code is meaningless. The RECORDED= line is the result.

    python scripts/record_start.py --headless --enable_cameras \
        --checkpoint <path> --attempts 5 --attempt_steps 480 --speed 3
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Drone-Racer-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--attempts", type=int, default=5,
                    help="separate pad starts, each written to its own file -- "
                         "one strip of near-identical flights reads as a single "
                         "flight, which is what the first attempt at this made")
parser.add_argument("--attempt_steps", type=int, default=480,
                    help="max sim steps per attempt; 60 steps = 1 s of flight")
parser.add_argument("--speed", type=float, default=3.0,
                    help="playback multiple; encodes at speed x 60 fps")
parser.add_argument("--out", default="/workspace/videos")
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

    # Take the start geometry from the *training* config, not from this one.
    #
    # DroneRacerEnvCfg_PLAY subclasses ManagerBasedRLEnvCfg, not
    # DroneRacerEnvCfg -- it is assembled from the same components rather than
    # derived -- so none of the racing __post_init__ start settings reach it
    # and race_start_xy is simply None. Recording then silently shows ordinary
    # gate run-ins while announcing pad starts, which is the same failure this
    # whole day has been about: a config field that does nothing and says so
    # nowhere. Copying from the training config also means the pad cannot drift
    # between what the policy trains on and what we film.
    train_cfg = parse_env_cfg("Isaac-Drone-Racer-v0", device=args.device, num_envs=1)
    for field in ("race_start_xy", "race_start_idx", "race_start_scatter_m",
                  "spawn_z_range"):
        setattr(cfg.commands.target, field,
                getattr(train_cfg.commands.target, field))

    # Every episode is the competition start, and nothing else.
    cfg.commands.target.race_start_fraction = 1.0
    cfg.commands.target.floor_start_fraction = 0.0

    pad = cfg.commands.target.race_start_xy
    if pad is None:
        raise SystemExit(
            "race_start_xy is None after copying from the training config, so "
            "no episode would start at the pad and this recording would show "
            "something else entirely. Refusing to record."
        )
    print(f"\nrecording pad starts at {pad} -> Isaac gate "
          f"{cfg.commands.target.race_start_idx} (official G1), "
          f"spawn z {cfg.commands.target.spawn_z_range}", flush=True)

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
    cmd = unwrapped.command_manager.get_term("target")
    robot = unwrapped.scene["robot"]
    origins = unwrapped.scene.env_origins

    import imageio.v2 as imageio  # noqa: PLC0415 - only needed once Isaac is up

    # Speed is baked into the frame count, not declared in the header.
    #
    # Writing 1200 frames and stamping 180 fps on the container produced a file
    # that players simply ignored: QuickTime ran it at 60 and showed 20 seconds
    # of real-time flight instead of 6.7 seconds at 3x. Dropping frames instead
    # means the speed-up survives any player, because there is nothing left to
    # honour or ignore.
    skip = max(1, int(round(args.speed)))
    out_fps = 60
    print(f"  capturing every {skip} sim step at {out_fps} fps "
          f"= {skip * out_fps / 60:g}x real time", flush=True)

    results = []
    for attempt in range(1, args.attempts + 1):
        # skrl's IsaacLabWrapper.reset() resets exactly once and is a no-op
        # after that -- it returns a cached observation and never touches the
        # simulation. An earlier version of this script called it five times
        # believing it was restarting the run, and recorded one continuous
        # flight while reporting six. Re-arming the flag is what actually
        # reaches ManagerBasedRLEnv.reset().
        wrapped._reset_once = True
        obs, _ = wrapped.reset()

        path = os.path.join(args.out, f"race_start_{attempt:02d}.mp4")
        writer = imageio.get_writer(path, fps=out_fps, macro_block_size=None)

        z0 = float((robot.data.root_pos_w[0, 2] - origins[0, 2]))
        peak_z, gates, died = z0, 0, -1
        try:
            with torch.inference_mode():
                for step in range(args.attempt_steps):
                    z = float((robot.data.root_pos_w[0, 2] - origins[0, 2]))
                    peak_z = max(peak_z, z)
                    if step % skip == 0:
                        writer.append_data(env.render())
                    a = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                    obs, _, term, trunc, _ = wrapped.step(a)
                    gates += int(cmd.gate_passed.sum())
                    if bool((term | trunc).any()):
                        died = step
                        break
        finally:
            writer.close()

        size = os.path.getsize(path) / 1e6
        outcome = f"ended at step {died} ({died/60:.1f} s)" if died >= 0 else "still flying at cut"
        results.append((attempt, z0, peak_z, gates, outcome, size, path))
        print(f"  attempt {attempt}: spawn z={z0:.2f} peak z={peak_z:.2f} "
              f"gates={gates}  {outcome}  [{size:.1f} MB]", flush=True)

    env.close()
    print("\n  per-attempt summary")
    for a, z0, pz, g, oc, sz, pth in results:
        print(f"    {a:>2}  spawn {z0:4.2f} m  peak {pz:5.2f} m  "
              f"{g} gates  {oc}", flush=True)
    print(f"\nRECORDED={len(results)} files in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    sys.exit(status)
