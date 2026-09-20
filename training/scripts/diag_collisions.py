"""What is the racing policy actually hitting?

Racing plateaus at about a quarter of the course and every episode ends in a
crash -- ``time_out`` is exactly zero across ~450,000 episodes. The contact
sensor covers ``Robot/.*`` against everything with a 0.01 N threshold, so
gates, ground and arena walls are indistinguishable in the training logs, and
the fix is completely different for each:

    ground  -> altitude control; suspect the low-gate dive term
    gate    -> threading too tight; wants clearance shaping or a wider approach
    wall    -> overshooting turns at speed

So: fly the trained policy, and every time a collision ends an episode, record
where the aircraft was and classify it. Classification is by position rather
than by contact filtering, because the sensor reports a net force per body and
not what the body touched.

    python scripts/diag_collisions.py --headless --checkpoint <path>

Reports the split, plus the state at impact -- height, speed, and distance to
the gate it was aiming at -- because "hit a gate at 2 m/s while centred" and
"hit a gate at 12 m/s while 3 m off axis" are different problems wearing the
same label.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Drone-Racer-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1200, help="20 s at 60 Hz")
parser.add_argument("--ground_z", type=float, default=0.35,
                    help="below this counts as ground contact")
parser.add_argument("--gate_r", type=float, default=1.5,
                    help="within this of a gate centre counts as gate contact")
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


def percentile(t: torch.Tensor, q: float) -> float:
    return float(t.quantile(q)) if t.numel() else float("nan")


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
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

    robot = unwrapped.scene["robot"]
    cmd = unwrapped.command_manager.get_term("target")
    tm = unwrapped.termination_manager

    # (num_envs, num_gates, 3) in world frame.
    gate_pos = cmd.track.data.object_com_pos_w
    num_gates = gate_pos.shape[1]
    print(f"\ncourse: {num_gates} gates", flush=True)
    lo = gate_pos.reshape(-1, 3).min(dim=0).values
    hi = gate_pos.reshape(-1, 3).max(dim=0).values
    print(f"gate extent: x {float(lo[0]):.1f}..{float(hi[0]):.1f}  "
          f"y {float(lo[1]):.1f}..{float(hi[1]):.1f}  "
          f"z {float(lo[2]):.1f}..{float(hi[2]):.1f}", flush=True)

    heights, speeds, gate_dists, target_dists = [], [], [], []
    xy_x, xy_y = [], []
    n_collisions = 0

    obs, _ = wrapped.reset()
    with torch.inference_mode():
        for _ in range(args.steps):
            # Snapshot BEFORE stepping. Isaac resets terminated envs inside
            # step(), so reading position afterwards gives the respawn point,
            # not the impact: zero velocity, 1 m past a gate, every time. The
            # smoke test had the same defect with the thrust buffer.
            prev_pos = robot.data.root_pos_w.clone()
            prev_vel = robot.data.root_lin_vel_w.clone()

            actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
            obs, _, terminated, truncated, _ = wrapped.step(actions)

            hit = tm.get_term("collision")
            if not bool(hit.any()):
                continue

            idx = hit.nonzero(as_tuple=False).squeeze(-1)
            pos = prev_pos[idx]
            vel = prev_vel[idx]

            # Distance to the nearest gate of any kind, and to the one it was
            # actually aiming at -- a crash next to the wrong gate is a
            # navigation failure, not a threading failure.
            d_all = torch.norm(gate_pos[idx] - pos.unsqueeze(1), dim=-1)
            nearest = d_all.min(dim=1).values
            target = torch.norm(cmd.command[idx, :3] - pos, dim=-1)

            xy_x.append(pos[:, 0].clone())
            xy_y.append(pos[:, 1].clone())
            heights.append(pos[:, 2].clone())
            speeds.append(torch.norm(vel, dim=-1).clone())
            gate_dists.append(nearest.clone())
            target_dists.append(target.clone())
            n_collisions += int(idx.numel())

    if not heights:
        print("\nno collisions recorded", flush=True)
        env.close()
        return 1

    z = torch.cat(heights)
    v = torch.cat(speeds)
    dg = torch.cat(gate_dists)
    dt = torch.cat(target_dists)

    is_ground = z < args.ground_z
    is_gate = (~is_ground) & (dg < args.gate_r)
    is_other = ~is_ground & ~is_gate

    total = z.numel()
    print(f"\n{total} collisions over {args.steps} steps, "
          f"{args.num_envs} envs\n", flush=True)
    print(f"  {'what it hit':<16} {'share':>8}   {'median speed':>13} {'median z':>10}",
          flush=True)
    print("  " + "-" * 54, flush=True)
    for label, mask in (("ground", is_ground), ("gate", is_gate),
                        ("other/wall", is_other)):
        n = int(mask.sum())
        if not n:
            print(f"  {label:<16} {0.0:>7.1%}", flush=True)
            continue
        print(f"  {label:<16} {n / total:>7.1%}   "
              f"{float(v[mask].median()):>12.2f}m/s {float(z[mask].median()):>9.2f}m",
              flush=True)

    print(f"\n  impact speed      median {float(v.median()):.2f}  "
          f"p95 {percentile(v, 0.95):.2f} m/s", flush=True)
    print(f"  height at impact  median {float(z.median()):.2f}  "
          f"p05 {percentile(z, 0.05):.2f} m", flush=True)
    print(f"  dist to nearest gate  median {float(dg.median()):.2f} m", flush=True)
    print(f"  dist to TARGET gate   median {float(dt.median()):.2f} m", flush=True)

    # Dump the impact coordinates so the crashes can be drawn on the course.
    # A per-gate failure rate says which gate is hard; a map says *where* on
    # the leg the aircraft is losing it, which is a different question and the
    # one that tells you whether it is overshooting turns or clipping frames.
    import json
    xs = torch.cat([h for h in xy_x]).tolist()
    ys = torch.cat([h for h in xy_y]).tolist()
    with open("/workspace/logs/crash_xy.json", "w") as fh:
        json.dump({"x": [round(v, 2) for v in xs],
                   "y": [round(v, 2) for v in ys],
                   "z": [round(float(v), 2) for v in z.tolist()],
                   "speed": [round(float(v), 2) for v in v.tolist()]}, fh)
    print(f"wrote /workspace/logs/crash_xy.json ({len(xs)} points)", flush=True)

    print(f"\nCOLLISION_SPLIT ground={float(is_ground.float().mean()):.3f} "
          f"gate={float(is_gate.float().mean()):.3f} "
          f"other={float(is_other.float().mean()):.3f} "
          f"n={total}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        status = main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    sys.exit(status)
