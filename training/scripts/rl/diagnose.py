"""Instrumented probe for why the drone racer will not learn.

Runs the env under scripted action policies (no PPO) and reports, per step:
  * whether the observation is finite, and which channel first goes non-finite
  * whether the target gate is visible in the camera at all
  * geometry at spawn: range and bearing to the target gate vs the camera FoV
  * which termination fires, and how often a gate is actually passed

The question this exists to answer is whether the policy is ever given a usable
learning signal. If the gate is outside the frame on most steps, the keypoint
channels are all NOT_SEEN and there is nothing for the policy to servo on, which
would explain a flat reward curve regardless of PPO settings.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose drone racer training signal.")
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument(
    "--policy",
    type=str,
    default="hover",
    choices=["zero", "hover", "random"],
    help="zero = no input, hover = mid-throttle, random = uniform in [-1, 1].",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tasks  # noqa: F401, E402

# Kit swallows late stdout on shutdown, so mirror every line to a file.
_OUT = open("/tmp/diag_result.txt", "w")


def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    _OUT.write(line + "\n")
    _OUT.flush()


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    from utils.aigp_obs import KEYPOINT_COUNT, NOT_SEEN  # noqa: F401

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    n_env = env.num_envs
    feat = obs.shape[1]
    # History is flattened as [oldest ... newest]; the live frame is the tail.
    hist = getattr(env_cfg.observations.policy.aigp_state, "history_length", 0) or 1
    base = feat // hist
    emit(f"\nobs {tuple(obs.shape)} = {hist} frames x {base} features; probing newest frame")

    cmd = env.command_manager.get_term("target")
    robot = env.scene["robot"]

    stats = {
        "steps": 0,
        "nonfinite_steps": 0,
        "first_nonfinite_step": None,
        "first_nonfinite_channel": None,
        "vis_any": 0.0,
        "vis_count": 0.0,
        "gate_passes": 0.0,
        "gate_misses": 0.0,
        "act_sat": 0.0,
    }
    term_counts: dict[str, float] = {}
    spawn_range: list[float] = []
    spawn_bearing: list[float] = []
    spawn_visible: list[float] = []

    def gate_geometry():
        """Range and horizontal bearing from drone to its target gate."""
        tgt = cmd.command[:, :3]
        rel = tgt - robot.data.root_pos_w
        rng = torch.linalg.norm(rel, dim=-1)
        q = robot.data.root_quat_w
        # yaw of the body, then bearing of the gate relative to nose
        siny = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
        cosy = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        yaw = torch.atan2(siny, cosy)
        bearing = torch.atan2(rel[:, 1], rel[:, 0]) - yaw
        bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
        return rng, bearing.abs()

    rng0, bear0 = gate_geometry()
    spawn_range += rng0.tolist()
    spawn_bearing += bear0.degrees().tolist() if hasattr(bear0, "degrees") else (bear0 * 180 / math.pi).tolist()

    for step in range(args_cli.steps):
        if args_cli.policy == "zero":
            act = torch.zeros(n_env, env.action_space.shape[1], device=env.device)
        elif args_cli.policy == "hover":
            act = torch.zeros(n_env, env.action_space.shape[1], device=env.device)
            act[:, 0] = 0.2  # slight positive collective
        else:
            act = torch.rand(n_env, env.action_space.shape[1], device=env.device) * 2.0 - 1.0

        obs_dict, rew, terminated, truncated, info = env.step(act)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        newest = obs[:, -base:]

        stats["steps"] += 1
        bad = ~torch.isfinite(newest)
        if bool(bad.any()):
            stats["nonfinite_steps"] += 1
            if stats["first_nonfinite_step"] is None:
                stats["first_nonfinite_step"] = step
                chans = bad.any(dim=0).nonzero(as_tuple=False).flatten().tolist()
                stats["first_nonfinite_channel"] = chans[:12]

        vis = newest[:, 16:24]
        stats["vis_any"] += float((vis.sum(dim=-1) > 0).float().mean())
        stats["vis_count"] += float(vis.sum(dim=-1).mean())
        stats["gate_passes"] += float(cmd.gate_passed.float().sum())
        stats["gate_misses"] += float(cmd.gate_missed.float().sum())
        stats["act_sat"] += float((act.abs() >= 0.999).float().mean())

        done = terminated | truncated
        if bool(done.any()):
            for name, buf in env.termination_manager._term_dones.items():
                term_counts[name] = term_counts.get(name, 0.0) + float(buf.float().sum())
            ids = done.nonzero(as_tuple=False).flatten()
            r, b = gate_geometry()
            spawn_range += r[ids].tolist()
            spawn_bearing += (b[ids] * 180 / math.pi).tolist()
            spawn_visible += (vis[ids].sum(dim=-1) > 0).float().tolist()

    s = stats["steps"]
    emit("\n================ DIAGNOSIS ================")
    emit(f"policy={args_cli.policy}  envs={n_env}  steps={s}")
    emit("\n-- observation health --")
    if stats["first_nonfinite_step"] is None:
        emit("  all observations finite")
    else:
        emit(f"  NON-FINITE from step {stats['first_nonfinite_step']} "
              f"({stats['nonfinite_steps']}/{s} steps affected)")
        emit(f"  first bad channels (0-15 uv, 16-23 vis, 24-28 att/gyro, 29-31 vel, 32+ ctx): "
              f"{stats['first_nonfinite_channel']}")

    emit("\n-- can the policy SEE the gate? --")
    emit(f"  steps with >=1 keypoint visible : {100*stats['vis_any']/s:5.1f} %")
    emit(f"  mean visible keypoints (of 8)   : {stats['vis_count']/s:5.2f}")

    emit("\n-- geometry at spawn/reset --")
    if spawn_range:
        sr = sorted(spawn_range)
        sb = sorted(spawn_bearing)
        emit(f"  range to target gate : median {sr[len(sr)//2]:6.1f} m   "
              f"p90 {sr[int(0.9*len(sr))-1]:6.1f} m   max {sr[-1]:6.1f} m")
        emit(f"  |bearing| to gate    : median {sb[len(sb)//2]:6.1f} deg  "
              f"p90 {sb[int(0.9*len(sb))-1]:6.1f} deg  max {sb[-1]:6.1f} deg")
        emit("  camera HFoV is 90 deg, so |bearing| > 45 deg means the gate is off-frame")
        if spawn_visible:
            emit(f"  gate visible right after reset  : {100*sum(spawn_visible)/len(spawn_visible):5.1f} %")

    emit("\n-- learning signal --")
    emit(f"  gate passes : {stats['gate_passes']:.0f}")
    emit(f"  gate misses : {stats['gate_misses']:.0f}")
    emit(f"  action saturation (|a|>=1)      : {100*stats['act_sat']/s:5.1f} %")

    emit("\n-- terminations --")
    tot = sum(term_counts.values()) or 1.0
    for k, v in sorted(term_counts.items(), key=lambda kv: -kv[1]):
        emit(f"  {k:20s} {v:8.0f}  ({100*v/tot:5.1f} %)")
    emit("===========================================\n")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emit("EXCEPTION:\n" + traceback.format_exc())
    simulation_app.close()
