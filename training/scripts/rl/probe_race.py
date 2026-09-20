# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure *why* a trained racing policy stops improving.

Two questions the TensorBoard scalars cannot answer:

1. **Why do episodes end?** Isaac Lab reports per-term episode counts, but skrl's
   trainer only forwards ``info["log"]`` entries that are ``torch.Tensor`` and
   Isaac Lab writes those counts as Python ints, so they never reach TensorBoard.
2. **Is the gate-speed bonus saturating?** ``gate_pass_speed`` is clamped at
   ``max_scale``; once crossings sit at the cap the term is a flat bonus with no
   gradient, and raising the cap is the only way to keep buying speed.

Runs the checkpoint headless over many environments and reports both, plus the
distribution of gates reached per episode.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe a trained drone racing policy.")
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--ml_framework", type=str, default="torch")
parser.add_argument("--out", type=str, default="/tmp/probe_race.txt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import tasks  # noqa: F401, E402

_fh = open(args_cli.out, "w")


def emit(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    _fh.write(msg + "\n")
    _fh.flush()


def pct(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q)) if values.numel() else float("nan")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    if "SEPARATE_NETS" in os.environ:
        experiment_cfg["models"]["separate"] = bool(int(os.environ["SEPARATE_NETS"]))
    runner = Runner(env, experiment_cfg)

    ckpt = os.path.abspath(args_cli.checkpoint)
    emit(f"checkpoint : {ckpt}")
    runner.agent.load(ckpt)
    runner.agent.set_running_mode("eval")

    base = env.unwrapped
    cmd = base.command_manager.get_term("target")
    term_mgr = base.termination_manager
    reward_cfg = base.cfg.rewards
    ref_speed = float(reward_cfg.gate_pass_speed.params.get("ref_speed", 12.0))
    max_scale = float(reward_cfg.gate_pass_speed.params.get("max_scale", 1.5))
    cap_speed = ref_speed * max_scale

    n = base.num_envs
    dev = base.device
    speeds: list[torch.Tensor] = []
    term_counts = {name: 0 for name in term_mgr.active_terms}
    gates_this_ep = torch.zeros(n, dtype=torch.int32, device=dev)
    # Counted here rather than read from ``episode_length_buf``: Isaac Lab
    # resets finished envs inside ``step``, so by the time the done mask is
    # visible that buffer has already been zeroed.
    steps_alive = torch.zeros(n, dtype=torch.int32, device=dev)
    ep_gates: list[torch.Tensor] = []
    ep_lengths: list[torch.Tensor] = []
    crash_gate: list[torch.Tensor] = []
    episodes = 0

    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            # Snapshot the target before stepping: finished envs are reset
            # inside ``step``, which resamples the command.
            target_before = cmd.next_gate_idx.clone()
            obs, _, terminated, truncated, _ = env.step(actions)
            steps_alive += 1

            crashed = term_mgr.get_term("collision")
            if bool(crashed.any()):
                crash_gate.append(target_before[crashed].clone())

            passed = cmd.gate_passed
            if bool(passed.any()):
                speeds.append(cmd.gate_pass_speed[passed].clone())
                gates_this_ep += passed.int()

            for name in term_counts:
                term_counts[name] += int(term_mgr.get_term(name).sum())

            done = (terminated | truncated).view(-1)
            if bool(done.any()):
                ep_gates.append(gates_this_ep[done].clone())
                ep_lengths.append(steps_alive[done].clone())
                gates_this_ep[done] = 0
                steps_alive[done] = 0
                episodes += int(done.sum())

    all_speeds = torch.cat(speeds) if speeds else torch.zeros(0, device=dev)
    all_gates = torch.cat(ep_gates).float() if ep_gates else torch.zeros(0, device=dev)
    all_len = torch.cat(ep_lengths).float() if ep_lengths else torch.zeros(0, device=dev)
    dt = float(base.step_dt)

    emit(f"\nenvs={n}  steps={args_cli.steps}  episodes={episodes}  policy_dt={dt:.4f}s")

    emit("\n-- why episodes end --")
    total_term = sum(term_counts.values())
    for name, c in sorted(term_counts.items(), key=lambda kv: -kv[1]):
        share = 100.0 * c / total_term if total_term else 0.0
        emit(f"  {name:18s} {c:7d}  {share:5.1f} %")

    emit("\n-- gates reached per episode --")
    if all_gates.numel():
        emit(f"  mean {float(all_gates.mean()):5.2f}   median {pct(all_gates, 0.5):5.1f}   "
             f"p90 {pct(all_gates, 0.9):5.1f}   max {int(all_gates.max())}")
        for k in (0, 1, 3, 5, 10, 16):
            emit(f"  >= {k:2d} gates : {100.0 * float((all_gates >= k).float().mean()):5.1f} %")

    emit("\n-- episode length --")
    if all_len.numel():
        emit(f"  mean {float(all_len.mean()) * dt:5.2f} s   p90 {pct(all_len, 0.9) * dt:5.2f} s   "
             f"max {float(all_len.max()) * dt:5.2f} s   (limit {base.max_episode_length * dt:.1f} s)")

    emit("\n-- which gate were they heading to when they crashed --")
    if crash_gate:
        cg = torch.cat(crash_gate)
        total_c = cg.numel()
        counts = torch.bincount(cg.long(), minlength=cmd.num_gates)
        for i, c in enumerate(counts.tolist()):
            if c:
                emit(f"  gate {i + 1:2d} : {c:6d}  {100.0 * c / total_c:5.1f} %")

    emit("\n-- gate crossing speed --")
    emit(f"  reward ref_speed={ref_speed:.1f} m/s  max_scale={max_scale:.2f}  -> cap at {cap_speed:.1f} m/s")
    if all_speeds.numel():
        emit(f"  crossings {all_speeds.numel()}")
        emit(f"  mean {float(all_speeds.mean()):5.2f} m/s   median {pct(all_speeds, 0.5):5.2f}   "
             f"p90 {pct(all_speeds, 0.9):5.2f}   max {float(all_speeds.max()):5.2f}")
        at_cap = 100.0 * float((all_speeds >= cap_speed).float().mean())
        emit(f"  at or above cap : {at_cap:5.1f} %   <- if high, the bonus is flat and buys no more speed")
    else:
        emit("  no gates passed")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emit("EXCEPTION:\n" + traceback.format_exc())
    finally:
        _fh.close()
        simulation_app.close()
