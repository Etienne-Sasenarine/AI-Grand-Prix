"""How many environments does this card actually want?

Renting a GPU by the hour makes throughput a budget question, not a tuning
one. The answer is not "as many as fit": env-steps per second climbs with
``num_envs`` while the GPU has idle capacity and then flattens, and past that
point every extra environment costs memory and startup time for nothing.

Measures the flat part directly, on the task that will actually be trained,
because the shape depends on the scene -- contact sensors, articulation count
and the physics substep rate all move it.

    python scripts/bench_throughput.py --headless
    python scripts/bench_throughput.py --headless --counts 4096,8192,16384

Reports env-steps/s and peak GPU memory per configuration, and names the
knee. Run it before a long run, not during one -- it needs the card to itself.
"""

from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Drone-Racer-v0")
parser.add_argument("--counts", default="2048,4096,8192,16384,32768",
                    help="comma-separated environment counts to try")
parser.add_argument("--warmup", type=int, default=40,
                    help="steps discarded before timing; the first are always slow")
parser.add_argument("--steps", type=int, default=150, help="timed steps")
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


def measure(num_envs: int) -> dict | None:
    """Build, warm up, time. Returns None if the configuration will not run."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    build_start = time.perf_counter()
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
        env = gym.make(args.task, cfg=env_cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"  {num_envs:>6}  BUILD FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None
    build_s = time.perf_counter() - build_start

    unwrapped = env.unwrapped
    device = unwrapped.device

    try:
        env.reset()
        # Random actions rather than zeros. Zero action is hover, which keeps
        # every aircraft in free flight and never exercises the contact and
        # reset paths -- and those are a real part of the per-step cost.
        for _ in range(args.warmup):
            env.step(torch.rand(num_envs, 4, device=device) * 2.0 - 1.0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(torch.rand(num_envs, 4, device=device) * 2.0 - 1.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        print(f"  {num_envs:>6}  RUN FAILED: {type(exc).__name__}: {exc}", flush=True)
        env.close()
        return None

    # Both figures, because the interesting one is not the obvious one.
    # torch.cuda only sees its own allocator: PhysX holds the simulation state
    # outside it, and at 4096 envs that is ~7 GB against torch's 0.14. Sizing a
    # run off the torch number would suggest the card is nearly empty when it
    # is not, so the device-wide reading is what gets reported.
    torch_gb = torch.cuda.max_memory_allocated() / 1024**3
    free_b, total_b = torch.cuda.mem_get_info()
    device_gb = (total_b - free_b) / 1024**3
    steps_per_s = args.steps / elapsed
    env_steps_per_s = steps_per_s * num_envs

    env.close()
    return {
        "num_envs": num_envs,
        "it_per_s": steps_per_s,
        "env_steps_per_s": env_steps_per_s,
        "device_gb": device_gb,
        "torch_gb": torch_gb,
        "build_s": build_s,
    }


def main() -> int:
    counts = [int(c) for c in args.counts.split(",") if c.strip()]
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\ndevice: {torch.cuda.get_device_name(0)}  ({total_gb:.1f} GB)", flush=True)
    print(f"task:   {args.task}", flush=True)
    print(f"\n  {'envs':>6}  {'it/s':>8}  {'env-steps/s':>12}  {'GPU GB':>7}  "
          f"{'torch':>6}  {'build s':>8}  {'vs prev':>8}", flush=True)
    print("  " + "-" * 70, flush=True)

    results = []
    for num_envs in counts:
        row = measure(num_envs)
        if row is None:
            # A failure at one size does not invalidate the smaller ones, but
            # everything above it will fail too.
            break
        gain = ""
        if results:
            ratio = row["env_steps_per_s"] / results[-1]["env_steps_per_s"]
            gain = f"{ratio:+.2f}x" if ratio < 1 else f"{ratio:.2f}x"
        print(f"  {row['num_envs']:>6}  {row['it_per_s']:>8.2f}  "
              f"{row['env_steps_per_s']:>12,.0f}  {row['device_gb']:>7.1f}  "
              f"{row['torch_gb']:>6.2f}  {row['build_s']:>8.1f}  {gain:>8}", flush=True)
        results.append(row)

    if not results:
        print("\nnothing ran", flush=True)
        return 1

    best = max(results, key=lambda r: r["env_steps_per_s"])

    # The knee, not the maximum. Once each doubling buys less than 15% more
    # throughput, the extra environments are costing memory and startup time
    # for almost nothing -- and on a rented box startup time is money too.
    knee = results[0]
    for prev, cur in zip(results, results[1:]):
        if cur["env_steps_per_s"] / prev["env_steps_per_s"] < 1.15:
            break
        knee = cur

    print(f"\nfastest:     {best['num_envs']} envs at "
          f"{best['env_steps_per_s']:,.0f} env-steps/s", flush=True)
    print(f"knee:        {knee['num_envs']} envs at "
          f"{knee['env_steps_per_s']:,.0f} env-steps/s "
          f"({knee['device_gb']:.1f} GB device-wide)", flush=True)
    print(f"\nuse --num_envs {knee['num_envs']} unless memory is needed elsewhere.",
          flush=True)
    print(f"BENCH_KNEE={knee['num_envs']} "
          f"BENCH_RATE={knee['env_steps_per_s']:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    sys.exit(status)
