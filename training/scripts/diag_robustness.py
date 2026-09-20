"""What breaks the policy that training could never show us?

Every metric we track -- reward, gates per episode, collision rate, the
per-gate table -- measures the distribution we chose to train on. None of them
can report a mismatch between that distribution and the aircraft, because in
simulation the mismatched input is always correct.

That is not hypothetical. The best racing policy averages 9.5 gates and fails
100% of episodes at the real race start, and no training metric showed it. It
took rendering a video and someone noticing the drone looked wrong.

So this perturbs the observation in the ways the real aircraft will, and
measures what each one costs:

    gate_shift    the one-hot gate index off by one. On the drone nothing
                  publishes which gate is next -- gate_tracker.py counts
                  crossings, and a counter that slips hands the policy a
                  confidently wrong context.
    gate_zero     the counter lost entirely.
    kp_noise      keypoints jitter. Training projects gate corners
                  analytically and exactly; YOLO does not.
    kp_dropout    a fraction of keypoints not detected at all.
    gyro_noise    IMU noise on the body rates.
    att_noise     IMU noise on roll and pitch.
    stale_frame   the newest frame repeated, i.e. a dropped telemetry update.
                  The organizer guides warn attitude arrives at 30-50 Hz while
                  the policy runs at 60.

Each is applied as a *standing condition* across the history, not a one-off
glitch, because that is what a drifted counter or a noisy sensor actually is.

    python scripts/diag_robustness.py --headless --checkpoint <path>
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

# Channel map within one frame, from contract/observation.py.
CORNERS = slice(0, 16)
VIS = slice(16, 24)
ATT = slice(24, 26)      # roll, pitch
GYRO = slice(26, 29)     # gx, gy, gz
VEL = slice(29, 32)
CONTEXT = slice(32, 51)  # 18 gate one-hot + lap fraction
GATE_ONEHOT = slice(32, 50)
NOT_SEEN = obs_contract.NOT_SEEN


def verify_layout(frames: torch.Tensor) -> None:
    """Confirm the reshape actually found frames before anything is perturbed.

    Reshaping a flat vector the wrong way round produces plausible-looking
    numbers and silently perturbs the wrong channels, which is the failure this
    whole file exists to catch elsewhere. Visibility is the cheapest witness:
    those eight channels are strictly 0 or 1, and nothing else in the frame is.
    """
    vis = frames[..., VIS]
    binary = torch.all((vis == 0.0) | (vis == 1.0))
    if not bool(binary):
        raise SystemExit(
            "layout check FAILED: channels 16-23 are not binary, so the "
            "reshape is not frame-major and every perturbation below would "
            "hit the wrong channels"
        )
    onehot = frames[..., GATE_ONEHOT].sum(dim=-1)
    if not bool(torch.all((onehot >= 0.0) & (onehot <= 1.0 + 1e-4))):
        raise SystemExit("layout check FAILED: gate one-hot does not sum to <=1")
    print("  layout verified: visibility binary, gate one-hot sums to <=1", flush=True)


def perturb(frames: torch.Tensor, kind: str, gen: torch.Generator) -> torch.Tensor:
    """Return a copy of (N, HISTORY, FRAME) with one standing fault applied."""
    f = frames.clone()
    if kind == "none":
        return f

    if kind == "gate_shift":
        # Roll the one-hot by one slot: the counter thinks it is a gate ahead.
        f[..., GATE_ONEHOT] = torch.roll(f[..., GATE_ONEHOT], shifts=1, dims=-1)
    elif kind == "gate_zero":
        f[..., GATE_ONEHOT] = 0.0
    elif kind == "kp_noise":
        # Corners are normalised, so 0.02 is roughly 1% of the frame.
        noise = torch.randn(f[..., CORNERS].shape, generator=gen,
                            device=f.device, dtype=f.dtype) * 0.02
        seen = f[..., CORNERS] != NOT_SEEN
        f[..., CORNERS] = torch.where(seen, f[..., CORNERS] + noise, f[..., CORNERS])
    elif kind == "kp_dropout":
        drop = (torch.rand(f[..., VIS].shape, generator=gen,
                           device=f.device, dtype=f.dtype) < 0.30)
        f[..., VIS] = torch.where(drop, torch.zeros_like(f[..., VIS]), f[..., VIS])
        # A keypoint that is not detected carries no position either.
        drop_uv = drop.repeat_interleave(2, dim=-1)
        f[..., CORNERS] = torch.where(
            drop_uv, torch.full_like(f[..., CORNERS], NOT_SEEN), f[..., CORNERS]
        )
    elif kind == "gyro_noise":
        f[..., GYRO] += torch.randn(f[..., GYRO].shape, generator=gen,
                                    device=f.device, dtype=f.dtype) * 0.05
    elif kind == "att_noise":
        f[..., ATT] += torch.randn(f[..., ATT].shape, generator=gen,
                                   device=f.device, dtype=f.dtype) * 0.02
    elif kind == "stale_frame":
        # Newest frame replaced by the one before it: an update that did not
        # arrive. Which frame is newest depends on the packing order, so do
        # both ends -- the cost is that this is a slightly harsher fault than
        # a single dropped update, not that it tests the wrong thing.
        f[:, -1, :] = f[:, -2, :]
        f[:, 0, :] = f[:, 1, :]
    else:
        raise SystemExit(f"unknown perturbation {kind!r}")
    return f


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    xc = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
    xc["trainer"]["close_environment_at_exit"] = False
    xc["agent"]["experiment"]["write_interval"] = 0
    xc["agent"]["experiment"]["checkpoint_interval"] = 0
    wrapped = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
    runner = Runner(wrapped, xc)
    runner.agent.load(args.checkpoint)
    runner.agent.set_running_mode("eval")

    cmd = unwrapped.command_manager.get_term("target")
    obs, _ = wrapped.reset()
    width = obs.shape[-1]
    frame_dim = width // obs_contract.HISTORY
    print(f"\nobservation {width} = {obs_contract.HISTORY} frames x {frame_dim}",
          flush=True)
    verify_layout(obs.view(args.num_envs, obs_contract.HISTORY, frame_dim))

    KINDS = ["none", "gate_shift", "gate_zero", "kp_noise", "kp_dropout",
             "gyro_noise", "att_noise", "stale_frame"]
    results = {}

    for kind in KINDS:
        gen = torch.Generator(device=obs.device).manual_seed(0)
        obs, _ = wrapped.reset()
        passed = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                frames = obs.view(args.num_envs, obs_contract.HISTORY, frame_dim)
                a = runner.agent.act(
                    perturb(frames, kind, gen).view(args.num_envs, width),
                    timestep=0, timesteps=0,
                )[0]
                obs, _, _, _, _ = wrapped.step(a)
                passed += int(cmd.gate_passed.sum())
        # gates per env per episode-equivalent; steps/60 s of flying
        per_env = passed / args.num_envs
        results[kind] = per_env
        base = results.get("none", per_env)
        rel = per_env / base if base > 0 else float("nan")
        print(f"  {kind:<13} {per_env:7.3f} gates/env   {rel:6.1%} of baseline",
              flush=True)

    print("\nROBUSTNESS " + " ".join(f"{k}={v:.3f}" for k, v in results.items()),
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
