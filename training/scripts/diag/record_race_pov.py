"""Record the racing policy the way the stack recordings were made: chase | onboard.

Left: Isaac's chase view of the aircraft. Right: the drone's own tiled camera
(640x360, the calibrated lens, 20 deg up-tilt) with two overlays in the same
style as the stack recordings:

  green  the target gate's corners as the policy is actually fed them -- the
         last frame of its observation history, so exactly what the network
         saw that step, latch, delay and dropout included; an edge is drawn
         only when both of its corners are visible
  cyan   (with --detector) every gate the real hand497 ONNX detector finds on
         the same rendered frame, corners and box confidence, so the
         perception side can be judged against the policy's input

A status line carries time, gates passed, speed and height.

Built from scripts/diag/record_start.py (policy loading, pad starts, the
skrl reset trap, baked-in speed-up) and the perception-sim branch's
stack_sim/record_stack.py (side-by-side composition). Every episode is a
competition-pad start. Written on 23 Sep 2026 after the training pod had
gone away, so it has NOT been run on a live Isaac install yet; the RECORDED=
line, not the exit code, says whether it worked.

    ENABLE_CAMERAS=1 AIGP_POLICY_HZ=40 AIGP_KP_DROP=0.13 AIGP_KP_STICKY=0.81 \\
    python scripts/diag/record_race_pov.py --headless --enable_cameras \\
        --checkpoint archive/race40drop_best_agent.pt --attempts 3 --seconds 40 --speed 1

AIGP_POLICY_HZ must match the checkpoint (40 for race40/race40drop). Set the
dropout variables to film the policy under the measured detector, leave them
unset for perfect corners. For the cyan overlay add

        --detector /workspace/aigp-perception/models/gate_pose_hand497.onnx \\
        --perception-repo /workspace/aigp-perception

(needs `pip install onnxruntime-gpu` or `onnxruntime` on the pod). The
detector was trained on real gates, not renders; how well it fires on Isaac's
orange squares is itself one of the things worth seeing.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-Drone-Racer-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--attempts", type=int, default=3, help="separate pad starts, one file each")
parser.add_argument("--seconds", type=float, default=40.0, help="cut each attempt here if it is still flying")
parser.add_argument("--speed", type=float, default=1.0, help="playback multiple, baked in by dropping frames")
parser.add_argument("--out", default="/workspace/videos/race_pov")
parser.add_argument("--ml_framework", default="torch")
parser.add_argument("--detector", default=None, help="hand497 .onnx; draws what the real detector sees in cyan")
parser.add_argument("--perception-repo", default="/workspace/aigp-perception", help="checkout of bojro/aigp-perception")
parser.add_argument("--detector-conf", type=float, default=0.4)
parser.add_argument("--detector-kpt-conf", type=float, default=0.25)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# The Play config keeps its onboard camera only under this switch, and it is
# read at config-parse time, so it has to be set before anything Isaac Lab.
os.environ["ENABLE_CAMERAS"] = "1"
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import tasks  # noqa: E402,F401
from utils.keypoint_overlay import annotate_rgb  # noqa: E402

HISTORY = 32
GREEN, CYAN = (60, 230, 60), (60, 220, 230)     # RGB, matching the stack recordings' legend

try:
    import cv2  # noqa: E402
except ImportError:  # the overlay degrades to the NumPy-only annotate_rgb
    cv2 = None


def _text(img: np.ndarray, s: str, y: int, scale: float = 0.45) -> None:
    if cv2 is not None:
        cv2.putText(img, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_gate(img: np.ndarray, uv: np.ndarray, visible: np.ndarray, colour) -> None:
    """Corner dots plus each ring's edges, an edge only where both corners were seen."""
    for (u, v), vis in zip(uv, visible):
        if vis:
            cv2.circle(img, (int(round(u)), int(round(v))), 4, colour, -1)
    for base in (0, 4):
        for a in range(4):
            i, j = base + a, base + (a + 1) % 4
            if visible[i] and visible[j]:
                cv2.line(img, tuple(int(round(x)) for x in uv[i]), tuple(int(round(x)) for x in uv[j]),
                         colour, 1, cv2.LINE_AA)


def policy_corners(frame_obs: np.ndarray, w: int, h: int):
    """The target gate as the policy is fed it: 16 normalised coords then 8 visibility flags."""
    uv = frame_obs[:16].reshape(8, 2) * np.array([w, h], dtype=np.float32)
    visible = frame_obs[16:24] > 0.5
    return uv, visible


def load_detector():
    if args.detector is None:
        return None
    sys.path.insert(0, os.path.join(args.perception_repo, "deploy", "onnx"))
    from gate_detector import GateDetector  # noqa: PLC0415

    det = GateDetector(args.detector, conf=args.detector_conf, kpt_conf=args.detector_kpt_conf)
    print(f"  detector {os.path.basename(args.detector)} on {det.provider}", flush=True)
    return det


def main() -> int:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)

    # Start geometry comes from the training config: the Play config is
    # assembled, not derived, and carries none of the racing start fields.
    train_cfg = parse_env_cfg("Isaac-Drone-Racer-v0", device=args.device, num_envs=1)
    for field in ("race_start_xy", "race_start_idx", "race_start_scatter_m", "spawn_z_range"):
        setattr(cfg.commands.target, field, getattr(train_cfg.commands.target, field))
    cfg.commands.target.race_start_fraction = 1.0
    cfg.commands.target.floor_start_fraction = 0.0
    if cfg.commands.target.race_start_xy is None:
        raise SystemExit("race_start_xy is None after copying from the training config; refusing to record")
    if getattr(cfg.scene, "tiled_camera", None) is None:
        raise SystemExit("no onboard camera in the scene: ENABLE_CAMERAS did not take effect")

    policy_hz = 1.0 / (float(cfg.sim.dt) * int(cfg.decimation))
    print(f"\nrecording pad starts at {cfg.commands.target.race_start_xy}, policy at {policy_hz:g} Hz "
          f"(AIGP_POLICY_HZ={os.environ.get('AIGP_POLICY_HZ', 'unset')}), "
          f"dropout q={os.environ.get('AIGP_KP_DROP', '0')} rho={os.environ.get('AIGP_KP_STICKY', '0')}", flush=True)

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

    uw = env.unwrapped
    cmd = uw.command_manager.get_term("target")
    robot = uw.scene["robot"]
    cam = uw.scene.sensors["tiled_camera"]
    origins = uw.scene.env_origins

    import imageio.v2 as imageio  # noqa: PLC0415

    detector = load_detector()
    skip = max(1, int(round(args.speed)))
    out_fps = int(round(policy_hz))
    steps = int(args.seconds * policy_hz)
    results = []
    for attempt in range(1, args.attempts + 1):
        wrapped._reset_once = True  # skrl resets once and caches; re-arm it (see record_start.py)
        obs, _ = wrapped.reset()
        path = os.path.join(args.out, f"race_pov_{attempt:02d}.mp4")
        writer = imageio.get_writer(path, fps=out_fps, macro_block_size=None)
        gates, end = 0, "cut"
        try:
            with torch.inference_mode():
                for step in range(steps):
                    if step % skip == 0:
                        flat = obs[0].detach().cpu().numpy().reshape(-1)
                        frame_dim = flat.size // HISTORY          # 51 for v1, 55 for v2
                        current = flat[-frame_dim:]              # the newest frame is last
                        fpv = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8).copy()
                        if cv2 is None:
                            fpv = annotate_rgb(fpv, current)
                        else:
                            h, w = fpv.shape[:2]
                            n_det = 0
                            if detector is not None:
                                for g in detector.detect(fpv[..., ::-1]):   # detector wants BGR
                                    draw_gate(fpv, g.keypoints, g.kpt_visible, CYAN)
                                    x1, y1 = g.box[:2]
                                    cv2.putText(fpv, f"{g.conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1, cv2.LINE_AA)
                                    n_det += 1
                            uv, vis = policy_corners(current, w, h)
                            draw_gate(fpv, uv, vis, GREEN)
                            legend = f"green = corners the policy is fed ({int(vis.sum())}/8)"
                            if detector is not None:
                                legend += f"   cyan = {os.path.basename(args.detector)} on this frame ({n_det} gates)"
                            _text(fpv, legend, 18, 0.4)
                        chase = np.asarray(env.render())[..., :3].astype(np.uint8)
                        if chase.shape[:2] != fpv.shape[:2]:
                            try:
                                import cv2  # noqa: PLC0415

                                chase = cv2.resize(chase, (fpv.shape[1], fpv.shape[0]))
                            except ImportError:
                                ys = np.linspace(0, chase.shape[0] - 1, fpv.shape[0]).astype(int)
                                xs = np.linspace(0, chase.shape[1] - 1, fpv.shape[1]).astype(int)
                                chase = chase[ys][:, xs]
                        pos = (robot.data.root_pos_w[0] - origins[0]).cpu().numpy()
                        spd = float(robot.data.root_lin_vel_w[0].norm())
                        t = step / policy_hz
                        _text(fpv, f"t {t:5.1f}s  gates {gates}  {spd:4.1f} m/s  z {pos[2]:4.2f} m", 352)
                        _text(chase, f"Isaac chase view   attempt {attempt}   {os.path.basename(args.checkpoint)}", 18)
                        writer.append_data(np.hstack([chase, fpv]))
                    a = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                    obs, _, term, trunc, _ = wrapped.step(a)
                    gates += int(cmd.gate_passed.sum())
                    if bool((term | trunc).any()):
                        end = f"ended at {step / policy_hz:.1f} s"
                        break
        finally:
            writer.close()
        results.append((attempt, gates, end, path))
        print(f"  attempt {attempt}: {gates} gates, {end} -> {path}", flush=True)

    env.close()
    for a, g, e, p in results:
        print(f"RECORDED attempt {a}: {g} gates, {e}: {p}", flush=True)
    print(f"\nRECORDED={len(results)} files in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    sys.exit(status)
