# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Start at a 1x real-time pace (T cycles this).")
parser.add_argument(
    "--speed-cap",
    type=float,
    default=None,
    help="Post-policy speed limiter (m/s). 0 = off. Overrides PLAY_SPEED_CAP_MPS.",
)
parser.add_argument(
    "--time-scale",
    type=float,
    default=None,
    help="Playback scale: 1 = real-time, 0.5 = half-speed, 0.25 = quarter-speed. Overrides --real-time.",
)
parser.add_argument(
    "--renderer",
    type=str,
    default="RayTracedLighting",
    choices=["RayTracedLighting", "PathTracing"],
    help="Renderer to use.",
)
parser.add_argument("--log", type=int, default=None, help="Log the observations and metrics.")
parser.add_argument(
    "--record_yolo",
    type=int,
    default=0,
    help="Save this many FPV frames as a YOLO-pose keypoint dataset (0 = off).",
)
parser.add_argument(
    "--record_yolo_every",
    type=int,
    default=2,
    help="Save every N policy steps when --record_yolo is set (default 2 ≈ 30 Hz).",
)
parser.add_argument(
    "--record_yolo_dir",
    type=str,
    default="datasets",
    help="Directory under which the YOLO dataset folder is created.",
)
parser.add_argument(
    "--no-kp-overlay",
    action="store_true",
    default=False,
    help="Do not open the FPV + derived-keypoint overlay window.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video / YOLO frames / keypoint overlay
if args_cli.video or args_cli.record_yolo or not args_cli.no_kp_overlay:
    args_cli.enable_cameras = True
    os.environ["ENABLE_CAMERAS"] = "1"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import tempfile
import time

import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)

import tasks  # noqa: F401
from utils.aigp_obs import FRAME_H, FRAME_W
from utils.keypoint_overlay import annotate_rgb, current_race_obs_frame
from utils.logger import CSVLogger
from utils.speed_cap import cap_from_env, limit_policy_action
from utils.yolo_pose_recorder import YoloPoseRecorder

# Organizer PDF: 85x165 ft, origin top-left, Y down. Used only for HUD labels.
_PDF_FT = {
    1: (12.0, 71.0),
    2: (16.0, 39.0),
    3: (40.0, 17.0),
    4: (72.0, 40.0),
    5: (66.0, 70.0),
    6: (41.0, 86.0),
    7: (70.5, 106.6),
    8: (60.0, 129.0),
    9: (39.7, 147.7),
    10: (13.0, 110.0),
}


def official_number_from_xy(x: float, y: float) -> int:
    best_n, best_d = 1, float("inf")
    for n, (x_ft, y_ft) in _PDF_FT.items():
        ox, oy = x_ft * 0.3048, (165.0 - y_ft) * 0.3048
        d = (x - ox) ** 2 + (y - oy) ** 2
        if d < best_d:
            best_d, best_n = d, n
    return best_n

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.log and args_cli.num_envs > 1:
        raise ValueError("Logging is only supported for a single agent. Set --num_envs to 1.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.log:
        logger = CSVLogger(log_dir)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    # Must match the checkpoint: shared vs separate trunks have different
    # state_dict layouts. Pairs with GATE_LOOKAHEAD, which sets the obs width.
    # VISUAL_GATE_COUNTER=1 replaces privileged next_gate_idx with the onboard
    # pass counter (same 51-D one-hot the real bird will have to fill).
    if "SEPARATE_NETS" in os.environ:
        experiment_cfg["models"]["separate"] = bool(int(os.environ["SEPARATE_NETS"]))

    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if os.environ.get("VISUAL_GATE_COUNTER", "").strip().lower() in ("1", "true", "yes"):
        print("[INFO] VISUAL_GATE_COUNTER=1: one-hot and HUD pass use the onboard lock, not sim plane-cross")
    print(
        f"[INFO] PLAY_START_OFFICIAL={os.environ.get('PLAY_START_OFFICIAL', '1')} "
        "(PDF gate number, not Isaac index)"
    )
    speed_cap_mps = cap_from_env(args_cli.speed_cap)
    if speed_cap_mps > 0.0:
        print(f"[INFO] PLAY_SPEED_CAP_MPS={speed_cap_mps:g}: post-policy brake, weights unchanged")
    else:
        print("[INFO] PLAY_SPEED_CAP_MPS off")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # Viewer controls. Kit swallows Python stdout, so anything printed here is
    # invisible; mirror it to a file that can actually be read back.
    PERSP_CAM = "/OmniverseKit_Persp"
    _events_path = os.path.join(tempfile.gettempdir(), "play_events.txt")

    def _note(msg: str) -> None:
        print(msg, flush=True)
        try:
            with open(_events_path, "a") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass

    def _find_fpv_camera() -> str | None:
        """Locate the drone's camera prim by walking the stage.

        The configured prim path is templated per-env and the sensor may rename
        or nest the prim it spawns, so search for the camera under env_0's robot
        rather than trusting a hard-coded path.
        """
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return None
            found = [
                p.GetPath().pathString
                for p in stage.Traverse()
                if p.IsA(UsdGeom.Camera) and p.GetPath().pathString.startswith("/World/envs/env_0")
            ]
            _note(f"[INFO] camera prims under env_0: {found or 'none'}")
            for path in found:
                if "Robot" in path:
                    return path
            return found[0] if found else None
        except Exception as exc:  # noqa: BLE001 - viewer nicety, never abort a run
            _note(f"[WARN] camera discovery failed: {exc}")
            return None

    def _set_camera(path: str) -> bool:
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                _note("[WARN] no active viewport")
                return False
            viewport.camera_path = path
            return True
        except Exception as exc:  # noqa: BLE001 - viewer nicety, never abort a run
            _note(f"[WARN] could not switch viewport to {path}: {exc}")
            return False

    # 'R' restarts the run, 'V' toggles between the drone camera and the free
    # perspective view. Episodes otherwise only end on a crash, a flyaway or the
    # time limit, so there is no way to replay a start worth watching.
    # 'T' cycles real-time / half / quarter. The lap clock is always sim time;
    # a slow GPU cannot invent real-time, it can only stop adding extra delay.
    PACE_STEPS = (1.0, 0.5, 0.25)
    if args_cli.time_scale is not None:
        start_scale = max(0.05, float(args_cli.time_scale))
    elif args_cli.real_time:
        start_scale = 1.0
    else:
        start_scale = 1.0
    pace = {"scale": start_scale, "step_s": None, "overlay_i": 0}
    restart_requested = [False]
    toggle_requested = [False]
    pace_requested = [False]

    def _pace_label(scale: float) -> str:
        if abs(scale - 1.0) < 1e-6:
            return "real-time"
        if abs(scale - 0.5) < 1e-6:
            return "1/2 speed"
        if abs(scale - 0.25) < 1e-6:
            return "1/4 speed"
        return f"{scale:.2f}x"

    def _apply_kit_pace() -> None:
        # Kit will dilate physics when the viewport drops below minFrameRate.
        # Turn that off so "slow" is only our own sleep, not a hidden extra lag.
        try:
            import carb

            settings = carb.settings.get_settings()
            settings.set("/persistent/simulation/minFrameRate", 1)
            settings.set("/app/runLoops/main/rateLimitEnabled", False)
            settings.set("/app/vsync", False)
            settings.set("/rtx/ecoMode/enabled", True)
        except Exception as exc:  # noqa: BLE001 - viewer nicety
            _note(f"[WARN] could not set Kit frame pacing: {exc}")

    def _shrink_viewport() -> None:
        # Viewport + tiled camera both ray-trace. Keep the free viewport
        # at 960×540; the FPV camera itself is 1920×1080 at 30 Hz.
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.resolution = (960, 540)
                _note("[INFO] viewport resolution -> 960x540 for real-time pace")
        except Exception as exc:  # noqa: BLE001 - viewer nicety
            _note(f"[WARN] could not lower viewport resolution: {exc}")

    def _cycle_pace() -> None:
        nearest = min(range(len(PACE_STEPS)), key=lambda i: abs(PACE_STEPS[i] - pace["scale"]))
        pace["scale"] = PACE_STEPS[(nearest + 1) % len(PACE_STEPS)]
        _note(f"[INFO] pace -> {_pace_label(pace['scale'])} (T to cycle)")

    _apply_kit_pace()
    try:
        import carb.input
        import omni.appwindow

        def _on_keyboard_event(event, *args):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if event.input.name == "R":
                    restart_requested[0] = True
                elif event.input.name == "V":
                    toggle_requested[0] = True
                elif event.input.name == "T":
                    pace_requested[0] = True
            return True

        _input_iface = carb.input.acquire_input_interface()
        _keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        _kb_sub = _input_iface.subscribe_to_keyboard_events(_keyboard, _on_keyboard_event)
        _note("[INFO] keys: R = restart, V = camera, T = real-time / 1/2 / 1/4 pace")
    except Exception as exc:  # noqa: BLE001 - viewer nicety, never worth aborting a run
        _note(f"[WARN] hotkeys unavailable: {exc}")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    num_episode = 0

    # Sit in the drone's own camera by default. The configured viewer sits ~190 m
    # back to frame the whole 269 m course, where a 280 mm drone is roughly one
    # pixel -- so the run looks empty and a restart looks like nothing happened.
    # Speedometer. The FPV view gives no sense of scale, so a run either looks
    # fast or it does not, with nothing to check that against.
    hud: dict = {}
    try:
        import omni.ui as ui

        hud["window"] = ui.Window("Telemetry", width=280, height=350)
        with hud["window"].frame:
            with ui.VStack(spacing=2, style={"margin": 6}):
                hud["speed"] = ui.Label("--", style={"font_size": 42, "color": 0xFF44DD44})
                hud["kmh"] = ui.Label("", style={"font_size": 30, "color": 0xFF44DD44})
                hud["cap"] = ui.Label("", style={"font_size": 16, "color": 0xFFFFAA66})
                hud["gate"] = ui.Label("", style={"font_size": 16, "color": 0xFFFFFFFF})
                hud["lap"] = ui.Label("", style={"font_size": 22, "color": 0xFFFFCC44})
                hud["last"] = ui.Label("", style={"font_size": 16, "color": 0xFFDDDDDD})
                hud["best"] = ui.Label("", style={"font_size": 16, "color": 0xFF88DDFF})
                hud["pace"] = ui.Label("", style={"font_size": 16, "color": 0xFFFFAA66})
                hud["alt"] = ui.Label("", style={"font_size": 16, "color": 0xFFAAAAAA})
                hud["kps"] = ui.Label("", style={"font_size": 16, "color": 0xFF88DDFF})
        _note("[INFO] telemetry HUD created")
    except Exception as exc:  # noqa: BLE001 - viewer nicety, never abort a run
        _note(f"[WARN] telemetry HUD unavailable: {exc}")

    kp_overlay: dict = {}
    if not args_cli.no_kp_overlay:
        try:
            import numpy as np
            import omni.ui as ui

            kp_overlay["provider"] = ui.ByteImageProvider()
            _prev_w, _prev_h = 960, 540
            kp_overlay["window"] = ui.Window(
                "Derived keypoints", width=_prev_w + 16, height=_prev_h + 40
            )
            with kp_overlay["window"].frame:
                with ui.VStack():
                    kp_overlay["image"] = ui.ImageWithProvider(
                        kp_overlay["provider"],
                        width=_prev_w,
                        height=_prev_h,
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                    )
            _note("[INFO] derived-keypoint overlay window created")
        except Exception as exc:  # noqa: BLE001 - viewer nicety
            _note(f"[WARN] keypoint overlay UI unavailable: {exc}")
            kp_overlay.clear()

    _robot = env.unwrapped.scene["robot"]
    _target = env.unwrapped.command_manager.get_term("target")
    lap_clock = {"t0": 0.0, "n": 1, "wrapped": False, "last": None, "best": None}

    def _pdf_gate(idx: int) -> int:
        pos = _target.track.data.object_com_pos_w[0, int(idx)]
        return official_number_from_xy(float(pos[0]), float(pos[1]))

    def _bind_pdf_start(official: int = 1) -> int:
        """Teleport onto the organizer PDF gate. env.reset() always returns to G6."""
        import isaaclab.utils.math as math_utils

        n = int(_target.num_gates)
        xy = _target.track.data.object_com_pos_w[0, :n]
        x_ft, y_ft = _PDF_FT[int(official)]
        tx, ty = x_ft * 0.3048, (165.0 - y_ft) * 0.3048
        d2 = (xy[:, 0] - tx) ** 2 + (xy[:, 1] - ty) ** 2
        idx = int(torch.argmin(d2).item())
        pos = _target.track.data.object_com_pos_w[0, idx]
        quat = _target.track.data.object_quat_w[0, idx]
        back = math_utils.quat_apply(
            quat.unsqueeze(0),
            torch.tensor([[-3.0, 0.0, 0.0]], device=pos.device, dtype=pos.dtype),
        )[0]
        spawn = pos + back
        aim = pos - spawn
        yaw = torch.atan2(aim[1], aim[0])
        zeros = torch.zeros(1, device=pos.device, dtype=pos.dtype)
        heading = math_utils.quat_from_euler_xyz(zeros, zeros, yaw.view(1))[0]
        env_ids = torch.tensor([0], device=pos.device, dtype=torch.int64)
        _robot.write_root_pose_to_sim(
            torch.cat([spawn, heading]).unsqueeze(0), env_ids=env_ids
        )
        _robot.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=pos.device, dtype=pos.dtype), env_ids=env_ids
        )
        _target.next_gate_idx[0] = idx
        _target.next_gate_w[0, :3] = pos
        _target.next_gate_w[0, 3:7] = quat
        _target.cfg.randomise_start = False
        _target.cfg.fixed_start_idx = idx
        if hasattr(_target.cfg, "official_start_gate"):
            _target.cfg.official_start_gate = int(official)
        if hasattr(_target, "_resolved_play_start"):
            _target._resolved_play_start = idx
        vc = getattr(env.unwrapped, "_visual_gate_counter", None)
        if vc is not None:
            vc.reset(start_idx=idx)
        _note(
            f"[INFO] teleport PDF G{official} Isaac idx {idx} "
            f"gate=({float(pos[0]):.2f}, {float(pos[1]):.2f}) "
            f"spawn=({float(spawn[0]):.2f}, {float(spawn[1]):.2f}) "
            f"label PDF G{_pdf_gate(idx)}"
        )
        return idx

    def _policy_vec(raw):
        if isinstance(raw, dict):
            raw = raw.get("policy", next(iter(raw.values())))
        if torch.is_tensor(raw):
            raw = raw[0] if raw.ndim > 1 else raw
            return raw.detach().cpu().numpy()
        arr = np.asarray(raw)
        return arr[0] if arr.ndim > 1 else arr

    def _fmt_time(seconds: float | None) -> str:
        if seconds is None:
            return "--"
        if seconds >= 60.0:
            minutes = int(seconds // 60.0)
            return f"{minutes}:{seconds % 60.0:05.2f}"
        return f"{seconds:.2f} s"

    def _episode_time_s() -> float:
        return float(env.unwrapped.episode_length_buf[0].item()) * float(env.unwrapped.step_dt)

    def _reset_lap_clock() -> None:
        lap_clock["t0"] = _episode_time_s()
        lap_clock["n"] = 1
        lap_clock["wrapped"] = False

    def _update_hud(n_vis: int | None = None, *, next_gate: int | None = None) -> None:
        if "speed" not in hud:
            return
        vel = _robot.data.root_lin_vel_w[0]
        speed = float(torch.linalg.vector_norm(vel))
        # Race order is 1-based. Use the target from *before* the step so a
        # crossing still reads as that gate instead of jumping to the next one.
        if next_gate is None:
            next_gate = int(_target.next_gate_idx[0])
        hud["speed"].text = f"{speed:.1f} m/s"
        hud["kmh"].text = f"{speed * 3.6:.1f} km/h"
        if "cap" in hud:
            hud["cap"].text = f"cap {speed_cap_mps:.0f} m/s" if speed_cap_mps > 0.0 else "cap off"
        sim_gate = int(_target.next_gate_idx[0])
        sim_pdf = _pdf_gate(sim_gate)
        vc = getattr(env.unwrapped, "_visual_gate_counter", None)
        if vc is not None:
            vis_gate = int(vc.index[0].item())
            vis_pdf = _pdf_gate(vis_gate)
            mark = "ok" if vis_gate == sim_gate else "DRIFT"
            hud["gate"].text = f"PDF G{vis_pdf}  sim G{sim_pdf}  {mark}"
        else:
            hud["gate"].text = f"PDF G{_pdf_gate(next_gate)}"
        current = _episode_time_s() - lap_clock["t0"]
        hud["lap"].text = f"lap {lap_clock['n']}  {_fmt_time(current)}"
        hud["last"].text = f"last  {_fmt_time(lap_clock['last'])}"
        hud["best"].text = f"best  {_fmt_time(lap_clock['best'])}"
        if "pace" in hud:
            hud["pace"].text = f"pace {_pace_label(pace['scale'])}  (T)"
        hud["alt"].text = f"alt {float(_robot.data.root_pos_w[0, 2]):.1f} m   vs {float(vel[2]):+.1f} m/s"
        if "kps" in hud and n_vis is not None:
            hud["kps"].text = f"keypoints {n_vis}/8"

    def _update_kp_overlay(raw_obs) -> int | None:
        if "provider" not in kp_overlay:
            return None
        try:
            import numpy as np

            frame = current_race_obs_frame(_policy_vec(raw_obs))
            n_vis = int((frame[16:24] > 0.5).sum())
            rgb = None
            if _cam is not None:
                rgb = _cam.data.output["rgb"][0]
                if torch.is_tensor(rgb):
                    rgb = rgb.detach().cpu().numpy()
            if rgb is None:
                rgb = np.zeros((int(FRAME_H), int(FRAME_W), 3), dtype=np.uint8)
            painted = annotate_rgb(rgb, frame)
            h, w = painted.shape[:2]
            buf = kp_overlay.get("rgba")
            if buf is None or buf.shape != (h, w, 4):
                buf = np.empty((h, w, 4), dtype=np.uint8)
                kp_overlay["rgba"] = buf
            buf[..., :3] = painted[..., :3]
            buf[..., 3] = 255
            # tobytes() is a memcpy. flatten().tolist() builds ~900k Python ints
            # and is why the overlay hitchs and the sim falls off real-time.
            try:
                kp_overlay["provider"].set_bytes_data(buf.tobytes(), [w, h])
            except (TypeError, RuntimeError):
                kp_overlay["provider"].set_bytes_data(memoryview(buf).cast("B").tolist(), [w, h])
            return n_vis
        except Exception as exc:  # noqa: BLE001 - overlay must not abort play
            _note(f"[WARN] keypoint overlay update failed: {exc}")
            kp_overlay.pop("provider", None)
            return None

    fpv_cam = _find_fpv_camera()
    on_fpv = [bool(fpv_cam) and _set_camera(fpv_cam)]
    if on_fpv[0]:
        _note(f"[INFO] viewport -> drone camera ({fpv_cam})")
    else:
        _note("[INFO] no drone camera on stage; relaunch with ENABLE_CAMERAS=1 for FPV")
    _shrink_viewport()

    yolo_rec: YoloPoseRecorder | None = None
    if args_cli.record_yolo:
        try:
            _ = env.unwrapped.scene["tiled_camera"]
        except KeyError as exc:
            raise RuntimeError(
                "--record_yolo needs the tiled camera; launch with ENABLE_CAMERAS=1 "
                "(play.py sets this automatically when --record_yolo is passed)."
            ) from exc
        yolo_rec = YoloPoseRecorder(
            args_cli.record_yolo_dir,
            every_n=args_cli.record_yolo_every,
            max_frames=args_cli.record_yolo,
        )
        _note(
            f"[INFO] YOLO-pose recorder -> {yolo_rec.root} "
            f"(cap {args_cli.record_yolo}, every {args_cli.record_yolo_every} steps)"
        )

    _track = env.unwrapped.scene["track"]
    try:
        _cam = env.unwrapped.scene["tiled_camera"]
    except (KeyError, ValueError):
        _cam = None
        if not args_cli.no_kp_overlay:
            _note("[WARN] no tiled camera; overlay will be blank without ENABLE_CAMERAS=1")

    n_gates = int(_target.num_gates)
    for i in range(n_gates):
        p = _target.track.data.object_com_pos_w[0, i]
        _note(
            f"[INFO] track[{i}] xy=({float(p[0]):.2f}, {float(p[1]):.2f}, {float(p[2]):.2f}) "
            f"PDF G{_pdf_gate(i)}"
        )
    _start_idx = _bind_pdf_start(int(os.environ.get("PLAY_START_OFFICIAL", "1")))
    _reset_lap_clock()
    _update_hud(next_gate=_start_idx)

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            if speed_cap_mps > 0.0:
                vel_w = _robot.data.root_lin_vel_w
                quat_w = _robot.data.root_quat_w
                if isinstance(actions, dict):
                    actions = {
                        name: limit_policy_action(act, vel_w, quat_w, cap_mps=speed_cap_mps)
                        for name, act in actions.items()
                    }
                else:
                    actions = limit_policy_action(actions, vel_w, quat_w, cap_mps=speed_cap_mps)
            # Snapshot before step: pass detection advances the index in-step.
            vc = getattr(env.unwrapped, "_visual_gate_counter", None)
            if vc is not None:
                next_gate = int(vc.index[0].item())
                last_gate = int(vc.n_course) - 1
            else:
                next_gate = int(_target.next_gate_idx[0])
                last_gate = int(_target.num_gates) - 1

            # env stepping
            obs, rew, terminated, truncated, info = env.step(actions)

            if vc is not None:
                passed = bool(vc.passed[0].item())
            else:
                passed = bool(_target.gate_passed[0].item())
            if passed and next_gate == last_gate:
                lap_clock["wrapped"] = True
            if passed and next_gate == 0 and lap_clock["wrapped"]:
                lap_s = _episode_time_s() - lap_clock["t0"]
                lap_clock["last"] = lap_s
                if lap_clock["best"] is None or lap_s < lap_clock["best"]:
                    lap_clock["best"] = lap_s
                lap_clock["n"] += 1
                lap_clock["t0"] = _episode_time_s()
                lap_clock["wrapped"] = False
                _note(f"[INFO] lap {lap_clock['n'] - 1}  {_fmt_time(lap_s)}")

            done = terminated if not torch.is_tensor(terminated) else bool(terminated.reshape(-1)[0].item())
            timed_out = truncated if not torch.is_tensor(truncated) else bool(truncated.reshape(-1)[0].item())
            if done or timed_out:
                _bind_pdf_start(int(os.environ.get("PLAY_START_OFFICIAL", "1")))
                _reset_lap_clock()

            n_vis = _update_kp_overlay(obs)
            _update_hud(n_vis, next_gate=next_gate)

            if yolo_rec is not None and _cam is not None:
                rgb = _cam.data.output["rgb"][0]
                n_gates = _track.num_objects
                gate_pos = _track.data.object_com_pos_w[0, :n_gates]
                gate_quat = _track.data.object_quat_w[0, :n_gates]
                if yolo_rec.maybe_record(
                    rgb=rgb,
                    drone_pos_w=_robot.data.root_pos_w[0],
                    drone_quat_w=_robot.data.root_quat_w[0],
                    gate_pos_w=gate_pos,
                    gate_quat_w=gate_quat,
                ):
                    out_dir = yolo_rec.finalize()
                    _note(f"[INFO] YOLO dataset complete: {out_dir} ({yolo_rec.saved} frames)")
                    break
                if yolo_rec.saved and yolo_rec.saved % 200 == 0 and (yolo_rec._step % yolo_rec.every_n) == 0:
                    _note(f"[INFO] YOLO frames saved: {yolo_rec.saved}/{yolo_rec.max_frames}")

            if restart_requested[0]:
                restart_requested[0] = False
                obs, _ = env.reset()
                _bind_pdf_start(int(os.environ.get("PLAY_START_OFFICIAL", "1")))
                _reset_lap_clock()
                _update_hud(next_gate=int(_target.next_gate_idx[0]))
                _note("[INFO] run restarted")

            if toggle_requested[0] and fpv_cam:
                toggle_requested[0] = False
                on_fpv[0] = not on_fpv[0]
                target = fpv_cam if on_fpv[0] else PERSP_CAM
                if not _set_camera(target):
                    on_fpv[0] = not on_fpv[0]
                else:
                    _note(f"[INFO] viewport -> {'drone camera' if on_fpv[0] else 'free view'}")

            if pace_requested[0]:
                pace_requested[0] = False
                _cycle_pace()
                _update_hud(n_vis, next_gate=next_gate)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # Frames already cost more than dt, so sleeping to dt/scale is a no-op
        # and T looks broken. Slow modes wait relative to the *measured* step
        # so 1/2 and 1/4 are always slower than whatever the GPU can do.
        work = time.time() - start_time
        if pace["step_s"] is None:
            pace["step_s"] = work
        else:
            pace["step_s"] = 0.85 * pace["step_s"] + 0.15 * work
        if pace["scale"] < 1.0:
            remain = pace["step_s"] / pace["scale"] - (time.time() - start_time)
            if remain > 0.0:
                time.sleep(remain)

        if args_cli.log:
            if truncated or terminated:
                num_episode += 1
                logger.save()
                if num_episode >= args_cli.log:
                    break
            logger.log(info["metrics"])

    if yolo_rec is not None and not yolo_rec.done and yolo_rec.saved > 0:
        out_dir = yolo_rec.finalize()
        _note(f"[INFO] YOLO dataset finalized early: {out_dir} ({yolo_rec.saved} frames)")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
