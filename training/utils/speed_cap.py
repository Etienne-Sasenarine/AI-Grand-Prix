# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""Post-policy speed limiter. Does not change weights.

The PPO action is thrust + NED rates, not velocity. This wraps the decoded
stick after the policy: over the cap it kills nose-down and extra thrust;
well over it adds a nose-up brake. Same idea as ``ASSIST_SPEED_CAP_MPS``.

Play: ``PLAY_SPEED_CAP_MPS`` or ``play.py --speed-cap``. Unset / 0 = off.
"""

from __future__ import annotations

import os

import torch

from utils.aigp_obs import quat_rotate_inverse

# Start easing before the hard number so the policy is not slammed at the line.
SOFT_START = 0.85
# Past this multiple, add an explicit nose-up / cut-thrust brake.
HARD_RATIO = 1.25
# NED pitch stick added when well over (positive = nose up = slow down).
BRAKE_PITCH = 0.35


def cap_from_env(override: float | None = None) -> float:
    """CLI override, else ``PLAY_SPEED_CAP_MPS``, else 0 (off)."""
    if override is not None:
        return max(0.0, float(override))
    raw = os.environ.get("PLAY_SPEED_CAP_MPS", "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def body_forward_speed(
    vel_w: torch.Tensor,
    quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Isaac FLU body-x speed (m/s). Positive is along the nose."""
    v_flu = quat_rotate_inverse(quat_wxyz, vel_w)
    return v_flu[..., 0]


def limit_policy_action(
    action: torch.Tensor,
    vel_w: torch.Tensor,
    quat_wxyz: torch.Tensor,
    *,
    cap_mps: float,
    hard_ratio: float = HARD_RATIO,
    brake_pitch: float = BRAKE_PITCH,
) -> torch.Tensor:
    """Return a copy of ``action`` with speed-over-cap braking applied.

    Action layout matches ``decode_aigp_action``: ``[thrust, roll, pitch, yaw]``
    in ``[-1, 1]``. NED pitch: negative is nose-down / add speed.
    """
    if cap_mps <= 0.0:
        return action
    a = action.clone()
    if a.ndim == 1:
        a = a.unsqueeze(0)
        vel_w = vel_w.unsqueeze(0) if vel_w.ndim == 1 else vel_w
        quat_wxyz = quat_wxyz.unsqueeze(0) if quat_wxyz.ndim == 1 else quat_wxyz
        squeeze = True
    else:
        squeeze = False

    speed = torch.linalg.vector_norm(vel_w, dim=-1)
    v_fwd = body_forward_speed(vel_w, quat_wxyz)
    # Brake on total energy, but only scrub the forward channel (assist-style).
    # A sideways slide should not get a nose-up punch.
    measure = torch.maximum(speed, v_fwd.clamp(min=0.0))

    soft = float(cap_mps) * SOFT_START
    hard = float(cap_mps) * float(hard_ratio)
    fade = ((measure - soft) / max(float(cap_mps) - soft, 1e-6)).clamp(0.0, 1.0)
    hard_fade = ((measure - float(cap_mps)) / max(hard - float(cap_mps), 1e-6)).clamp(0.0, 1.0)

    thrust = a[..., 0]
    pitch = a[..., 2]
    # Fade out extra collective and any nose-down.
    extra_thrust = thrust.clamp(min=0.0)
    nose_down = pitch.clamp(max=0.0)
    thrust = thrust - extra_thrust * fade
    pitch = pitch - nose_down * fade
    # Well over: nose up, and pull remaining stick toward hover.
    pitch = pitch + float(brake_pitch) * hard_fade
    thrust = thrust * (1.0 - 0.5 * hard_fade)

    a[..., 0] = thrust.clamp(-1.0, 1.0)
    a[..., 2] = pitch.clamp(-1.0, 1.0)
    return a.squeeze(0) if squeeze else a
