"""Kaufmann-style velocity-to-gate alignment (no Isaac Sim import)."""

from __future__ import annotations

import math

import torch

from utils.aigp_obs import CAMERA_TILT_UP_DEG, quat_rotate


def vel_align_from_vel(
    vel_w: torch.Tensor,
    n_hat: torch.Tensor,
    min_speed: float = 0.25,
) -> torch.Tensor:
    """Dense term: ``max(0, v̂ · n̂)``. Zero when nearly stopped."""
    speed = torch.linalg.vector_norm(vel_w, dim=-1)
    vhat = vel_w / speed.clamp_min(1e-6).unsqueeze(-1)
    align = (vhat * n_hat).sum(dim=-1).clamp(min=0.0)
    return torch.where(speed >= min_speed, align, torch.zeros_like(align))


def cam_horizontal_align(
    drone_quat_w: torch.Tensor,
    n_hat: torch.Tensor,
    tilt_deg: float = CAMERA_TILT_UP_DEG,
) -> torch.Tensor:
    """``max(0, ĉam_xy · n̂)``: FPV optical axis, flattened to world XY.

    The 20° camera tilt is removed by the XY projection so a level drone
    looking straight through the hole scores 1.0. Same ``n̂ = [cos θ, sin θ, 0]``
    as the velocity term.
    """
    t = math.radians(tilt_deg)
    optical_b = torch.tensor(
        [math.cos(t), 0.0, math.sin(t)],
        device=drone_quat_w.device,
        dtype=drone_quat_w.dtype,
    ).expand(drone_quat_w.shape[0], 3)
    cam_w = quat_rotate(drone_quat_w, optical_b)
    cam_h = cam_w[:, :2]
    cam_h = cam_h / cam_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    n_xy = n_hat[..., :2]
    n_xy = n_xy / n_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (cam_h * n_xy).sum(dim=-1).clamp(min=0.0)


def cam_cone_score(align: torch.Tensor, psi_deg: float = 40.0) -> torch.Tensor:
    """1 on-axis, 0 at ``ψ`` (and outside). Linear in cosine."""
    cos_psi = math.cos(math.radians(psi_deg))
    return ((align - cos_psi) / (1.0 - cos_psi)).clamp(0.0, 1.0)


def approach_aim_pos(
    curr_pos: torch.Tensor,
    gate_pos: torch.Tensor,
    n_hat: torch.Tensor,
    run_in_m: float = 3.0,
) -> torch.Tensor:
    """Gate centre on the approach side; 3 m run-in when already in front.

    After the last southbound gate the drone is east of gate 1. Aiming at the
    opening then pulls into the face, and a pass only counts west→east, so the
    policy circuits the hall. The run-in is the short hook.
    """
    waypoint = gate_pos - run_in_m * n_hat
    on_exit = ((curr_pos - gate_pos) * n_hat).sum(dim=-1) > 0.0
    return torch.where(on_exit.unsqueeze(-1), waypoint, gate_pos)


def progress_to_aim(
    prev_pos: torch.Tensor,
    curr_pos: torch.Tensor,
    gate_pos: torch.Tensor,
    n_hat: torch.Tensor,
    run_in_m: float = 3.0,
) -> torch.Tensor:
    """Metres closed toward ``approach_aim_pos`` (same target for both ends)."""
    aim = approach_aim_pos(curr_pos, gate_pos, n_hat, run_in_m=run_in_m)
    return torch.linalg.norm(prev_pos - aim, dim=-1) - torch.linalg.norm(curr_pos - aim, dim=-1)


def run_in_progress_from_pos(
    prev_pos: torch.Tensor,
    curr_pos: torch.Tensor,
    gate_pos: torch.Tensor,
    n_hat: torch.Tensor,
    run_in_m: float = 3.0,
    only_exit_side: bool = True,
) -> torch.Tensor:
    """Metres closed toward the approach waypoint ``gate − run_in · n̂``.

    After the last southbound gate the drone is on the *exit* side of gate 1
    (east of a +X opening). Euclidean progress to the centre then pulls into
    the face; a pass only counts west→east, so the cheap policy is to circuit
    the hall and set up from behind. This pays for the short hook to the
    3 m run-in instead. Silent on the approach side, where ordinary progress
    already points through the hole.
    """
    waypoint = gate_pos - run_in_m * n_hat
    progress = torch.linalg.norm(prev_pos - waypoint, dim=-1) - torch.linalg.norm(
        curr_pos - waypoint, dim=-1
    )
    if not only_exit_side:
        return progress
    signed = ((curr_pos - gate_pos) * n_hat).sum(dim=-1)
    return torch.where(signed > 0.0, progress, torch.zeros_like(progress))


def center_passage_score(
    pos_w: torch.Tensor,
    gate_pos_w: torch.Tensor,
    n_hat: torch.Tensor,
    half_size: float = 0.75,
) -> torch.Tensor:
    """Linear closeness: ``1 - r_⊥ / R``. Not a threshold.

    0.00 m off → 1.0, 0.375 m → 0.5, 0.75 m (rim) → 0. Offset is in the
    gate plane (right × up), not along ``n̂``. ``half_size`` is half the
    inner opening (1.5 m / 2).
    """
    delta = pos_w - gate_pos_w
    n_xy = n_hat[..., :2]
    n_xy = n_xy / n_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # right = through × world_up = [n_y, -n_x, 0]
    right = torch.stack([n_xy[:, 1], -n_xy[:, 0]], dim=-1)
    r_right = (delta[:, :2] * right).sum(dim=-1)
    r_up = delta[:, 2]
    r_perp = torch.sqrt(r_right * r_right + r_up * r_up)
    return (1.0 - r_perp / half_size).clamp(min=0.0)
