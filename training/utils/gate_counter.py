"""Onboard pass counter for Isaac: Python (N=1) plus a GPU-batched twin.

The Python class in the repo-root ``gate_counter.py`` is what the Jetson flies.
Training at 4096 envs cannot loop that in Python every 60 Hz step, so
``TorchVisualGateCounter`` is the same state machine on tensors.
"""
from __future__ import annotations

import torch

from utils.flight_repo import add_to_sys_path

if add_to_sys_path("gate_counter.py") is None:
    raise ImportError(
        "the flight repo's gate_counter.py was not found; set AIGP_FLIGHT_REPO "
        "to the ai-grand-prix checkout (see utils/flight_repo.py)"
    )

from gate_counter import (  # noqa: E402
    CLOSE_SPAN,
    COOLDOWN_S,
    DEFAULT_DT,
    JUMP,
    LOOP,
    MIN_ALIGN,
    MIN_VIS_LOCK,
    MIN_VIS_LOST,
    N_COURSE,
    SPAN_KEEP,
    STABLE_S,
    START_IDX,
    THROUGH_SPAN,
    WALK_OFF,
    BatchedVisualGateCounter,
    VisualGateCounter,
    lock_metrics,
)


def lock_metrics_torch(uv: torch.Tensor, visible: torch.Tensor) -> dict[str, torch.Tensor]:
    """Batched ``lock_metrics``. ``uv`` is (N, 8, 2), ``visible`` is (N, 8)."""
    vis = visible.bool()
    n_vis = vis.sum(dim=-1)
    inner = vis[:, 4:8]
    n_inner = inner.sum(dim=-1)
    use_inner = n_inner >= 2
    inner_mask = torch.cat([torch.zeros_like(vis[:, :4]), inner], dim=-1)
    mask = torch.where(use_inner.unsqueeze(-1), inner_mask, vis)
    enough = mask.sum(dim=-1) >= 2
    u = uv[..., 0]
    v = uv[..., 1]
    neg = torch.finfo(u.dtype).min
    pos = torch.finfo(u.dtype).max
    u_lo = u.masked_fill(~mask, pos).amin(dim=-1)
    u_hi = u.masked_fill(~mask, neg).amax(dim=-1)
    v_lo = v.masked_fill(~mask, pos).amin(dim=-1)
    v_hi = v.masked_fill(~mask, neg).amax(dim=-1)
    span = torch.hypot(u_hi - u_lo, v_hi - v_lo)
    count = mask.sum(dim=-1).clamp_min(1).to(u.dtype)
    cx = (u * mask).sum(dim=-1) / count
    cy = (v * mask).sum(dim=-1) / count
    nan = torch.full_like(cx, float("nan"))
    cx = torch.where(enough, cx, nan)
    cy = torch.where(enough, cy, nan)
    span = torch.where(enough, span, torch.zeros_like(span))
    cx_f = torch.nan_to_num(cx, nan=0.5)
    cy_f = torch.nan_to_num(cy, nan=0.5)
    offset = torch.hypot(cx_f - 0.5, cy_f - 0.5)
    align = (1.0 - offset / (0.5**0.5)).clamp(min=0.0)
    align = torch.where(enough, align, torch.zeros_like(align))
    return {
        "n_vis": n_vis,
        "n_inner": n_inner,
        "span": span,
        "cx": cx,
        "cy": cy,
        "align": align,
        "offset": offset,
    }


class TorchVisualGateCounter:
    """GPU-batched copy of ``VisualGateCounter`` for Isaac training."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        start_idx: int | torch.Tensor = START_IDX,
        n_course: int = N_COURSE,
        loop: bool = LOOP,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_course = max(1, int(n_course))
        self.loop = bool(loop)
        self.index = self._as_start(start_idx)
        self.passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._was_close = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._saw_far = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._peak = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._last_span = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._last_cx = torch.full((self.num_envs,), 0.5, device=self.device, dtype=dtype)
        self._last_cy = torch.full((self.num_envs,), 0.5, device=self.device, dtype=dtype)
        self._last_n_vis = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._last_n_inner = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._last_align = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._cooldown = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._stable = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._have_lock = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _as_start(self, start_idx: int | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(start_idx):
            idx = start_idx.to(device=self.device, dtype=torch.long).reshape(-1)
            if idx.numel() == 1:
                idx = idx.expand(self.num_envs)
            elif idx.numel() != self.num_envs:
                raise ValueError("start_idx length must match num_envs")
        else:
            idx = torch.full(
                (self.num_envs,), int(start_idx), device=self.device, dtype=torch.long
            )
        return idx % self.n_course

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        start_idx: int | torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if start_idx is None:
            start = self.index[ids]
        elif torch.is_tensor(start_idx):
            start = start_idx.to(device=self.device, dtype=torch.long).reshape(-1)
            if start.numel() == 1:
                start = start.expand(ids.numel())
            elif start.numel() != ids.numel():
                raise ValueError("start_idx length must match env_ids")
        else:
            start = torch.full((ids.numel(),), int(start_idx), device=self.device, dtype=torch.long)
        start = start % self.n_course
        self.index[ids] = start
        self.passed[ids] = False
        self._was_close[ids] = False
        self._saw_far[ids] = False
        self._peak[ids] = 0
        self._last_span[ids] = 0
        self._last_cx[ids] = 0.5
        self._last_cy[ids] = 0.5
        self._last_n_vis[ids] = 0
        self._last_n_inner[ids] = 0
        self._last_align[ids] = 0
        self._cooldown[ids] = 0
        self._stable[ids] = 0
        self._have_lock[ids] = False

    def _advance(self, hit: torch.Tensor) -> None:
        nxt = self.index + 1
        if self.loop:
            nxt = nxt % self.n_course
        else:
            nxt = nxt.clamp(max=self.n_course - 1)
        self.index = torch.where(hit, nxt, self.index)
        self._was_close = self._was_close & ~hit
        self._saw_far = self._saw_far & ~hit
        self._peak = torch.where(hit, torch.zeros_like(self._peak), self._peak)
        self._last_span = torch.where(hit, torch.zeros_like(self._last_span), self._last_span)
        self._last_n_vis = torch.where(hit, torch.zeros_like(self._last_n_vis), self._last_n_vis)
        self._last_n_inner = torch.where(hit, torch.zeros_like(self._last_n_inner), self._last_n_inner)
        self._last_align = torch.where(hit, torch.zeros_like(self._last_align), self._last_align)
        self._cooldown = torch.where(
            hit, torch.full_like(self._cooldown, COOLDOWN_S), self._cooldown
        )
        self._stable = torch.where(hit, torch.zeros_like(self._stable), self._stable)
        self._have_lock = self._have_lock & ~hit

    def step(
        self,
        uv: torch.Tensor,
        visible: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Update from current-lock corners. Returns a (N,) bool pass mask.

        Call after packing the observation so this frame's one-hot matches
        the corners. ``index`` then advances for the next frame.
        """
        dt = DEFAULT_DT if dt is None else max(0.0, float(dt))
        m = lock_metrics_torch(uv, visible)
        n_vis = m["n_vis"]
        n_inner = m["n_inner"]
        span = m["span"]
        align = m["align"]
        cx, cy = m["cx"], m["cy"]
        offset = m["offset"]
        finite_c = torch.isfinite(cx)
        self.passed.zero_()

        cooling = self._cooldown > 0
        self._cooldown = (self._cooldown - dt).clamp(min=0.0)
        locked = n_vis >= MIN_VIS_LOCK
        update_last = cooling & locked
        self._last_n_vis = torch.where(update_last, n_vis, self._last_n_vis)
        self._last_n_inner = torch.where(update_last, n_inner, self._last_n_inner)
        self._last_span = torch.where(update_last, span, self._last_span)
        self._last_cx = torch.where(update_last & finite_c, cx, self._last_cx)
        self._last_cy = torch.where(update_last & finite_c, cy, self._last_cy)
        self._last_align = torch.where(update_last, align, self._last_align)
        active = ~cooling

        jump = (
            active
            & locked
            & (self._last_n_vis >= MIN_VIS_LOCK)
            & finite_c
            & torch.isfinite(self._last_cx)
            & (torch.hypot(cx - self._last_cx, cy - self._last_cy) > JUMP)
        )
        self._was_close = torch.where(jump, torch.zeros_like(self._was_close), self._was_close)
        self._saw_far = torch.where(jump, span < CLOSE_SPAN, self._saw_far)
        self._peak = torch.where(jump, span, self._peak)
        self._stable = torch.where(jump, torch.zeros_like(self._stable), self._stable)
        self._have_lock = self._have_lock | jump
        self._last_n_vis = torch.where(jump, n_vis, self._last_n_vis)
        self._last_n_inner = torch.where(jump, n_inner, self._last_n_inner)
        self._last_span = torch.where(jump, span, self._last_span)
        self._last_cx = torch.where(jump & finite_c, cx, self._last_cx)
        self._last_cy = torch.where(jump & finite_c, cy, self._last_cy)
        self._last_align = torch.where(jump, align, self._last_align)

        grow = active & ~jump & locked
        self._saw_far = torch.where(grow & (span < CLOSE_SPAN), torch.ones_like(self._saw_far), self._saw_far)
        self._peak = torch.where(grow, torch.maximum(self._peak, span), self._peak)
        self._have_lock = self._have_lock | grow
        cx_d = torch.where(finite_c, cx, self._last_cx)
        cy_d = torch.where(torch.isfinite(cy), cy, self._last_cy)
        drifted = torch.hypot(cx_d - self._last_cx, cy_d - self._last_cy)
        hold = grow & (drifted < 0.08) & (align >= MIN_ALIGN) & (span >= CLOSE_SPAN * 0.8)
        self._stable = torch.where(grow, torch.where(hold, self._stable + dt, torch.zeros_like(self._stable)), self._stable)
        approached = self._saw_far & (span >= CLOSE_SPAN) & (align >= MIN_ALIGN)
        held = (self._stable >= STABLE_S) & (span >= CLOSE_SPAN) & (align >= MIN_ALIGN)
        through = (span >= THROUGH_SPAN) & (align >= MIN_ALIGN * 0.7)
        arm = grow & (approached | held | through)
        self._was_close = self._was_close | arm
        abort_side = grow & self._was_close & (offset >= WALK_OFF)
        self._was_close = self._was_close & ~abort_side
        self._peak = torch.where(abort_side, span, self._peak)
        self._stable = torch.where(abort_side, torch.zeros_like(self._stable), self._stable)
        self._last_n_vis = torch.where(grow, n_vis, self._last_n_vis)
        self._last_n_inner = torch.where(grow, n_inner, self._last_n_inner)
        self._last_span = torch.where(grow, span, self._last_span)
        self._last_cx = torch.where(grow & finite_c, cx, self._last_cx)
        self._last_cy = torch.where(grow & finite_c, cy, self._last_cy)
        self._last_align = torch.where(grow, align, self._last_align)

        collapsed = active & ~jump & ~locked
        lost = (n_vis <= MIN_VIS_LOST) | ((n_inner == 0) & (self._last_n_inner >= 2))
        candidate = collapsed & lost & self._was_close & self._have_lock
        walked = (self._last_align < MIN_ALIGN) | ((self._last_cx - 0.5).abs() > WALK_OFF)
        shrinking = self._last_span < SPAN_KEEP * self._peak.clamp_min(1e-6)
        still_large = (self._last_span >= CLOSE_SPAN * 0.85) | (self._last_span >= THROUGH_SPAN)
        abort = candidate & (walked | shrinking | ~still_large)
        hit = candidate & ~abort
        miss = collapsed & ~hit
        blank = miss & (n_vis == 0)
        self._have_lock = self._have_lock & ~blank
        self._stable = torch.where(blank, torch.zeros_like(self._stable), self._stable)
        self._was_close = self._was_close & ~abort
        self._have_lock = self._have_lock & ~abort
        self._peak = torch.where(abort, torch.zeros_like(self._peak), self._peak)
        self._stable = torch.where(abort, torch.zeros_like(self._stable), self._stable)
        touch = miss | abort
        self._last_n_vis = torch.where(touch, n_vis, self._last_n_vis)
        self._last_n_inner = torch.where(touch, n_inner, self._last_n_inner)

        self.passed = hit
        if hit.any():
            self._advance(hit)
        return self.passed
