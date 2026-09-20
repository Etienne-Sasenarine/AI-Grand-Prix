"""Torch batched pass counter matches the Python / Jetson state machine."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gate_counter import VisualGateCounter
from utils.gate_counter import TorchVisualGateCounter, lock_metrics_torch


def _square(cx: float, cy: float, half: float, inner_scale: float = 0.55):
    o = half
    i = half * inner_scale
    outer = [(cx - o, cy - o), (cx + o, cy - o), (cx + o, cy + o), (cx - o, cy + o)]
    inner = [(cx - i, cy - i), (cx + i, cy - i), (cx + i, cy + i), (cx - i, cy + i)]
    return outer + inner


def _blank():
    return [(math.nan, math.nan)] * 8, [False] * 8


def _to_torch(uv, vis, n=1):
    t_uv = torch.tensor([[p for p in uv]], dtype=torch.float32)
    t_vis = torch.tensor([list(vis)], dtype=torch.bool)
    if n > 1:
        t_uv = t_uv.repeat(n, 1, 1)
        t_vis = t_vis.repeat(n, 1)
    return t_uv, t_vis


def test_torch_matches_python_on_through():
    py = VisualGateCounter(start_idx=6, n_course=11)
    th = TorchVisualGateCounter(1, "cpu", start_idx=6, n_course=11)
    for i in range(14):
        t = i / 13
        uv = _square(0.5, 0.5, 0.06 + 0.16 * t)
        vis = [True] * 8
        py.step(uv, vis, dt=1.0 / 30.0)
        th.step(*_to_torch(uv, vis), dt=1.0 / 30.0)
        assert int(th.index[0]) == py.index
    blank_uv, blank_vis = _blank()
    assert py.step(blank_uv, blank_vis)
    assert bool(th.step(*_to_torch(blank_uv, blank_vis))[0])
    assert int(th.index[0]) == py.index == 7


def test_torch_jump_is_not_a_pass():
    th = TorchVisualGateCounter(1, "cpu", start_idx=6, n_course=11)
    for _ in range(8):
        uv = _square(0.35, 0.5, 0.16)
        th.step(*_to_torch(uv, [True] * 8))
    jumped = _square(0.78, 0.5, 0.16)
    th.step(*_to_torch(jumped, [True] * 8))
    assert int(th.index[0]) == 6
    assert not bool(th.step(*_to_torch(*_blank()))[0])
    assert int(th.index[0]) == 6


def test_lock_metrics_torch_center():
    uv, vis = _to_torch(_square(0.5, 0.5, 0.20), [True] * 8)
    m = lock_metrics_torch(uv, vis)
    assert m["n_inner"].item() == 4
    assert abs(m["cx"].item() - 0.5) < 1e-5
    assert m["align"].item() > 0.9
