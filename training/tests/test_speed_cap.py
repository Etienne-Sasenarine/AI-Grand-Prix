# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Speed-cap wrapper tests (no Isaac Sim)."""

from __future__ import annotations

import torch

from utils.speed_cap import body_forward_speed, cap_from_env, limit_policy_action


def _ident(n: int = 1) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def test_cap_from_env_reads_override_and_env(monkeypatch=None):
    assert cap_from_env(10.0) == 10.0
    assert cap_from_env(0.0) == 0.0
    assert cap_from_env(-3.0) == 0.0


def test_body_forward_is_world_x_when_level():
    vel = torch.tensor([[12.0, 3.0, -1.0]])
    assert float(body_forward_speed(vel, _ident())) == 12.0


def test_below_cap_leaves_action_alone():
    act = torch.tensor([[0.4, 0.1, -0.6, 0.0]])
    vel = torch.tensor([[4.0, 0.0, 0.0]])
    out = limit_policy_action(act, vel, _ident(), cap_mps=10.0)
    assert torch.allclose(out, act)


def test_over_cap_kills_nose_down_and_extra_thrust():
    act = torch.tensor([[0.8, 0.0, -0.9, 0.2]])
    vel = torch.tensor([[16.0, 0.0, 0.0]])
    out = limit_policy_action(act, vel, _ident(), cap_mps=10.0)
    assert float(out[0, 2]) > float(act[0, 2])  # less nose-down
    assert float(out[0, 0]) < float(act[0, 0])  # less punch
    assert torch.allclose(out[0, 1], act[0, 1])
    assert torch.allclose(out[0, 3], act[0, 3])


def test_well_over_cap_adds_nose_up():
    act = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    vel = torch.tensor([[20.0, 0.0, 0.0]])
    out = limit_policy_action(act, vel, _ident(), cap_mps=10.0)
    assert float(out[0, 2]) > 0.2


def test_disabled_cap_is_noop():
    act = torch.tensor([[0.8, 0.1, -0.9, 0.2]])
    vel = torch.tensor([[40.0, 0.0, 0.0]])
    out = limit_policy_action(act, vel, _ident(), cap_mps=0.0)
    assert torch.allclose(out, act)
