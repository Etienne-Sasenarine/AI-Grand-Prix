"""Miss termination and low-gate dive reward, without launching Isaac Sim."""

from types import SimpleNamespace

import torch

from tasks.drone_racer.mdp.rewards import low_gate_dive
from tasks.drone_racer.mdp.terminations import gate_missed


def _quat_from_euler_zyx(roll, pitch, yaw):
    """wxyz quaternion. Pitch about Y, FLU / Z-up."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


class _Cmd:
    def __init__(self, missed, gate_xyz):
        self.gate_missed = missed
        self.command = torch.zeros(missed.shape[0], 7)
        self.command[:, :3] = gate_xyz
        self.command[:, 3] = 1.0


class _Robot:
    def __init__(self, pos, quat, vel):
        self.data = SimpleNamespace(
            root_pos_w=pos,
            root_quat_w=quat,
            root_lin_vel_w=vel,
        )


def _env(cmd, robot, thrust=None):
    env = SimpleNamespace(num_envs=cmd.gate_missed.shape[0], action_manager=None)
    env.command_manager = SimpleNamespace(get_term=lambda _name: cmd)
    env.scene = {"robot": robot}
    if thrust is not None:
        term = SimpleNamespace(processed_actions=torch.stack([thrust, thrust, thrust, thrust], dim=-1))
        env.action_manager = SimpleNamespace(get_term=lambda _name: term)
    return env


def test_gate_missed_termination_follows_command_flag():
    missed = torch.tensor([True, False, True])
    env = _env(_Cmd(missed, torch.zeros(3, 3)), _Robot(torch.zeros(3, 3), torch.zeros(3, 4), torch.zeros(3, 3)))
    out = gate_missed(env, "target")
    assert torch.equal(out, missed)


def test_dive_reward_zero_when_gate_is_not_below():
    n = 1
    pos = torch.tensor([[0.0, 0.0, 2.0]])
    gate = torch.tensor([[8.0, 0.0, 4.0]])
    quat = _quat_from_euler_zyx(torch.zeros(n), torch.full((n,), -0.4), torch.zeros(n))
    env = _env(
        _Cmd(torch.zeros(n, dtype=torch.bool), gate),
        _Robot(pos, quat, torch.tensor([[0.0, 0.0, -2.0]])),
        thrust=torch.tensor([3.0]),
    )
    assert float(low_gate_dive(env, "target")) == 0.0


def test_dive_reward_positive_on_low_gate_nose_down_cut_sink():
    n = 1
    pos = torch.tensor([[0.0, 0.0, 4.2]])
    gate = torch.tensor([[10.0, 0.0, 2.0]])
    quat = _quat_from_euler_zyx(torch.zeros(n), torch.full((n,), -0.4), torch.zeros(n))
    env = _env(
        _Cmd(torch.zeros(n, dtype=torch.bool), gate),
        _Robot(pos, quat, torch.tensor([[0.0, 0.0, -3.0]])),
        thrust=torch.tensor([2.0]),
    )
    score = float(low_gate_dive(env, "target"))
    assert score > 0.4
