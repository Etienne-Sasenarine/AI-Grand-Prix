"""Perception latency and the v2 action channels.

``KeypointDelayLine`` models the gap between the shutter opening and the
keypoints existing -- readout, the copy to the Orin, YOLO-pose inference. It is
the larger half of the round trip: the scenario that took our stress runs from
2.6 to 55 crashes per 100 gates was 33 ms of vision against 17 ms of command.

The observation half is checked here too, because the two only make sense
together. A delay the policy has no record of causing is a disturbance it
cannot learn to anticipate.
"""

from __future__ import annotations

import pytest
import torch

from utils.aigp_obs import (
    ACTION_DIM,
    KEYPOINT_COUNT,
    KeypointDelayLine,
    pack_observation,
)

N_ENVS = 4


def _frame(value: float, num_envs: int = N_ENVS):
    """A projection whose every pixel carries a recognisable stamp."""
    uv = torch.full((num_envs, KEYPOINT_COUNT, 2), value)
    visible = torch.ones(num_envs, KEYPOINT_COUNT, dtype=torch.bool)
    return uv, visible


# --- delay line -------------------------------------------------------------


def test_zero_delay_returns_the_frame_untouched():
    """The rollback path must be exact: v1 has to reproduce bit for bit."""
    line = KeypointDelayLine(N_ENVS, "cpu", max_delay=0)
    uv, vis = _frame(123.0)
    out_uv, out_vis = line.step(uv, vis)
    assert torch.equal(out_uv, uv)
    assert torch.equal(out_vis, vis)


def test_a_frame_comes_back_after_exactly_its_delay():
    line = KeypointDelayLine(1, "cpu", max_delay=4)
    line.delay[0] = 3
    line.reset(torch.tensor([0]))

    seen = []
    for i in range(8):
        uv, vis = _frame(float(i), num_envs=1)
        out_uv, _ = line.step(uv, vis)
        seen.append(float(out_uv[0, 0, 0]))

    # At step k the policy sees frame k-3. Steps 0, 1 and 2 predate any real
    # frame and show the backfill; step 3 shows frame 0 for real. Those four
    # values coincide here only because the backfill and frame 0 are the same
    # projection -- which is the point of backfilling with the current frame.
    assert seen[:4] == [0.0, 0.0, 0.0, 0.0]
    assert seen[4:] == [1.0, 2.0, 3.0, 4.0]


def test_each_env_gets_its_own_delay():
    """Per-env, because a single global lag is something a policy can learn
    exactly and quietly depend on."""
    line = KeypointDelayLine(3, "cpu", max_delay=4)
    line.delay = torch.tensor([0, 2, 4])
    line.reset(torch.arange(3))
    for i in range(8):
        out_uv, _ = line.step(*_frame(float(i), num_envs=3))
    assert float(out_uv[0, 0, 0]) == 7.0
    assert float(out_uv[1, 0, 0]) == 5.0
    assert float(out_uv[2, 0, 0]) == 3.0


def test_visibility_travels_with_the_pixels():
    """A stale frame must carry its own visibility flags. Pairing fresh flags
    with old pixels would tell the policy it can see a gate at coordinates
    that no longer describe one."""
    line = KeypointDelayLine(1, "cpu", max_delay=2)
    line.delay[0] = 2
    line.reset(torch.tensor([0]))

    uv_seen, vis_seen = _frame(10.0, num_envs=1)
    line.step(uv_seen, vis_seen)
    uv_gone, vis_gone = _frame(20.0, num_envs=1)
    vis_gone[:] = False
    for _ in range(2):
        out_uv, out_vis = line.step(uv_gone, vis_gone)
    assert float(out_uv[0, 0, 0]) == 10.0
    assert bool(out_vis.all()), "visibility did not travel with its own frame"


def test_the_ring_wraps_without_losing_alignment():
    line = KeypointDelayLine(1, "cpu", max_delay=2)
    line.delay[0] = 2
    line.reset(torch.tensor([0]))
    seen = []
    for i in range(20):
        out_uv, _ = line.step(*_frame(float(i), num_envs=1))
        seen.append(float(out_uv[0, 0, 0]))
    assert seen[3:] == [float(i) for i in range(1, 18)]


def test_reset_backfills_with_the_current_frame_not_zeros():
    """A fresh episode has no older frame to show. Handing it blank keypoints
    would teach the policy that every episode opens blind, which is an artefact
    of the simulator -- the aircraft has been sitting there looking at a gate.
    """
    line = KeypointDelayLine(1, "cpu", max_delay=3)
    line.delay[0] = 3
    for i in range(5):
        line.step(*_frame(float(i), num_envs=1))

    line.reset(torch.tensor([0]))
    out_uv, out_vis = line.step(*_frame(99.0, num_envs=1))
    assert float(out_uv[0, 0, 0]) == 99.0, "replayed a frame from the last episode"
    assert bool(out_vis.all())


def test_resetting_one_env_leaves_the_others_delayed():
    line = KeypointDelayLine(2, "cpu", max_delay=3)
    line.delay = torch.tensor([3, 3])
    line.reset(torch.arange(2))
    for i in range(1, 5):
        line.step(*_frame(float(i), num_envs=2))

    line.reset(torch.tensor([0]))
    out_uv, _ = line.step(
        torch.full((2, KEYPOINT_COUNT, 2), 99.0),
        torch.ones(2, KEYPOINT_COUNT, dtype=torch.bool),
    )
    assert float(out_uv[0, 0, 0]) == 99.0
    assert float(out_uv[1, 0, 0]) != 99.0


def test_pending_resets_accumulate_rather_than_overwrite():
    """Two resets can land between steps when episodes end on different ticks."""
    line = KeypointDelayLine(3, "cpu", max_delay=2)
    line.delay = torch.full((3,), 2, dtype=torch.long)
    for i in range(1, 4):
        line.step(torch.full((3, KEYPOINT_COUNT, 2), float(i)),
                  torch.ones(3, KEYPOINT_COUNT, dtype=torch.bool))
    line.reset(torch.tensor([0]))
    line.reset(torch.tensor([2]))
    out_uv, _ = line.step(torch.full((3, KEYPOINT_COUNT, 2), 50.0),
                          torch.ones(3, KEYPOINT_COUNT, dtype=torch.bool))
    assert float(out_uv[0, 0, 0]) == 50.0
    assert float(out_uv[2, 0, 0]) == 50.0, "second reset was dropped"
    assert float(out_uv[1, 0, 0]) != 50.0


def test_negative_max_delay_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        KeypointDelayLine(1, "cpu", max_delay=-1)


def test_resample_draws_within_the_requested_band():
    line = KeypointDelayLine(256, "cpu", max_delay=4)
    line.resample(torch.arange(256), 1, 4)
    assert int(line.delay.min()) >= 1
    assert int(line.delay.max()) <= 4
    assert len(line.delay.unique()) > 1, "every env drew the same delay"


# --- v2 action channels -----------------------------------------------------


def _packed(actions=None):
    uv = torch.rand(N_ENVS, KEYPOINT_COUNT, 2) * 640.0
    vis = torch.ones(N_ENVS, KEYPOINT_COUNT, dtype=torch.bool)
    zeros = torch.zeros(N_ENVS)
    return pack_observation(
        uv, vis, zeros, zeros, torch.zeros(N_ENVS, 3),
        velocity_body_ned=torch.zeros(N_ENVS, 3),
        gate_index=torch.zeros(N_ENVS, dtype=torch.long),
        with_velocity=True, with_context=True, actions=actions,
    )


def test_v1_frame_is_fifty_one_channels():
    assert _packed().shape[-1] == 51


def test_v2_frame_is_fifty_five():
    assert _packed(torch.zeros(N_ENVS, ACTION_DIM)).shape[-1] == 55


def test_v2_appends_rather_than_interleaves():
    """A v1 reader must see exactly v1 in the first 51 channels."""
    torch.manual_seed(0)
    uv = torch.rand(N_ENVS, KEYPOINT_COUNT, 2) * 640.0
    vis = torch.ones(N_ENVS, KEYPOINT_COUNT, dtype=torch.bool)
    zeros = torch.zeros(N_ENVS)
    kwargs = dict(
        velocity_body_ned=torch.zeros(N_ENVS, 3),
        gate_index=torch.zeros(N_ENVS, dtype=torch.long),
        with_velocity=True, with_context=True,
    )
    v1 = pack_observation(uv, vis, zeros, zeros, torch.zeros(N_ENVS, 3), **kwargs)
    v2 = pack_observation(uv, vis, zeros, zeros, torch.zeros(N_ENVS, 3),
                          actions=torch.rand(N_ENVS, ACTION_DIM), **kwargs)
    assert torch.allclose(v2[:, :51], v1)


def test_action_channels_carry_the_policy_output():
    actions = torch.tensor([[0.5, -0.25, 0.75, -1.0]] * N_ENVS)
    obs = _packed(actions)
    assert torch.allclose(obs[:, 51:55], actions)


def test_action_channels_are_clamped_to_the_policy_range():
    """A Gaussian policy can emit outside [-1, 1]; the decode clamps, so the
    observation must record the clamped value or it disagrees with the plant."""
    obs = _packed(torch.full((N_ENVS, ACTION_DIM), 5.0))
    assert torch.allclose(obs[:, 51:55], torch.ones(N_ENVS, ACTION_DIM))
    obs = _packed(torch.full((N_ENVS, ACTION_DIM), -5.0))
    assert torch.allclose(obs[:, 51:55], -torch.ones(N_ENVS, ACTION_DIM))


def test_extra_action_columns_are_ignored():
    obs = _packed(torch.zeros(N_ENVS, ACTION_DIM + 3))
    assert obs.shape[-1] == 55
