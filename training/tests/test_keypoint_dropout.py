"""The detector-dropout augmentation, and that it stays calibrated.

Corners blink out on the real aircraft and never in simulation, where they are
projected analytically. Measured at a gate on 2026-09-21 over 1345 frames: a
corner's visibility flips between consecutive frames 4.2% of the time, and
about 13% of the corners the geometry offers are missed.

These are pinned because the augmentation is only worth having if it matches
what was measured -- an arbitrary dropout rate teaches an arbitrary lesson.

The helper is exec'd out of the module rather than imported, because importing
`observations` pulls in Isaac Lab, which is not available in this suite.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SRC = Path(__file__).resolve().parents[1] / "tasks/drone_racer/mdp/observations.py"


def _load():
    src = SRC.read_text()
    start = src.index("def _keypoint_dropout")
    end = src.index("def aigp_race_observation")
    ns: dict = {"os": os, "torch": torch, "NOT_SEEN": -1.0}
    exec(src[start:end], ns)          # noqa: S102 - our own source, not input
    return ns["_keypoint_dropout"]


class _Env:
    episode_length_buf = None


def _run(q: str, sticky: str, steps: int = 1500, n: int = 128):
    os.environ["AIGP_KP_DROP"] = q
    os.environ["AIGP_KP_STICKY"] = sticky
    try:
        fn = _load()
        env = _Env()
        env.episode_length_buf = torch.ones(n, dtype=torch.long)
        base = torch.zeros(n, 24)
        base[:, :16] = 1.0
        base[:, 16:24] = 1.0
        dropped = flips = 0.0
        prev = None
        for _ in range(steps):
            out = fn(env, base.clone())
            assert out is not None, "the helper must return the observation"
            vis = out[:, 16:24]
            dropped += float((vis == 0).float().mean())
            if prev is not None:
                flips += float((vis != prev).float().mean())
            prev = vis
            # A corner that was not detected carries the absent sentinel,
            # NOT_SEEN = -1.0. Zero would be a valid on-screen position.
            corners = out[:, :16].reshape(n, 8, 2)
            assert torch.all(corners[vis == 0] == -1.0)
        return dropped / steps, flips / (steps - 1)
    finally:
        os.environ.pop("AIGP_KP_DROP", None)
        os.environ.pop("AIGP_KP_STICKY", None)


def test_unset_changes_nothing():
    """Every existing run must be unaffected."""
    os.environ.pop("AIGP_KP_DROP", None)
    fn = _load()
    env = _Env()
    env.episode_length_buf = torch.ones(8, dtype=torch.long)
    obs = torch.rand(8, 24)
    out = fn(env, obs.clone())
    assert torch.equal(out, obs)


def test_matches_the_measured_dropout_and_flip_rate():
    dropped, flips = _run("0.13", "0.81")
    assert dropped == pytest.approx(0.13, abs=0.02)
    assert flips == pytest.approx(0.042, abs=0.008)


def test_failures_are_clustered_not_independent():
    """The property that makes the task trainable at all.

    At the same 13% dropout, independent draws flip ten times as often. An
    independent-frames model predicts the policy would be ready 9% of the time
    where the aircraft measured 58%: the bad frames clump, leaving long clean
    stretches between them. Training against independent dropout would present
    a world with no usable runs in it.
    """
    _, clustered = _run("0.13", "0.81", steps=800)
    _, independent = _run("0.13", "0.0", steps=800)
    assert independent > 4 * clustered
