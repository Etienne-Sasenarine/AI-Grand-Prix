"""The detector-jitter augmentation: corners that are present but not exact.

Dropout teaches the policy to cope with a corner being absent. Jitter is the
other half: a corner that is *there and wrong*. That is the worse failure,
because the runner treats visibility as truth and nothing downstream discounts
a corner it believes it can see.

The magnitude is deliberately NOT pinned here, because it has not been
measured. What is pinned is the mechanism: off unless asked for, only moves
corners that exist, respects the sentinel, scales correctly in each axis, and
stays inside the frame.

The helper is exec'd out of the module rather than imported, because importing
`observations` pulls in Isaac Lab, which is not available in this suite.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SRC = Path(__file__).resolve().parents[1] / "tasks/drone_racer/mdp/observations.py"
FRAME_W, FRAME_H, NOT_SEEN = 640.0, 360.0, -1.0


def _load():
    src = SRC.read_text()
    start = src.index("def _keypoint_jitter")
    end = src.index("def aigp_race_observation")
    ns: dict = {"os": os, "torch": torch, "NOT_SEEN": NOT_SEEN,
                "FRAME_W": FRAME_W, "FRAME_H": FRAME_H}
    exec(src[start:end], ns)          # noqa: S102 - our own source, not input
    return ns["_keypoint_jitter"]


class _Env:
    episode_length_buf = None


def _obs(n=512, uv=0.5, visible=True):
    """n environments, eight corners each, all at the same place."""
    obs = torch.zeros((n, 55))
    obs[:, :16] = uv
    obs[:, 16:24] = 1.0 if visible else 0.0
    return obs


@pytest.fixture(autouse=True)
def _clean_env():
    os.environ.pop("AIGP_KP_JITTER_PX", None)
    yield
    os.environ.pop("AIGP_KP_JITTER_PX", None)


def test_off_by_default_changes_nothing():
    """It must be opt-in: an uncalibrated augmentation left on by accident
    would quietly change every run that did not ask for it."""
    jitter = _load()
    before = _obs()
    after = jitter(_Env(), before.clone())
    assert torch.equal(after[:, :16], before[:, :16])

    os.environ["AIGP_KP_JITTER_PX"] = "0.0"
    assert torch.equal(jitter(_Env(), before.clone())[:, :16], before[:, :16])


def test_a_missing_corner_keeps_its_sentinel_exactly():
    """NOT_SEEN is a sentinel, not a position.

    Nudging -1.0 by a pixel would turn "not seen" into a number the contract
    reads as a real, if odd, location. The dropout augmentation had the mirror
    of this bug -- it wrote 0.0, a valid top-left position -- and it collapsed
    racing from 12.80 gates per episode to 0.13.
    """
    jitter = _load()
    os.environ["AIGP_KP_JITTER_PX"] = "40.0"      # deliberately huge
    obs = _obs()
    obs[:, :16] = NOT_SEEN
    after = jitter(_Env(), obs.clone())
    assert torch.all(after[:, :16] == NOT_SEEN)


def test_only_the_corners_that_exist_are_moved():
    jitter = _load()
    os.environ["AIGP_KP_JITTER_PX"] = "8.0"
    obs = _obs()
    obs[:, :8] = NOT_SEEN                          # first four corners absent
    after = jitter(_Env(), obs.clone())
    assert torch.all(after[:, :8] == NOT_SEEN)
    assert not torch.equal(after[:, 8:16], obs[:, 8:16])


def test_the_magnitude_is_the_requested_pixels_in_each_axis():
    """The same pixel error is a bigger fraction of v than of u: 640 vs 360."""
    jitter = _load()
    sigma_px = 6.0
    os.environ["AIGP_KP_JITTER_PX"] = str(sigma_px)
    obs = _obs(n=20000)
    after = jitter(_Env(), obs.clone())
    delta = after[:, :16] - obs[:, :16]
    u_px = delta[:, 0::2].std().item() * FRAME_W
    v_px = delta[:, 1::2].std().item() * FRAME_H
    assert u_px == pytest.approx(sigma_px, rel=0.05)
    assert v_px == pytest.approx(sigma_px, rel=0.05)
    # And in normalised units v moves further, which is the point of scaling
    # per axis rather than applying one number to both.
    assert delta[:, 1::2].std() > delta[:, 0::2].std()


def test_noise_is_independent_per_corner_and_per_environment():
    """A single draw shared across corners would be a camera shake, not
    detector error, and the policy could cancel it as a common-mode term."""
    jitter = _load()
    os.environ["AIGP_KP_JITTER_PX"] = "6.0"
    after = jitter(_Env(), _obs(n=4096).clone())
    corner_u = after[:, 0::2]
    # Correlation between two different corners' u across environments.
    a, b = corner_u[:, 0], corner_u[:, 1]
    r = torch.corrcoef(torch.stack([a, b]))[0, 1].abs().item()
    assert r < 0.1, f"corners move together (r={r:.3f})"


def test_a_corner_never_leaves_the_frame():
    """Clamped to [0, 1], the same range the runner's own normalisation gives."""
    jitter = _load()
    os.environ["AIGP_KP_JITTER_PX"] = "200.0"      # far wider than the frame
    for edge in (0.0, 1.0, 0.5):
        after = jitter(_Env(), _obs(n=4096, uv=edge).clone())
        assert after[:, :16].min() >= 0.0
        assert after[:, :16].max() <= 1.0


def test_dropout_runs_first_so_a_dropped_corner_is_never_nudged():
    """Order is asserted against the source: jitter after dropout.

    The other order would move a corner and then drop it, which is harmless, or
    drop it and then move the sentinel, which is not.
    """
    src = SRC.read_text()
    assert src.index("obs = _keypoint_dropout(env, obs)") < src.index(
        "obs = _keypoint_jitter(env, obs)"
    )


def test_it_says_it_is_uncalibrated():
    """An assumption presented as a measurement is how q=0.20 nearly shipped."""
    src = SRC.read_text()
    body = src[src.index("def _keypoint_jitter"):src.index("def aigp_race_observation")]
    assert "not calibrated" in body
    assert "live_compare.py --stats-log" in body
