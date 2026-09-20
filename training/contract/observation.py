"""The observation vector, defined once for the simulator and the aircraft.

The policy is trained in Isaac against ``aigp_obs.py`` and flown on the Orin
against ``race_obs.py``. Those are two implementations of one vector, and if
they disagree by so much as a channel order the policy does not fail loudly --
it flies confidently and wrongly. This module is the single definition both
build from, and :mod:`contract.verify` turns it into a hash so a mismatch is
caught at import rather than in the air.

Layout, one frame, in order::

    corners      16   u,v per keypoint, normalised; NOT_SEEN when off-frame
    visibility    8   1.0 seen, 0.0 not
    state         5   roll, pitch, gx, gy, gz
    velocity      3   vx, vy, vz -- *commanded* body velocity, never IMU accel
    context      19   one-hot over 18 gates + fraction through the lap
                 --
                 51

A policy stacks ``HISTORY`` of these. Two versions exist:

``v1`` (1632 = 51 x 32) is what every checkpoint through ``pq_speed_best``
trained against, and what the client emits today.

``v2`` (1760 = 55 x 32) appends the four action channels to each frame. This
is not cosmetic. With zero latency a policy can infer what it commanded by
watching what happened next, so action history is redundant. The moment the
plant has delay -- and the real aircraft has roughly 40 ms of it -- that stops
being true: the policy commands roll, sees no roll for two or three steps
because the command has not landed, commands more, and the accumulated
commands arrive together and overshoot. Eschmann's ablation measured this
directly: with delay simulated but action history removed, trajectory tracking
went from 10/10 to 0/10.

Pick a version explicitly. Do not add channels to ``v1``.
"""

from __future__ import annotations

# --- sentinels and clips ----------------------------------------------------
KEYPOINT_COUNT = 8
NOT_SEEN = -1.0
# A keypoint may sit this far outside the frame and still count as seen, as a
# fraction of frame size. A YOLO-pose model does predict keypoints past the
# image edge when it can see enough of the object, so a small margin is honest;
# a large one would tell the policy it can see corners no detector will report.
OFF_FRAME_MARGIN = 0.15
GYRO_CLIP = 8.0
VEL_CLIP = 20.0
N_GATES = 18

# --- channel names, in vector order -----------------------------------------
CORNER_CHANNELS: tuple[str, ...] = tuple(
    f"{axis}{i}" for i in range(KEYPOINT_COUNT) for axis in ("u", "v")
)
VIS_CHANNELS: tuple[str, ...] = tuple(f"vis{i}" for i in range(KEYPOINT_COUNT))
STATE_CHANNELS: tuple[str, ...] = ("roll", "pitch", "gx", "gy", "gz")
VEL_CHANNELS: tuple[str, ...] = ("vx", "vy", "vz")
CONTEXT_CHANNELS: tuple[str, ...] = tuple(f"gate{i}" for i in range(N_GATES)) + ("gate_frac",)
ACTION_CHANNELS: tuple[str, ...] = ("a_thrust", "a_roll_rate", "a_pitch_rate", "a_yaw_rate")

# --- segment boundaries -----------------------------------------------------
VISUAL_END = 2 * KEYPOINT_COUNT + KEYPOINT_COUNT   # 24
STATE_END = VISUAL_END + len(STATE_CHANNELS)       # 29
VEL_END = STATE_END + len(VEL_CHANNELS)            # 32
CONTEXT_END = VEL_END + len(CONTEXT_CHANNELS)      # 51

FRAME_V1: tuple[str, ...] = (
    CORNER_CHANNELS + VIS_CHANNELS + STATE_CHANNELS + VEL_CHANNELS + CONTEXT_CHANNELS
)
FRAME_V2: tuple[str, ...] = FRAME_V1 + ACTION_CHANNELS

# How many frames the policy stacks. 32 at the 60 Hz control rate spans 0.53 s.
# A single frame cannot tell a gate drifting out of view from one already gone,
# and detection falls off hard past 40 degrees of pitch, so the policy needs
# enough context to fly through a dropout.
HISTORY = 32

# The lookahead term, when enabled, appends the *next* gate's pose in the
# current gate frame. It is static course geometry rather than sim state, so it
# survives deployment -- but only where a track map exists. Checkpoints are
# only loadable under the setting they trained with, because it changes width.
LOOKAHEAD_DIM = 7

VERSIONS = {"v1": FRAME_V1, "v2": FRAME_V2}


def frame_channels(version: str = "v1") -> tuple[str, ...]:
    """Channel names for one frame, in vector order."""
    try:
        return VERSIONS[version]
    except KeyError:
        raise ValueError(f"unknown observation version {version!r}; have {sorted(VERSIONS)}") from None


def frame_dim(version: str = "v1") -> int:
    """Width of one frame. 51 for v1, 55 for v2."""
    return len(frame_channels(version))


def observation_dim(version: str = "v1", *, history: int = HISTORY, lookahead: bool = False) -> int:
    """Full flattened policy input width.

    v1 with the default history is 1632, and 1639 with lookahead. v2 is 1760.
    """
    return frame_dim(version) * history + (LOOKAHEAD_DIM if lookahead else 0)


def channel_index(name: str, version: str = "v1") -> int:
    """Position of a named channel within one frame."""
    channels = frame_channels(version)
    try:
        return channels.index(name)
    except ValueError:
        raise ValueError(f"{name!r} is not a channel of observation {version}") from None


def describe(version: str = "v1") -> str:
    """One-line human summary, for logs and checkpoint metadata."""
    return (
        f"aigp-obs/{version} frame={frame_dim(version)} history={HISTORY} "
        f"flat={observation_dim(version)}"
    )
