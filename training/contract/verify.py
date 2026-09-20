"""Turn the contract into a hash, so drift is an error instead of a surprise.

The simulator and the aircraft each build the observation vector themselves.
That is deliberate -- one runs torch on 4096 parallel environments, the other
runs numpy on a single Orin at 60 Hz, and forcing them through a shared
implementation would make both worse. What must not differ is the *definition*:
channel order, clips, intrinsics, the plant constants the vector is derived
from.

A mismatch there is the worst class of bug this project can have. Nothing
raises. Nothing logs. The policy receives a vector whose channels mean
something slightly different from what it trained on, and flies into a gate
with complete confidence.

So: every constant that both ends must agree on is hashed. A checkpoint records
the hash it trained under, the client checks the hash it is flying under, and a
difference is refused at load time rather than discovered at 20 m/s.

    >>> from contract import verify
    >>> verify.contract_hash()[:12]      # doctest: +SKIP
    'a3f9c21e4b07'

To change the contract deliberately, change it here, let the hash move, and
retrain. That is the intended cost: it is what stops it moving by accident.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from . import camera, observation, plant

CONTRACT_NAME = "aigp-obs"


def contract_fields(version: str = "v1") -> dict[str, Any]:
    """Every value both ends must agree on, in a stable, serialisable form.

    Only things that change the *meaning* of the vector belong here. Anything
    randomised per episode (the plant's hover point, its thrust curve) is
    deliberately absent: the policy is supposed to see a spread of those, and
    hashing them would make every run incompatible with every other.
    """
    return {
        "name": CONTRACT_NAME,
        "version": version,
        # -- vector layout
        "channels": list(observation.frame_channels(version)),
        "history": observation.HISTORY,
        "frame_dim": observation.frame_dim(version),
        "flat_dim": observation.observation_dim(version),
        "not_seen": observation.NOT_SEEN,
        "off_frame_margin": observation.OFF_FRAME_MARGIN,
        "gyro_clip": observation.GYRO_CLIP,
        "vel_clip": observation.VEL_CLIP,
        "n_gates": observation.N_GATES,
        # -- camera
        "frame_w": camera.FRAME_W,
        "frame_h": camera.FRAME_H,
        "fx": round(camera.FX, 9),
        "fy": round(camera.FY, 9),
        "cx": round(camera.CX, 9),
        "cy": round(camera.CY, 9),
        "distortion": [round(d, 12) for d in camera.DIST_OPENCV],
        "camera_tilt_up_deg": camera.CAMERA_TILT_UP_DEG,
        "gate_outer_m": camera.GATE_OUTER_M,
        "gate_inner_m": camera.GATE_INNER_M,
        # -- action decode: what the policy's [-1, 1] means in physical units
        "thrust_min": plant.THRUST_MIN,
        "thrust_hover": plant.THRUST_HOVER,
        "thrust_max": plant.THRUST_MAX,
        "rate_limit": plant.RATE_LIMIT,
        "action_channels": list(plant.ACTION_NAMES),
    }


def contract_hash(version: str = "v1") -> str:
    """Stable hex digest of :func:`contract_fields`."""
    blob = json.dumps(contract_fields(version), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def short_hash(version: str = "v1") -> str:
    """First twelve hex characters -- enough for a filename or a log line."""
    return contract_hash(version)[:12]


class ContractMismatch(RuntimeError):
    """Raised when a checkpoint and the running code disagree."""


def check(expected_hash: str, version: str = "v1", *, what: str = "checkpoint") -> None:
    """Refuse to continue if ``expected_hash`` is not what this code produces.

    Call this wherever a policy is loaded, on both the training and the flying
    side. The error names the mismatch rather than describing it vaguely,
    because whoever hits this is usually mid-session and needs to know which
    end moved.
    """
    actual = contract_hash(version)
    if expected_hash != actual:
        raise ContractMismatch(
            f"{what} was built against {CONTRACT_NAME}/{version} "
            f"{expected_hash[:12]}, but this code is {actual[:12]}.\n"
            f"The observation vector means something different than it did.\n"
            f"Current: {observation.describe(version)}\n"
            f"Either load a checkpoint trained under this contract, or check "
            f"out the revision of contract/ that produced {expected_hash[:12]}."
        )


def stamp(version: str = "v1") -> dict[str, str]:
    """Metadata to embed in a checkpoint at save time."""
    return {
        "contract_name": CONTRACT_NAME,
        "contract_version": version,
        "contract_hash": contract_hash(version),
        "contract_summary": observation.describe(version),
    }
