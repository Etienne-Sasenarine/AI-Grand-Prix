"""One definition of the observation vector, shared by the simulator and the aircraft.

``aigp-sim`` trains the policy; ``AI_GP`` flies it. Each builds the observation
itself, because one is torch across 4096 environments and the other is numpy on
a single Orin. This package is the definition they both build *from*, and
:mod:`contract.verify` hashes it so the two cannot drift apart quietly.

    from contract import camera, observation, plant, verify

    verify.check(checkpoint["contract_hash"])     # refuse a stale policy
    width = observation.observation_dim("v1")     # 1632

Vendoring note: this package is duplicated into the flight repo rather than
imported across repositories, because the Orin should not need this one
checked out to fly. The hash is what makes the duplication safe -- copy the
directory wholesale, never edit one copy.
"""

from __future__ import annotations

from . import camera, observation, plant, verify

__all__ = ["camera", "observation", "plant", "verify"]

# Bump when the *meaning* of the contract changes in a way the hash cannot
# express on its own -- a new version string, a new segment. The hash catches
# value drift; this catches structural drift in how it is read.
CONTRACT_API = 1
