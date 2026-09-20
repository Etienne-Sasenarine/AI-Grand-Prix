"""Hover MDP: the racing terms, plus station-keeping rewards.

Everything except the reward is deliberately reused from ``drone_racer`` --
same observation, same action, same plant, same events. The star import keeps
those available under one namespace so the env config reads like the racing
one; only the terms below are new.
"""

from tasks.drone_racer.mdp import *  # noqa: F401, F403

from .rewards import (  # noqa: F401
    hold_point,
    settled,
    station_keep,
    stillness,
    upright,
)
