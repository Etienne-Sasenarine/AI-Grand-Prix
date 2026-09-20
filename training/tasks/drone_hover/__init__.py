"""Gate-relative station keeping: the cage deployment test, and a racing seed."""

import gymnasium as gym

from tasks.drone_racer import agents

gym.register(
    id="Isaac-Drone-Hover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_hover_env_cfg:DroneHoverEnvCfg",
        # The same PPO config as racing. The observation and action are
        # identical, so a policy trained here loads into the racing task
        # unchanged -- which is the point of the exercise.
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Hover-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_hover_env_cfg:DroneHoverEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)
