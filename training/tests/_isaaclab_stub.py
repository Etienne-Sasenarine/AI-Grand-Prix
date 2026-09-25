"""One Isaac Lab stand-in, shared by every test that needs one.

Several test modules exercise code that imports ``isaaclab`` but does not
actually use the simulator -- reward arithmetic, the inertia write, the
observation packing. Each used to install its own stub at import time, which
broke in two ways:

* **First one wins.** Whichever test module pytest collected first installed
  its stub, and a later module needing a submodule the first did not provide
  got ``No module named 'isaaclab.managers'``. The failure depended on
  collection order, so a file could pass alone and fail in the suite.

* **It would shadow the real thing.** A guard of "is isaaclab in sys.modules"
  is false on the training box too, where Isaac Lab *is* installed and simply
  has not been imported yet. The stub would then take precedence over the real
  library and the tests would quietly stop testing anything real.

So: one stub, built once, and only when Isaac Lab is genuinely absent.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

import torch

_MARKER = "_aigp_test_stub"


def _euler_xyz_from_quat(q: torch.Tensor):
    """Real conversion, not a placeholder.

    ``gate_normal_w`` derives a gate's facing direction from this, so a fake
    would make every geometry assertion meaningless -- tests would pass against
    arithmetic nobody performs. Isaac's quaternion convention is wxyz.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def ensure() -> bool:
    """Install the stub if Isaac Lab is absent. True if the stub is in use."""
    # Order matters. find_spec raises ValueError on a hand-built module whose
    # __spec__ is None, so check what is already loaded before asking the
    # import system to look.
    existing = sys.modules.get("isaaclab")
    if existing is not None:
        return bool(getattr(existing, _MARKER, False))
    if importlib.util.find_spec("isaaclab") is not None:
        return False  # the real library is installed; never shadow it

    isaaclab = types.ModuleType("isaaclab")
    setattr(isaaclab, _MARKER, True)

    assets = types.ModuleType("isaaclab.assets")
    for name in ("Articulation", "RigidObject", "AssetBaseCfg", "ArticulationCfg"):
        setattr(assets, name, type(name, (), {}))

    managers = types.ModuleType("isaaclab.managers")

    class SceneEntityCfg:
        def __init__(self, name="robot", **kw):
            self.name = name
            for k, v in kw.items():
                setattr(self, k, v)

    managers.SceneEntityCfg = SceneEntityCfg
    for name in ("RewardTermCfg", "TerminationTermCfg", "ObservationTermCfg",
                 "ObservationGroupCfg", "EventTermCfg", "ActionTerm", "ActionTermCfg"):
        setattr(managers, name, type(name, (), {}))

    envs = types.ModuleType("isaaclab.envs")
    for name in ("ManagerBasedEnv", "ManagerBasedRLEnv", "ManagerBasedRLEnvCfg"):
        setattr(envs, name, type(name, (), {}))

    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.euler_xyz_from_quat = _euler_xyz_from_quat
    utils.math = math_mod

    isaaclab.assets = assets
    isaaclab.managers = managers
    isaaclab.envs = envs
    isaaclab.utils = utils

    for mod_name, mod in (("isaaclab", isaaclab), ("isaaclab.assets", assets),
                          ("isaaclab.managers", managers), ("isaaclab.envs", envs),
                          ("isaaclab.utils", utils), ("isaaclab.utils.math", math_mod)):
        mod.__spec__ = importlib.machinery.ModuleSpec(mod_name, loader=None)
    sys.modules.update({
        "isaaclab": isaaclab,
        "isaaclab.assets": assets,
        "isaaclab.managers": managers,
        "isaaclab.envs": envs,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_mod,
    })
    return True


def load_module(relative_path: str, name: str):
    """Import a module by file path, bypassing its package ``__init__``.

    ``tasks/__init__.py`` pulls in ``isaaclab_tasks``, which needs a full Isaac
    install. Loading the file directly sidesteps that for modules whose own
    imports the stub can satisfy.
    """
    from pathlib import Path

    full = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(name, full)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
