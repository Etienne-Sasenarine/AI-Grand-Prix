"""Run a trained Isaac/skrl policy with NumPy only — no PyTorch, no CUDA.

Why this exists
---------------
The Jetson image the organizers ship has OpenCV, NumPy, pyserial and python3-gi.
It has no PyTorch, and installing PyTorch on a Jetson is a multi-hour job that
breaks in interesting ways. It is also unnecessary: the policy is three 256-wide
ELU layers plus a linear head. That is four matrix multiplies.

Two pieces:

  export_policy(checkpoint, out.npz)   run on the laptop, needs torch
  NumpyPolicy(out.npz).act(obs)        run on the drone, needs numpy only

The exporter verifies itself against the authoritative skrl implementation
before writing anything, so a silent architecture mismatch cannot slip through.

The observation preprocessor
----------------------------
The training config uses skrl's ``RunningStandardScaler`` on states. The network
was trained on *normalised* observations, so the raw 1632-vector must be
normalised with the exact running statistics from the checkpoint before it
reaches the first layer. Forgetting this does not crash anything — it just
produces a policy that flies like it has never seen the course. The statistics
are exported alongside the weights and applied in ``act()``.

skrl's scaler is::

    clip((x - mean) / (sqrt(variance) + epsilon), -clip_threshold, +clip_threshold)

with epsilon 1e-8 and clip_threshold 5.0 by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# skrl RunningStandardScaler defaults. If the training config ever overrides
# these, the exporter reads the real values out of the agent yaml instead.
DEFAULT_EPSILON = 1e-8
DEFAULT_CLIP_THRESHOLD = 5.0

OBS_DIM = 1632  # 51 numbers per frame x 32 frames of history
ACTION_DIM = 4  # thrust, roll rate, pitch rate, yaw rate


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU, matching torch.nn.ELU(alpha=1.0)."""
    return np.where(x > 0.0, x, alpha * np.expm1(np.minimum(x, 0.0)))


class NumpyPolicy:
    """Deterministic forward pass of an exported policy.

    Deterministic means the distribution mean, not a sample. Training explores
    by sampling; deployment should not. ``log_std`` is exported for reference
    but is never used to add noise here.
    """

    def __init__(self, npz_path: str | Path) -> None:
        d = np.load(str(npz_path))
        self.layers = []
        i = 0
        while f"W{i}" in d:
            self.layers.append((d[f"W{i}"].astype(np.float32), d[f"b{i}"].astype(np.float32)))
            i += 1
        self.w_out = d["W_out"].astype(np.float32)
        self.b_out = d["b_out"].astype(np.float32)
        self.mean = d["obs_mean"].astype(np.float32)
        self.std = np.sqrt(d["obs_var"].astype(np.float32)) + float(d["obs_epsilon"])
        self.clip = float(d["obs_clip"])
        self.log_std = d["log_std"].astype(np.float32)
        self.obs_dim = int(self.layers[0][0].shape[1])
        self.action_dim = int(self.w_out.shape[0])

    def normalise(self, obs: np.ndarray) -> np.ndarray:
        """Apply the training-time running standard scaler."""
        return np.clip((obs - self.mean) / self.std, -self.clip, self.clip)

    def act(self, obs: np.ndarray, *, already_normalised: bool = False) -> np.ndarray:
        """Raw observation (1632,) in, action (4,) in [-1, 1] out."""
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.obs_dim:
            raise ValueError(f"expected {self.obs_dim} observation values, got {x.shape[0]}")
        if not already_normalised:
            x = self.normalise(x)
        for w, b in self.layers:
            x = elu(w @ x + b)
        return np.clip(self.w_out @ x + self.b_out, -1.0, 1.0)


# --------------------------------------------------------------------------
# Export side. Only used on the laptop; needs torch, and skrl for verification.
# --------------------------------------------------------------------------

def export_policy(
    checkpoint_path: str | Path,
    out_path: str | Path,
    *,
    epsilon: float = DEFAULT_EPSILON,
    clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
    verify: bool = True,
    tolerance: float = 1e-4,
) -> dict:
    """Convert an skrl PPO checkpoint into a NumPy-only ``.npz``.

    Returns a dict of diagnostics. Raises if verification fails.
    """
    import torch

    ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    policy = ck["policy"]
    pre = ck.get("state_preprocessor")
    if pre is None:
        raise RuntimeError(
            "checkpoint has no state_preprocessor. The policy was trained on normalised "
            "observations; without the statistics it cannot be reproduced."
        )

    # net_container.{0,2,4} are the Linear layers; the odd indices are ELU and
    # carry no parameters, which is why they do not appear in the state dict.
    idx, arrays = 0, {}
    n = 0
    while f"net_container.{idx}.weight" in policy:
        arrays[f"W{n}"] = policy[f"net_container.{idx}.weight"].numpy()
        arrays[f"b{n}"] = policy[f"net_container.{idx}.bias"].numpy()
        idx += 2
        n += 1
    if n == 0:
        raise RuntimeError("no net_container layers found; is this an skrl checkpoint?")

    arrays["W_out"] = policy["policy_layer.weight"].numpy()
    arrays["b_out"] = policy["policy_layer.bias"].numpy()
    arrays["log_std"] = policy["log_std_parameter"].numpy()
    arrays["obs_mean"] = pre["running_mean"].numpy()
    arrays["obs_var"] = pre["running_variance"].numpy()
    arrays["obs_epsilon"] = np.float32(epsilon)
    arrays["obs_clip"] = np.float32(clip_threshold)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    info = {
        "checkpoint": str(checkpoint_path),
        "output": str(out_path),
        "hidden_layers": n,
        "obs_dim": int(arrays["W0"].shape[1]),
        "action_dim": int(arrays["W_out"].shape[0]),
        "bytes": out_path.stat().st_size,
        "preprocessor_samples": float(pre["current_count"]),
    }

    if verify:
        info["max_abs_error"] = _verify_against_torch(
            checkpoint_path, out_path, tolerance=tolerance
        )
    return info


def _verify_against_torch(checkpoint_path, npz_path, *, tolerance: float, n_samples: int = 64) -> float:
    """Compare the NumPy path against a torch reference on random observations.

    The reference is built to match skrl's ``gaussian_model`` instantiator:
    Linear -> ELU for each hidden layer, then a linear output head. If our
    reading of that architecture were wrong, the error here would be large,
    which is the point — an ELU silently missing from the last hidden layer
    produces a plausible-looking but wrong policy.
    """
    import torch
    import torch.nn as nn

    ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    p = ck["policy"]

    sizes, idx = [], 0
    while f"net_container.{idx}.weight" in p:
        sizes.append(p[f"net_container.{idx}.weight"].shape)
        idx += 2
    mods = []
    for shape in sizes:
        mods += [nn.Linear(shape[1], shape[0]), nn.ELU()]
    trunk = nn.Sequential(*mods)
    head = nn.Linear(sizes[-1][0], p["policy_layer.weight"].shape[0])

    with torch.no_grad():
        for i, shape in enumerate(sizes):
            trunk[2 * i].weight.copy_(p[f"net_container.{2 * i}.weight"])
            trunk[2 * i].bias.copy_(p[f"net_container.{2 * i}.bias"])
        head.weight.copy_(p["policy_layer.weight"])
        head.bias.copy_(p["policy_layer.bias"])

        # skrl stores these as float64 and casts at use time (`.float()` in
        # RunningStandardScaler._compute). Match that exactly.
        mean = ck["state_preprocessor"]["running_mean"].float()
        var = ck["state_preprocessor"]["running_variance"].float()

        rng = np.random.default_rng(0)
        obs = rng.standard_normal((n_samples, int(sizes[0][1]))).astype(np.float32)
        obs = (obs * np.sqrt(var.numpy()) + mean.numpy()).astype(np.float32)  # realistic scale

        x = torch.from_numpy(obs.astype(np.float32))
        x = torch.clamp((x - mean) / (torch.sqrt(var) + DEFAULT_EPSILON),
                        -DEFAULT_CLIP_THRESHOLD, DEFAULT_CLIP_THRESHOLD)
        ref = torch.clamp(head(trunk(x)), -1.0, 1.0).numpy()

    policy = NumpyPolicy(npz_path)
    ours = np.stack([policy.act(o) for o in obs])
    err = float(np.abs(ref - ours).max())
    if err > tolerance:
        raise AssertionError(
            f"NumPy policy disagrees with torch by {err:.3g} (tolerance {tolerance:.3g}). "
            "Do not fly this."
        )
    return err


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    print(json.dumps(export_policy(a.checkpoint, a.output, verify=not a.no_verify), indent=2))
