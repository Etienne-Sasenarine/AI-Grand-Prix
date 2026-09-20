# Three asset files are unrecoverable pointers

`assets/5_in_drone/meshes/base_link.dae`, `.../prop.dae` and
`assets/gate/gate.glb` are 4 KB Git LFS pointer files with no content behind
them. `AI_GP/isaac_drone_racer/.gitattributes` tracks `*.dae` and `*.glb` in
LFS, but the objects were never uploaded — `git lfs pull` against
`Code-Red-Cables/AI_GP` returns **404, object does not exist on the server**.

So the content is not merely un-fetched. It is gone from the remote, and anyone
cloning fresh gets the same stubs.

## This does not affect training

Nothing loads them. The code loads USD, and every USD in the path is real:

| loaded by | file | |
|---|---|---|
| `assets/five_in_drone.py` | `5_in_drone/5_in_drone.usd` → `configuration/5_in_drone_base.usd` | real, 98 MB, geometry baked in |
| `tasks/drone_racer/track_generator.py` | `gate/gate_aigp.usd` | real |

The `.dae` meshes are the URDF's visual and collision references, used only if
somebody re-imports `5_in_drone.urdf` to regenerate the USD. `gate.glb` is an
alternate format of a gate that is otherwise present as `gate_aigp.usd`,
`gate.usd`, `gate_vq2.glb` and `vq2_mesh/gate_vq2.obj`.

## What it does affect

Regenerating the drone asset from the URDF. If that is ever needed, the meshes
have to come from whoever still has them on disk — most likely Gene, who
committed them. Worth recovering before that knowledge evaporates, because a
re-import without them produces an aircraft with no geometry.

This is also the second trap in the same corner of the repo. The URDF's inertia
was inherited from a 0.5 kg aircraft and sat unnoticed against a 1.745 kg
airframe because nothing reads the URDF at runtime either.

## Why they are not LFS-tracked here

`.gitattributes` in this repo covers `*.usd`, `*.usda`, `*.png`, `*.jpg` and
`*.pt` only. Ours are kilobytes; routing them through LFS buys nothing and adds
a way for a clone to arrive with pointers and no content — which is precisely
what happened upstream.
