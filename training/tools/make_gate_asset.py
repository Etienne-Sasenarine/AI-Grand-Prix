#!/usr/bin/env python3
"""Build the textured VQ2 gate USD used by the track.

Pipeline
--------
1. ``tools/make_gate_texture.py`` writes ``textures/gate_face_vq2.png``.
2. This script exports a UV'd ring OBJ + albedo, converts it with Isaac Lab's
   MeshConverter (OmniPBR + embedded texture bind), then patches emission so
   the gate reads orange/red instead of washed-out yellow.

Hand-authored PreviewSurface + opacity previously painted the gate black when
the texture failed to bind; MeshConverter's OmniPBR path is what Isaac
actually shades in the viewport.

Run (from repo root, env_isaaclab active)::

    python tools/make_gate_texture.py   # optional, if texture missing
    python tools/make_gate_asset.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(ROOT, "assets", "gate")
MESH = os.path.join(GATE, "vq2_mesh")
TEX_SRC = os.path.join(GATE, "textures", "gate_face_vq2.png")
ALBEDO = os.path.join(MESH, "material_0.png")
OBJ = os.path.join(MESH, "gate_vq2.obj")
USD = os.path.join(GATE, "gate_aigp.usd")

OUTER, INNER, DEPTH = 2.7, 1.5, 0.26
PROUD = 0.004


def _export_obj() -> None:
    import numpy as np
    from PIL import Image
    import trimesh

    os.makedirs(MESH, exist_ok=True)
    img = Image.open(TEX_SRC).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 48, 18, 255))
    albedo = Image.alpha_composite(bg, img).convert("RGB")
    albedo.save(ALBEDO)
    shutil.copy2(ALBEDO, os.path.join(GATE, "textures", "material_0.png"))

    o, i = OUTER / 2, INNER / 2

    def ring(x: float):
        outer = [(-o, -o), (o, -o), (o, o), (-o, o)]
        inner = [(-i, -i), (i, -i), (i, i), (-i, i)]
        verts = [[x, y, z] for y, z in outer + inner]
        quads = (
            [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            if x >= 0
            else [(1, 0, 4, 5), (2, 1, 5, 6), (3, 2, 6, 7), (0, 3, 7, 4)]
        )
        faces = []
        for a, b, c, d in quads:
            faces += [[a, b, c], [a, c, d]]
        uvs = [[(y + o) / OUTER, 1.0 - (z + o) / OUTER] for y, z in outer + inner]
        return np.array(verts), np.array(faces), np.array(uvs)

    vs, fs, uvs = [], [], []
    base = 0
    for x in (DEPTH / 2 + PROUD, -(DEPTH / 2 + PROUD)):
        v, f, uv = ring(x)
        vs.append(v)
        uvs.append(uv)
        fs.append(f + base)
        base += len(v)

    pbr = trimesh.visual.material.PBRMaterial(
        baseColorTexture=albedo,
        baseColorFactor=[1, 1, 1, 1],
        metallicFactor=0.0,
        roughnessFactor=0.4,
    )
    mesh = trimesh.Trimesh(
        vertices=np.vstack(vs),
        faces=np.vstack(fs),
        visual=trimesh.visual.TextureVisuals(uv=np.vstack(uvs), material=pbr),
        process=False,
    )
    mesh.export(OBJ, include_normals=True)
    with open(os.path.join(MESH, "material.mtl"), "w") as fh:
        fh.write(
            "newmtl material_0\n"
            "Ka 1.00000000 0.18800000 0.07000000\n"
            "Kd 1.00000000 0.18800000 0.07000000\n"
            "Ks 0.05000000 0.05000000 0.05000000\n"
            "Ns 20.00000000\n"
            "d 1.00000000\n"
            "illum 2\n"
            "map_Kd material_0.png\n"
        )
    print(f"exported {OBJ}")


def _convert_and_patch() -> None:
    """Must run under Isaac (pxr + MeshConverter)."""
    script = r"""
import os
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from pxr import Usd, Sdf

root = os.environ["GATE_ROOT"]
obj = os.path.join(root, "vq2_mesh", "gate_vq2.obj")
cfg = MeshConverterCfg(
    asset_path=obj,
    usd_dir=root,
    usd_file_name="gate_aigp.usd",
    force_usd_conversion=True,
    make_instanceable=False,
    mass_props=schemas_cfg.MassPropertiesCfg(mass=1.0),
    rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    collision_props=schemas_cfg.CollisionPropertiesCfg(),
    collision_approximation="convexDecomposition",
)
conv = MeshConverter(cfg)
print("USD_PATH", conv.usd_path)

stage = Usd.Stage.Open(conv.usd_path)
shader = stage.GetPrimAtPath("/gate_vq2/geometry/Looks/material_0/material_0")
shader.GetAttribute("inputs:diffuse_texture").Set(Sdf.AssetPath("./textures/material_0.png"))
shader.GetAttribute("inputs:emissive_intensity").Set(800.0)
shader.GetAttribute("inputs:emissive_color").Set((1.0, 0.18, 0.07))
shader.GetAttribute("inputs:diffuse_color_constant").Set((1.0, 0.18, 0.07))
stage.GetRootLayer().Save()
print("patched texture + emissive")
app.close()
"""
    env = os.environ.copy()
    env["GATE_ROOT"] = GATE
    env.setdefault("LD_LIBRARY_PATH", "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", ""))
    # Prefer the isaac lab conda python if available
    py = shutil.which("python") or sys.executable
    subprocess.check_call([py, "-c", script], env=env, cwd=ROOT)


def main() -> None:
    if not os.path.isfile(TEX_SRC):
        print(f"missing {TEX_SRC} — run tools/make_gate_texture.py first")
        sys.exit(1)
    _export_obj()
    _convert_and_patch()
    print(f"ready: {USD}")
    print("track_generator should reference assets/gate/gate_aigp.usd")


if __name__ == "__main__":
    main()
