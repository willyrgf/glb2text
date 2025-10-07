#!/usr/bin/env python3
"""
GLB Text Extractor
------------------

Reads a .glb or .gltf file and prints a human-readable text (or Markdown) report
with scenes, nodes, meshes, materials, skins (skeletons), and animations.

Dependencies:
    pip install pygltflib

Usage:
    python glb_text_extractor.py path/to/model.glb [-o report.txt] [--markdown] [--max-depth N] [--verbose]

Notes:
- This script does not require Blender. It inspects the GLB/GLTF structure directly.
- Triangle/vertex counts are computed from accessors and primitive modes; they are estimates
  and assume default glTF semantics.
- Animation durations are read from the input accessor's `max` when available.
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pygltflib import (
    GLTF2,
    BufferView,
    Accessor,
    Node,
    Mesh,
    Skin,
    Animation,
    PbrMetallicRoughness,
)

PRIM_MODES = {
    0: "POINTS",
    1: "LINES",
    2: "LINE_LOOP",
    3: "LINE_STRIP",
    4: "TRIANGLES",
    5: "TRIANGLE_STRIP",
    6: "TRIANGLE_FAN",
}

@dataclass
class Ctx:
    gltf: GLTF2
    node_name: Dict[int, str]


def idx_or_none(v):
    return None if v is None else int(v)


def safe_name(name: Optional[str], fallback: str) -> str:
    return name if (name and name.strip()) else fallback


def material_name(gltf: GLTF2, idx: Optional[int]) -> str:
    if idx is None:
        return "<none>"
    m = gltf.materials[idx]
    return safe_name(m.name, f"material[{idx}]")


def accessor(gltf: GLTF2, idx: Optional[int]) -> Optional[Accessor]:
    if idx is None:
        return None
    return gltf.accessors[idx]


def accessor_len(gltf: GLTF2, idx: Optional[int]) -> Optional[int]:
    acc = accessor(gltf, idx)
    return None if acc is None else int(acc.count)


def node_name_by_index(gltf: GLTF2, idx: Optional[int]) -> str:
    if idx is None:
        return "<none>"
    n = gltf.nodes[idx]
    return safe_name(n.name, f"node[{idx}]")


def mesh_name_by_index(gltf: GLTF2, idx: Optional[int]) -> str:
    if idx is None:
        return "<none>"
    m = gltf.meshes[idx]
    return safe_name(m.name, f"mesh[{idx}]")


def skin_name_by_index(gltf: GLTF2, idx: Optional[int]) -> str:
    if idx is None:
        return "<none>"
    s = gltf.skins[idx]
    return safe_name(s.name, f"skin[{idx}]")


def primitive_vertex_count(gltf: GLTF2, prim) -> Optional[int]:
    # POSITION accessor count is the number of vertices in this primitive
    pos_idx = getattr(prim.attributes, "POSITION", None) if prim.attributes else None
    return accessor_len(gltf, idx_or_none(pos_idx))


def primitive_triangle_count(gltf: GLTF2, prim) -> Optional[int]:
    mode = getattr(prim, "mode", 4) or 4  # default TRIANGLES
    mode_name = PRIM_MODES.get(mode, f"UNKNOWN({mode})")
    # Prefer indices accessor length; otherwise fall back to POSITION-based estimation
    indices_len = accessor_len(gltf, idx_or_none(prim.indices))
    if mode_name == "TRIANGLES":
        if indices_len is not None:
            return indices_len // 3
        verts = primitive_vertex_count(gltf, prim)
        return None if verts is None else verts // 3
    elif mode_name in ("TRIANGLE_STRIP", "TRIANGLE_FAN"):
        n = indices_len if indices_len is not None else primitive_vertex_count(gltf, prim)
        return None if (n is None or n < 3) else n - 2
    else:
        return None


def pbr_summary(pbr: Optional[PbrMetallicRoughness]) -> str:
    if pbr is None:
        return ""
    out = []
    if pbr.baseColorFactor is not None:
        out.append(f"baseColorFactor={tuple(pbr.baseColorFactor)}")
    if pbr.metallicFactor is not None:
        out.append(f"metallic={pbr.metallicFactor}")
    if pbr.roughnessFactor is not None:
        out.append(f"roughness={pbr.roughnessFactor}")
    if pbr.baseColorTexture is not None and pbr.baseColorTexture.index is not None:
        out.append(f"baseColorTex=texture[{pbr.baseColorTexture.index}]")
    if pbr.metallicRoughnessTexture is not None and pbr.metallicRoughnessTexture.index is not None:
        out.append(f"metallicRoughnessTex=texture[{pbr.metallicRoughnessTexture.index}]")
    return ", ".join(out)


def indent(s: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in s.splitlines())


def summarize_scenes(ctx: Ctx) -> str:
    gltf = ctx.gltf
    lines = ["Scenes:"]
    if not gltf.scenes:
        lines.append("  <none>")
        return "\n".join(lines)
    for si, scene in enumerate(gltf.scenes):
        sname = safe_name(scene.name, f"scene[{si}]")
        roots = scene.nodes or []
        lines.append(f"  - {sname} (root nodes: {len(roots)})")
        for ni in roots:
            lines.append(indent(f"• {node_name_by_index(gltf, ni)} [node {ni}]", 6))
    return "\n".join(lines)


def node_transform_str(n: Node) -> str:
    parts = []
    if n.matrix:
        parts.append("matrix=[...] (4x4)")
    if n.translation:
        parts.append(f"T={tuple(round(x, 6) for x in n.translation)}")
    if n.rotation:
        parts.append(f"R(quat)={tuple(round(x, 6) for x in n.rotation)}")
    if n.scale:
        parts.append(f"S={tuple(round(x, 6) for x in n.scale)}")
    return ", ".join(parts) if parts else "<identity>"


def summarize_hierarchy(ctx: Ctx, max_depth: int = 3) -> str:
    gltf = ctx.gltf
    lines = ["Node Hierarchy (truncated):"]
    # Build parent map and children list
    children_map: Dict[int, List[int]] = {i: [] for i in range(len(gltf.nodes or []))}
    for i, n in enumerate(gltf.nodes or []):
        for c in (n.children or []):
            children_map[i].append(c)
    # Roots are any nodes referenced by scenes' root nodes
    root_set = set()
    for scene in (gltf.scenes or []):
        for ni in (scene.nodes or []):
            root_set.add(ni)

    def walk(i: int, depth: int):
        prefix = "  " * depth + ("- " if depth else "")
        n = gltf.nodes[i]
        tag = []
        if n.mesh is not None:
            tag.append(f"mesh={mesh_name_by_index(gltf, n.mesh)}")
        if n.skin is not None:
            tag.append(f"skin={skin_name_by_index(gltf, n.skin)}")
        if n.camera is not None:
            tag.append(f"camera[{n.camera}]")
        xform = node_transform_str(n)
        lines.append(f"{prefix}{safe_name(n.name, f'node[{i}]')} ({', '.join(tag) if tag else 'node'}) | {xform}")
        if depth + 1 >= max_depth:
            if children_map.get(i):
                lines.append("  " * (depth + 1) + f"… ({len(children_map[i])} child nodes)")
            return
        for c in children_map.get(i, []):
            walk(c, depth + 1)

    for r in sorted(root_set):
        walk(r, 0)
    return "\n".join(lines)


def summarize_meshes(ctx: Ctx) -> str:
    gltf = ctx.gltf
    lines = ["Meshes:"]
    if not gltf.meshes:
        lines.append("  <none>")
        return "\n".join(lines)
    for mi, m in enumerate(gltf.meshes):
        mname = safe_name(m.name, f"mesh[{mi}]")
        lines.append(f"  - {mname} (primitives: {len(m.primitives)})")
        for pi, prim in enumerate(m.primitives):
            mode = PRIM_MODES.get(getattr(prim, "mode", 4) or 4, "UNKNOWN")
            # prim.attributes is an Attributes object, convert to dict to get keys
            if prim.attributes:
                attr_dict = {k: v for k, v in vars(prim.attributes).items() if v is not None}
                attrs = ", ".join(sorted(attr_dict.keys()))
            else:
                attrs = "<none>"
            v = primitive_vertex_count(gltf, prim)
            t = primitive_triangle_count(gltf, prim)
            mat = material_name(gltf, idx_or_none(prim.material))
            morphs = len(prim.targets) if prim.targets else 0
            lines.append(
                indent(
                    f"[{pi}] mode={mode}, material={mat}, vertices={v if v is not None else '?'}, "
                    f"triangles={t if t is not None else '?'}, attributes=[{attrs}], morphTargets={morphs}",
                    6,
                )
            )
    return "\n".join(lines)


def summarize_materials(ctx: Ctx) -> str:
    gltf = ctx.gltf
    lines = ["Materials:"]
    if not gltf.materials:
        lines.append("  <none>")
        return "\n".join(lines)
    for i, m in enumerate(gltf.materials):
        name = safe_name(m.name, f"material[{i}]")
        pbr = pbr_summary(m.pbrMetallicRoughness)
        alpha = getattr(m, "alphaMode", None)
        dbl = getattr(m, "doubleSided", False)
        lines.append(f"  - {name} | alpha={alpha}, doubleSided={dbl}{', ' + pbr if pbr else ''}")
    return "\n".join(lines)


def summarize_textures_images(ctx: Ctx) -> str:
    gltf = ctx.gltf
    lines = ["Textures & Images:"]
    if not gltf.textures and not gltf.images:
        lines.append("  <none>")
        return "\n".join(lines)
    if gltf.textures:
        lines.append(f"  Textures ({len(gltf.textures)}):")
        for ti, t in enumerate(gltf.textures):
            img = gltf.images[t.source] if t.source is not None else None
            imsg = ""
            if img is not None:
                src = f"image[{t.source}]"
                if img.mimeType:
                    src += f" ({img.mimeType})"
                imsg = f" -> {src}"
            lines.append(f"    - texture[{ti}] sampler={t.sampler if t.sampler is not None else '<default>'}{imsg}")
    if gltf.images:
        lines.append(f"  Images ({len(gltf.images)}):")
        for ii, img in enumerate(gltf.images):
            src = "uri=" + img.uri if img.uri else ("bufferView=" + str(img.bufferView) if img.bufferView is not None else "<embedded>")
            mt = f", mime={img.mimeType}" if img.mimeType else ""
            lines.append(f"    - image[{ii}] {src}{mt}")
    return "\n".join(lines)


def summarize_skins(ctx: Ctx) -> str:
    gltf = ctx.gltf
    lines = ["Skins (Skeletons):"]
    if not gltf.skins:
        lines.append("  <none>")
        return "\n".join(lines)
    for si, s in enumerate(gltf.skins):
        name = safe_name(s.name, f"skin[{si}]")
        root = node_name_by_index(gltf, idx_or_none(s.skeleton))
        ibm = accessor_len(gltf, idx_or_none(s.inverseBindMatrices))
        lines.append(f"  - {name} (root: {root}, joints: {len(s.joints or [])}, inverseBindMatrices: {ibm if ibm is not None else 'n/a'})")
        for j in (s.joints or []):
            lines.append(indent(f"• {node_name_by_index(gltf, j)} [node {j}]", 6))
    return "\n".join(lines)


def sampler_input_max_time(gltf: GLTF2, accessor_index: Optional[int]) -> Optional[float]:
    if accessor_index is None:
        return None
    acc = gltf.accessors[accessor_index]
    if acc.max is not None and len(acc.max) > 0:
        # For time inputs, max is a scalar [tmax]
        try:
            return float(acc.max[0])
        except Exception:
            pass
    return None  # Fallback requires decoding buffers; omitted for simplicity


def summarize_animations(ctx: Ctx) -> str:
    gltf = ctx.gltf
    lines = ["Animations:"]
    if not gltf.animations:
        lines.append("  <none>")
        return "\n".join(lines)
    for ai, a in enumerate(gltf.animations):
        name = safe_name(a.name, f"animation[{ai}]")
        # Compute rough duration from max of all sampler inputs
        durations = []
        for s in (a.samplers or []):
            tmax = sampler_input_max_time(gltf, idx_or_none(s.input))
            if tmax is not None:
                durations.append(tmax)
        dur = max(durations) if durations else None
        lines.append(f"  - {name} (channels: {len(a.channels)}, samplers: {len(a.samplers)}, duration: {dur if dur is not None else '?'}s)")
        for ci, ch in enumerate(a.channels or []):
            tgt = ch.target
            node = node_name_by_index(gltf, idx_or_none(tgt.node))
            path = tgt.path if tgt and tgt.path else "?"
            samp = a.samplers[idx_or_none(ch.sampler)] if ch.sampler is not None else None
            interp = samp.interpolation if samp and samp.interpolation else "LINEAR"
            in_len = accessor_len(gltf, idx_or_none(samp.input)) if samp else None
            out_len = accessor_len(gltf, idx_or_none(samp.output)) if samp else None
            lines.append(
                indent(
                    f"[{ci}] target={node}.{path} | sampler={interp}, keys={in_len if in_len is not None else '?'}, values={out_len if out_len is not None else '?'}",
                    6,
                )
            )
    return "\n".join(lines)


def summarize_extensions(ctx: Ctx) -> str:
    gltf = ctx.gltf
    used = ", ".join(gltf.extensionsUsed or []) if gltf.extensionsUsed else "<none>"
    reqd = ", ".join(gltf.extensionsRequired or []) if gltf.extensionsRequired else "<none>"
    return f"Extensions: used=[{used}], required=[{reqd}]"


def build_report(gltf: GLTF2, markdown: bool, max_depth: int) -> str:
    ctx = Ctx(gltf=gltf, node_name={})

    sections = [
        ("File", [
            f"generator: {getattr(gltf.asset, 'generator', '<unknown>')}",
            f"version:   {getattr(gltf.asset, 'version', '<unknown>')}",
        ]),
        ("Extensions", [summarize_extensions(ctx)]),
        ("Scenes", [summarize_scenes(ctx)]),
        ("Hierarchy", [summarize_hierarchy(ctx, max_depth=max_depth)]),
        ("Meshes", [summarize_meshes(ctx)]),
        ("Materials", [summarize_materials(ctx)]),
        ("Textures & Images", [summarize_textures_images(ctx)]),
        ("Skins", [summarize_skins(ctx)]),
        ("Animations", [summarize_animations(ctx)]),
    ]

    if markdown:
        out = ["# GLB Report", ""]
        for title, blocks in sections:
            out.append(f"## {title}")
            for b in blocks:
                out.append("\n" + b + "\n")
        return "\n".join(out)
    else:
        out = []
        sep = "=" * 72
        for title, blocks in sections:
            out.append(sep)
            out.append(title.upper())
            out.append(sep)
            for b in blocks:
                out.append(b)
            out.append("")
        return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Extract a text report from a GLB/GLTF file.")
    p.add_argument("input", help="Path to .glb or .gltf")
    p.add_argument("-o", "--output", help="Write report to this file (otherwise prints to stdout)")
    p.add_argument("--markdown", action="store_true", help="Emit Markdown instead of plain text")
    p.add_argument("--max-depth", type=int, default=3, help="Max depth when printing node hierarchy (default: 3)")
    p.add_argument("--verbose", action="store_true", help="Print extra info to stderr")
    args = p.parse_args(argv)

    gltf = GLTF2().load(args.input)

    if args.verbose:
        sys.stderr.write(f"Loaded {args.input}: scenes={len(gltf.scenes or [])}, nodes={len(gltf.nodes or [])}, "
                         f"meshes={len(gltf.meshes or [])}, skins={len(gltf.skins or [])}, animations={len(gltf.animations or [])}\n")

    report = build_report(gltf, markdown=args.markdown, max_depth=args.max_depth)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        if args.verbose:
            sys.stderr.write(f"Wrote {args.output}\n")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

