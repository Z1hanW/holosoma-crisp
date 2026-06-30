#!/usr/bin/env python3
"""Fuse paired CRISP stair motions and OBJ terrains for multi-terrain WBT.

The fused motion stays in each clip's local coordinate frame and carries a
terrain_origins array. MotionCommand uses that array to bind each sampled motion
to its matching translated terrain tile at reset time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REQUIRED_MOTION_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
)

TIME_AXIS_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crisp-root", type=Path, default=Path("crisp_stairs"))
    parser.add_argument("--motion-dir", type=Path)
    parser.add_argument("--geometry-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clips", type=str, default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix", type=str, default="motion_stairs_16_multiterrain")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--floor-margin", type=float, default=2.0)
    parser.add_argument("--floor-top-z", type=float, default=0.0)
    parser.add_argument("--floor-thickness", type=float, default=0.1)
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--cols", type=int, default=0)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_clip_token(token: str, aliases: dict[str, str]) -> str:
    token = token.strip()
    if not token:
        raise ValueError("Empty clip token")
    if token in aliases:
        return aliases[token]
    if token.isdigit():
        candidate = f"stair_{token}"
        if candidate in aliases:
            return aliases[candidate]
        return candidate
    if token.startswith("stair_") or token.endswith("_stairs_up_down"):
        return token
    return token


def resolve_clips(args: argparse.Namespace, manifest: dict[str, Any] | None) -> list[str]:
    aliases: dict[str, str] = {}
    manifest_clips: list[str] = []
    if manifest is not None:
        for clip in manifest.get("clips", []):
            clip_id = str(clip["clip_id"])
            manifest_clips.append(clip_id)
            aliases[clip_id] = clip_id
            requested = clip.get("requested")
            if requested is not None:
                aliases[str(requested)] = clip_id

    if args.clips:
        return [normalize_clip_token(token, aliases) for token in args.clips.replace(",", " ").split()]
    if manifest_clips:
        return manifest_clips
    raise ValueError("No clips provided and no manifest clips found.")


def load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a Trimesh: {type(mesh)}")
    return mesh


def make_floor(bounds: np.ndarray, margin: float, top_z: float, thickness: float) -> trimesh.Trimesh:
    min_corner, max_corner = bounds.astype(np.float64)
    extents = np.array(
        [
            max_corner[0] - min_corner[0] + 2.0 * max(margin, 0.0),
            max_corner[1] - min_corner[1] + 2.0 * max(margin, 0.0),
            max(thickness, 1e-4),
        ],
        dtype=np.float64,
    )
    center = np.array(
        [
            0.5 * (min_corner[0] + max_corner[0]),
            0.5 * (min_corner[1] + max_corner[1]),
            top_z - 0.5 * extents[2],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = center
    return trimesh.creation.box(extents=extents, transform=transform)


def load_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = [key for key in REQUIRED_MOTION_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing required motion keys: {missing}")
        return {key: data[key] for key in data.files}


def validate_motion_compatible(reference: dict[str, np.ndarray], current: dict[str, np.ndarray], path: Path) -> None:
    for key in ("joint_names", "body_names"):
        if not np.array_equal(reference[key], current[key]):
            raise ValueError(f"{path} has different {key}; cannot fuse safely.")
    if not np.array_equal(reference["fps"], current["fps"]):
        raise ValueError(f"{path} has fps {current['fps']} but expected {reference['fps']}.")
    for key in TIME_AXIS_KEYS:
        if reference[key].shape[1:] != current[key].shape[1:]:
            raise ValueError(
                f"{path} has incompatible {key} trailing shape {current[key].shape[1:]}; "
                f"expected {reference[key].shape[1:]}."
            )


def pair_bounds_xy(motion: dict[str, np.ndarray], mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
    motion_min = body_pos[..., :2].reshape(-1, 2).min(axis=0)
    motion_max = body_pos[..., :2].reshape(-1, 2).max(axis=0)
    terrain_min = mesh.bounds[0, :2].astype(np.float64)
    terrain_max = mesh.bounds[1, :2].astype(np.float64)
    return np.minimum(motion_min, terrain_min), np.maximum(motion_max, terrain_max)


def build_layout(bounds: list[tuple[np.ndarray, np.ndarray]], cols: int, margin: float) -> tuple[np.ndarray, float, float, int]:
    spans = np.stack([max_xy - min_xy for min_xy, max_xy in bounds], axis=0)
    pitch_x = float(spans[:, 0].max() + margin)
    pitch_y = float(spans[:, 1].max() + margin)
    n = len(bounds)
    if cols <= 0:
        cols = int(math.ceil(math.sqrt(n)))
    origins = []
    for idx in range(n):
        row, col = divmod(idx, cols)
        origins.append([col * pitch_x, row * pitch_y, 0.0])
    return np.asarray(origins, dtype=np.float32), pitch_x, pitch_y, cols


def main() -> None:
    args = parse_args()
    crisp_root = args.crisp_root
    manifest_path = args.manifest or crisp_root / "terrain_traversal_manifest.json"
    manifest = load_manifest(manifest_path)
    clips = resolve_clips(args, manifest)

    motion_dir = args.motion_dir or crisp_root / "___crisp_clean_motion"
    geometry_dir = args.geometry_dir or crisp_root / "___crisp_clean_geometry"
    output_dir = args.output_dir or crisp_root / "_fused"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for clip_id in clips:
        motion_path = motion_dir / f"{clip_id}.npz"
        terrain_path = geometry_dir / f"{clip_id}.obj"
        if not motion_path.exists():
            raise FileNotFoundError(f"Missing motion file for {clip_id}: {motion_path}")
        if not terrain_path.exists():
            raise FileNotFoundError(f"Missing terrain OBJ for {clip_id}: {terrain_path}")

        motion = load_motion(motion_path)
        mesh = load_trimesh(terrain_path)
        records.append(
            {
                "clip_id": clip_id,
                "motion_path": motion_path,
                "terrain_path": terrain_path,
                "motion": motion,
                "mesh": mesh,
                "bounds_xy": pair_bounds_xy(motion, mesh),
            }
        )

    reference_motion = records[0]["motion"]
    for record in records[1:]:
        validate_motion_compatible(reference_motion, record["motion"], record["motion_path"])

    terrain_origins, pitch_x, pitch_y, cols = build_layout(
        [record["bounds_xy"] for record in records],
        cols=args.cols,
        margin=args.margin,
    )

    save_motion: dict[str, np.ndarray] = {}
    for key in TIME_AXIS_KEYS:
        save_motion[key] = np.concatenate([record["motion"][key] for record in records], axis=0)
    save_motion["fps"] = reference_motion["fps"]
    save_motion["joint_names"] = reference_motion["joint_names"]
    save_motion["body_names"] = reference_motion["body_names"]

    total_frames = save_motion["joint_pos"].shape[0]
    motion_ends = np.zeros(total_frames, dtype=bool)
    cursor = 0
    motion_lengths = []
    for record in records:
        length = int(record["motion"]["joint_pos"].shape[0])
        cursor += length
        motion_lengths.append(length)
        motion_ends[cursor - 1] = True
    save_motion["motion_ends"] = motion_ends
    save_motion["motion_names"] = np.asarray([record["clip_id"] for record in records])
    save_motion["terrain_origins"] = terrain_origins

    fused_parts = []
    metadata_clips = []
    for idx, record in enumerate(records):
        origin = terrain_origins[idx].astype(np.float64)
        mesh = record["mesh"].copy()
        mesh.apply_translation(origin)
        fused_parts.append(mesh)
        if not args.no_floor:
            fused_parts.append(
                make_floor(
                    mesh.bounds,
                    margin=args.floor_margin,
                    top_z=args.floor_top_z,
                    thickness=args.floor_thickness,
                )
            )

        metadata_clips.append(
            {
                "motion_id": idx,
                "clip_id": record["clip_id"],
                "motion_file": str(record["motion_path"]),
                "terrain_file": str(record["terrain_path"]),
                "motion_length": motion_lengths[idx],
                "terrain_origin": terrain_origins[idx].astype(float).tolist(),
                "local_bounds_xy": [
                    record["bounds_xy"][0].astype(float).tolist(),
                    record["bounds_xy"][1].astype(float).tolist(),
                ],
                "translated_mesh_bounds": mesh.bounds.astype(float).tolist(),
            }
        )

    fused_mesh = trimesh.util.concatenate(fused_parts)
    motion_out = output_dir / f"{args.prefix}.npz"
    terrain_out = output_dir / f"{args.prefix}.obj"
    metadata_out = output_dir / f"{args.prefix}.json"

    np.savez_compressed(motion_out, **save_motion)
    fused_mesh.export(terrain_out)
    metadata = {
        "schema_version": 1,
        "format": "holosoma_crisp_stairs_multiterrain_fused",
        "clip_count": len(records),
        "total_frames": int(total_frames),
        "cols": int(cols),
        "pitch": [pitch_x, pitch_y],
        "margin": float(args.margin),
        "floor": {
            "enabled": not args.no_floor,
            "margin": float(args.floor_margin),
            "top_z": float(args.floor_top_z),
            "thickness": float(args.floor_thickness),
        },
        "motion_file": str(motion_out),
        "terrain_file": str(terrain_out),
        "clips": metadata_clips,
        "mesh": {
            "vertices": int(len(fused_mesh.vertices)),
            "faces": int(len(fused_mesh.faces)),
            "bounds": fused_mesh.bounds.astype(float).tolist(),
        },
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote fused motion: {motion_out}")
    print(f"Wrote fused terrain: {terrain_out}")
    print(f"Wrote metadata: {metadata_out}")
    print(f"clips={len(records)} total_frames={total_frames} mesh_faces={len(fused_mesh.faces)}")


if __name__ == "__main__":
    main()
