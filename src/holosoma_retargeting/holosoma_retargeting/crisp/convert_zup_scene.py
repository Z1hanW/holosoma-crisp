from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - exercised only in missing-env setup
    raise SystemExit("trimesh is required. Activate the holosoma/CRISP python env first.") from exc


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_URDF = PACKAGE_ROOT / "models" / "g1" / "g1_29dof_spherehand.urdf"
DEFAULT_ROBOT_XML = PACKAGE_ROOT / "models" / "g1" / "g1_29dof_spherehand.xml"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "demo_data" / "crisp_terrain"


@dataclass(frozen=True)
class ConvertedSequence:
    sequence: str
    source_scene_dir: Path
    output_dir: Path
    piece_count: int
    scene_xml: Path
    object_urdf: Path
    object_mesh: Path
    motion_file: Path | None
    bounds_min: list[float]
    bounds_max: list[float]
    motion_metadata: dict | None = None


@dataclass(frozen=True)
class SceneTransform:
    rotation: np.ndarray
    translation: np.ndarray
    source_root: Path
    rotation_path: Path
    translation_path: Path


def _relpath(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def _natural_key(path: Path) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    token = ""
    for char in path.name:
        if char.isdigit():
            token += char
        else:
            if token:
                parts.append((1, int(token)))
                token = ""
            parts.append((0, char.lower()))
    if token:
        parts.append((1, int(token)))
    return tuple(parts)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geometries = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh from {path}, got {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Empty mesh in {path}")
    return mesh


def _find_scene_dir(root: Path, sequence: str, hmr_key: str) -> Path:
    names = [sequence]
    if sequence.isdigit():
        names.append(f"stair_{sequence}")
    elif not sequence.startswith("stair_"):
        suffix = sequence.split("_")[-1]
        if suffix.isdigit():
            names.append(f"stair_{suffix}")

    candidates: list[Path] = []
    for name in dict.fromkeys(names):
        seq_dir = root / name
        candidates.extend(
            [
                seq_dir / hmr_key / "scene_mesh_sqs",
                seq_dir / "scene_mesh_sqs",
                seq_dir,
            ]
        )

    for candidate in candidates:
        pieces = candidate / "pieces"
        if pieces.is_dir() and any(pieces.glob("*.obj")):
            return candidate.resolve()

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find CRISP pieces for sequence {sequence!r}. Tried:\n{tried}")


def _discover_scene_dirs(root: Path, hmr_key: str) -> list[tuple[str, Path]]:
    if (root / "pieces").is_dir() and any((root / "pieces").glob("*.obj")):
        return [(root.parent.name, root.resolve())]

    found: list[tuple[str, Path]] = []
    for seq_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=_natural_key):
        for candidate in (seq_dir / hmr_key / "scene_mesh_sqs", seq_dir / "scene_mesh_sqs", seq_dir):
            pieces = candidate / "pieces"
            if pieces.is_dir() and any(pieces.glob("*.obj")):
                found.append((seq_dir.name, candidate.resolve()))
                break
    if not found:
        raise FileNotFoundError(f"No CRISP z-up scene_mesh_sqs/pieces folders found under {root}")
    return found


def _sequence_name_from_scene_dir(scene_dir: Path) -> str:
    if scene_dir.name == "scene_mesh_sqs":
        if scene_dir.parent.name in {"gv", "hmr", "vggt_omega"}:
            return scene_dir.parent.parent.name
        return scene_dir.parent.name
    return scene_dir.name


def _load_scene_transform(scene_dir: Path) -> SceneTransform | None:
    candidates = []
    if scene_dir.name == "scene_mesh_sqs":
        candidates.append(scene_dir.parent)
    candidates.extend([scene_dir, scene_dir.parent, scene_dir.parent.parent])

    for root in dict.fromkeys(path.resolve() for path in candidates):
        rotation_path = root / "world_rotation.npy"
        translation_path = root / "shared_translation.txt"
        if not rotation_path.is_file() or not translation_path.is_file():
            continue
        rotation = np.asarray(np.load(rotation_path), dtype=np.float32)
        translation = np.asarray(np.loadtxt(translation_path), dtype=np.float32).reshape(-1)
        if rotation.shape != (3, 3):
            raise ValueError(f"Bad world_rotation shape in {rotation_path}: {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"Bad shared_translation shape in {translation_path}: {translation.shape}")
        return SceneTransform(
            rotation=rotation,
            translation=translation,
            source_root=root,
            rotation_path=rotation_path,
            translation_path=translation_path,
        )
    return None


def _copy_scene_transform(scene_transform: SceneTransform | None, output_dir: Path) -> None:
    if scene_transform is None:
        return
    shutil.copy2(scene_transform.rotation_path, output_dir / "world_rotation.npy")
    shutil.copy2(scene_transform.translation_path, output_dir / "shared_translation.txt")
    np.savetxt(output_dir / "world_rotation.txt", scene_transform.rotation, fmt="%.8f")


def _robot_meshdir_abs(robot_xml: Path) -> Path | None:
    root = ET.parse(robot_xml).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        return None
    meshdir = compiler.get("meshdir")
    if not meshdir:
        return None
    meshdir_path = Path(meshdir)
    if meshdir_path.is_absolute():
        return meshdir_path.resolve()
    return (robot_xml.parent / meshdir_path).resolve()


PieceMesh = tuple[str, Path, trimesh.Trimesh]


def _copy_and_load_pieces(scene_dir: Path, output_dir: Path, overwrite: bool) -> list[PieceMesh]:
    src_piece_paths = sorted((scene_dir / "pieces").glob("*.obj"), key=_natural_key)
    if not src_piece_paths:
        raise FileNotFoundError(f"No OBJ pieces found in {scene_dir / 'pieces'}")

    dst_piece_dir = output_dir / "pieces"
    if dst_piece_dir.exists() and overwrite:
        shutil.rmtree(dst_piece_dir)
    dst_piece_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[PieceMesh] = []
    for idx, src_path in enumerate(src_piece_paths):
        name = f"piece_{idx:03d}"
        dst_path = dst_piece_dir / f"{name}.obj"
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"{dst_path} exists. Pass --overwrite to regenerate.")
        shutil.copy2(src_path, dst_path)
        pieces.append((name, dst_path, _load_mesh(dst_path)))
    return pieces


def _write_combined_mesh(pieces: Iterable[PieceMesh], output_path: Path) -> tuple[list[float], list[float]]:
    meshes = [mesh.copy() for _, _, mesh in pieces]
    combined = trimesh.util.concatenate(meshes)
    combined.export(output_path)
    bounds = np.asarray(combined.bounds, dtype=float)
    return bounds[0].tolist(), bounds[1].tolist()


def _palette(index: int) -> tuple[float, float, float, float]:
    colors = (
        (0.30, 0.55, 0.85, 0.65),
        (0.86, 0.44, 0.34, 0.65),
        (0.38, 0.68, 0.43, 0.65),
        (0.82, 0.64, 0.28, 0.65),
        (0.58, 0.48, 0.82, 0.65),
        (0.35, 0.72, 0.72, 0.65),
    )
    return colors[index % len(colors)]


def _write_box_assets(
    pieces: list[PieceMesh],
    output_path: Path,
    mesh_file_base: Path,
) -> None:
    lines = ["<mujocoinclude>"]
    for idx, (name, piece_path, _) in enumerate(pieces):
        mesh_file = _relpath(piece_path, mesh_file_base)
        lines.append(f'  <mesh name="{name}" file="{escape(mesh_file)}" scale="1 1 1"/>')
        rgba = " ".join(f"{value:.3f}" for value in _palette(idx))
        lines.append(f'  <material name="{name}_material" rgba="{rgba}"/>')
    lines.append("</mujocoinclude>")
    output_path.write_text("\n".join(lines) + "\n")


def _write_box_body(pieces: list[PieceMesh], output_path: Path) -> None:
    lines = ["<mujocoinclude>"]
    for name, _, _ in pieces:
        lines.extend(
            [
                f'  <body name="multi_boxes_{name}_link" pos="0 0 0" quat="1 0 0 0">',
                (
                    f'    <geom name="multi_boxes_{name}" type="mesh" mesh="{name}" '
                    f'pos="0 0 0" quat="1 0 0 0" material="{name}_material" '
                    'contype="1" conaffinity="1" friction="1 0.005 0.0001"/>'
                ),
                "  </body>",
            ]
        )
    lines.append("</mujocoinclude>")
    output_path.write_text("\n".join(lines) + "\n")


def _write_urdf(pieces: list[PieceMesh], output_path: Path) -> None:
    lines = ['<?xml version="1.0"?>', '<robot name="multi_boxes">', '  <link name="world"/>']
    mass = 1.0 / max(len(pieces), 1)
    for idx, (name, piece_path, _) in enumerate(pieces):
        mesh_file = Path("pieces") / piece_path.name
        rgba = " ".join(f"{value:.3f}" for value in _palette(idx))
        lines.extend(
            [
                f'  <link name="{name}_link">',
                "    <visual>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                "      <geometry>",
                f'        <mesh filename="{mesh_file.as_posix()}" scale="1 1 1"/>',
                "      </geometry>",
                f'      <material name="{name}_material">',
                f'        <color rgba="{rgba}"/>',
                "      </material>",
                "    </visual>",
                f'    <collision name="{name}">',
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                "      <geometry>",
                f'        <mesh filename="{mesh_file.as_posix()}" scale="1 1 1"/>',
                "      </geometry>",
                "    </collision>",
                "    <inertial>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <mass value="{mass:.8f}"/>',
                '      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>',
                "    </inertial>",
                "  </link>",
                f'  <joint name="world_to_{name}" type="fixed">',
                '    <parent link="world"/>',
                f'    <child link="{name}_link"/>',
                '    <origin xyz="0 0 0" rpy="0 0 0"/>',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    output_path.write_text("\n".join(lines) + "\n")


def _write_scene_xml(robot_xml: Path, robot_urdf: Path, object_name: str, output_dir: Path) -> Path:
    tree = ET.parse(robot_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir_abs = _robot_meshdir_abs(robot_xml)
    if compiler is not None and meshdir_abs is not None:
        compiler.set("meshdir", _relpath(meshdir_abs, output_dir))

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    asset.append(ET.Element("include", {"file": "box_assets.xml"}))

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    worldbody.append(ET.Element("include", {"file": "box_body.xml"}))

    scene_name = robot_urdf.name.replace(".urdf", f"_w_{object_name}.xml")
    scene_path = output_dir / scene_name
    ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="utf-8", xml_declaration=False)
    return scene_path


def _copy_motion(sequence: str, output_dir: Path, motion_root: Path | None, motion_glob: str) -> Path | None:
    if motion_root is None:
        return None

    candidates: list[Path] = []
    candidates.extend(sorted((motion_root / sequence).glob(motion_glob.format(sequence=sequence))))
    candidates.extend(sorted(motion_root.glob(motion_glob.format(sequence=sequence))))
    candidates.extend([motion_root / sequence / f"{sequence}.npy", motion_root / f"{sequence}.npy"])

    for candidate in candidates:
        if candidate.is_file():
            dst = output_dir / candidate.name
            if candidate.resolve() != dst.resolve():
                shutil.copy2(candidate, dst)
            return dst
    return None


def _find_crisp_smplx_motion(sequence: str, crisp_hmr_root: Path, hmr_key: str, motion_filename: str) -> Path | None:
    candidates = [
        crisp_hmr_root / sequence / hmr_key / "hmr" / motion_filename,
        crisp_hmr_root / sequence / "hmr" / motion_filename,
        crisp_hmr_root / sequence / motion_filename,
    ]
    if sequence.startswith("stair_"):
        numeric = sequence.removeprefix("stair_")
        candidates.extend(
            [
                crisp_hmr_root / numeric / hmr_key / "hmr" / motion_filename,
                crisp_hmr_root / numeric / "hmr" / motion_filename,
                crisp_hmr_root / numeric / motion_filename,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _write_crisp_smplx_motion(
    sequence: str,
    output_dir: Path,
    crisp_hmr_root: Path | None,
    hmr_key: str,
    source_filename: str,
    output_filename: str,
    scene_transform: SceneTransform | None,
) -> tuple[Path | None, dict | None]:
    if crisp_hmr_root is None:
        return None, None

    source_path = _find_crisp_smplx_motion(sequence, crisp_hmr_root, hmr_key, source_filename)
    if source_path is None:
        return None, None
    if scene_transform is None:
        raise FileNotFoundError(
            f"Found CRISP HMR motion for {sequence} at {source_path}, but could not find "
            "world_rotation.npy/shared_translation.txt next to the z-up scene. Refusing to "
            "write unaligned motion."
        )

    source_data = np.load(source_path)
    if "global_joint_positions" not in source_data:
        raise KeyError(f"{source_path} does not contain global_joint_positions")

    joints_raw = np.asarray(source_data["global_joint_positions"], dtype=np.float32)
    if joints_raw.ndim != 3 or joints_raw.shape[-1] != 3:
        raise ValueError(f"Expected global_joint_positions shape (T, J, 3), got {joints_raw.shape} in {source_path}")

    joints = joints_raw @ scene_transform.rotation.T + scene_transform.translation.reshape(1, 1, 3)

    output_name = output_filename.format(sequence=sequence)
    if not output_name.endswith(".npy"):
        output_name += ".npy"
    output_path = output_dir / output_name
    np.save(output_path, joints)

    z_extent_per_frame = joints[:, :, 2].max(axis=1) - joints[:, :, 2].min(axis=1)
    metadata = {
        "source": str(source_path),
        "output": output_path.name,
        "format": "smplx",
        "shape": list(joints.shape),
        "coordinate_policy": "transformed_to_zup_scene_frame_no_scale",
        "transform": {
            "formula": "joints_zup = joints_raw @ world_rotation.T + shared_translation",
            "world_rotation": scene_transform.rotation.tolist(),
            "shared_translation": scene_transform.translation.tolist(),
            "source_root": str(scene_transform.source_root),
            "world_rotation_file": str(scene_transform.rotation_path),
            "shared_translation_file": str(scene_transform.translation_path),
        },
        "source_height_key": float(source_data["height"]) if "height" in source_data else None,
        "median_frame_z_extent": float(np.median(z_extent_per_frame)),
        "mean_frame_z_extent": float(np.mean(z_extent_per_frame)),
        "note": (
            "source_height_key is the source file value and may describe trajectory z range, not body height; "
            "this converter does not use it to scale the motion."
        ),
    }
    (output_dir / "crisp_motion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_path, metadata


def convert_sequence(
    sequence: str,
    scene_dir: Path,
    output_root: Path,
    robot_xml: Path,
    robot_urdf: Path,
    object_name: str,
    overwrite: bool,
    motion_root: Path | None,
    motion_glob: str,
    crisp_hmr_root: Path | None,
    hmr_key: str,
    crisp_hmr_motion_filename: str,
    motion_output_filename: str,
    require_motion: bool,
) -> ConvertedSequence:
    output_dir = output_root / sequence
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = [
        output_dir / "box_assets.xml",
        output_dir / "box_body.xml",
        output_dir / f"{object_name}.obj",
        output_dir / f"{object_name}.urdf",
        output_dir / robot_urdf.name.replace(".urdf", f"_w_{object_name}.xml"),
        output_dir / "manifest.json",
    ]
    if overwrite:
        for path in generated_paths:
            if path.exists():
                path.unlink()

    scene_transform = _load_scene_transform(scene_dir)
    _copy_scene_transform(scene_transform, output_dir)

    pieces = _copy_and_load_pieces(scene_dir, output_dir, overwrite)
    object_mesh = output_dir / f"{object_name}.obj"
    bounds_min, bounds_max = _write_combined_mesh(pieces, object_mesh)

    mesh_file_base = _robot_meshdir_abs(robot_xml) or output_dir
    _write_box_assets(pieces, output_dir / "box_assets.xml", mesh_file_base)
    _write_box_body(pieces, output_dir / "box_body.xml")

    object_urdf = output_dir / f"{object_name}.urdf"
    _write_urdf(pieces, object_urdf)

    scene_xml = _write_scene_xml(robot_xml, robot_urdf, object_name, output_dir)
    motion_file = _copy_motion(sequence, output_dir, motion_root, motion_glob)
    motion_metadata = None
    if motion_file is None:
        motion_file, motion_metadata = _write_crisp_smplx_motion(
            sequence=sequence,
            output_dir=output_dir,
            crisp_hmr_root=crisp_hmr_root,
            hmr_key=hmr_key,
            source_filename=crisp_hmr_motion_filename,
            output_filename=motion_output_filename,
            scene_transform=scene_transform,
        )
    if require_motion and motion_file is None:
        raise FileNotFoundError(
            f"No motion found for {sequence}. Checked --motion-root={motion_root} "
            f"and --crisp-hmr-root={crisp_hmr_root}"
        )

    converted = ConvertedSequence(
        sequence=sequence,
        source_scene_dir=scene_dir,
        output_dir=output_dir,
        piece_count=len(pieces),
        scene_xml=scene_xml,
        object_urdf=object_urdf,
        object_mesh=object_mesh,
        motion_file=motion_file,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        motion_metadata=motion_metadata,
    )

    manifest = {
        "sequence": converted.sequence,
        "source_scene_dir": str(converted.source_scene_dir),
        "coordinate_frame": "z-up",
        "geometry_policy": "faithful_copy_no_rotation_no_translation_no_viewer_scale",
        "scene_transform": (
            {
                "formula": "points_zup = points_raw @ world_rotation.T + shared_translation",
                "source_root": str(scene_transform.source_root),
                "world_rotation": "world_rotation.npy",
                "world_rotation_txt": "world_rotation.txt",
                "shared_translation": "shared_translation.txt",
            }
            if scene_transform is not None
            else None
        ),
        "object_name": object_name,
        "piece_count": converted.piece_count,
        "bounds_min": converted.bounds_min,
        "bounds_max": converted.bounds_max,
        "files": {
            "object_mesh": converted.object_mesh.name,
            "object_urdf": converted.object_urdf.name,
            "scene_xml": converted.scene_xml.name,
            "box_assets_xml": "box_assets.xml",
            "box_body_xml": "box_body.xml",
            "pieces_dir": "pieces",
            "motion_file": converted.motion_file.name if converted.motion_file else None,
            "motion_metadata": "crisp_motion_metadata.json" if converted.motion_metadata else None,
        },
        "retargeting_note": (
            "Holosoma climbing loads the first .npy motion file in this folder and may generate scaled "
            "URDF/XML copies using its SMPL scale. The generated CRISP terrain files themselves remain scale=1."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return converted


def validate_generated(converted: ConvertedSequence, validate_mujoco: bool) -> None:
    for xml_file in (
        converted.output_dir / "box_assets.xml",
        converted.output_dir / "box_body.xml",
        converted.object_urdf,
    ):
        ET.parse(xml_file)
    ET.parse(converted.scene_xml)

    if validate_mujoco:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(converted.scene_xml))
        terrain_geoms = [
            model.geom(i).name
            for i in range(model.ngeom)
            if model.geom(i).name.startswith("multi_boxes_")
        ]
        if len(terrain_geoms) != converted.piece_count:
            raise RuntimeError(
                f"MuJoCo loaded {len(terrain_geoms)} terrain geoms, expected {converted.piece_count}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CRISP z-up scene pieces into Holosoma climbing terrain.")
    parser.add_argument(
        "--crisp-zup-root",
        type=Path,
        required=True,
        help="Root containing stair_<id>/gv/scene_mesh_sqs.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Holosoma climbing data root.")
    parser.add_argument(
        "--sequence",
        action="append",
        help="Sequence name or numeric id. May be passed multiple times.",
    )
    parser.add_argument("--hmr-key", default="gv", help="Subfolder under each sequence containing scene_mesh_sqs.")
    parser.add_argument("--object-name", default="multi_boxes", help="Holosoma climbing object name.")
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument(
        "--motion-root",
        type=Path,
        default=None,
        help="Optional root containing Holosoma motion .npy files.",
    )
    parser.add_argument("--motion-glob", default="{sequence}*.npy", help="Glob used under --motion-root.")
    parser.add_argument(
        "--crisp-hmr-root",
        type=Path,
        default=None,
        help="Optional CRISP root containing <sequence>/<hmr-key>/hmr/hps_track_smplx.npz.",
    )
    parser.add_argument(
        "--crisp-hmr-motion-filename",
        default="hps_track_smplx.npz",
        help="CRISP SMPL-X motion file name under each hmr folder.",
    )
    parser.add_argument(
        "--motion-output-filename",
        default="{sequence}.npy",
        help="Output .npy name written into each Holosoma sequence folder.",
    )
    parser.add_argument("--require-motion", action="store_true", help="Fail if no matching motion .npy is found.")
    parser.add_argument("--max-sequences", type=int, default=None, help="Limit discovered sequences for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing sequence outputs.")
    parser.add_argument("--validate", action="store_true", help="Parse generated XML/URDF files.")
    parser.add_argument("--validate-mujoco", action="store_true", help="Load generated scene XML with MuJoCo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.crisp_zup_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    robot_xml = args.robot_xml.expanduser().resolve()
    robot_urdf = args.robot_urdf.expanduser().resolve()
    motion_root = args.motion_root.expanduser().resolve() if args.motion_root else None
    crisp_hmr_root = args.crisp_hmr_root.expanduser().resolve() if args.crisp_hmr_root else None

    if args.sequence:
        scene_dirs = [(seq, _find_scene_dir(root, seq, args.hmr_key)) for seq in args.sequence]
        scene_dirs = [(_sequence_name_from_scene_dir(scene_dir), scene_dir) for _, scene_dir in scene_dirs]
    else:
        scene_dirs = _discover_scene_dirs(root, args.hmr_key)

    if args.max_sequences is not None:
        scene_dirs = scene_dirs[: args.max_sequences]

    converted_items: list[ConvertedSequence] = []
    for sequence, scene_dir in scene_dirs:
        converted = convert_sequence(
            sequence=sequence,
            scene_dir=scene_dir,
            output_root=output_root,
            robot_xml=robot_xml,
            robot_urdf=robot_urdf,
            object_name=args.object_name,
            overwrite=args.overwrite,
            motion_root=motion_root,
            motion_glob=args.motion_glob,
            crisp_hmr_root=crisp_hmr_root,
            hmr_key=args.hmr_key,
            crisp_hmr_motion_filename=args.crisp_hmr_motion_filename,
            motion_output_filename=args.motion_output_filename,
            require_motion=args.require_motion,
        )
        if args.validate or args.validate_mujoco:
            validate_generated(converted, args.validate_mujoco)
        converted_items.append(converted)
        motion_status = converted.motion_file.name if converted.motion_file else "no motion .npy copied"
        print(
            f"{converted.sequence}: {converted.piece_count} pieces -> {converted.output_dir} "
            f"({motion_status})"
        )

    print(f"Converted {len(converted_items)} sequence(s) into {output_root}")


if __name__ == "__main__":
    main()
