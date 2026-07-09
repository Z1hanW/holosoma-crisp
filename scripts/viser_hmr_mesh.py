#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


SMPLX_22_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (4, 7),
    (7, 10),
    (2, 5),
    (5, 8),
    (8, 11),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (12, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (12, 14),
    (14, 17),
    (17, 19),
    (19, 21),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CRISP HMR joints with a limb mesh proxy.")
    parser.add_argument("--hmr-npz", type=Path, required=True, help="HMR npz/npy with global_joint_positions.")
    parser.add_argument("--mesh-npz", type=Path, default=None, help="Optional dynamic mesh npz with vertices/faces.")
    parser.add_argument("--terrain-obj", type=Path, default=None, help="Optional scene mesh OBJ.")
    parser.add_argument("--terrain-scale", type=float, default=1.0)
    parser.add_argument("--world-rotation", type=Path, default=None, help="Optional world_rotation.npy for raw HMR joints.")
    parser.add_argument("--shared-translation", type=Path, default=None, help="Optional shared_translation.txt/.npy.")
    parser.add_argument("--port", type=int, default=9302)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--joint-size", type=float, default=0.045)
    parser.add_argument("--limb-radius", type=float, default=0.035)
    parser.add_argument("--limb-sections", type=int, default=10)
    parser.add_argument("--grid-width", type=float, default=20.0)
    parser.add_argument("--grid-height", type=float, default=20.0)
    parser.add_argument("--autoplay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Load inputs and print shapes without starting Viser.")
    return parser.parse_args()


def load_hmr_joints(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in ("global_joint_positions", "human_joints", "joints", "hmr_joints"):
            if key in data:
                joints = data[key]
                break
        else:
            raise KeyError(f"No supported joint key found in {path}. Keys: {data.files}")
    else:
        joints = data

    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"HMR joints must have shape (T, J, 3), got {joints.shape}")
    return joints


def load_shared_translation(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path)
    else:
        value = np.loadtxt(path)
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.shape[0] != 3:
        raise ValueError(f"shared translation must have 3 values, got {value.shape}")
    return value


def apply_world_transform(
    points: np.ndarray,
    world_rotation: Path | None,
    shared_translation: Path | None,
) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    if world_rotation is not None:
        rotation = np.asarray(np.load(world_rotation), dtype=np.float32).reshape(3, 3)
        out = out @ rotation.T
    if shared_translation is not None:
        out = out + load_shared_translation(shared_translation)
    return out


def skeleton_edges(joint_count: int) -> list[tuple[int, int]]:
    return [(a, b) for a, b in SMPLX_22_EDGES if a < joint_count and b < joint_count]


def line_segments(joints: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros((joints.shape[0], 0, 2, 3), dtype=np.float32)
    return np.asarray([[[frame[a], frame[b]] for a, b in edges] for frame in joints], dtype=np.float32)


def rotation_from_z_axis(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(3)
    target = direction / norm
    source = np.array([0.0, 0.0, 1.0])
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))
    if dot > 1.0 - 1e-8:
        return np.eye(3)
    if dot < -1.0 + 1e-8:
        return np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / float(np.dot(cross, cross)))


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float, sections: int) -> trimesh.Trimesh:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    vector = end - start
    length = max(float(np.linalg.norm(vector)), radius * 2.0)
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    transform = np.eye(4)
    transform[:3, :3] = rotation_from_z_axis(vector)
    transform[:3, 3] = (start + end) * 0.5
    mesh.apply_transform(transform)
    return mesh


def build_limb_meshes(
    joints: np.ndarray,
    edges: list[tuple[int, int]],
    radius: float,
    sections: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not edges:
        raise ValueError("No skeleton edges available for mesh proxy.")

    vertices_per_frame: list[np.ndarray] = []
    faces: np.ndarray | None = None
    for frame in joints:
        meshes = [cylinder_between(frame[a], frame[b], radius, sections) for a, b in edges]
        merged = trimesh.util.concatenate(meshes)
        frame_vertices = np.asarray(merged.vertices, dtype=np.float32)
        frame_faces = np.asarray(merged.faces, dtype=np.int32)
        if faces is None:
            faces = frame_faces
        elif faces.shape != frame_faces.shape or not np.array_equal(faces, frame_faces):
            raise RuntimeError("Generated limb mesh topology changed across frames.")
        vertices_per_frame.append(frame_vertices)

    if faces is None:
        raise RuntimeError("Failed to build limb mesh faces.")
    return np.stack(vertices_per_frame, axis=0), faces


def load_dynamic_mesh(path: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError("--mesh-npz must be an npz with vertices/faces.")

    vertices_key = next((key for key in ("vertices", "verts", "smplx_vertices") if key in data), None)
    faces_key = next((key for key in ("faces", "smplx_faces") if key in data), None)
    if vertices_key is None or faces_key is None:
        raise KeyError(f"--mesh-npz requires vertices/faces keys. Keys: {data.files}")

    vertices = np.asarray(data[vertices_key], dtype=np.float32)
    faces = np.asarray(data[faces_key], dtype=np.int32)
    if vertices.ndim == 2:
        vertices = np.repeat(vertices[None, ...], n_frames, axis=0)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"mesh vertices must have shape (T, V, 3) or (V, 3), got {vertices.shape}")
    if vertices.shape[0] != n_frames:
        raise ValueError(f"mesh has {vertices.shape[0]} frames, HMR joints have {n_frames}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"mesh faces must have shape (F, 3), got {faces.shape}")
    return vertices, faces


def load_terrain(path: Path, scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if scale != 1.0:
        mesh.apply_scale(float(scale))
    return mesh


def main() -> None:
    args = parse_args()
    joints = load_hmr_joints(args.hmr_npz)
    joints = apply_world_transform(joints, args.world_rotation, args.shared_translation)
    n_frames = int(joints.shape[0])
    edges = skeleton_edges(joints.shape[1])
    lines = line_segments(joints, edges)

    if args.mesh_npz is not None:
        mesh_vertices, mesh_faces = load_dynamic_mesh(args.mesh_npz, n_frames)
        mesh_vertices = apply_world_transform(mesh_vertices, args.world_rotation, args.shared_translation)
        mesh_source = str(args.mesh_npz)
    else:
        mesh_vertices, mesh_faces = build_limb_meshes(joints, edges, args.limb_radius, args.limb_sections)
        mesh_source = "limb cylinder proxy"

    print(
        f"[viser_hmr_mesh] frames={n_frames} joints={joints.shape[1]} "
        f"mesh_vertices={mesh_vertices.shape[1]} mesh_faces={mesh_faces.shape[0]} source={mesh_source}",
        flush=True,
    )
    if args.dry_run:
        return

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=args.grid_width, height=args.grid_height, position=(0.0, 0.0, 0.0))

    terrain_handle = None
    if args.terrain_obj is not None and args.terrain_obj.exists():
        terrain_handle = server.scene.add_mesh_trimesh(
            "/terrain",
            load_terrain(args.terrain_obj, args.terrain_scale),
            visible=True,
        )

    mesh_handle = server.scene.add_mesh_simple(
        "/hmr/mesh",
        vertices=mesh_vertices[0],
        faces=mesh_faces,
        color=(255, 150, 70),
        opacity=0.45,
    )
    joint_handle = server.scene.add_point_cloud(
        "/hmr/joints",
        joints[0],
        colors=np.tile(np.array([255, 80, 200], dtype=np.uint8), (joints.shape[1], 1)),
        point_size=args.joint_size,
        point_shape="circle",
        precision="float32",
        visible=True,
    )
    line_handle = server.scene.add_line_segments(
        "/hmr/skeleton",
        lines[0],
        colors=np.tile(np.array([255, 220, 40], dtype=np.uint8), (lines.shape[1], 2, 1)),
        line_width=2.5,
        visible=True,
    )

    root = joints[:, 0, :]
    if n_frames > 1:
        server.scene.add_line_segments(
            "/hmr/root_trajectory",
            np.stack([root[:-1], root[1:]], axis=1).astype(np.float32),
            colors=np.tile(np.array([255, 180, 0], dtype=np.uint8), (n_frames - 1, 2, 1)),
            line_width=2.0,
        )

    state = {
        "frame": 0,
        "playing": bool(args.autoplay),
        "loop": bool(args.loop),
        "show_mesh": True,
        "show_joints": True,
        "show_skeleton": True,
    }

    def show_frame(frame: int) -> None:
        frame = int(np.clip(frame, 0, n_frames - 1))
        mesh_handle.vertices = mesh_vertices[frame]
        mesh_handle.visible = state["show_mesh"]
        joint_handle.points = joints[frame]
        joint_handle.visible = state["show_joints"]
        line_handle.points = lines[frame]
        line_handle.visible = state["show_skeleton"]
        state["frame"] = frame

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
        play_button = server.gui.add_button("Play / Pause")
        fps_input = server.gui.add_number("FPS", initial_value=args.fps, min=1, max=120, step=1)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=bool(args.loop))

    with server.gui.add_folder("Layers"):
        show_mesh = server.gui.add_checkbox("HMR mesh", initial_value=True)
        show_joints = server.gui.add_checkbox("HMR joints", initial_value=True)
        show_skeleton = server.gui.add_checkbox("HMR skeleton", initial_value=True)
        show_terrain = server.gui.add_checkbox("Terrain", initial_value=terrain_handle is not None)

    @frame_slider.on_update
    def _(_) -> None:
        show_frame(int(frame_slider.value))

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]

    @loop_checkbox.on_update
    def _(_) -> None:
        state["loop"] = bool(loop_checkbox.value)

    @show_mesh.on_update
    def _(_) -> None:
        state["show_mesh"] = bool(show_mesh.value)
        mesh_handle.visible = state["show_mesh"]

    @show_joints.on_update
    def _(_) -> None:
        state["show_joints"] = bool(show_joints.value)
        joint_handle.visible = state["show_joints"]

    @show_skeleton.on_update
    def _(_) -> None:
        state["show_skeleton"] = bool(show_skeleton.value)
        line_handle.visible = state["show_skeleton"]

    @show_terrain.on_update
    def _(_) -> None:
        if terrain_handle is not None:
            terrain_handle.visible = bool(show_terrain.value)

    def playback_loop() -> None:
        last = time.perf_counter()
        while True:
            if not state["playing"]:
                time.sleep(0.02)
                last = time.perf_counter()
                continue
            fps_value = max(1, int(fps_input.value))
            now = time.perf_counter()
            if now - last >= 1.0 / fps_value:
                if state["frame"] >= n_frames - 1:
                    if not state["loop"]:
                        state["playing"] = False
                        last = now
                        continue
                    next_frame = 0
                else:
                    next_frame = state["frame"] + 1
                show_frame(next_frame)
                frame_slider.value = next_frame
                last = now
            else:
                time.sleep(0.002)

    threading.Thread(target=playback_loop, daemon=True).start()
    print("[viser_hmr_mesh] ready", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
