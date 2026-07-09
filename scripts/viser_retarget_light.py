#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight Viser playback for retargeted MuJoCo qpos.")
    parser.add_argument("--xml", type=Path, required=True, help="MuJoCo XML matching the retarget qpos.")
    parser.add_argument("--qpos-npz", type=Path, required=True, help="Retarget output npz containing qpos.")
    parser.add_argument("--terrain-obj", type=Path, default=None, help="Optional terrain mesh OBJ to display.")
    parser.add_argument("--terrain-scale", type=float, default=1.0, help="Scale applied to --terrain-obj.")
    parser.add_argument("--port", type=int, default=9303)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-size", type=float, default=0.045)
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--show-g1-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mesh-alpha", type=float, default=0.82)
    parser.add_argument("--grid-width", type=float, default=20.0)
    parser.add_argument("--grid-height", type=float, default=20.0)
    parser.add_argument("--autoplay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--align-g1-to-human-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Translate G1 each frame so its pelvis body overlays human joint 0.",
    )
    return parser.parse_args()


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def robot_body_ids(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    for body_id in range(1, model.nbody):
        name = body_name(model, body_id)
        if name.startswith("multi_boxes_"):
            continue
        ids.append(body_id)
    return ids


def robot_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    seen: set[tuple[int, int, int, tuple[float, ...]]] = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body = body_name(model, body_id)
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if body_id == 0 or body.startswith("multi_boxes_") or geom_name == "ground":
            continue
        geom_type = int(model.geom_type[geom_id])
        data_id = int(model.geom_dataid[geom_id])
        key = (
            body_id,
            geom_type,
            data_id,
            tuple(np.asarray(model.geom_size[geom_id], dtype=np.float32).round(6).tolist()),
        )
        if key in seen:
            continue
        seen.add(key)
        ids.append(geom_id)
    return ids


def compute_body_frames(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = mujoco.MjData(model)
    ids = robot_body_ids(model)
    id_to_local = {body_id: idx for idx, body_id in enumerate(ids)}
    edges: list[tuple[int, int]] = []
    for body_id in ids:
        parent = int(model.body_parentid[body_id])
        if parent in id_to_local:
            edges.append((id_to_local[parent], id_to_local[body_id]))

    points_per_frame: list[np.ndarray] = []
    lines_per_frame: list[np.ndarray] = []
    for frame_qpos in qpos:
        data.qpos[: model.nq] = frame_qpos[: model.nq]
        mujoco.mj_forward(model, data)
        pts = np.asarray(data.xpos[ids], dtype=np.float32).copy()
        points_per_frame.append(pts)
        if edges:
            lines = np.asarray([[pts[a], pts[b]] for a, b in edges], dtype=np.float32)
        else:
            lines = np.zeros((0, 2, 3), dtype=np.float32)
        lines_per_frame.append(lines)

    names = [body_name(model, body_id) for body_id in ids]
    return np.stack(points_per_frame), np.stack(lines_per_frame), names


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat().astype(np.float32)
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def compute_geom_frames(model: mujoco.MjModel, qpos: np.ndarray, geom_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    for frame_qpos in qpos:
        data.qpos[: model.nq] = frame_qpos[: model.nq]
        mujoco.mj_forward(model, data)
        frame_pos = []
        frame_quat = []
        for geom_id in geom_ids:
            frame_pos.append(np.asarray(data.geom_xpos[geom_id], dtype=np.float32).copy())
            frame_quat.append(matrix_to_wxyz(np.asarray(data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)))
        positions.append(np.stack(frame_pos, axis=0) if frame_pos else np.zeros((0, 3), dtype=np.float32))
        rotations.append(np.stack(frame_quat, axis=0) if frame_quat else np.zeros((0, 4), dtype=np.float32))
    return np.stack(positions, axis=0), np.stack(rotations, axis=0)


def geom_trimesh(model: mujoco.MjModel, geom_id: int, alpha: float) -> trimesh.Trimesh | None:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float32)
    mesh: trimesh.Trimesh | None = None
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        face_adr = int(model.mesh_faceadr[mesh_id])
        face_num = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float32).copy()
        faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int64).copy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        mesh = trimesh.creation.box(extents=(size[:3] * 2.0).astype(float))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(size[1]) * 2.0, sections=24)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(size[1]) * 2.0, count=[16, 16])

    if mesh is None:
        return None
    rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float32).copy()
    if not np.isfinite(rgba).all() or rgba[3] <= 0:
        rgba = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32)
    rgba[3] = min(float(alpha), float(rgba[3]))
    mesh.visual.face_colors = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    return mesh


def human_skeleton_edges(joint_count: int) -> list[tuple[int, int]]:
    # SMPLX_DEMO_JOINTS order used by real2sim2real:
    # Pelvis, L/R hip, spine, knees, ankles, feet, neck/head, shoulders, elbows, wrists.
    edges = [
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
    return [(a, b) for a, b in edges if a < joint_count and b < joint_count]


def make_segments(points: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros((points.shape[0], 0, 2, 3), dtype=np.float32)
    return np.asarray([[[frame[a], frame[b]] for a, b in edges] for frame in points], dtype=np.float32)


def main() -> None:
    args = parse_args()
    payload = np.load(args.qpos_npz, allow_pickle=True)
    qpos = np.asarray(payload["qpos"])
    human_joints = np.asarray(payload["human_joints"], dtype=np.float32) if "human_joints" in payload else None
    fps = int(payload["fps"]) if "fps" in payload else int(args.fps)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    if qpos.shape[1] < model.nq:
        raise ValueError(f"qpos has {qpos.shape[1]} columns, but MuJoCo model expects nq={model.nq}")

    points_raw, lines_raw, names = compute_body_frames(model, qpos)
    geom_ids = robot_geom_ids(model) if args.show_g1_mesh else []
    geom_positions_raw, geom_quats = compute_geom_frames(model, qpos, geom_ids) if geom_ids else (
        np.zeros((qpos.shape[0], 0, 3), dtype=np.float32),
        np.zeros((qpos.shape[0], 0, 4), dtype=np.float32),
    )
    points = points_raw.copy()
    lines = lines_raw.copy()
    geom_positions = geom_positions_raw.copy()
    offsets = np.zeros((points.shape[0], 3), dtype=np.float32)
    if human_joints is not None and args.align_g1_to_human_root:
        count = min(points.shape[0], human_joints.shape[0])
        offsets = human_joints[:count, 0, :] - points[:count, 0, :]
        points[:count] += offsets[:, None, :]
        lines[:count] += offsets[:, None, None, :]
        geom_positions[:count] += offsets[:, None, :]
    n_frames, n_bodies, _ = points.shape
    if human_joints is not None and human_joints.shape[0] != n_frames:
        raise ValueError(f"human_joints has {human_joints.shape[0]} frames, qpos has {n_frames}")
    raw_root = points_raw[:, 0, :]
    print(
        "[viser_retarget_light] raw_g1_root_min="
        f"{raw_root.min(axis=0).round(4).tolist()} raw_g1_root_max={raw_root.max(axis=0).round(4).tolist()}",
        flush=True,
    )
    if human_joints is not None:
        human_root = human_joints[:, 0, :]
        print(
            "[viser_retarget_light] human_root_min="
            f"{human_root.min(axis=0).round(4).tolist()} human_root_max={human_root.max(axis=0).round(4).tolist()}",
            flush=True,
        )
    print(
        f"[viser_retarget_light] frames={n_frames} bodies={n_bodies} nq={model.nq} fps={fps} "
        f"human_joints={'yes' if human_joints is not None else 'no'} "
        f"g1_mesh_geoms={len(geom_ids)} align_g1_to_human_root={args.align_g1_to_human_root}",
        flush=True,
    )

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=args.grid_width, height=args.grid_height, position=(0.0, 0.0, 0.0))

    terrain_handle = None
    if args.terrain_obj is not None and args.terrain_obj.exists():
        terrain = trimesh.load(args.terrain_obj, process=False, force="mesh")
        if isinstance(terrain, trimesh.Scene):
            terrain = trimesh.util.concatenate(tuple(terrain.geometry.values()))
        if args.terrain_scale != 1.0:
            terrain.apply_scale(float(args.terrain_scale))
        terrain_handle = server.scene.add_mesh_trimesh("/terrain", terrain, visible=True)

    pelvis = points[:, 0, :]
    traj_segments = np.stack([pelvis[:-1], pelvis[1:]], axis=1).astype(np.float32)
    traj_colors = np.tile(np.array([255, 180, 0], dtype=np.uint8), (max(n_frames - 1, 1), 2, 1))
    if n_frames > 1:
        server.scene.add_line_segments("/pelvis_trajectory", traj_segments, traj_colors, line_width=2.0)

    point_handles = []
    line_handles = []
    human_point_handles = []
    human_line_handles = []
    mesh_handles = []
    point_colors = np.tile(np.array([30, 145, 255], dtype=np.uint8), (n_bodies, 1))
    line_colors = np.tile(np.array([245, 245, 245], dtype=np.uint8), (lines.shape[1], 2, 1))
    human_lines = None
    human_point_colors = None
    human_line_colors = None
    if human_joints is not None:
        human_edges = human_skeleton_edges(human_joints.shape[1])
        human_lines = make_segments(human_joints, human_edges)
        human_point_colors = np.tile(np.array([255, 60, 190], dtype=np.uint8), (human_joints.shape[1], 1))
        human_line_colors = np.tile(np.array([255, 210, 40], dtype=np.uint8), (human_lines.shape[1], 2, 1))
    for mesh_idx, geom_id in enumerate(geom_ids):
        mesh = geom_trimesh(model, geom_id, args.mesh_alpha)
        if mesh is None:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        body = body_name(model, int(model.geom_bodyid[geom_id]))
        handle = server.scene.add_mesh_trimesh(
            f"/g1_mesh/{mesh_idx:03d}_{body}_{geom_name}",
            mesh,
            position=geom_positions[0, mesh_idx],
            wxyz=geom_quats[0, mesh_idx],
            visible=True,
        )
        mesh_handles.append((mesh_idx, handle))

    for idx in range(n_frames):
        visible = idx == 0
        point_handles.append(
            server.scene.add_point_cloud(
                f"/robot/frame_{idx:04d}/points",
                points[idx],
                point_colors,
                point_size=args.point_size,
                point_shape="circle",
                precision="float32",
                visible=False,
            )
        )
        line_handles.append(
            server.scene.add_line_segments(
                f"/robot/frame_{idx:04d}/skeleton",
                lines[idx],
                line_colors,
                line_width=args.line_width,
                visible=False,
            )
        )
        if human_joints is not None and human_lines is not None:
            human_point_handles.append(
                server.scene.add_point_cloud(
                    f"/human/frame_{idx:04d}/points",
                    human_joints[idx],
                    human_point_colors,
                    point_size=args.point_size * 1.25,
                    point_shape="circle",
                    precision="float32",
                    visible=visible,
                )
            )
            human_line_handles.append(
                server.scene.add_line_segments(
                    f"/human/frame_{idx:04d}/skeleton",
                    human_lines[idx],
                    human_line_colors,
                    line_width=args.line_width * 1.25,
                    visible=visible,
                )
            )

    state = {
        "frame": 0,
        "playing": bool(args.autoplay),
        "loop": bool(args.loop),
        "show_g1_mesh": bool(mesh_handles),
        "show_g1_points": False,
        "show_g1_skeleton": False,
        "show_human_points": human_joints is not None,
        "show_human_skeleton": human_joints is not None,
    }

    def show_frame(frame: int) -> None:
        frame = int(np.clip(frame, 0, n_frames - 1))
        old = state["frame"]
        if old != frame:
            point_handles[old].visible = False
            line_handles[old].visible = False
            if human_point_handles:
                human_point_handles[old].visible = False
                human_line_handles[old].visible = False
        point_handles[frame].visible = state["show_g1_points"]
        line_handles[frame].visible = state["show_g1_skeleton"]
        for mesh_idx, handle in mesh_handles:
            handle.position = geom_positions[frame, mesh_idx]
            handle.wxyz = geom_quats[frame, mesh_idx]
            handle.visible = state["show_g1_mesh"]
        if human_point_handles:
            human_point_handles[frame].visible = state["show_human_points"]
            human_line_handles[frame].visible = state["show_human_skeleton"]
        state["frame"] = frame

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
        play_button = server.gui.add_button("Play / Pause")
        fps_input = server.gui.add_number("FPS", initial_value=fps, min=1, max=120, step=1)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=bool(args.loop))
    with server.gui.add_folder("Layers"):
        show_g1_mesh = server.gui.add_checkbox("G1 real mesh", initial_value=bool(mesh_handles))
        show_g1_points = server.gui.add_checkbox("G1 body points", initial_value=False)
        show_g1_skeleton = server.gui.add_checkbox("G1 skeleton", initial_value=False)
        show_human_points = server.gui.add_checkbox("Human joints", initial_value=human_joints is not None)
        show_human_skeleton = server.gui.add_checkbox("Human skeleton", initial_value=human_joints is not None)
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

    @show_g1_points.on_update
    def _(_) -> None:
        state["show_g1_points"] = bool(show_g1_points.value)
        point_handles[state["frame"]].visible = state["show_g1_points"]

    @show_g1_skeleton.on_update
    def _(_) -> None:
        state["show_g1_skeleton"] = bool(show_g1_skeleton.value)
        line_handles[state["frame"]].visible = state["show_g1_skeleton"]

    @show_g1_mesh.on_update
    def _(_) -> None:
        state["show_g1_mesh"] = bool(show_g1_mesh.value)
        for _, handle in mesh_handles:
            handle.visible = state["show_g1_mesh"]

    @show_human_points.on_update
    def _(_) -> None:
        state["show_human_points"] = bool(show_human_points.value)
        if human_point_handles:
            human_point_handles[state["frame"]].visible = state["show_human_points"]

    @show_human_skeleton.on_update
    def _(_) -> None:
        state["show_human_skeleton"] = bool(show_human_skeleton.value)
        if human_line_handles:
            human_line_handles[state["frame"]].visible = state["show_human_skeleton"]

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
    print("[viser_retarget_light] ready", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
