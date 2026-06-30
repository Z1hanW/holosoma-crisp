#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import viser
from loguru import logger
from viser.extras import ViserUrdf


DEFAULT_CRISP_ROOT = Path("/home/ubuntu/FAR/holosoma/crisp_stairs")
DEFAULT_CLIP_TOKENS = (
    "45",
    "3",
    "56_outdoor",
    "78_outdoor_stairs_up_down",
    "48",
    "50",
    "51",
    "53",
    "54",
    "61",
    "69",
    "75",
    "78",
    "83",
    "295",
    "101",
)
ROBOT_URDF = Path("/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf")


@dataclass(frozen=True)
class ClipRecord:
    name: str
    requested: str
    motion_path: Path
    geometry_path: Path
    frames: int
    fps: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinematic replay viewer for CRISP stair motions and paired OBJ geometry."
    )
    parser.add_argument("--crisp-root", type=Path, default=DEFAULT_CRISP_ROOT)
    parser.add_argument("--motion-dir", type=Path, default=None)
    parser.add_argument("--geometry-dir", type=Path, default=None)
    parser.add_argument("--port", type=int, default=2101)
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIP_TOKENS))
    parser.add_argument("--all", action="store_true", help="Ignore --clips and show every motion/geometry pair.")
    parser.add_argument("--start-clip", default=None)
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS override. Defaults to motion fps.")
    parser.add_argument("--autoplay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-robot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action="store_true", help="Fail if any requested clip is missing.")
    return parser.parse_args()


def _motion_dir(args: argparse.Namespace) -> Path:
    return args.motion_dir or (args.crisp_root / "___crisp_clean_motion")


def _geometry_dir(args: argparse.Namespace) -> Path:
    return args.geometry_dir or (args.crisp_root / "___crisp_clean_geometry")


def _clip_name_from_token(token: str) -> str:
    token = str(token).strip()
    if not token:
        raise ValueError("Empty clip token.")
    if token == "56_outdoor":
        return "56_outdoor_stairs_up_down"
    if token.isdigit():
        return f"stair_{token}"
    if token.startswith("stair_") or token.endswith("_stairs_up_down"):
        return token
    return token


def _load_manifest_aliases(crisp_root: Path) -> dict[str, str]:
    manifest_path = crisp_root / "terrain_traversal_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read manifest {}: {}", manifest_path, exc)
        return {}
    aliases: dict[str, str] = {}
    for entry in payload.get("clips", []):
        if not isinstance(entry, dict):
            continue
        requested = str(entry.get("requested", "")).strip()
        clip_id = str(entry.get("clip_id", "")).strip()
        if requested and clip_id:
            aliases[requested] = clip_id
    return aliases


def _resolve_requested_clips(
    tokens: list[str],
    motion_dir: Path,
    geometry_dir: Path,
    crisp_root: Path,
    *,
    strict: bool,
) -> tuple[list[str], list[str]]:
    aliases = _load_manifest_aliases(crisp_root)
    resolved: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        candidates = [_clip_name_from_token(token)]
        if token in aliases:
            candidates.append(aliases[token])
        if token == "295":
            candidates.append("stair_95")

        chosen = None
        for candidate in dict.fromkeys(candidates):
            if (motion_dir / f"{candidate}.npz").is_file() and (geometry_dir / f"{candidate}.obj").is_file():
                chosen = candidate
                break
        if chosen is None:
            missing.append(token)
            continue
        if chosen != candidates[0]:
            msg = f"requested '{token}' resolved to existing clip '{chosen}'"
            logger.warning(msg)
            warnings.append(msg)
        if chosen not in resolved:
            resolved.append(chosen)

    if missing:
        msg = f"Missing requested clips: {missing}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        warnings.append(msg)
    return resolved, warnings


def _list_all_pairs(motion_dir: Path, geometry_dir: Path) -> list[str]:
    motion_names = {path.stem for path in motion_dir.glob("*.npz")}
    geometry_names = {path.stem for path in geometry_dir.glob("*.obj")}
    return sorted(motion_names & geometry_names)


def _motion_summary(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=True) as data:
        if "joint_pos" not in data:
            raise KeyError(f"{path} missing joint_pos")
        frames = int(np.asarray(data["joint_pos"]).shape[0])
        fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30
    return frames, fps


def _build_clip_records(args: argparse.Namespace) -> list[ClipRecord]:
    motion_dir = _motion_dir(args)
    geometry_dir = _geometry_dir(args)
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")
    if not geometry_dir.is_dir():
        raise FileNotFoundError(f"Geometry directory not found: {geometry_dir}")

    if args.all:
        clip_names = _list_all_pairs(motion_dir, geometry_dir)
    else:
        clip_names, _ = _resolve_requested_clips(
            list(args.clips),
            motion_dir,
            geometry_dir,
            args.crisp_root,
            strict=bool(args.strict),
        )
    if not clip_names:
        raise RuntimeError("No clips resolved for replay.")

    records: list[ClipRecord] = []
    for name in clip_names:
        motion_path = motion_dir / f"{name}.npz"
        geometry_path = geometry_dir / f"{name}.obj"
        frames, fps = _motion_summary(motion_path)
        records.append(
            ClipRecord(
                name=name,
                requested=name.removeprefix("stair_"),
                motion_path=motion_path,
                geometry_path=geometry_path,
                frames=frames,
                fps=fps,
            )
        )
    return records


def _load_motion(path: Path, viser_joint_names: list[str]) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=True) as data:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        if joint_pos.ndim != 2 or joint_pos.shape[1] < 7:
            raise ValueError(f"Invalid joint_pos shape in {path}: {joint_pos.shape}")
        root = joint_pos[:, :7]
        motion_joint_values = joint_pos[:, 7:]
        motion_joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
        name_to_idx = {name: idx for idx, name in enumerate(motion_joint_names)}
        missing = [name for name in viser_joint_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Motion joints missing for Viser URDF in {path}: {missing}")
        joint_indices = np.asarray([name_to_idx[name] for name in viser_joint_names], dtype=np.int64)
        fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30
    return np.concatenate([root, motion_joint_values[:, joint_indices]], axis=1), fps


def _load_geometry(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded geometry is not a trimesh: {type(mesh)}")
    return mesh


def run_viewer(args: argparse.Namespace) -> None:
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    records = _build_clip_records(args)
    record_by_name = {record.name: record for record in records}
    start_name = args.start_clip or records[0].name
    start_name = _clip_name_from_token(start_name)
    if start_name == "stair_295" and "stair_95" in record_by_name:
        logger.warning("start clip '295' resolved to 'stair_95'")
        start_name = "stair_95"
    if start_name not in record_by_name:
        raise ValueError(f"Start clip '{start_name}' is not in loaded clips: {list(record_by_name)}")

    server = viser.ViserServer(host="0.0.0.0", port=int(args.port), label="crisp_stairs_kinematic_replay")
    if args.show_grid:
        server.scene.add_grid("/grid", width=12.0, height=12.0, position=(0.0, 0.0, 0.0))

    robot = ViserUrdf(server, ROBOT_URDF, root_node_name="/robot", load_meshes=True, load_collision_meshes=False)
    robot.show_visual = bool(args.show_robot)
    robot_root = robot._visual_root_frame
    if robot_root is None:
        raise RuntimeError("ViserUrdf did not create a robot root frame.")
    viser_joint_names = list(robot.get_actuated_joint_names())

    motion_cache: dict[str, tuple[np.ndarray, int]] = {}
    geometry_cache: dict[str, trimesh.Trimesh] = {}
    geometry_handle: dict[str, viser.MeshHandle | None] = {"handle": None}
    state = {"clip": start_name, "frame": 0, "playing": bool(args.autoplay)}
    slider_guard = {"updating": False}

    def load_motion(name: str) -> tuple[np.ndarray, int]:
        if name not in motion_cache:
            motion_cache[name] = _load_motion(record_by_name[name].motion_path, viser_joint_names)
        return motion_cache[name]

    def load_geometry(name: str) -> trimesh.Trimesh:
        if name not in geometry_cache:
            geometry_cache[name] = _load_geometry(record_by_name[name].geometry_path)
        return geometry_cache[name]

    if args.preload:
        for record in records:
            load_motion(record.name)
            load_geometry(record.name)

    def set_geometry(name: str) -> None:
        handle = geometry_handle["handle"]
        if handle is not None:
            handle.remove()
            geometry_handle["handle"] = None
        mesh = load_geometry(name)
        geometry_handle["handle"] = server.scene.add_mesh_trimesh("/geometry", mesh)
        geometry_handle["handle"].visible = bool(show_geometry.value)

    def apply_frame(frame_idx: int) -> None:
        qpos, _ = load_motion(state["clip"])
        frame_idx = int(np.clip(frame_idx, 0, qpos.shape[0] - 1))
        frame = qpos[frame_idx]
        robot_root.position = frame[:3]
        robot_root.wxyz = frame[3:7]
        robot.update_cfg(frame[7 : 7 + len(viser_joint_names)])

    def set_clip(name: str) -> None:
        state["clip"] = name
        state["frame"] = 0
        qpos, motion_fps = load_motion(name)
        set_geometry(name)
        frame_slider.max = max(0, qpos.shape[0] - 1)
        slider_guard["updating"] = True
        frame_slider.value = 0
        slider_guard["updating"] = False
        if args.fps is None:
            fps_number.value = int(motion_fps)
        clip_info.content = (
            f"Clip: `{name}` | frames: `{qpos.shape[0]}` | source fps: `{motion_fps}` | "
            f"motion: `{record_by_name[name].motion_path.name}` | geometry: `{record_by_name[name].geometry_path.name}`"
        )
        apply_frame(0)

    qpos0, fps0 = load_motion(start_name)
    with server.gui.add_folder("Motion"):
        clip_dropdown = server.gui.add_dropdown(
            "Clip",
            options=tuple(record.name for record in records),
            initial_value=start_name,
        )
        clip_info = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, qpos0.shape[0] - 1),
            step=1,
            initial_value=0,
        )
        play_button = server.gui.add_button("Play / Pause")
        fps_number = server.gui.add_number(
            "FPS",
            initial_value=float(args.fps if args.fps is not None else fps0),
            min=1.0,
            max=240.0,
            step=1.0,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=bool(args.loop))
    with server.gui.add_folder("Display"):
        show_robot = server.gui.add_checkbox("Show robot", initial_value=bool(args.show_robot))
        show_geometry = server.gui.add_checkbox("Show geometry", initial_value=bool(args.show_geometry))

    @clip_dropdown.on_update
    def _(_evt) -> None:
        set_clip(str(clip_dropdown.value))

    @frame_slider.on_update
    def _(_evt) -> None:
        if slider_guard["updating"]:
            return
        state["frame"] = int(frame_slider.value)
        apply_frame(state["frame"])

    @play_button.on_click
    def _(_evt) -> None:
        state["playing"] = not state["playing"]

    @show_robot.on_update
    def _(_evt) -> None:
        robot.show_visual = bool(show_robot.value)

    @show_geometry.on_update
    def _(_evt) -> None:
        handle = geometry_handle["handle"]
        if handle is not None:
            handle.visible = bool(show_geometry.value)

    def player_loop() -> None:
        next_tick = time.monotonic()
        while True:
            if state["playing"]:
                fps = max(float(fps_number.value), 1.0)
                now = time.monotonic()
                if now >= next_tick:
                    next_tick = now + 1.0 / fps
                    qpos, _ = load_motion(state["clip"])
                    frame_idx = int(frame_slider.value) + 1
                    if frame_idx >= qpos.shape[0]:
                        if loop_checkbox.value:
                            frame_idx = 0
                        else:
                            frame_idx = qpos.shape[0] - 1
                            state["playing"] = False
                    slider_guard["updating"] = True
                    frame_slider.value = frame_idx
                    slider_guard["updating"] = False
                    state["frame"] = frame_idx
                    apply_frame(frame_idx)
            time.sleep(0.001)

    logger.info("Loaded {} CRISP stair clips: {}", len(records), [record.name for record in records])
    logger.info("Kinematic replay only: no physics rollout, no retiming/scaling/recentering.")
    logger.info("Viser listening on http://localhost:{}", args.port)
    set_clip(start_name)
    threading.Thread(target=player_loop, daemon=True).start()
    while True:
        time.sleep(1.0)


def main() -> None:
    run_viewer(_parse_args())


if __name__ == "__main__":
    main()
