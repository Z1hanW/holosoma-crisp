#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from viser_player import _hmr_edges_for_joint_count, load_hmr_joints, load_npz


@dataclass
class SequenceTrack:
    name: str
    qpos: np.ndarray
    fps: int
    hmr_joints: np.ndarray
    scaled_human_joints: np.ndarray | None
    scaled_scene_urdf: Path
    unscaled_scene_urdf: Path | None


def _sequence_key(name: str) -> tuple[int, str]:
    if name.startswith("stair_"):
        try:
            return int(name.split("_", 1)[1]), name
        except ValueError:
            pass
    return 10**9, name


def _discover_sequences(qpos_dir: Path, limit: int) -> list[str]:
    seqs = sorted(
        [path.stem.removesuffix("_original") for path in qpos_dir.glob("*_original.npz")],
        key=_sequence_key,
    )
    if limit > 0:
        return seqs[:limit]
    return seqs


def _find_scaled_scene_urdf(seq_dir: Path) -> Path:
    candidates = sorted(seq_dir.glob("multi_boxes_scaled_*.urdf"))
    if candidates:
        return candidates[0]
    fallback = seq_dir / "multi_boxes.urdf"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing scene URDF under {seq_dir}")


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _sample_motion(motion: np.ndarray, frame_value: float) -> np.ndarray:
    n_frames = int(motion.shape[0])
    if n_frames <= 1:
        return motion[0]
    frame_value = float(np.clip(frame_value, 0.0, n_frames - 1))
    i0 = int(np.floor(frame_value))
    i1 = min(i0 + 1, n_frames - 1)
    u = float(frame_value - i0)
    return ((1.0 - u) * motion[i0] + u * motion[i1]).astype(np.float32)


def _load_track(data_dir: Path, qpos_dir: Path, seq_name: str) -> SequenceTrack:
    seq_dir = data_dir / seq_name
    qpos_path = qpos_dir / f"{seq_name}_original.npz"
    hmr_path = seq_dir / f"{seq_name}.npy"
    if not qpos_path.is_file():
        raise FileNotFoundError(qpos_path)
    if not hmr_path.is_file():
        raise FileNotFoundError(hmr_path)

    qpos, fps, scaled_human_joints = load_npz(str(qpos_path))
    hmr_joints = load_hmr_joints(str(hmr_path))
    return SequenceTrack(
        name=seq_name,
        qpos=np.asarray(qpos, dtype=np.float32),
        fps=int(fps),
        hmr_joints=np.asarray(hmr_joints, dtype=np.float32),
        scaled_human_joints=scaled_human_joints,
        scaled_scene_urdf=_find_scaled_scene_urdf(seq_dir),
        unscaled_scene_urdf=(seq_dir / "multi_boxes.urdf") if (seq_dir / "multi_boxes.urdf").is_file() else None,
    )


def _remove(handle) -> None:
    if handle is not None:
        handle.remove()


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Viser player for CRISP Holosoma retargeting outputs.")
    parser.add_argument("--port", type=int, default=9331)
    parser.add_argument("--data-dir", type=Path, default=Path("demo_data/crisp_dataset"))
    parser.add_argument(
        "--qpos-dir",
        type=Path,
        default=Path("demo_results_parallel/g1/climbing/crisp_dataset"),
    )
    parser.add_argument("--robot-urdf", type=Path, default=Path("models/g1/g1_29dof_spherehand.urdf"))
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--initial-seq", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--grid-width", type=float, default=20.0)
    parser.add_argument("--grid-height", type=float, default=20.0)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    qpos_dir = args.qpos_dir.expanduser().resolve()
    robot_urdf = args.robot_urdf.expanduser().resolve()
    if args.sequences:
        sequences = list(dict.fromkeys(args.sequences))
    else:
        sequences = _discover_sequences(qpos_dir, int(args.limit))
    if args.initial_seq in sequences:
        sequences = [args.initial_seq] + [seq for seq in sequences if seq != args.initial_seq]
    if not sequences:
        raise FileNotFoundError(f"No qpos files found in {qpos_dir}")

    server = viser.ViserServer(host="0.0.0.0", port=int(args.port), label="crisp_batch_player")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.add_grid("/grid_z0", width=float(args.grid_width), height=float(args.grid_height), position=(0, 0, 0))

    robot_root = server.scene.add_frame("/robot", show_axes=False)
    robot_urdf_y = yourdfpy.URDF.load(str(robot_urdf), load_meshes=True, build_scene_graph=True)
    robot = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")
    robot_dof = len(robot.get_actuated_joint_limits())

    with server.gui.add_folder("Playback"):
        gui_sequence = server.gui.add_dropdown("Sequence", options=tuple(sequences), initial_value=sequences[0])
        gui_frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_number("FPS", initial_value=30, min=1, max=120, step=1)

    with server.gui.add_folder("Layers"):
        gui_show_g1 = server.gui.add_checkbox("G1", True)
        gui_show_scaled_scene = server.gui.add_checkbox("Scaled scene", True)
        gui_show_unscaled_scene = server.gui.add_checkbox("Unscaled scene", False)
        gui_show_hmr = server.gui.add_checkbox("Original HMR joints", True)
        gui_show_hmr_skeleton = server.gui.add_checkbox("Original HMR skeleton", True)
        gui_show_scaled_human = server.gui.add_checkbox("Scaled human joints", True)
        gui_show_scaled_human_skeleton = server.gui.add_checkbox("Scaled human skeleton", True)

    with server.gui.add_folder("Info"):
        gui_status = server.gui.add_markdown("Loading...")
        server.gui.add_markdown(f"Data dir: `{data_dir}`")
        server.gui.add_markdown(f"Qpos dir: `{qpos_dir}`")

    lock = threading.RLock()
    cache: dict[str, SequenceTrack] = {}
    state: dict[str, object] = {
        "track": None,
        "frame": 0.0,
        "scaled_scene": None,
        "unscaled_scene": None,
        "scaled_root": None,
        "unscaled_root": None,
        "hmr_points": None,
        "hmr_lines": None,
        "scaled_human_points": None,
        "scaled_human_lines": None,
    }

    def _get_track(seq_name: str) -> SequenceTrack:
        if seq_name not in cache:
            print(f"[stairs-batch-player] loading {seq_name}", flush=True)
            cache[seq_name] = _load_track(data_dir, qpos_dir, seq_name)
        return cache[seq_name]

    def _clear_sequence_handles() -> None:
        for key in (
            "scaled_scene",
            "unscaled_scene",
            "scaled_root",
            "unscaled_root",
            "hmr_points",
            "hmr_lines",
            "scaled_human_points",
            "scaled_human_lines",
        ):
            _remove(state.get(key))
            state[key] = None

    def _set_robot_visible(visible: bool) -> None:
        robot.show_visual = bool(visible)
        robot_root.visible = bool(visible)

    def _apply_frame(frame_value: float) -> None:
        track = state["track"]
        if track is None:
            return
        assert isinstance(track, SequenceTrack)
        frame_idx = int(np.clip(round(frame_value), 0, track.qpos.shape[0] - 1))
        q = track.qpos[frame_idx]
        joints = q[7 : 7 + robot_dof]
        if joints.shape[0] < robot_dof:
            joints = np.pad(joints, (0, robot_dof - joints.shape[0]))
        robot.update_cfg(joints[:robot_dof])
        robot_root.position = q[0:3]
        robot_root.wxyz = _quat_normalize(q[3:7])
        _set_robot_visible(bool(gui_show_g1.value))

        hmr_frame = _sample_motion(track.hmr_joints, frame_value)
        hmr_points = state.get("hmr_points")
        if hmr_points is not None:
            hmr_points.points = hmr_frame
            hmr_points.visible = bool(gui_show_hmr.value)
        hmr_lines = state.get("hmr_lines")
        if hmr_lines is not None:
            edges = _hmr_edges_for_joint_count(hmr_frame.shape[0])
            hmr_lines.points = hmr_frame[edges]
            hmr_lines.visible = bool(gui_show_hmr.value and gui_show_hmr_skeleton.value)

        if track.scaled_human_joints is not None:
            scaled_frame = _sample_motion(track.scaled_human_joints, frame_value)
            scaled_points = state.get("scaled_human_points")
            if scaled_points is not None:
                scaled_points.points = scaled_frame
                scaled_points.visible = bool(gui_show_scaled_human.value)
            scaled_lines = state.get("scaled_human_lines")
            if scaled_lines is not None:
                edges = _hmr_edges_for_joint_count(scaled_frame.shape[0])
                scaled_lines.points = scaled_frame[edges]
                scaled_lines.visible = bool(gui_show_scaled_human.value and gui_show_scaled_human_skeleton.value)

        scaled_scene = state.get("scaled_scene")
        if scaled_scene is not None:
            scaled_scene.show_visual = bool(gui_show_scaled_scene.value)
        unscaled_scene = state.get("unscaled_scene")
        if unscaled_scene is not None:
            unscaled_scene.show_visual = bool(gui_show_unscaled_scene.value)

    def _activate_sequence(seq_name: str) -> None:
        track = _get_track(seq_name)
        with lock:
            _clear_sequence_handles()
            state["track"] = track
            state["frame"] = 0.0

            scaled_root = server.scene.add_frame("/scaled_scene", show_axes=False)
            scaled_scene = ViserUrdf(server, track.scaled_scene_urdf, root_node_name="/scaled_scene")
            scaled_scene.show_visual = bool(gui_show_scaled_scene.value)
            state["scaled_root"] = scaled_root
            state["scaled_scene"] = scaled_scene

            if track.unscaled_scene_urdf is not None:
                unscaled_root = server.scene.add_frame("/unscaled_scene", show_axes=False)
                unscaled_scene = ViserUrdf(server, track.unscaled_scene_urdf, root_node_name="/unscaled_scene")
                unscaled_scene.show_visual = bool(gui_show_unscaled_scene.value)
                state["unscaled_root"] = unscaled_root
                state["unscaled_scene"] = unscaled_scene

            hmr_edges = _hmr_edges_for_joint_count(track.hmr_joints.shape[1])
            hmr0 = track.hmr_joints[0]
            state["hmr_points"] = server.scene.add_point_cloud(
                "/original_hmr/joints",
                points=hmr0,
                colors=(255, 120, 20),
                point_size=0.035,
                point_shape="circle",
                visible=bool(gui_show_hmr.value),
            )
            if hmr_edges.size > 0:
                state["hmr_lines"] = server.scene.add_line_segments(
                    "/original_hmr/skeleton",
                    points=hmr0[hmr_edges],
                    colors=(255, 190, 60),
                    line_width=3.0,
                    visible=bool(gui_show_hmr.value and gui_show_hmr_skeleton.value),
                )

            if track.scaled_human_joints is not None:
                scaled_edges = _hmr_edges_for_joint_count(track.scaled_human_joints.shape[1])
                scaled0 = track.scaled_human_joints[0]
                state["scaled_human_points"] = server.scene.add_point_cloud(
                    "/scaled_human/joints",
                    points=scaled0,
                    colors=(20, 170, 255),
                    point_size=0.035,
                    point_shape="circle",
                    visible=bool(gui_show_scaled_human.value),
                )
                if scaled_edges.size > 0:
                    state["scaled_human_lines"] = server.scene.add_line_segments(
                        "/scaled_human/skeleton",
                        points=scaled0[scaled_edges],
                        colors=(80, 220, 255),
                        line_width=3.0,
                        visible=bool(gui_show_scaled_human.value and gui_show_scaled_human_skeleton.value),
                    )

            gui_frame.max = max(0, int(track.qpos.shape[0]) - 1)
            gui_frame.value = 0
            gui_fps.value = int(track.fps)
            gui_status.content = (
                f"`{track.name}` frames={track.qpos.shape[0]} "
                f"hmr={track.hmr_joints.shape[0]} scaled_human="
                f"{None if track.scaled_human_joints is None else track.scaled_human_joints.shape[0]}"
            )
            _apply_frame(0.0)

            center = track.qpos[0, :3]
            for _, client in server.get_clients().items():
                client.camera.position = center + np.array([0.0, -3.0, 1.6])
                client.camera.look_at = center + np.array([0.0, 0.0, 0.8])

            print(
                f"[stairs-batch-player] active {track.name}: frames={track.qpos.shape[0]} "
                f"scaled_scene={track.scaled_scene_urdf}",
                flush=True,
            )

    @gui_sequence.on_update
    def _(_event) -> None:
        _activate_sequence(str(gui_sequence.value))

    @gui_frame.on_update
    def _(_event) -> None:
        with lock:
            state["frame"] = float(gui_frame.value)
            _apply_frame(float(gui_frame.value))

    for checkbox in (
        gui_show_g1,
        gui_show_scaled_scene,
        gui_show_unscaled_scene,
        gui_show_hmr,
        gui_show_hmr_skeleton,
        gui_show_scaled_human,
        gui_show_scaled_human_skeleton,
    ):

        @checkbox.on_update
        def _(_event) -> None:
            with lock:
                _apply_frame(float(gui_frame.value))

    def _player_loop() -> None:
        while True:
            time.sleep(1.0 / max(float(gui_fps.value), 1.0))
            with lock:
                track = state["track"]
                if track is None or not bool(gui_playing.value):
                    continue
                assert isinstance(track, SequenceTrack)
                frame = float(state["frame"]) + 1.0
                if frame > track.qpos.shape[0] - 1:
                    frame = 0.0
                state["frame"] = frame
                gui_frame.value = int(frame)
                _apply_frame(frame)

    print(f"[stairs-batch-player] sequences={len(sequences)} port={args.port}", flush=True)
    _activate_sequence(sequences[0])
    threading.Thread(target=_player_loop, daemon=True).start()
    print(f"[stairs-batch-player] ready: http://localhost:{args.port}", flush=True)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
