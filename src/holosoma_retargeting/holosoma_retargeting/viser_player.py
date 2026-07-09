#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # expected: qpos [T, ?], and optional fps
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data else 30
    scaled_human_joints = data["human_joints"] if "human_joints" in data else None
    if scaled_human_joints is not None:
        scaled_human_joints = np.asarray(scaled_human_joints, dtype=np.float32)
        if scaled_human_joints.ndim != 3 or scaled_human_joints.shape[-1] != 3:
            raise ValueError(
                f"Scaled human joints in {npz_path} must have shape (T, J, 3), got {scaled_human_joints.shape}"
            )
    return qpos, fps, scaled_human_joints


def load_hmr_joints(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in ("global_joint_positions", "human_joints", "joints", "hmr_joints"):
            if key in data:
                joints = data[key]
                break
        else:
            raise KeyError(f"No supported HMR joint key found in {path}. Keys: {data.files}")
    else:
        joints = data

    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"HMR joints must have shape (T, J, 3), got {joints.shape}")
    return joints


SMPLX_22_EDGES = np.asarray(
    [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 11),
        (9, 12),
        (9, 13),
        (9, 14),
        (12, 15),
        (13, 16),
        (14, 17),
        (16, 18),
        (17, 19),
        (18, 20),
        (19, 21),
    ],
    dtype=np.int32,
)


def _hmr_edges_for_joint_count(joint_count: int) -> np.ndarray:
    return SMPLX_22_EDGES[np.all(SMPLX_22_EDGES < joint_count, axis=1)]


def _sample_motion(motion: np.ndarray, frame_value: float, loop: bool) -> np.ndarray:
    n_frames = int(motion.shape[0])
    if n_frames == 1:
        return motion[0]

    if loop:
        frame_value = frame_value % n_frames
        i0 = int(np.floor(frame_value))
        i1 = (i0 + 1) % n_frames
    else:
        frame_value = float(np.clip(frame_value, 0.0, n_frames - 1))
        i0 = int(np.floor(frame_value))
        i1 = min(i0 + 1, n_frames - 1)
    u = float(frame_value - i0)
    return ((1.0 - u) * motion[i0] + u * motion[i1]).astype(np.float32)


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    scaled_human_joints: np.ndarray | None = None,
    fps: int | None = None,
):
    """
    qpos layout (MuJoCo order):
      [0:3]   robot base position (xyz)
      [3:7]   robot base quat (wxyz)
      [7:7+R] robot joint positions (R = actuated dof)
      [end-7:end-4] (optional) object position (xyz)
      [end-4:end]   (optional) object quat (wxyz)

    We'll infer R from the robot URDF's actuated joints in ViserUrdf.
    """
    server = viser.ViserServer(port=config.port)

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    # URDFs (using yourdfpy so meshes show up)
    robot_urdf_y = yourdfpy.URDF.load(config.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    vo = None
    if config.object_urdf:
        object_urdf_y = yourdfpy.URDF.load(config.object_urdf, load_meshes=True, build_scene_graph=True)
        vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

    vu = None
    if config.unscaled_object_urdf:
        unscaled_object_urdf_y = yourdfpy.URDF.load(
            config.unscaled_object_urdf,
            load_meshes=True,
            build_scene_graph=True,
        )
        vu = ViserUrdf(server, urdf_or_path=unscaled_object_urdf_y, root_node_name="/unscaled_object")

    hmr_joints = load_hmr_joints(config.hmr_joints_npy) if config.hmr_joints_npy else None
    hmr_points_handle = None
    hmr_lines_handle = None
    hmr_edges = None
    if hmr_joints is not None:
        hmr_edges = _hmr_edges_for_joint_count(hmr_joints.shape[1])
        hmr0 = hmr_joints[0]
        hmr_points_handle = server.scene.add_point_cloud(
            "/reference_hmr/joints",
            points=hmr0,
            colors=(255, 120, 20),
            point_size=config.hmr_point_size,
            point_shape="circle",
            visible=config.show_hmr,
        )
        if hmr_edges.size > 0:
            hmr_lines_handle = server.scene.add_line_segments(
                "/reference_hmr/skeleton",
                points=hmr0[hmr_edges],
                colors=(255, 190, 60),
                line_width=3.0,
                visible=config.show_hmr and config.show_hmr_skeleton,
            )

    scaled_human_points_handle = None
    scaled_human_lines_handle = None
    scaled_human_edges = None
    if scaled_human_joints is not None:
        scaled_human_edges = _hmr_edges_for_joint_count(scaled_human_joints.shape[1])
        scaled_human0 = scaled_human_joints[0]
        scaled_human_points_handle = server.scene.add_point_cloud(
            "/scaled_human/joints",
            points=scaled_human0,
            colors=(20, 170, 255),
            point_size=config.scaled_human_point_size,
            point_shape="circle",
            visible=config.show_scaled_human,
        )
        if scaled_human_edges.size > 0:
            scaled_human_lines_handle = server.scene.add_line_segments(
                "/scaled_human/skeleton",
                points=scaled_human0[scaled_human_edges],
                colors=(80, 220, 255),
                line_width=3.0,
                visible=config.show_scaled_human and config.show_scaled_human_skeleton,
            )

    # A tiny grid
    server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height, position=(0.0, 0.0, 0.0))

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)

    # Use fps from config if not provided, otherwise use the one from npz file
    actual_fps = fps if fps is not None else config.fps

    # Set initial mesh visibility
    vr.show_visual = config.show_meshes
    if vo is not None:
        vo.show_visual = config.show_meshes
    if vu is not None:
        vu.show_visual = config.show_unscaled_scene

    # ---------- Additional GUI controls (mesh visibility) ----------
    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show scaled robot/scene meshes", initial_value=config.show_meshes)
        show_unscaled_cb = (
            server.gui.add_checkbox("Show unscaled reference scene", initial_value=config.show_unscaled_scene)
            if vu is not None
            else None
        )
        show_hmr_cb = (
            server.gui.add_checkbox("Show original HMR", initial_value=config.show_hmr)
            if hmr_joints is not None
            else None
        )
        show_hmr_skeleton_cb = (
            server.gui.add_checkbox("Show HMR skeleton", initial_value=config.show_hmr_skeleton)
            if hmr_lines_handle is not None
            else None
        )
        show_scaled_human_cb = (
            server.gui.add_checkbox("Show scaled human joints", initial_value=config.show_scaled_human)
            if scaled_human_joints is not None
            else None
        )
        show_scaled_human_skeleton_cb = (
            server.gui.add_checkbox("Show scaled human skeleton", initial_value=config.show_scaled_human_skeleton)
            if scaled_human_lines_handle is not None
            else None
        )

    @show_meshes_cb.on_update
    def _(_):
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    if show_unscaled_cb is not None:

        @show_unscaled_cb.on_update
        def _(_):
            if vu is not None:
                vu.show_visual = bool(show_unscaled_cb.value)

    def _update_hmr_visibility() -> None:
        show_hmr = bool(show_hmr_cb.value) if show_hmr_cb is not None else config.show_hmr
        show_skel = bool(show_hmr_skeleton_cb.value) if show_hmr_skeleton_cb is not None else config.show_hmr_skeleton
        if hmr_points_handle is not None:
            hmr_points_handle.visible = show_hmr
        if hmr_lines_handle is not None:
            hmr_lines_handle.visible = show_hmr and show_skel

    if show_hmr_cb is not None:

        @show_hmr_cb.on_update
        def _(_):
            _update_hmr_visibility()

    if show_hmr_skeleton_cb is not None:

        @show_hmr_skeleton_cb.on_update
        def _(_):
            _update_hmr_visibility()

    def _update_scaled_human_visibility() -> None:
        show_scaled = (
            bool(show_scaled_human_cb.value) if show_scaled_human_cb is not None else config.show_scaled_human
        )
        show_skel = (
            bool(show_scaled_human_skeleton_cb.value)
            if show_scaled_human_skeleton_cb is not None
            else config.show_scaled_human_skeleton
        )
        if scaled_human_points_handle is not None:
            scaled_human_points_handle.visible = show_scaled
        if scaled_human_lines_handle is not None:
            scaled_human_lines_handle.visible = show_scaled and show_skel

    if show_scaled_human_cb is not None:

        @show_scaled_human_cb.on_update
        def _(_):
            _update_scaled_human_visibility()

    if show_scaled_human_skeleton_cb is not None:

        @show_scaled_human_skeleton_cb.on_update
        def _(_):
            _update_scaled_human_visibility()

    def _on_frame_update(frame_value: float, _q: np.ndarray) -> None:
        if hmr_joints is not None and hmr_points_handle is not None:
            hmr_frame = _sample_motion(hmr_joints, frame_value, config.loop)
            hmr_points_handle.points = hmr_frame
            if hmr_lines_handle is not None and hmr_edges is not None:
                hmr_lines_handle.points = hmr_frame[hmr_edges]

        if scaled_human_joints is not None and scaled_human_points_handle is not None:
            scaled_human_frame = _sample_motion(scaled_human_joints, frame_value, config.loop)
            scaled_human_points_handle.points = scaled_human_frame
            if scaled_human_lines_handle is not None and scaled_human_edges is not None:
                scaled_human_lines_handle.points = scaled_human_frame[scaled_human_edges]

    # ---------- Use reusable motion control sliders from viser_utils ----------
    create_motion_control_sliders(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        viser_object=vo if config.assume_object_in_qpos else None,
        object_base_frame=object_root if config.assume_object_in_qpos else None,
        contains_object_in_qpos=config.assume_object_in_qpos,
        initial_fps=actual_fps,
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
        on_frame_update=_on_frame_update if (hmr_joints is not None or scaled_human_joints is not None) else None,
    )
    n_frames = int(qpos.shape[0])
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={'yes' if (config.object_urdf and config.assume_object_in_qpos) else 'no'}"
    )
    if hmr_joints is not None:
        print(f"[viser_player] Loaded original HMR joints: {hmr_joints.shape} from {config.hmr_joints_npy}")
    if scaled_human_joints is not None:
        print(f"[viser_player] Loaded scaled human joints from qpos npz: {scaled_human_joints.shape}")
    if vu is not None:
        print(f"[viser_player] Loaded unscaled reference scene: {config.unscaled_object_urdf}")
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(cfg: ViserConfig) -> None:
    """Main function for viser player."""
    qpos, fps, scaled_human_joints = load_npz(cfg.qpos_npz)
    make_player(
        config=cfg,
        qpos=qpos,
        scaled_human_joints=scaled_human_joints,
        fps=fps,
    )

    # keep process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
