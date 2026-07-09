"""Configuration types for viser visualization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViserConfig:
    """Configuration for viser player visualization.

    This follows the pattern from holosoma's config_types.
    Uses a flat structure with default values.
    """

    qpos_npz: str = "rt_results/OMOMO_new/box_parallel/sub8_largebox_051_original.npz"
    """Path to .npz file with qpos data."""

    port: int = 8080
    """Port for the Viser web server."""

    robot_urdf: str = "models/g1/g1_29dof.urdf"
    """Path to robot URDF file."""

    object_urdf: str | None = None
    """Path to object URDF file (optional)."""

    unscaled_object_urdf: str | None = None
    """Path to an unscaled reference object/scene URDF file (optional)."""

    hmr_joints_npy: str | None = None
    """Path to original HMR world joints in .npy or .npz format (optional)."""

    fps: int = 30
    """Frames per second for playback."""

    assume_object_in_qpos: bool = True
    """Whether object pose is included in qpos array."""

    loop: bool = False
    """Whether to loop playback."""

    show_meshes: bool = True
    """Whether to show mesh visualizations."""

    show_unscaled_scene: bool = True
    """Whether to show the unscaled reference scene when provided."""

    show_hmr: bool = True
    """Whether to show original HMR joints when provided."""

    show_hmr_skeleton: bool = True
    """Whether to draw line segments between HMR joints when provided."""

    show_scaled_human: bool = True
    """Whether to show scaled human joints from the qpos npz when available."""

    show_scaled_human_skeleton: bool = True
    """Whether to draw line segments between scaled human joints when available."""

    hmr_point_size: float = 0.035
    """Point size for original HMR joint visualization."""

    scaled_human_point_size: float = 0.035
    """Point size for scaled human joint visualization."""

    grid_width: float = 8.0
    """Grid width for visualization."""

    grid_height: float = 8.0
    """Grid height for visualization."""

    visual_fps_multiplier: int = 2
    """Visual FPS multiplier for interpolation."""

    min_fps: int = 1
    """Minimum FPS setting."""

    max_fps: int = 240
    """Maximum FPS setting."""

    min_interp_mult: int = 1
    """Minimum interpolation multiplier."""

    max_interp_mult: int = 8
    """Maximum interpolation multiplier."""
