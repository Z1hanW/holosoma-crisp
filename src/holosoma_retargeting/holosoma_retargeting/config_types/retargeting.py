"""Configuration types for retargeting (top-level config)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig


@dataclass
class RetargetingConfig:
    """Top-level retargeting configuration used by the Tyro CLI.

    This combines all configuration types needed for retargeting.
    """

    # --- Task type selection ---
    task_type: Literal["robot_only", "object_interaction", "climbing"] = "object_interaction"
    """Type of retargeting task."""

    # --- top-level run knobs ---
    robot: str = "g1"
    """Robot type. Use str to allow dynamic robot types via _ROBOT_DEFAULTS."""

    data_format: str | None = None
    """Motion data format. Auto-determined by task_type if None.
    Can be any format registered in DEMO_JOINTS_REGISTRY
    (e.g., 'lafan', 'smplh', 'mocap', 'smplx', or custom formats)."""

    task_name: str = "sub3_largebox_003"
    """Name of the task/sequence."""

    data_path: Path = Path("demo_data/OMOMO_new")
    """Path to data directory."""

    save_dir: Path | None = None
    """Directory to save results. Auto-determined if None."""

    augmentation: bool = False
    """Whether to use augmentation."""

    first_frame_scene_z_translation: bool = False
    """Temporarily translate climbing human joints in z using frame-0 scene geometry clearance."""

    first_frame_scene_z_clearance: float = 0.10
    """Minimum clearance above frame-0 scene geometry for lower-body reference joints."""

    first_frame_scene_z_max_abs: float = 0.0
    """Optional absolute clamp for the first-frame z translation. 0 disables clamping."""

    scene_raycast_z_alignment: bool = False
    """For climbing data, align human joint z using -z ray hits against the scene mesh before scaling."""

    scene_raycast_z_mode: Literal["per_frame", "sequence", "global"] = "per_frame"
    """Raycast z alignment mode: per_frame adjusts each frame; sequence/global applies one global offset."""

    scene_raycast_z_clearance: float = 0.0
    """Additional clearance above the raycast scene hit, in the unscaled input frame."""

    scene_raycast_z_global_percentile: float = 95.0
    """Percentile of per-frame max raycast deficits used for sequence/global z alignment. Use 100 for strict non-penetration."""

    climbing_motion_root_nominal: bool = False
    """For climbing original runs, add a per-frame root nominal from the preprocessed human motion to prevent IK drift."""

    # --- Nested configs ---
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    """Robot configuration (nested - can override robot_urdf_file, robot_dof, etc.
    via --robot-config.robot-urdf-file)."""

    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplh", robot_type="g1")
    )
    """Motion data configuration (nested - can override demo_joints, joints_mapping, etc.
    via --motion-data-config.demo-joints).
    Note: data_format default will be set based on task_type in main()."""

    task_config: TaskConfig = field(default_factory=TaskConfig)
    """Task-specific configuration (nested - can override ground_size, surface_weight_threshold, etc.
    via --task-config.ground-size)."""

    retargeter: RetargeterConfig = field(default_factory=RetargeterConfig)
    """Retargeter configuration (nested - can override q_a_init_idx, activate_joint_limits, etc.
    via --retargeter.q-a-init-idx)."""


@dataclass
class ParallelRetargetingConfig(RetargetingConfig):
    """Extended retargeting config for parallel processing.

    Adds parallel-specific fields while inheriting all retargeting config fields.
    This config is used for processing multiple files in parallel.
    """

    # Parallel processing specific fields
    data_dir: Path = Path("demo_data/OMOMO_new")
    """Directory containing input data files for parallel processing.
    This overrides data_path from RetargetingConfig when processing multiple files."""

    max_workers: int | None = None
    """Maximum number of parallel workers. Auto-determined if None."""
