"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude, quat_rotate

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


# ================================================================================================
# Terrain Contact Rewards
# ================================================================================================


def _get_body_indices(body_names: tuple[str, ...], available_body_names: list[str], device: str) -> torch.Tensor:
    indexes = []
    for body_name in body_names:
        if body_name not in available_body_names:
            raise RuntimeError(f"Body '{body_name}' is not available. Available bodies: {available_body_names}")
        indexes.append(available_body_names.index(body_name))
    return torch.tensor(indexes, dtype=torch.long, device=device)


def _expand_ray_starts(ray_starts: torch.Tensor, num_envs: int) -> torch.Tensor:
    if ray_starts.ndim == 2:
        return ray_starts.unsqueeze(0).expand(num_envs, -1, -1)
    if ray_starts.ndim == 3 and ray_starts.shape[0] == 1:
        return ray_starts.expand(num_envs, -1, -1)
    return ray_starts


def zhen_penalty(
    env: WholeBodyTrackingManager,
    raycaster_names: tuple[str, str] = ("left_foot_raycaster", "right_foot_raycaster"),
    contact_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    foothold_epsilon: float = 0.1,
    contact_force_threshold: float = 50.0,
    sole_offset: float = 0.0347,
    height_scanner_name: str = "height_scanner",
    pelvis_window_half: float = 0.2,
    stair_ruggedness_thresh: float = 0.1,
) -> torch.Tensor:
    """Far-tracking-style foothold penalty for stair support.

    For each contacting foot, sample the sole footprint with an IsaacLab RayCaster.
    The raw penalty is the fraction of sole rays whose expected sole surface is
    more than ``foothold_epsilon`` above the terrain hit. A pelvis height scanner
    gates the penalty to rugged/stair-like terrain patches.
    """
    sensors = getattr(getattr(env.simulator, "scene", None), "sensors", {})
    missing_sensors = [sensor_name for sensor_name in raycaster_names if sensor_name not in sensors]
    if missing_sensors:
        raise RuntimeError(
            f"Foot raycaster sensor(s) {missing_sensors} are not available. "
            "Enable them with --simulator.config.foot-raycasters.enabled=True."
        )

    contact_body_indices = _get_body_indices(tuple(contact_body_names), env.simulator.body_names, env.device)
    contact_force_z = env.simulator.contact_forces_history[:, -1, contact_body_indices, 2]
    in_contact = contact_force_z > contact_force_threshold
    if not in_contact.any():
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    stair_present = torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
    if height_scanner_name in sensors:
        height_scanner = sensors[height_scanner_name]
        hs_starts = _expand_ray_starts(height_scanner.ray_starts, env.num_envs)
        hit_z_hs = height_scanner.data.ray_hits_w[..., 2]
        pelvis_window = (
            (hs_starts[..., 0].abs() <= pelvis_window_half)
            & (hs_starts[..., 1].abs() <= pelvis_window_half)
            & torch.isfinite(hit_z_hs)
        )
        hit_min = torch.where(pelvis_window, hit_z_hs, torch.full_like(hit_z_hs, float("inf"))).amin(dim=1)
        hit_max = torch.where(pelvis_window, hit_z_hs, torch.full_like(hit_z_hs, -float("inf"))).amax(dim=1)
        ruggedness = hit_max - hit_min
        stair_present = (
            torch.isfinite(hit_min) & torch.isfinite(hit_max) & (ruggedness > stair_ruggedness_thresh)
        ).float()

    per_foot_frac = torch.zeros(env.num_envs, len(raycaster_names), dtype=torch.float32, device=env.device)
    for foot_idx, sensor_name in enumerate(raycaster_names):
        if not in_contact[:, foot_idx].any():
            continue

        ray = sensors[sensor_name]
        ray_starts = _expand_ray_starts(ray.ray_starts, env.num_envs)
        num_rays = max(ray_starts.shape[1], 1)

        sole_local = ray_starts.clone()
        sole_local[..., 2] = -sole_offset
        foot_pos_w = ray.data.pos_w[:, None, :]
        foot_quat_w = ray.data.quat_w[:, None, :].expand(env.num_envs, num_rays, 4)
        sole_w = foot_pos_w + quat_rotate(
            foot_quat_w.reshape(-1, 4),
            sole_local.reshape(-1, 3),
            w_last=False,
        ).reshape(env.num_envs, num_rays, 3)

        hit_z = ray.data.ray_hits_w[..., 2]
        bad = (sole_w[..., 2] - hit_z > foothold_epsilon) & torch.isfinite(hit_z)
        per_foot_frac[:, foot_idx] = bad.float().sum(dim=1) / float(num_rays)

    return stair_present * torch.sum(in_contact.float() * per_foot_frac, dim=-1)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
