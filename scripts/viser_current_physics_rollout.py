#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import tyro
from loguru import logger

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_checkpoint, load_saved_experiment_config
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Current-repo IsaacSim physics rollout streamed to Viser.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint path or wandb:// checkpoint URI.")
    parser.add_argument("--port", type=int, default=2099, help="Viser server port.")
    parser.add_argument("--env-id", type=int, default=0, help="Environment index to stream.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until stopped.")
    parser.add_argument("--update-hz", type=float, default=30.0, help="Viser publish rate.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-tiles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xy-offset-range", type=float, default=1.0)
    parser.add_argument("--disable-randomization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Additional ExperimentConfig overrides after '--', using the same Tyro syntax as eval_agent.py.",
    )
    return parser.parse_args()


def _resolve_holosoma_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _robot_urdf_path(config: ExperimentConfig) -> Path:
    asset_root = _resolve_holosoma_path(config.robot.asset.asset_root)
    return asset_root / config.robot.asset.urdf_file


def _disable_randomization(config: ExperimentConfig) -> ExperimentConfig:
    if config.randomization is None:
        return config

    def _disable_term(term: RandomizationTermCfg) -> RandomizationTermCfg:
        params = dict(term.params)
        if "enabled" in params:
            params["enabled"] = False
        return dataclasses.replace(term, params=params)

    setup_terms = {name: _disable_term(term) for name, term in config.randomization.setup_terms.items()}
    reset_terms = {
        name: term
        for name, term in config.randomization.reset_terms.items()
        if name
        not in {
            "push_randomizer_state",
            "randomize_push_schedule",
            "randomize_action_delay",
            "randomize_dof_state",
            "actuator_randomizer_state",
        }
    }
    step_terms = {
        name: term
        for name, term in config.randomization.step_terms.items()
        if name not in {"push_randomizer_state", "apply_pushes"}
    }
    randomization = RandomizationManagerCfg(
        setup_terms=setup_terms,
        reset_terms=reset_terms,
        step_terms=step_terms,
        ignore_unsupported=config.randomization.ignore_unsupported,
    )
    return dataclasses.replace(config, randomization=randomization)


def _make_eval_config(args: argparse.Namespace, saved_config: ExperimentConfig) -> ExperimentConfig:
    eval_config = saved_config.get_eval_config()
    spawn = dataclasses.replace(
        eval_config.terrain.terrain_term.spawn,
        randomize_tiles=bool(args.randomize_tiles),
        xy_offset_range=float(args.xy_offset_range),
    )
    eval_config = dataclasses.replace(
        eval_config,
        terrain=dataclasses.replace(
            eval_config.terrain,
            terrain_term=dataclasses.replace(eval_config.terrain.terrain_term, spawn=spawn),
        ),
        training=dataclasses.replace(
            eval_config.training,
            headless=bool(args.headless),
            num_envs=max(int(args.env_id) + 1, 1),
            max_eval_steps=None if args.max_steps <= 0 else int(args.max_steps),
            export_onnx=False,
        ),
    )
    if args.disable_randomization:
        eval_config = _disable_randomization(eval_config)

    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    if overrides:
        eval_config = tyro.cli(
            ExperimentConfig,
            default=eval_config,
            args=overrides,
            description="ExperimentConfig overrides.",
            config=TYRO_CONIFG,
        )
    return eval_config


def _load_mesh_for_viser(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh or Scene, got {type(mesh)}")
    return mesh


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _tensor_row_to_numpy(value: Any, env_id: int) -> np.ndarray:
    return value[env_id].detach().cpu().numpy()


def main() -> None:
    args = _parse_args()
    init_eval_logging()
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    checkpoint_cfg = CheckpointConfig(checkpoint=args.checkpoint)
    saved_config, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_config = _make_eval_config(args, saved_config)

    env, device, simulation_app = setup_simulation_environment(eval_config)
    try:
        eval_log_dir = get_experiment_dir(eval_config.logger, eval_config.training, get_timestamp(), task_name="eval")
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        eval_config.save_config(str(eval_log_dir / CONFIG_NAME))

        checkpoint = load_checkpoint(args.checkpoint, str(eval_log_dir))
        algo_class = get_class(eval_config.algo._target_)
        algo = algo_class(device=device, env=env, config=eval_config.algo.config, log_dir=str(eval_log_dir), multi_gpu_cfg=None)
        algo.setup()
        algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
        algo.load(str(checkpoint))

        import torch
        import viser
        from viser.extras import ViserUrdf

        server = viser.ViserServer(host="0.0.0.0", port=int(args.port), label="holosoma_current_physics")
        terrain_state = env.terrain_manager.get_state("locomotion_terrain")
        terrain_mesh = _load_mesh_for_viser(terrain_state.mesh)
        server.scene.add_mesh_simple(
            "/terrain",
            vertices=np.asarray(terrain_mesh.vertices, dtype=np.float32),
            faces=np.asarray(terrain_mesh.faces, dtype=np.int32),
            color=(95, 95, 95),
            opacity=0.78,
            side="double",
        )

        urdf_path = _robot_urdf_path(eval_config)
        logger.info("Viser using current-repo URDF: {}", urdf_path)
        robot_viser = ViserUrdf(server, urdf_path, root_node_name="/robot", load_meshes=True, load_collision_meshes=False)
        robot_root = robot_viser._visual_root_frame
        if robot_root is None:
            raise RuntimeError("ViserUrdf did not create a visual root frame.")

        viser_joint_names = list(robot_viser.get_actuated_joint_names())
        dof_name_to_idx = {name: i for i, name in enumerate(env.dof_names)}
        missing = [name for name in viser_joint_names if name not in dof_name_to_idx]
        if missing:
            raise RuntimeError(f"Viser joints missing from simulator DOFs: {missing}")
        viser_to_sim = torch.tensor([dof_name_to_idx[name] for name in viser_joint_names], device=device, dtype=torch.long)

        logger.info("Viser listening on http://localhost:{}", args.port)
        logger.info(
            "Streaming env {} from true IsaacSim physics; randomize_tiles={} xy_offset_range={} randomization_disabled={}",
            args.env_id,
            args.randomize_tiles,
            args.xy_offset_range,
            args.disable_randomization,
        )

        algo._create_eval_callbacks()
        algo._pre_evaluate_policy()
        actor_state = algo._create_actor_state()
        algo.eval_policy = algo.get_inference_policy()

        obs_dict = env.reset_all()
        init_actions = torch.zeros(env.num_envs, algo.num_act, device=device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        critic_obs = torch.cat([actor_state["obs"][key] for key in algo.critic_obs_keys], dim=1)
        actor_state["obs"]["critic_obs"] = critic_obs
        actor_state = algo._pre_eval_env_step(actor_state)

        motion_state = env.command_manager.get_state("motion_command")
        min_period = 0.0 if args.update_hz <= 0 else 1.0 / float(args.update_hz)
        last_publish = 0.0
        step = 0
        try:
            while args.max_steps <= 0 or step < args.max_steps:
                actor_state["step"] = step
                actor_state = algo._pre_eval_env_step(actor_state)
                actor_state = algo.env_step(actor_state)
                actor_state = algo._post_eval_env_step(actor_state)

                now = time.monotonic()
                if now - last_publish >= min_period:
                    env.simulator.refresh_sim_tensors()
                    root_state = _tensor_row_to_numpy(env.simulator.robot_root_states, int(args.env_id))
                    dof_pos = _tensor_row_to_numpy(env.simulator.dof_pos[:, viser_to_sim], int(args.env_id))
                    robot_root.position = root_state[:3].astype(np.float32)
                    robot_root.wxyz = _xyzw_to_wxyz(root_state[3:7])
                    robot_viser.update_cfg(dof_pos.astype(np.float32))
                    last_publish = now

                if args.log_every > 0 and step % int(args.log_every) == 0:
                    root_state = _tensor_row_to_numpy(env.simulator.robot_root_states, int(args.env_id))
                    msg = f"step={step} root_xyz=({root_state[0]:.3f}, {root_state[1]:.3f}, {root_state[2]:.3f})"
                    if motion_state is not None and getattr(motion_state, "metrics", None):
                        metrics = motion_state.metrics
                        if "motion/error_ref_pos" in metrics:
                            err = float(metrics["motion/error_ref_pos"][int(args.env_id)].detach().cpu())
                            msg += f" error_ref_pos={err:.4f}"
                        if "motion/error_body_pos" in metrics:
                            err = float(metrics["motion/error_body_pos"][int(args.env_id)].detach().cpu())
                            msg += f" error_body_pos={err:.4f}"
                    logger.info(msg)
                step += 1
        finally:
            algo._post_evaluate_policy()
            logger.info("Viser rollout loop exited at step {}", step)
            server.stop()
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
