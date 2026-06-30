# Holosoma

Holosoma (Greek: "whole-body") is a comprehensive humanoid robotics framework for training and deploying reinforcement learning policies on humanoid robots, as well as motion retargeting. Supports locomotion (velocity tracking) and whole-body tracking tasks across multiple simulators (IsaacGym, IsaacSim, MJWarp, MuJoCo) with algorithms like PPO and FastSAC.

## Features

- **Multi-simulator support**: IsaacGym, IsaacSim, MuJoCo Warp (MJWarp), and MuJoCo (inference only)
- **Multiple RL algorithms**: PPO and FastSAC
- **Robot support**: Unitree G1 and Booster T1 humanoids
- **Task types**: Locomotion (velocity tracking) and whole-body tracking
- **Sim-to-sim and sim-to-real deployment**: Shared inference pipeline across simulation and real robot control
- **Motion retargeting**: Convert human motion capture data to robot motions while preserving interactions with objects and terrain
- **Wandb integration**: Video logging, automatic ONNX checkpoint uploads, and direct checkpoint loading from Wandb

## Repository Structure

```
src/
├── holosoma/              # Core training framework (locomotion & whole-body tracking)
├── holosoma_inference/    # Inference and deployment pipeline
└── holosoma_retargeting/  # Motion retargeting from human motion data to robots
```

## Documentation

- **[Training Guide](src/holosoma/README.md)** - Train locomotion and whole-body tracking policies in IsaacGym/IsaacSim
- **[Inference & Deployment Guide](src/holosoma_inference/README.md)** - Deploy policies to real robots or evaluate in MuJoCo simulation
- **[Retargeting Guide](src/holosoma_retargeting/holosoma_retargeting/README.md)** - Convert human motion capture data to robot motions

## Quick Start

### Setup

Choose the appropriate setup script based on your use case:

```bash
# For IsaacGym training
bash scripts/setup_isaacgym.sh

# For IsaacSim training
# Requires Ubuntu 22.04 or later due to IsaacSim dependencies
bash scripts/setup_isaacsim.sh

# For MJWarp training and MuJoCo simulation (inference) — conda
bash scripts/setup_mujoco.sh

# For MJWarp training and MuJoCo simulation (inference) — uv (alternative)
bash scripts/setup_mujoco_via_uv.sh

# For inference/deployment
bash scripts/setup_inference.sh

# For motion retargeting
bash scripts/setup_retargeting.sh
```

### Training

Train a G1 robot with FastSAC on IsaacGym:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1
```

> **Note:** For headless servers, see the [training guide](src/holosoma/README.md#video-recording) for video recording configuration.

See the [Training Guide](src/holosoma/README.md) for more examples and configuration options.

### CSP WBT Stair45 Debug Runs

`csp_blindwbt.sh` launches the no-heightmap stair_45 WBT debug training run that uses the checked-in CRISP stair motion and OBJ terrain:

```bash
cd /home/ubuntu/FAR/holosoma
./csp_blindwbt.sh
```

The heightmap-aware variant uses the same motion and OBJ terrain, but switches to the height-scan experiment:

```bash
cd /home/ubuntu/FAR/holosoma
./csp_heightmapwbt.sh
```

Both scripts start a detached tmux session by default, log shell output under `logs/run_commands/`, and push metrics to W&B project `zihanw22/holosomatest`. They use:

- 8 GPUs with 4096 envs per GPU by default, for 32768 envs total.
- `crisp_stairs/___crisp_clean_motion/stair_45.npz` as the motion file.
- `crisp_stairs/___crisp_clean_geometry/stair_45.obj` as the loaded OBJ terrain.
- PhysX GPU collision stack size `536870912`.
- Checkpoint save interval `1000`.

The blind script uses `exp:g1-29dof-wbt`, so there is no heightmap or height scanner observation. The heightmap script uses `exp:g1-29dof-wbt-height-scan`, explicitly enables `simulator.config.height_scanner`, and adds the `height_scan` term to actor and critic observations.

For the current 4-GPU stair45 heightmap debugging run:

```bash
NUM_GPUS=4 ENVS_PER_GPU=4096 ./csp_heightmapwbt.sh
```

The heightmap script also enables a flat floor patch under the loaded OBJ terrain, matching the far-tracking obstacle-plus-floor convention. This keeps pelvis-mounted RayCaster height scans from missing finite OBJ terrain before or beside the stairs. The default margin is 2m and can be changed with `LOAD_OBJ_FLOOR_MARGIN`.

Multi-GPU height-scan training relies on empirical observation normalization. The distributed variance path clamps variance to be non-negative before `sqrt()` because height scans contain many near-constant values and `E[x^2] - E[x]^2` can produce tiny negative values in float32; without that clamp the actor distribution can receive NaNs before the first rollout.

### CSP Multi-Terrain Heightmap WBT

`csp_multiterrain_heightmapwbt.sh` trains the heightmap-aware WBT policy on the CRISP motion-stairs batch as a true physics rollout. It is not a kinematics replay: the policy is trained in IsaacSim/PhysX against the loaded OBJ terrain, with the height scanner enabled.

The multi-terrain fuse follows the far-tracking convention: many motion/terrain pairs are represented as one combined terrain mesh. The important Holosoma-specific detail is that the fused motion NPZ carries a `terrain_origins` array. On every WBT reset, after `motion_id` is sampled, `MotionCommand` writes the corresponding `terrain_origins[motion_id]` into `scene.env_origins`, `simulator.env_origins`, and the locomotion terrain state. This keeps each sampled motion aligned with its matching translated terrain tile while preserving the existing motion position code that adds `env_origins` at read time.

Generate or refresh the fused CRISP stair assets:

```bash
python scripts/fuse_crisp_stairs_multiterrain.py
```

Default outputs:

- `crisp_stairs/_fused/motion_stairs_16_multiterrain.npz`
- `crisp_stairs/_fused/motion_stairs_16_multiterrain.obj`
- `crisp_stairs/_fused/motion_stairs_16_multiterrain.json`

Run the multi-terrain heightmap training entrypoint:

```bash
cd /home/ubuntu/FAR/holosoma
./csp_multiterrain_heightmapwbt.sh
```

The script defaults to 8 GPUs with 4096 envs per GPU and checkpoint save interval `1000`. It automatically builds the fused assets when missing, uses `exp:g1-29dof-wbt-height-scan`, and loads the fused OBJ with `num_rows=1` and `num_cols=1`. Those terrain grid overrides are required because the OBJ is already the full fused multi-terrain world; the WBT command handles per-motion origin placement. The multi-terrain script uses PhysX GPU collision stack size `1073741824` by default; the 512MB single-stair setting can overflow on the fused stair mesh and drop contacts.

`zhen_penalty` is an optional far-tracking-style foothold penalty for true physics rollout training. When enabled, IsaacSim/PhysX registers left/right foot RayCaster sensors on the ankle roll links, samples each contacting sole footprint against the loaded static triangle-mesh terrain, and penalizes the fraction of sole rays whose expected sole surface is more than `foothold_epsilon` above the terrain hit. A pelvis height scanner gates the penalty to locally rugged/stair-like terrain, so flat patches do not receive the same foothold penalty. The reward term exists in the G1 WBT reward config with weight `0.0` by default.

For multi-terrain debugging, the script defaults `USE_ADAPTIVE_TIMESTEPS_SAMPLER=False` and adds `noadaptive` to the run name. The original global adaptive timestep sampler bins failures over the concatenated fused motion frame axis. On the 16-motion stair batch this can collapse almost all resets onto one hard global bin, for example W&B run `h5xzojtc` showed sampler entropy near `0.02`, top1 probability around `0.989`, top1 bin around `0.897`, and episode length around `30`. That bin falls inside the later stair clip range, so the policy stops seeing a balanced distribution of terrains. Keep it off until we replace it with a per-motion or motion-balanced adaptive sampler.

### CSP Depth Student Distillation

`csp_depth_distill.sh` distills a trained terrain-aware FastSAC tracking teacher into a depth-based student. This is a true physics rollout, not kinematics replay: IsaacSim/PhysX steps the robot and static OBJ terrain, the frozen teacher produces rollout actions from its original tracking observations, and the student learns an MSE action loss from proprioception plus a ray-cast depth image.

The depth camera follows the far-tracking ZED2i-style setup: raw `106x60`, horizontal FOV `101.41` degrees, range `[0.3, 2.0]`, mounted on `torso_link` with offset `[0.125, 0.06, 0.04]` and RPY `[0, 71, 0]` degrees. IsaacLab's pinhole ray pattern already converts optical camera rays into the robotics camera frame, so we do not apply far-tracking's `offset_rot_base=[-90, 0, -90]` a second time. Distillation resizes the normalized depth to `58x87` before the student CNN.

Run distillation from an explicit teacher checkpoint:

```bash
cd /home/ubuntu/FAR/holosoma
TEACHER_CHECKPOINT=logs/holosomatest/.../model_01000.pt ./csp_depth_distill.sh
```

The script defaults to 8 GPUs and 1024 envs per GPU. Depth camera ray-casting is much heavier than the height scan, so this is intentionally lower than the 4096 env/GPU tracking default; override with `ENVS_PER_GPU=4096` only after confirming memory headroom. Outputs are saved under `logs/holosomatest/` as `student_*.pt` and `student_*.onnx`, and metrics go to W&B project `zihanw22/holosomatest`.

To start distillation from the latest checkpoint of the current multi-terrain tracking run after a delay:

```bash
DELAY_SECONDS=25200 \
TRACKING_SESSION=csp_multiterrain_heightmapwbt_20260630_053640 \
scripts/schedule_depth_distill_from_latest.sh
```

The scheduler reads `logs/run_commands/<tracking-session>.run_name`, finds the highest `model_*.pt` under `logs/holosomatest/`, stops the tracking tmux session, and launches `csp_depth_distill.sh`.

Useful overrides:

```bash
# Rebuild the fused assets before launch.
REBUILD_FUSED_ASSETS=1 ./csp_multiterrain_heightmapwbt.sh

# Use 4 GPUs for a smaller debug run.
NUM_GPUS=4 ENVS_PER_GPU=4096 ./csp_multiterrain_heightmapwbt.sh

# Enable the far-tracking-style foot RayCaster support penalty.
ENABLE_ZHEN_PENALTY=1 ZHEN_PENALTY_WEIGHT=-10.0 ./csp_multiterrain_heightmapwbt.sh

# Launch the same multi-terrain heightmap + zhen_penalty run on 4 remote nodes
# with 8 GPUs per node and 4096 envs per GPU.
./csp_multinode_multiterrain_heightmapwbt.sh

# Re-enable the old global adaptive sampler only for controlled experiments.
USE_ADAPTIVE_TIMESTEPS_SAMPLER=True ./csp_multiterrain_heightmapwbt.sh

# Run in the foreground and forward extra train_agent.py flags.
RUN_IN_TMUX=0 ./csp_multiterrain_heightmapwbt.sh --run --training.seed=3

# Fuse a smaller debug subset by requested clip ids or resolved clip names.
FUSE_CLIPS="45 3 56_outdoor 78_outdoor_stairs_up_down" \
REBUILD_FUSED_ASSETS=1 ./csp_multiterrain_heightmapwbt.sh
```

The multi-node launcher defaults to these non-local nodes:

```bash
NODE_HOSTS="10.0.74.86 10.0.100.200 10.0.72.226 10.0.90.122" \
./csp_multinode_multiterrain_heightmapwbt.sh
```

It starts one tmux session per node, uses `10.0.74.86` as the default torchrun master, and writes per-node logs as `logs/run_commands/<session>_node<rank>_<host>.log` on each remote node. By default it clones/syncs `https://github.com/Z1hanW/holosoma-crisp.git` into `/home/ubuntu/FAR/holosoma_crisp` so it does not touch any existing remote checkout at `/home/ubuntu/FAR/holosoma`. Override `NODE_HOSTS` to swap in the spare node `10.0.123.134`, and set `KILL_EXISTING=1` if reusing an existing session name intentionally.

Single-stair useful overrides:

```bash
# Run in the foreground instead of tmux.
RUN_IN_TMUX=0 ./csp_blindwbt.sh --run
RUN_IN_TMUX=0 ./csp_heightmapwbt.sh --run

# Change the W&B name, iteration count, or GPU/env layout.
RUN_NAME=my_debug_run NUM_ITERATIONS=2000 NUM_GPUS=8 ENVS_PER_GPU=4096 ./csp_heightmapwbt.sh

# Forward extra train_agent.py flags after --run in foreground mode.
RUN_IN_TMUX=0 ./csp_heightmapwbt.sh --run --training.seed=3

# Adjust the height scanner ray grid resolution.
HEIGHT_SCANNER_RESOLUTION=0.08 ./csp_heightmapwbt.sh

# Adjust the loaded OBJ floor patch used by heightmap training.
LOAD_OBJ_FLOOR_MARGIN=3.0 ./csp_heightmapwbt.sh
```

### Quick Demo

We provide scripts to run the complete pipeline: (data downloading and processing for LAFAN), retargeting, data conversion, and whole-body tracking policy training.

```bash
# Run retargeting and whole-body tracking policy training using OMOMO data
bash demo_scripts/demo_omomo_wb_tracking.sh

# Run retargeting and whole-body tracking policy training using LAFAN data
bash demo_scripts/demo_lafan_wb_tracking.sh
```

### Deployment & Evaluation

After training, deploy your policies:

- **Real Robot**: See [Real Robot Locomotion](src/holosoma_inference/docs/workflows/real-robot-locomotion.md) or [Real Robot WBT](src/holosoma_inference/docs/workflows/real-robot-wbt.md)
- **MuJoCo Simulation**: See [Sim-to-Sim Locomotion](src/holosoma_inference/docs/workflows/sim-to-sim-locomotion.md) or [Sim-to-Sim WBT](src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md)

Or browse all deployment options in the [Inference & Deployment Guide](src/holosoma_inference/README.md).

### Demo Videos

Watch real-world deployments of Holosoma policies *(click thumbnails to play)*

<table>
  <tr>
    <th>G1 Locomotion</th>
    <th>T1 Locomotion</th>
    <th>G1 Dancing</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="https://youtu.be/YYMgj5BDIMI">
        <img src="https://img.youtube.com/vi/YYMgj5BDIMI/hqdefault.jpg" width="100%" alt="▶ G1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/Q6rNHJZ2a6Y">
        <img src="https://img.youtube.com/vi/Q6rNHJZ2a6Y/hqdefault.jpg" width="100%" alt="▶ T1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/ouPk69_eFfE">
        <img src="https://img.youtube.com/vi/ouPk69_eFfE/hqdefault.jpg" width="100%" alt="▶ G1 Dancing">
      </a>
    </td>
  </tr>
</table>


## Issue Reporting

We welcome feedback and issue reports to help improve holosoma. Please use issues to:

- Report bugs and technical issues
- Request new features

## Support

If you need help with anything aside from issues feel free to join our [discord server](https://discord.gg/TPupMvpqHc).

Use the discord to discuss larger plans and other more involved problems.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Citation

If you use Holosoma in your research, please cite it according to the "Cite this repository" panel on the right sidebar of the Github repo.

## License

This project is licensed under the Apache-2.0 License.
