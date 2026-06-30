#!/usr/bin/env bash
set -euo pipefail

# CSP multi-terrain heightmap-aware WBT training.
# Fuses the CRISP stair motion/OBJ pairs into one motion NPZ + one terrain OBJ,
# then binds each sampled motion id to its matching terrain origin at reset.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

find_free_port() {
  python - "$@" <<'PY'
import socket
import sys

start = int(sys.argv[1]) if len(sys.argv) > 1 else 29565
for port in range(start, start + 200):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit("No free port found")
PY
}

quote() {
  printf "%q" "$1"
}

ensure_fused_assets() {
  if [[ "${BUILD_FUSED_ASSETS}" != "1" ]]; then
    return
  fi

  if [[ "${REBUILD_FUSED_ASSETS}" == "1" || ! -f "${MOTION_FILE}" || ! -f "${TERRAIN_OBJ}" ]]; then
    mkdir -p "${FUSED_DIR}"
    local fuse_args=(
      --crisp-root "${CRISP_ROOT}"
      --output-dir "${FUSED_DIR}"
      --prefix "${FUSED_PREFIX}"
    )
    if [[ -n "${FUSE_CLIPS}" ]]; then
      fuse_args+=(--clips "${FUSE_CLIPS}")
    fi
    python "${FUSE_SCRIPT}" "${fuse_args[@]}"
  fi
}

HOSTNAME_SHORT="${HOSTNAME_SHORT:-$(hostname)}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

WANDB_ENTITY="${WANDB_ENTITY:-zihanw22}"
WANDB_PROJECT="${WANDB_PROJECT:-holosomatest}"
NUM_GPUS="${NUM_GPUS:-8}"
ENVS_PER_GPU="${ENVS_PER_GPU:-4096}"
TOTAL_ENVS="${TOTAL_ENVS:-$((NUM_GPUS * ENVS_PER_GPU))}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-1073741824}"
PHYSX_STACK_LABEL="${PHYSX_STACK_LABEL:-physxstack$((PHYSX_GPU_COLLISION_STACK_SIZE / 1048576))m}"
HEIGHT_SCANNER_BODY_NAME="${HEIGHT_SCANNER_BODY_NAME:-pelvis}"
HEIGHT_SCANNER_RESOLUTION="${HEIGHT_SCANNER_RESOLUTION:-0.1}"
HEIGHT_SCANNER_DEBUG_VIS="${HEIGHT_SCANNER_DEBUG_VIS:-False}"
ENABLE_ZHEN_PENALTY="${ENABLE_ZHEN_PENALTY:-0}"
ZHEN_PENALTY_WEIGHT="${ZHEN_PENALTY_WEIGHT:--10.0}"
ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD="${ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD:-50.0}"
ZHEN_PENALTY_FOOTHOLD_EPSILON="${ZHEN_PENALTY_FOOTHOLD_EPSILON:-0.1}"
ZHEN_PENALTY_SOLE_OFFSET="${ZHEN_PENALTY_SOLE_OFFSET:-0.0347}"
ZHEN_PENALTY_PELVIS_WINDOW_HALF="${ZHEN_PENALTY_PELVIS_WINDOW_HALF:-0.2}"
ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH="${ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH:-0.1}"
FOOT_RAYCASTERS_DEBUG_VIS="${FOOT_RAYCASTERS_DEBUG_VIS:-False}"
if [[ "${ENABLE_ZHEN_PENALTY,,}" == "true" || "${ENABLE_ZHEN_PENALTY}" == "1" ]]; then
  ZHEN_PENALTY_LABEL="${ZHEN_PENALTY_LABEL:-_zhenpenalty}"
else
  ZHEN_PENALTY_LABEL="${ZHEN_PENALTY_LABEL:-}"
fi
USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}"
if [[ "${USE_ADAPTIVE_TIMESTEPS_SAMPLER,,}" == "true" || "${USE_ADAPTIVE_TIMESTEPS_SAMPLER}" == "1" ]]; then
  ADAPTIVE_TIMESTEPS_LABEL="${ADAPTIVE_TIMESTEPS_LABEL:-adaptive}"
else
  ADAPTIVE_TIMESTEPS_LABEL="${ADAPTIVE_TIMESTEPS_LABEL:-noadaptive}"
fi

CRISP_ROOT="${CRISP_ROOT:-${SCRIPT_DIR}/crisp_stairs}"
FUSED_DIR="${FUSED_DIR:-${CRISP_ROOT}/_fused}"
FUSED_PREFIX="${FUSED_PREFIX:-motion_stairs_16_multiterrain}"
FUSE_SCRIPT="${FUSE_SCRIPT:-${SCRIPT_DIR}/scripts/fuse_crisp_stairs_multiterrain.py}"
FUSE_CLIPS="${FUSE_CLIPS:-}"
BUILD_FUSED_ASSETS="${BUILD_FUSED_ASSETS:-1}"
REBUILD_FUSED_ASSETS="${REBUILD_FUSED_ASSETS:-0}"

MOTION_FILE="${MOTION_FILE:-${FUSED_DIR}/${FUSED_PREFIX}.npz}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${FUSED_DIR}/${FUSED_PREFIX}.obj}"
RUN_NAME="${RUN_NAME:-${HOSTNAME_SHORT}_g1_29dof_wbt_motionstairs16_csp_multiterrain_heightmapwbt${ZHEN_PENALTY_LABEL}_${ADAPTIVE_TIMESTEPS_LABEL}_${PHYSX_STACK_LABEL}_${NUM_GPUS}gpu_${ENVS_PER_GPU}env_${TIMESTAMP}}"
SESSION="${SESSION:-csp_multiterrain_heightmapwbt_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-logs/run_commands}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SESSION}.log}"
MASTER_PORT="${MASTER_PORT:-$(find_free_port 29565)}"

ensure_fused_assets

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "Missing motion file: ${MOTION_FILE}" >&2
  exit 1
fi

if [[ ! -f "${TERRAIN_OBJ}" ]]; then
  echo "Missing terrain OBJ: ${TERRAIN_OBJ}" >&2
  exit 1
fi

if [[ "${1:-}" != "--run" && "${RUN_IN_TMUX:-1}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
  printf "%s\n" "${RUN_NAME}" > "${LOG_DIR}/${SESSION}.run_name"

  TMUX_ENV="RUN_IN_TMUX=0 TIMESTAMP=$(quote "${TIMESTAMP}") HOSTNAME_SHORT=$(quote "${HOSTNAME_SHORT}") WANDB_ENTITY=$(quote "${WANDB_ENTITY}") WANDB_PROJECT=$(quote "${WANDB_PROJECT}") NUM_GPUS=$(quote "${NUM_GPUS}") ENVS_PER_GPU=$(quote "${ENVS_PER_GPU}") TOTAL_ENVS=$(quote "${TOTAL_ENVS}") NUM_ITERATIONS=$(quote "${NUM_ITERATIONS}") SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}") PHYSX_GPU_COLLISION_STACK_SIZE=$(quote "${PHYSX_GPU_COLLISION_STACK_SIZE}") PHYSX_STACK_LABEL=$(quote "${PHYSX_STACK_LABEL}") HEIGHT_SCANNER_BODY_NAME=$(quote "${HEIGHT_SCANNER_BODY_NAME}") HEIGHT_SCANNER_RESOLUTION=$(quote "${HEIGHT_SCANNER_RESOLUTION}") HEIGHT_SCANNER_DEBUG_VIS=$(quote "${HEIGHT_SCANNER_DEBUG_VIS}") ENABLE_ZHEN_PENALTY=$(quote "${ENABLE_ZHEN_PENALTY}") ZHEN_PENALTY_WEIGHT=$(quote "${ZHEN_PENALTY_WEIGHT}") ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD=$(quote "${ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD}") ZHEN_PENALTY_FOOTHOLD_EPSILON=$(quote "${ZHEN_PENALTY_FOOTHOLD_EPSILON}") ZHEN_PENALTY_SOLE_OFFSET=$(quote "${ZHEN_PENALTY_SOLE_OFFSET}") ZHEN_PENALTY_PELVIS_WINDOW_HALF=$(quote "${ZHEN_PENALTY_PELVIS_WINDOW_HALF}") ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH=$(quote "${ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH}") FOOT_RAYCASTERS_DEBUG_VIS=$(quote "${FOOT_RAYCASTERS_DEBUG_VIS}") ZHEN_PENALTY_LABEL=$(quote "${ZHEN_PENALTY_LABEL}") USE_ADAPTIVE_TIMESTEPS_SAMPLER=$(quote "${USE_ADAPTIVE_TIMESTEPS_SAMPLER}") ADAPTIVE_TIMESTEPS_LABEL=$(quote "${ADAPTIVE_TIMESTEPS_LABEL}") CRISP_ROOT=$(quote "${CRISP_ROOT}") FUSED_DIR=$(quote "${FUSED_DIR}") FUSED_PREFIX=$(quote "${FUSED_PREFIX}") FUSE_SCRIPT=$(quote "${FUSE_SCRIPT}") FUSE_CLIPS=$(quote "${FUSE_CLIPS}") BUILD_FUSED_ASSETS=$(quote "${BUILD_FUSED_ASSETS}") REBUILD_FUSED_ASSETS=$(quote "${REBUILD_FUSED_ASSETS}") MOTION_FILE=$(quote "${MOTION_FILE}") TERRAIN_OBJ=$(quote "${TERRAIN_OBJ}") RUN_NAME=$(quote "${RUN_NAME}") SESSION=$(quote "${SESSION}") LOG_DIR=$(quote "${LOG_DIR}") LOG_FILE=$(quote "${LOG_FILE}") MASTER_PORT=$(quote "${MASTER_PORT}")"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    TMUX_ENV="CUDA_VISIBLE_DEVICES=$(quote "${CUDA_VISIBLE_DEVICES}") ${TMUX_ENV}"
  fi
  TMUX_CMD="cd $(quote "${SCRIPT_DIR}") && env ${TMUX_ENV} bash $(quote "${SCRIPT_DIR}/csp_multiterrain_heightmapwbt.sh") --run > $(quote "${LOG_FILE}") 2>&1"

  tmux new-session -d -s "${SESSION}" "${TMUX_CMD}"
  echo "Started CSP multi-terrain heightmap-aware WBT training."
  echo "  session: ${SESSION}"
  echo "  run_name: ${RUN_NAME}"
  echo "  log: ${LOG_FILE}"
  echo "  master_port: ${MASTER_PORT}"
  echo "  total_envs: ${TOTAL_ENVS} (${NUM_GPUS} x ${ENVS_PER_GPU})"
  echo "  fused_motion: ${MOTION_FILE}"
  echo "  fused_terrain: ${TERRAIN_OBJ}"
  echo "  physx_gpu_collision_stack_size: ${PHYSX_GPU_COLLISION_STACK_SIZE}"
  echo "  height_scanner_body: ${HEIGHT_SCANNER_BODY_NAME}"
  echo "  height_scanner_resolution: ${HEIGHT_SCANNER_RESOLUTION}"
  echo "  enable_zhen_penalty: ${ENABLE_ZHEN_PENALTY}"
  echo "  zhen_penalty_weight: ${ZHEN_PENALTY_WEIGHT}"
  echo "  use_adaptive_timesteps_sampler: ${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
  exit 0
fi

if [[ "${1:-}" == "--run" ]]; then
  shift
fi

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
  || source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh

conda activate "${CONDA_ENV_NAME:-hssim}"
source scripts/source_isaacsim_setup.sh

export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

ZHEN_ARGS=()
if [[ "${ENABLE_ZHEN_PENALTY,,}" == "true" || "${ENABLE_ZHEN_PENALTY}" == "1" ]]; then
  ZHEN_ARGS=(
    --simulator.config.foot-raycasters.enabled=True
    --simulator.config.foot-raycasters.debug-vis="${FOOT_RAYCASTERS_DEBUG_VIS}"
    --reward.terms.zhen_penalty.weight="${ZHEN_PENALTY_WEIGHT}"
    --reward.terms.zhen_penalty.params.contact_force_threshold="${ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD}"
    --reward.terms.zhen_penalty.params.foothold_epsilon="${ZHEN_PENALTY_FOOTHOLD_EPSILON}"
    --reward.terms.zhen_penalty.params.sole_offset="${ZHEN_PENALTY_SOLE_OFFSET}"
    --reward.terms.zhen_penalty.params.pelvis_window_half="${ZHEN_PENALTY_PELVIS_WINDOW_HALF}"
    --reward.terms.zhen_penalty.params.stair_ruggedness_thresh="${ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH}"
  )
fi

torchrun \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NUM_GPUS}" \
  src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-height-scan \
  terrain:terrain-load-obj \
  logger:wandb \
  --training.multigpu=True \
  --training.num-envs="${TOTAL_ENVS}" \
  --training.project="${WANDB_PROJECT}" \
  --training.name="${RUN_NAME}" \
  --logger.entity="${WANDB_ENTITY}" \
  --logger.project="${WANDB_PROJECT}" \
  --logger.name="${RUN_NAME}" \
  --logger.video.enabled=False \
  --terrain.terrain-term.obj-file-path="${TERRAIN_OBJ}" \
  --terrain.terrain-term.num-rows=1 \
  --terrain.terrain-term.num-cols=1 \
  --terrain.terrain-term.load-obj-add-floor=False \
  --command.setup_terms.motion_command.params.motion_config.motion_file="${MOTION_FILE}" \
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler="${USE_ADAPTIVE_TIMESTEPS_SAMPLER}" \
  --simulator.config.scene.env-spacing=0.0 \
  --simulator.config.height-scanner.enabled=True \
  --simulator.config.height-scanner.body-name="${HEIGHT_SCANNER_BODY_NAME}" \
  --simulator.config.height-scanner.resolution="${HEIGHT_SCANNER_RESOLUTION}" \
  --simulator.config.height-scanner.debug-vis="${HEIGHT_SCANNER_DEBUG_VIS}" \
  --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE}" \
  --algo.config.num-learning-iterations="${NUM_ITERATIONS}" \
  --algo.config.save-interval="${SAVE_INTERVAL}" \
  "${ZHEN_ARGS[@]}" \
  "$@"
