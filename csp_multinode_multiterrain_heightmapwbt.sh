#!/usr/bin/env bash
set -euo pipefail

# CSP multi-node multi-terrain heightmap-aware WBT training.
# Default topology excludes the local node and uses four remote g6e.48xlarge nodes:
#   4 nodes x 8 GPUs x 4096 envs/GPU = 131072 total envs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

quote() {
  printf "%q" "$1"
}

find_remote_free_port() {
  local host="$1"
  local start="${2:-29632}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" "python3 - <<PY
import socket
start = int(${start})
for port in range(start, start + 200):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('', port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit('No free port found')
PY"
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

DEFAULT_NODE_HOSTS="10.0.74.86 10.0.100.200 10.0.72.226 10.0.90.122"
NODE_HOSTS="${NODE_HOSTS:-${DEFAULT_NODE_HOSTS}}"
read -r -a NODE_HOST_ARRAY <<< "${NODE_HOSTS}"
if [[ "${#NODE_HOST_ARRAY[@]}" -lt 1 ]]; then
  echo "NODE_HOSTS is empty." >&2
  exit 1
fi

NNODES="${NNODES:-${#NODE_HOST_ARRAY[@]}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ENVS_PER_GPU="${ENVS_PER_GPU:-4096}"
TOTAL_GPUS="${TOTAL_GPUS:-$((NNODES * GPUS_PER_NODE))}"
TOTAL_ENVS="${TOTAL_ENVS:-$((TOTAL_GPUS * ENVS_PER_GPU))}"

HOSTNAME_SHORT="${HOSTNAME_SHORT:-$(hostname)}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
MASTER_ADDR="${MASTER_ADDR:-${NODE_HOST_ARRAY[0]}}"
MASTER_PORT="${MASTER_PORT:-$(find_remote_free_port "${MASTER_ADDR}" 29632)}"

WANDB_ENTITY="${WANDB_ENTITY:-zihanw22}"
WANDB_PROJECT="${WANDB_PROJECT:-holosomatest}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-1073741824}"
PHYSX_STACK_LABEL="${PHYSX_STACK_LABEL:-physxstack$((PHYSX_GPU_COLLISION_STACK_SIZE / 1048576))m}"
HEIGHT_SCANNER_BODY_NAME="${HEIGHT_SCANNER_BODY_NAME:-pelvis}"
HEIGHT_SCANNER_RESOLUTION="${HEIGHT_SCANNER_RESOLUTION:-0.1}"
HEIGHT_SCANNER_DEBUG_VIS="${HEIGHT_SCANNER_DEBUG_VIS:-False}"
FOOT_RAYCASTERS_DEBUG_VIS="${FOOT_RAYCASTERS_DEBUG_VIS:-False}"
ENABLE_ZHEN_PENALTY="${ENABLE_ZHEN_PENALTY:-1}"
ZHEN_PENALTY_WEIGHT="${ZHEN_PENALTY_WEIGHT:--10.0}"
ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD="${ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD:-50.0}"
ZHEN_PENALTY_FOOTHOLD_EPSILON="${ZHEN_PENALTY_FOOTHOLD_EPSILON:-0.1}"
ZHEN_PENALTY_SOLE_OFFSET="${ZHEN_PENALTY_SOLE_OFFSET:-0.0347}"
ZHEN_PENALTY_PELVIS_WINDOW_HALF="${ZHEN_PENALTY_PELVIS_WINDOW_HALF:-0.2}"
ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH="${ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH:-0.1}"
USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/FAR/holosoma_crisp}"
REMOTE_GIT_URL="${REMOTE_GIT_URL:-https://github.com/Z1hanW/holosoma-crisp.git}"
SYNC_REPO="${SYNC_REPO:-1}"
KILL_EXISTING="${KILL_EXISTING:-0}"

CRISP_ROOT="${CRISP_ROOT:-${REMOTE_REPO}/crisp_stairs}"
FUSED_DIR="${FUSED_DIR:-${CRISP_ROOT}/_fused}"
FUSED_PREFIX="${FUSED_PREFIX:-motion_stairs_16_multiterrain}"
FUSE_SCRIPT="${FUSE_SCRIPT:-${REMOTE_REPO}/scripts/fuse_crisp_stairs_multiterrain.py}"
FUSE_CLIPS="${FUSE_CLIPS:-}"
BUILD_FUSED_ASSETS="${BUILD_FUSED_ASSETS:-1}"
REBUILD_FUSED_ASSETS="${REBUILD_FUSED_ASSETS:-0}"
MOTION_FILE="${MOTION_FILE:-${FUSED_DIR}/${FUSED_PREFIX}.npz}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${FUSED_DIR}/${FUSED_PREFIX}.obj}"

MASTER_LABEL="${MASTER_ADDR//./-}"
RUN_NAME="${RUN_NAME:-${HOSTNAME_SHORT}_g1_29dof_wbt_motionstairs16_csp_multinode4x8_multiterrain_heightmapwbt_zhenpenalty_noadaptive_${PHYSX_STACK_LABEL}_${TOTAL_GPUS}gpu_${ENVS_PER_GPU}env_master${MASTER_LABEL}_${TIMESTAMP}}"
SESSION="${SESSION:-csp_multinode_heightmapwbt_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-logs/run_commands}"
SCRIPT_BASENAME="$(basename "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--node-run" ]]; then
  shift
  NODE_RANK="${NODE_RANK:?NODE_RANK must be set for --node-run}"

  ensure_fused_assets
  if [[ ! -f "${MOTION_FILE}" ]]; then
    echo "Missing motion file: ${MOTION_FILE}" >&2
    exit 1
  fi
  if [[ ! -f "${TERRAIN_OBJ}" ]]; then
    echo "Missing terrain OBJ: ${TERRAIN_OBJ}" >&2
    exit 1
  fi

  source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
    || source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh

  conda activate "${CONDA_ENV_NAME:-hssim}"
  source scripts/source_isaacsim_setup.sh

  export PYTHONPATH="${REMOTE_REPO}/src/holosoma:${PYTHONPATH:-}"
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
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
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
  exit 0
fi

mkdir -p "${LOG_DIR}"
printf "%s\n" "${RUN_NAME}" > "${LOG_DIR}/${SESSION}.run_name"
printf "%s\n" "${NODE_HOST_ARRAY[@]}" > "${LOG_DIR}/${SESSION}.nodes"

echo "Launching CSP multi-node heightmap WBT training."
echo "  session: ${SESSION}"
echo "  run_name: ${RUN_NAME}"
echo "  nodes: ${NODE_HOST_ARRAY[*]}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  total_envs: ${TOTAL_ENVS} (${NNODES} nodes x ${GPUS_PER_NODE} GPUs x ${ENVS_PER_GPU} envs/GPU)"
echo "  zhen_penalty_weight: ${ZHEN_PENALTY_WEIGHT}"

for node_rank in "${!NODE_HOST_ARRAY[@]}"; do
  host="${NODE_HOST_ARRAY[$node_rank]}"
  remote_log="${REMOTE_REPO}/${LOG_DIR}/${SESSION}_node${node_rank}_${host//./-}.log"

  if [[ "${SYNC_REPO}" == "1" ]]; then
    remote_parent="$(dirname "${REMOTE_REPO}")"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
      "if [[ ! -d $(quote "${REMOTE_REPO}/.git") ]]; then mkdir -p $(quote "${remote_parent}") && git clone $(quote "${REMOTE_GIT_URL}") $(quote "${REMOTE_REPO}"); fi && cd $(quote "${REMOTE_REPO}") && git fetch $(quote "${REMOTE_GIT_URL}") main && git merge --ff-only FETCH_HEAD"
  fi

  if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
    "tmux has-session -t $(quote "${SESSION}") 2>/dev/null"; then
    if [[ "${KILL_EXISTING}" == "1" ]]; then
      ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
        "tmux kill-session -t $(quote "${SESSION}")"
    else
      echo "Remote tmux session already exists on ${host}: ${SESSION}" >&2
      exit 1
    fi
  fi

  remote_env="NODE_RANK=$(quote "${node_rank}") NNODES=$(quote "${NNODES}") GPUS_PER_NODE=$(quote "${GPUS_PER_NODE}") ENVS_PER_GPU=$(quote "${ENVS_PER_GPU}") TOTAL_GPUS=$(quote "${TOTAL_GPUS}") TOTAL_ENVS=$(quote "${TOTAL_ENVS}") MASTER_ADDR=$(quote "${MASTER_ADDR}") MASTER_PORT=$(quote "${MASTER_PORT}") TIMESTAMP=$(quote "${TIMESTAMP}") HOSTNAME_SHORT=$(quote "${HOSTNAME_SHORT}") WANDB_ENTITY=$(quote "${WANDB_ENTITY}") WANDB_PROJECT=$(quote "${WANDB_PROJECT}") NUM_ITERATIONS=$(quote "${NUM_ITERATIONS}") SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}") PHYSX_GPU_COLLISION_STACK_SIZE=$(quote "${PHYSX_GPU_COLLISION_STACK_SIZE}") PHYSX_STACK_LABEL=$(quote "${PHYSX_STACK_LABEL}") HEIGHT_SCANNER_BODY_NAME=$(quote "${HEIGHT_SCANNER_BODY_NAME}") HEIGHT_SCANNER_RESOLUTION=$(quote "${HEIGHT_SCANNER_RESOLUTION}") HEIGHT_SCANNER_DEBUG_VIS=$(quote "${HEIGHT_SCANNER_DEBUG_VIS}") FOOT_RAYCASTERS_DEBUG_VIS=$(quote "${FOOT_RAYCASTERS_DEBUG_VIS}") ENABLE_ZHEN_PENALTY=$(quote "${ENABLE_ZHEN_PENALTY}") ZHEN_PENALTY_WEIGHT=$(quote "${ZHEN_PENALTY_WEIGHT}") ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD=$(quote "${ZHEN_PENALTY_CONTACT_FORCE_THRESHOLD}") ZHEN_PENALTY_FOOTHOLD_EPSILON=$(quote "${ZHEN_PENALTY_FOOTHOLD_EPSILON}") ZHEN_PENALTY_SOLE_OFFSET=$(quote "${ZHEN_PENALTY_SOLE_OFFSET}") ZHEN_PENALTY_PELVIS_WINDOW_HALF=$(quote "${ZHEN_PENALTY_PELVIS_WINDOW_HALF}") ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH=$(quote "${ZHEN_PENALTY_STAIR_RUGGEDNESS_THRESH}") USE_ADAPTIVE_TIMESTEPS_SAMPLER=$(quote "${USE_ADAPTIVE_TIMESTEPS_SAMPLER}") REMOTE_REPO=$(quote "${REMOTE_REPO}") REMOTE_GIT_URL=$(quote "${REMOTE_GIT_URL}") CRISP_ROOT=$(quote "${CRISP_ROOT}") FUSED_DIR=$(quote "${FUSED_DIR}") FUSED_PREFIX=$(quote "${FUSED_PREFIX}") FUSE_SCRIPT=$(quote "${FUSE_SCRIPT}") FUSE_CLIPS=$(quote "${FUSE_CLIPS}") BUILD_FUSED_ASSETS=$(quote "${BUILD_FUSED_ASSETS}") REBUILD_FUSED_ASSETS=$(quote "${REBUILD_FUSED_ASSETS}") MOTION_FILE=$(quote "${MOTION_FILE}") TERRAIN_OBJ=$(quote "${TERRAIN_OBJ}") RUN_NAME=$(quote "${RUN_NAME}") SESSION=$(quote "${SESSION}") LOG_DIR=$(quote "${LOG_DIR}")"

  remote_cmd="cd $(quote "${REMOTE_REPO}") && mkdir -p $(quote "${LOG_DIR}") && env ${remote_env} bash $(quote "${REMOTE_REPO}/${SCRIPT_BASENAME}") --node-run > $(quote "${remote_log}") 2>&1"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
    "tmux new-session -d -s $(quote "${SESSION}") $(quote "${remote_cmd}")"
  echo "  node_rank ${node_rank}: ${host} -> ${remote_log}"
done
