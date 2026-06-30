#!/usr/bin/env bash
set -euo pipefail

# CSP blind WBT debug training: no heightmap / no height scanner observation.
# Defaults reproduce the current stair_45 run:
#   - 8 GPUs
#   - 4096 envs per GPU (32768 total)
#   - stair_45 motion + stair_45 OBJ terrain
#   - W&B project holosomatest
#   - checkpoints every 1000 iterations

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

find_free_port() {
  python - "$@" <<'PY'
import socket
import sys

start = int(sys.argv[1]) if len(sys.argv) > 1 else 29545
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

HOSTNAME_SHORT="${HOSTNAME_SHORT:-$(hostname)}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

WANDB_ENTITY="${WANDB_ENTITY:-zihanw22}"
WANDB_PROJECT="${WANDB_PROJECT:-holosomatest}"
NUM_GPUS="${NUM_GPUS:-8}"
ENVS_PER_GPU="${ENVS_PER_GPU:-4096}"
TOTAL_ENVS="${TOTAL_ENVS:-$((NUM_GPUS * ENVS_PER_GPU))}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-536870912}"

MOTION_FILE="${MOTION_FILE:-${SCRIPT_DIR}/crisp_stairs/___crisp_clean_motion/stair_45.npz}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${SCRIPT_DIR}/crisp_stairs/___crisp_clean_geometry/stair_45.obj}"
RUN_NAME="${RUN_NAME:-${HOSTNAME_SHORT}_g1_29dof_wbt_stair45_csp_blindwbt_physxstack512m_${NUM_GPUS}gpu_${ENVS_PER_GPU}env_${TIMESTAMP}}"
SESSION="${SESSION:-csp_blindwbt_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-logs/run_commands}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SESSION}.log}"
MASTER_PORT="${MASTER_PORT:-$(find_free_port 29545)}"

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

  TMUX_CMD="cd $(quote "${SCRIPT_DIR}") && env RUN_IN_TMUX=0 TIMESTAMP=$(quote "${TIMESTAMP}") HOSTNAME_SHORT=$(quote "${HOSTNAME_SHORT}") WANDB_ENTITY=$(quote "${WANDB_ENTITY}") WANDB_PROJECT=$(quote "${WANDB_PROJECT}") NUM_GPUS=$(quote "${NUM_GPUS}") ENVS_PER_GPU=$(quote "${ENVS_PER_GPU}") TOTAL_ENVS=$(quote "${TOTAL_ENVS}") NUM_ITERATIONS=$(quote "${NUM_ITERATIONS}") SAVE_INTERVAL=$(quote "${SAVE_INTERVAL}") PHYSX_GPU_COLLISION_STACK_SIZE=$(quote "${PHYSX_GPU_COLLISION_STACK_SIZE}") MOTION_FILE=$(quote "${MOTION_FILE}") TERRAIN_OBJ=$(quote "${TERRAIN_OBJ}") RUN_NAME=$(quote "${RUN_NAME}") SESSION=$(quote "${SESSION}") LOG_DIR=$(quote "${LOG_DIR}") LOG_FILE=$(quote "${LOG_FILE}") MASTER_PORT=$(quote "${MASTER_PORT}") bash $(quote "${SCRIPT_DIR}/csp_blindwbt.sh") --run > $(quote "${LOG_FILE}") 2>&1"

  tmux new-session -d -s "${SESSION}" "${TMUX_CMD}"
  echo "Started CSP blind WBT training."
  echo "  session: ${SESSION}"
  echo "  run_name: ${RUN_NAME}"
  echo "  log: ${LOG_FILE}"
  echo "  master_port: ${MASTER_PORT}"
  echo "  total_envs: ${TOTAL_ENVS} (${NUM_GPUS} x ${ENVS_PER_GPU})"
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

torchrun \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NUM_GPUS}" \
  src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
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
  --command.setup_terms.motion_command.params.motion_config.motion_file="${MOTION_FILE}" \
  --simulator.config.scene.env-spacing=0.0 \
  --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE}" \
  --algo.config.num-learning-iterations="${NUM_ITERATIONS}" \
  --algo.config.save-interval="${SAVE_INTERVAL}" \
  "$@"
