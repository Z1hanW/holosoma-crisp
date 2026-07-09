#!/usr/bin/env bash

set -eo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RETARGET_ROOT="$REPO_ROOT/src/holosoma_retargeting/holosoma_retargeting"
MOTION_DATA_ROOT="$REPO_ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/crisp_s2r"

RETARGET_NPZ=""
TERRAIN_OBJ=""
MOTION_OUT=""
TERRAIN_OUT=""
OUTPUT_FPS=50
INPUT_FPS=30
TERRAIN_SCALE="0.7415730337"
HEIGHTMAP=1
SKIP_CONVERT=0
CONVERT_ONLY=0
SETUP_MODE="${CRISP_S2R_SETUP_MODE:-none}"
TRAIN_ARGS=()
LOGGER_ARG=""

usage() {
  cat <<'EOF'
Usage:
  scripts/train_crisp_s2r.sh --retarget-npz PATH [options] -- [train_agent args]

Options:
  --retarget-npz PATH   Required retargeting output, e.g. stair_75_original.npz.
  --terrain-obj PATH    Terrain OBJ used when --heightmap is enabled.
  --motion-out PATH     Output Holosoma WBT motion npz.
  --terrain-out PATH    Output scaled terrain OBJ.
  --output-fps N        Output fps for WBT motion conversion. Default: 50.
  --input-fps N         Fallback input fps for conversion. Default: 30.
  --terrain-scale S     Terrain OBJ vertex scale. Default: 0.7415730337.
  --heightmap           Train with terrain:terrain-load-obj. Default.
  --no-heightmap        Train without loading CRISP terrain.
  --skip-convert        Reuse --motion-out.
  --convert-only        Prepare data and exit before training.
  --setup-mode MODE     Environment setup mode: none or holosoma. Default: none.
                        none uses the active Python environment.
                        holosoma sources scripts/source_*_setup.sh.
  --skip-setup          Alias for --setup-mode none.
  -h, --help            Show this help.
EOF
}

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retarget-npz)
      RETARGET_NPZ="$2"
      shift 2
      ;;
    --terrain-obj)
      TERRAIN_OBJ="$2"
      shift 2
      ;;
    --motion-out)
      MOTION_OUT="$2"
      shift 2
      ;;
    --terrain-out)
      TERRAIN_OUT="$2"
      shift 2
      ;;
    --output-fps)
      OUTPUT_FPS="$2"
      shift 2
      ;;
    --input-fps)
      INPUT_FPS="$2"
      shift 2
      ;;
    --terrain-scale)
      TERRAIN_SCALE="$2"
      shift 2
      ;;
    --heightmap)
      HEIGHTMAP=1
      shift
      ;;
    --no-heightmap)
      HEIGHTMAP=0
      shift
      ;;
    --skip-convert)
      SKIP_CONVERT=1
      shift
      ;;
    --convert-only)
      CONVERT_ONLY=1
      shift
      ;;
    --setup-mode)
      SETUP_MODE="$2"
      shift 2
      ;;
    --skip-setup)
      SETUP_MODE="none"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      fail "Unknown option '$1'. Put train_agent.py args after --."
      ;;
  esac
done

[[ -n "$RETARGET_NPZ" ]] || fail "--retarget-npz is required"
[[ "$SETUP_MODE" == "none" || "$SETUP_MODE" == "holosoma" ]] || fail "--setup-mode must be 'none' or 'holosoma'"

RETARGET_NPZ="$(realpath "$RETARGET_NPZ")"
[[ -f "$RETARGET_NPZ" ]] || fail "Missing retarget npz: $RETARGET_NPZ"

export PYTHONPATH="$REPO_ROOT/src/holosoma_retargeting:$REPO_ROOT/src/holosoma:$REPO_ROOT/src/holosoma_inference:${PYTHONPATH:-}"

SEQ_NAME="$(basename "$RETARGET_NPZ")"
SEQ_NAME="${SEQ_NAME%.npz}"
SEQ_NAME="${SEQ_NAME%_original}"

if [[ -z "$MOTION_OUT" ]]; then
  MOTION_OUT="$MOTION_DATA_ROOT/${SEQ_NAME}_mj_fps${OUTPUT_FPS}.npz"
fi
MOTION_OUT="$(realpath -m "$MOTION_OUT")"

if [[ "$HEIGHTMAP" == "1" ]]; then
  if [[ -z "$TERRAIN_OBJ" ]]; then
    INFERRED_TERRAIN_DIR="$(find "$RETARGET_ROOT/demo_data" -mindepth 2 -maxdepth 2 -type d -name "$SEQ_NAME" -print -quit)"
    if [[ -n "$INFERRED_TERRAIN_DIR" ]]; then
      TERRAIN_OBJ="$INFERRED_TERRAIN_DIR/multi_boxes.obj"
    fi
  fi
  [[ -n "$TERRAIN_OBJ" ]] || fail "--terrain-obj is required when --heightmap is enabled"
  TERRAIN_OBJ="$(realpath "$TERRAIN_OBJ")"
  [[ -f "$TERRAIN_OBJ" ]] || fail "Missing terrain obj: $TERRAIN_OBJ"

  if [[ -z "$TERRAIN_OUT" ]]; then
    SCALE_TAG="$(printf "%s" "$TERRAIN_SCALE" | sed 's/\./p/g')"
    TERRAIN_OUT="$MOTION_DATA_ROOT/${SEQ_NAME}_multi_boxes_scaled_${SCALE_TAG}.obj"
  fi
  TERRAIN_OUT="$(realpath -m "$TERRAIN_OUT")"
fi

mkdir -p "$(dirname "$MOTION_OUT")"

if [[ "$SETUP_MODE" == "holosoma" ]]; then
  echo "[INFO] Sourcing Holosoma retargeting setup for motion conversion"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/source_retargeting_setup.sh"
  pip install -e "$REPO_ROOT/src/holosoma_retargeting" --quiet
else
  echo "[INFO] Using active Python environment for motion conversion"
fi

if [[ "$SKIP_CONVERT" != "1" ]]; then
  echo "[INFO] Converting retarget qpos to Holosoma WBT motion: $MOTION_OUT"
  python "$RETARGET_ROOT/data_conversion/convert_data_format_mj.py" \
    --input_file "$RETARGET_NPZ" \
    --input_fps "$INPUT_FPS" \
    --output_fps "$OUTPUT_FPS" \
    --output_name "$MOTION_OUT" \
    --data_format smplh \
    --object_name ground \
    --once
else
  [[ -f "$MOTION_OUT" ]] || fail "--skip-convert requested but motion file is missing: $MOTION_OUT"
fi

if [[ "$HEIGHTMAP" == "1" ]]; then
  mkdir -p "$(dirname "$TERRAIN_OUT")"
  echo "[INFO] Writing scaled terrain OBJ: $TERRAIN_OUT"
  python - "$TERRAIN_OBJ" "$TERRAIN_OUT" "$TERRAIN_SCALE" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
scale = float(sys.argv[3])

with src.open("r", encoding="utf-8", errors="ignore") as f_in, dst.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.startswith("v "):
            f_out.write(line)
            continue
        parts = line.rstrip("\n").split()
        if len(parts) < 4:
            f_out.write(line)
            continue
        x, y, z = (float(parts[i]) * scale for i in range(1, 4))
        rest = parts[4:]
        f_out.write(f"v {x:.9g} {y:.9g} {z:.9g}")
        if rest:
            f_out.write(" " + " ".join(rest))
        f_out.write("\n")
PY
fi

if [[ "$CONVERT_ONLY" == "1" ]]; then
  echo "[INFO] convert-only requested. Motion: $MOTION_OUT"
  [[ "$HEIGHTMAP" == "1" ]] && echo "[INFO] Terrain: $TERRAIN_OUT"
  exit 0
fi

if [[ "$SETUP_MODE" == "holosoma" ]]; then
  echo "[INFO] Sourcing Holosoma IsaacSim setup for WBT training"
  cd "$REPO_ROOT"
  unset CONDA_ENV_NAME
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/source_isaacsim_setup.sh"
  HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}"
  pip install -e "$REPO_ROOT/src/holosoma[unitree,booster]" --quiet
  if ! python -c "import isaaclab" 2>/dev/null; then
    pip install 'setuptools<81' --quiet
    echo 'setuptools<81' > /tmp/hs-build-constraints.txt
    PIP_BUILD_CONSTRAINT=/tmp/hs-build-constraints.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      pip install -e "$HOLOSOMA_DEPS_DIR/IsaacLab/source/isaaclab" --quiet
    rm /tmp/hs-build-constraints.txt
  fi
else
  echo "[INFO] Using active Python environment for WBT training"
fi

cd "$REPO_ROOT"

FORWARD_TRAIN_ARGS=()
for arg in "${TRAIN_ARGS[@]}"; do
  if [[ "$arg" == logger:* ]]; then
    LOGGER_ARG="$arg"
  else
    FORWARD_TRAIN_ARGS+=("$arg")
  fi
done

TRAIN_CMD=(
  python "$REPO_ROOT/src/holosoma/holosoma/train_agent.py"
  exp:g1-29dof-wbt
)

if [[ -n "$LOGGER_ARG" ]]; then
  TRAIN_CMD+=("$LOGGER_ARG")
fi

TRAIN_CMD+=(
  "--command.setup_terms.motion_command.params.motion_config.motion_file=$MOTION_OUT"
  "--training.name=crisp_s2r_${SEQ_NAME}"
)

if [[ "$HEIGHTMAP" == "1" ]]; then
  TRAIN_CMD+=(
    terrain:terrain-load-obj
    "--terrain.terrain-term.obj-file-path=$TERRAIN_OUT"
    "--simulator.config.scene.env_spacing=0.0"
  )
else
  TRAIN_CMD+=(terrain:terrain-locomotion-plane)
fi

TRAIN_CMD+=("${FORWARD_TRAIN_ARGS[@]}")

echo "[INFO] Training command:"
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"
