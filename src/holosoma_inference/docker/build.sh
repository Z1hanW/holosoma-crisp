#!/bin/bash

# Build the Docker image using the holosoma directory as context
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # holosoma/src/holosoma_inference/docker
SRC_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")" # holosoma
IMAGE_NAME="holosoma-inference"
DOCKER_IMAGE="${DOCKER_IMAGE:-holosoma-inference:latest}"

docker build "$SRC_DIR" -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE_NAME" -t "$DOCKER_IMAGE"

[[ "$1" == "--push" ]] && docker push "$DOCKER_IMAGE"

rm -f "$SCRIPT_DIR"/*.whl
