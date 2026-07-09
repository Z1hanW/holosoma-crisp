# Holosoma CRISP Real2Sim2Real

This repository contains Holosoma plus the CRISP Real2Sim2Real retargeting
workflow. The release entry points are environment-agnostic: activate any Python
environment that satisfies the package dependencies, install the packages from
source, and run the scripts with repo-relative paths.

## Repository Layout

```text
src/
  holosoma/              # training framework
  holosoma_inference/    # inference and deployment
  holosoma_retargeting/  # human-to-robot retargeting and CRISP conversion
scripts/
  viser_hmr_mesh.py      # HMR joints + mesh-proxy viewer
  viser_retarget_light.py # retargeted G1 / human / scene overlay viewer
  train_crisp_s2r.sh     # retargeted motion conversion and WBT training helper
s2r_readme.md            # CRISP S2R workflow
```

## Environment

Use your own environment manager. For a source checkout, the minimal editable
install pattern is:

```bash
python -m pip install -e src/holosoma_retargeting
python -m pip install -e src/holosoma
```

Training in IsaacSim/IsaacLab still requires the simulator dependencies described
in [src/holosoma/README.md](src/holosoma/README.md). The repository setup scripts
are optional convenience wrappers, not required by the CRISP S2R scripts.

## CRISP S2R

See [s2r_readme.md](s2r_readme.md) for the complete conversion, retargeting,
visualization, and training commands.

The short flow is:

```bash
python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root <crisp_zup_root> \
  --crisp-hmr-root <crisp_hmr_root> \
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset> \
  --overwrite \
  --validate

cd src/holosoma_retargeting/holosoma_retargeting

python examples/parallel_robot_retarget.py \
  --data-dir demo_data/<dataset> \
  --task-type climbing \
  --data_format smplx \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results_parallel/g1/climbing/<dataset>
```

Scene Laplacian points default to primitive-aware sampling: each `pieces/*.obj`
mesh contributes its vertices plus center, with bbox corners used for large
pieces. Single-mesh scenes fall back to surface sampling.

## Viewers

HMR input viewer:

```bash
python scripts/viser_hmr_mesh.py \
  --hmr-npz <crisp_hmr_root>/<sequence>/gv/hmr/hps_track_smplx.npz \
  --terrain-obj <crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/scene_mesh_sqs.obj \
  --world-rotation <crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/world_rotation.npy \
  --shared-translation <crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/shared_translation.txt
```

Retarget overlay viewer:

```bash
python scripts/viser_retarget_light.py \
  --xml src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset>/<sequence>/g1_29dof_spherehand_w_multi_boxes_scaled_0.74_0.74_0.74.xml \
  --terrain-obj src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset>/<sequence>/multi_boxes.obj \
  --terrain-scale 0.7415730337 \
  --qpos-npz src/holosoma_retargeting/holosoma_retargeting/demo_results_parallel/g1/climbing/<dataset>/<sequence>_original.npz \
  --show-g1-mesh \
  --align-g1-to-human-root
```

## Training Helper

`scripts/train_crisp_s2r.sh` defaults to `--setup-mode none`, so it uses the
currently active Python environment and does not source a conda environment.

```bash
scripts/train_crisp_s2r.sh \
  --retarget-npz src/holosoma_retargeting/holosoma_retargeting/demo_results_parallel/g1/climbing/<dataset>/<sequence>_original.npz \
  --terrain-obj src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset>/<sequence>/multi_boxes.obj \
  --heightmap \
  -- \
  --algo.config.num-learning-iterations=30000 \
  --training.num-envs=4096
```

If you intentionally use the repository's Holosoma conda setup scripts, pass
`--setup-mode holosoma`.

## Documentation

- [Training guide](src/holosoma/README.md)
- [Inference guide](src/holosoma_inference/README.md)
- [Retargeting guide](src/holosoma_retargeting/holosoma_retargeting/README.md)
- [CRISP S2R workflow](s2r_readme.md)
