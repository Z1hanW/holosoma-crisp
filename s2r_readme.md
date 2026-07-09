# CRISP S2R Pipeline

Input is a CRISP z-up scene plus HMR SMPL-X joints in the same frame.

Required files:

```text
<crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/
  pieces/*.obj
  scene_mesh_sqs.obj
  world_rotation.npy
  shared_translation.txt

<crisp_hmr_root>/<sequence>/gv/hmr/
  hps_track_smplx.npz
```

## Environment

Activate any Python environment with the required dependencies. From the repo
root, install the source packages you need:

```bash
python -m pip install -e src/holosoma_retargeting
python -m pip install -e src/holosoma
```

The commands below do not require a specific conda root or environment name. If
you use the repository setup scripts, source them before running the commands;
otherwise keep using your active environment.

## Prepare Data

```bash
PYTHONPATH=src/holosoma_retargeting \
python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root <crisp_zup_root> \
  --crisp-hmr-root <crisp_hmr_root> \
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset> \
  --overwrite \
  --validate
```

This writes:

```text
demo_data/<dataset>/<sequence>/
  pieces/*.obj
  multi_boxes.obj
  multi_boxes.urdf
  box_assets.xml
  box_body.xml
  g1_29dof_spherehand_w_multi_boxes.xml
  <sequence>.npy
```

Collision is per piece. `multi_boxes.obj` is only the merged terrain mesh.

## Retarget

```bash
cd src/holosoma_retargeting/holosoma_retargeting

python examples/parallel_robot_retarget.py \
  --data-dir demo_data/<dataset> \
  --task-type climbing \
  --data_format smplx \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results_parallel/g1/climbing/<dataset> \
  --retargeter.allow-infeasible-fallback
```

Scene Laplacian points default to `--task-config.object-point-mode auto`.
For CRISP primitive scenes this uses every `pieces/*.obj` mesh's unique vertices
plus its center. If a piece has more than
`--task-config.max-vertices-per-primitive` vertices, it uses bbox corners plus
center. If no pieces exist, it falls back to surface sampling.

Use `--task-config.object-point-mode surface_sample` only for the old random
surface-sampling behavior.

Output:

```text
demo_results_parallel/g1/climbing/<dataset>/<sequence>_original.npz
```

Check `failed_frames` before training.

## Visualize

HMR with mesh proxy:

```bash
python scripts/viser_hmr_mesh.py \
  --port 9302 \
  --hmr-npz <crisp_hmr_root>/<sequence>/gv/hmr/hps_track_smplx.npz \
  --terrain-obj <crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/scene_mesh_sqs.obj \
  --world-rotation <crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/world_rotation.npy \
  --shared-translation <crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/shared_translation.txt
```

Retargeting overlap:

```bash
python scripts/viser_retarget_light.py \
  --port 9303 \
  --xml src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset>/<sequence>/g1_29dof_spherehand_w_multi_boxes_scaled_0.74_0.74_0.74.xml \
  --terrain-obj src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset>/<sequence>/multi_boxes.obj \
  --terrain-scale 0.7415730337 \
  --qpos-npz src/holosoma_retargeting/holosoma_retargeting/demo_results_parallel/g1/climbing/<dataset>/<sequence>_original.npz \
  --show-g1-mesh \
  --align-g1-to-human-root
```

## Train

```bash
scripts/train_crisp_s2r.sh \
  --retarget-npz src/holosoma_retargeting/holosoma_retargeting/demo_results_parallel/g1/climbing/<dataset>/<sequence>_original.npz \
  --terrain-obj src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset>/<sequence>/multi_boxes.obj \
  --heightmap \
  -- \
  --algo.config.num-learning-iterations=30000 \
  --training.num-envs=4096
```

The training helper defaults to `--setup-mode none`, so it uses the active
environment and does not source conda. Use `--setup-mode holosoma` only if you
intentionally want the script to source this repository's setup scripts.
