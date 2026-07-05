# CRISP S2R Pipeline

Input to S2R is CRISP **z-up reconstruction** plus aligned HMR motion.

## Need From CRISP

```text
<crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/
  pieces/*.obj
  scene_mesh_sqs.obj
  world_rotation.npy
  shared_translation.txt

<crisp_hmr_root>/<sequence>/gv/hmr/
  hps_track_smplx.npz
```

The scene and HMR must already be in the same z-up frame.

## 1. Prepare Retargeting Assets

Convert CRISP z-up reconstruction into Holosoma climbing data:

```bash
python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root <crisp_zup_root> \
  --crisp-hmr-root <crisp_hmr_root> \
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset> \
  --overwrite \
  --validate
```

This prepares:

```text
demo_data/<dataset>/<sequence>/
  pieces/*.obj
  multi_boxes.obj
  multi_boxes.urdf
  g1_29dof_spherehand_w_multi_boxes.xml
  <sequence>.npy
  manifest.json
```

`pieces/*.obj` are the per-piece collision geometry used by retargeting.

## 2. Retarget To G1

```bash
cd src/holosoma_retargeting/holosoma_retargeting

python examples/parallel_robot_retarget.py \
  --data-dir demo_data/<dataset> \
  --task-type climbing \
  --data_format smplx \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results_parallel/g1/climbing/<dataset> \
  --retargeter.no-activate-foot-sticking \
  --retargeter.allow-infeasible-fallback
```

Output:

```text
demo_results_parallel/g1/climbing/<dataset>/<sequence>_original.npz
```

Check `failed_frames` before training.

## 3. Train Policy

With CRISP terrain:

```bash
scripts/train_crisp_s2r.sh \
  --retarget-npz src/holosoma_retargeting/holosoma_retargeting/demo_results_parallel/g1/climbing/<dataset>/<sequence>_original.npz \
  --terrain-obj src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset>/<sequence>/multi_boxes.obj \
  --heightmap \
  -- \
  --algo.config.num-learning-iterations=30000 \
  --training.num-envs=4096
```

Without terrain:

```bash
scripts/train_crisp_s2r.sh \
  --retarget-npz src/holosoma_retargeting/holosoma_retargeting/demo_results_parallel/g1/climbing/<dataset>/<sequence>_original.npz \
  --no-heightmap \
  -- \
  --algo.config.num-learning-iterations=30000 \
  --training.num-envs=4096
```

The script converts retargeted `qpos` into Holosoma WBT motion data, then runs
`src/holosoma/holosoma/train_agent.py`.
