# CRISP Real2Sim2Real Terrain Conversion

This folder contains the bridge from CRISP z-up scene reconstruction outputs into
Holosoma climbing/terrain retargeting inputs.

## Contract

- Input geometry is already z-up. The converter does not rotate, translate, align,
  rescale, or viewer-correct the geometry.
- Generated mesh assets use `scale="1 1 1"`. This faithfully reflects the CRISP
  data on disk.
- If CRISP HMR SMPL-X motion is imported, joints are transformed into the same
  z-up scene frame using:

  ```python
  joints_zup = joints_raw @ world_rotation.T + shared_translation
  ```

  This matches the established post-scene contract in `rot_baby.py`.
- Each terrain piece is loaded separately in URDF and MJCF. This is required so
  MuJoCo collision works per primitive/piece instead of through one combined
  convex hull.
- `multi_boxes.obj` is only the combined sampling mesh used by Holosoma's
  interaction point sampling. It is not the collision representation.

Holosoma's climbing retargeting code may later create `_scaled_*.urdf` and
`_scaled_*.xml` files from its SMPL/human scale logic. That is a retargeting
normalization step, not a hidden viewer correction in this converter.

## Expected CRISP Input

The default CRISP input layout is:

```text
<crisp-zup-root>/
  stair_75/
    gv/
      scene_mesh_sqs/
        scene_mesh_sqs.obj
        pieces/
          part_000.obj
          part_001.obj
          ...
```

For our current v2 z-up output this is:

```bash
/tmp/crisp_stairs_same75_post_visualizer_all115_zup/v2
```

## Generated Holosoma Layout

For each sequence, the converter writes:

```text
demo_data/crisp_terrain_zup_motion_aligned/stair_75/
  pieces/
    piece_000.obj
    piece_001.obj
    ...
  multi_boxes.obj
  multi_boxes.urdf
  box_assets.xml
  box_body.xml
  g1_29dof_spherehand_w_multi_boxes.xml
  manifest.json
```

`multi_boxes.urdf` contains one fixed link and one collision mesh per piece.
`box_assets.xml` contains one MuJoCo mesh asset per piece.
`box_body.xml` contains one static MuJoCo body/geom per piece.

## Convert Terrain

From the `real2sim2real` repo root:

```bash
source /home/ubuntu/miniconda3/bin/activate gmr
PYTHONPATH=src/holosoma_retargeting python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root /tmp/crisp_stairs_same75_post_visualizer_all115_zup/v2 \
  --sequence stair_75 \
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain_zup_motion_aligned \
  --overwrite \
  --validate-mujoco
```

Batch all available CRISP stair outputs:

```bash
source /home/ubuntu/miniconda3/bin/activate gmr
PYTHONPATH=src/holosoma_retargeting python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root /tmp/crisp_stairs_same75_post_visualizer_all115_zup/v2 \
  --crisp-hmr-root /tmp/crisp_stairs_legacy_stair75_112 \
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain_zup_motion_aligned \
  --overwrite \
  --validate
```

If Holosoma motion `.npy` files are already available, copy them into each
sequence folder during conversion:

```bash
PYTHONPATH=src/holosoma_retargeting python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root /tmp/crisp_stairs_same75_post_visualizer_all115_zup/v2 \
  --motion-root /path/to/holosoma_joint_positions \
  --motion-glob "{sequence}*.npy" \
  --require-motion \
  --overwrite
```

Holosoma climbing must load the canonical `<sequence>/<sequence>.npy` file as
global joint positions with shape `(T, J, 3)`. CRISP folders also contain
sidecar arrays such as `world_rotation.npy`; these are metadata and must not be
treated as motion.

## Retarget With CRISP Terrain

Once the terrain folder also contains a motion `.npy`, run:

```bash
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
  --data_path demo_data/crisp_terrain_zup_motion_aligned \
  --task-type climbing \
  --task-name stair_75 \
  --data_format smplx \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results/g1/climbing/crisp_terrain_zup_motion_aligned \
  --retargeter.no-activate-foot-sticking \
  --retargeter.allow-infeasible-fallback
```

For the current CRISP stair pass, terrain non-penetration and joint limits stay
enabled, foot sticking is disabled, and infeasible frames reuse the previous
qpos. Output `.npz` files record these frames in `failed_frames` and
`failed_frame_errors`.

Current local batch status:

- 112 SMPL-X motion inputs were retargeted.
- 112 `_original.npz` outputs were written.
- No sequence was missing from the output folder.
- 69 sequences have no fallback frames.
- 12 sequences have partial fallback frames:
  `stair_3`, `stair_4`, `stair_11`, `stair_19`, `stair_20`, `stair_38`,
  `stair_50`, `stair_59`, `stair_62`, `stair_66`, `stair_67`, `stair_104`.
- 31 sequences are full-fallback outputs and should be treated as infeasible
  retarget results until inspected or re-run with a different retargeting setup:
  `stair_2`, `stair_7`, `stair_13`, `stair_23`, `stair_25`, `stair_27`,
  `stair_31`, `stair_32`, `stair_33`, `stair_34`, `stair_36`, `stair_41`,
  `stair_43`, `stair_47`, `stair_49`, `stair_52`, `stair_53`, `stair_56`,
  `stair_57`, `stair_60`, `stair_72`, `stair_81`, `stair_84`, `stair_86`,
  `stair_89`, `stair_90`, `stair_94`, `stair_98`, `stair_99`, `stair_100`,
  `stair_102`.

For visualization of retargeted results:

```bash
python viser_player.py \
  --port 9303 \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --object-urdf demo_data/crisp_terrain_zup_motion_aligned/stair_75/multi_boxes.urdf \
  --qpos-npz demo_results_parallel/g1/climbing/crisp_terrain_zup_motion_aligned/stair_75_original.npz \
  --no-assume-object-in-qpos \
  --grid-width 20 \
  --grid-height 20
```
