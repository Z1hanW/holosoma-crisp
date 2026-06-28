# CRISP Real2Sim2Real Terrain Conversion

This folder contains the bridge from CRISP z-up scene reconstruction outputs into
Holosoma climbing/terrain retargeting inputs.

## Contract

- Input geometry is already z-up. The converter does not rotate, translate, align,
  rescale, or viewer-correct the geometry.
- Generated mesh assets use `scale="1 1 1"`. This faithfully reflects the CRISP
  data on disk.
- Each terrain piece is loaded separately in URDF and MJCF. This is required so
  MuJoCo collision works per primitive/piece instead of through one combined
  convex hull.
- `multi_boxes.obj` is only the combined sampling mesh used by Holosoma's
  interaction point sampling. It is not the collision representation.

Holosoma's climbing retargeting code may later create `_scaled_*.urdf` and
`_scaled_*.xml` files from its SMPL/human scale logic. That is a retargeting
normalization step, not a hidden change in this converter.

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
demo_data/crisp_terrain/stair_75/
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
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain \
  --overwrite \
  --validate-mujoco
```

Batch all available CRISP stair outputs:

```bash
source /home/ubuntu/miniconda3/bin/activate gmr
PYTHONPATH=src/holosoma_retargeting python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root /tmp/crisp_stairs_same75_post_visualizer_all115_zup/v2 \
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain \
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

Holosoma climbing loads the first `.npy` in each sequence folder as global joint
positions with shape `(T, J, 3)`.

## Retarget With CRISP Terrain

Once the terrain folder also contains a motion `.npy`, run:

```bash
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
  --data_path demo_data/crisp_terrain \
  --task-type climbing \
  --task-name stair_75 \
  --data_format mocap \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results/g1/climbing/crisp_terrain
```

For visualization of retargeted results:

```bash
python viser_player.py \
  --robot_urdf models/g1/g1_29dof_spherehand.urdf \
  --object_urdf demo_data/crisp_terrain/stair_75/multi_boxes.urdf \
  --qpos_npz demo_results/g1/climbing/crisp_terrain/stair_75_original.npz
```

