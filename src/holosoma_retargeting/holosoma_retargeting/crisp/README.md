# CRISP Terrain Conversion

This package converts CRISP z-up scene outputs into Holosoma climbing data.

## Environment

Use any Python environment with the required dependencies installed. From the
repo root, either install the package:

```bash
python -m pip install -e src/holosoma_retargeting
```

or run commands with `PYTHONPATH=src/holosoma_retargeting`.

## Contract

- Geometry is already z-up. The converter does not rotate, scale, or viewer-fix it.
- HMR SMPL-X joints are transformed with the scene's `world_rotation.npy` and
  `shared_translation.txt`.
- Collision is per primitive: every CRISP piece becomes its own URDF link and
  MuJoCo geom.
- Retargeting scene points default to each primitive's vertices plus center.
  Large pieces use bbox corners plus center. Single-mesh scenes fall back to
  surface sampling.

## Input

```text
<crisp_zup_root>/<sequence>/gv/scene_mesh_sqs/
  pieces/*.obj
  scene_mesh_sqs.obj
  world_rotation.npy
  shared_translation.txt

<crisp_hmr_root>/<sequence>/gv/hmr/hps_track_smplx.npz
```

## Convert

Run from the `real2sim2real` repo root:

```bash
PYTHONPATH=src/holosoma_retargeting python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root <crisp_zup_root> \
  --crisp-hmr-root <crisp_hmr_root> \
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/<dataset> \
  --overwrite \
  --validate
```

Output per sequence:

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
