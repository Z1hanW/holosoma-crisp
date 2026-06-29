# CRISP Real2Sim2Real Retargeting

This is the CRISP-specific entry point for using Holosoma as the
real2sim2real retargeting backend.

## What This Pipeline Consumes

The terrain input is CRISP z-up scene output:

```text
/tmp/crisp_stairs_same75_post_visualizer_all115_zup/v2/<sequence>/gv/scene_mesh_sqs/
  pieces/*.obj
  scene_mesh_sqs.obj
```

The motion input is CRISP HMR SMPL-X output:

```text
/tmp/crisp_stairs_legacy_stair75_112/<sequence>/gv/hmr/hps_track_smplx.npz
```

`hps_track_smplx.npz["global_joint_positions"]` is written into the Holosoma
sequence folder as `<sequence>.npy` after applying the same z-up scene transform
as the geometry:

```python
joints_zup = joints_raw @ world_rotation.T + shared_translation
```

The converter does not scale this motion. Raw HMR joints must not be written
directly into a z-up terrain folder, because that puts motion and terrain in
different frames.

## Important Geometry Contract

- CRISP terrain is assumed to already be z-up.
- The converter writes terrain mesh assets with `scale="1 1 1"`.
- CRISP HMR joints are transformed into the same z-up frame using
  `world_rotation.npy` and `shared_translation.txt` from the scene output.
- No viewer-side correction is used to hide coordinate or scale mistakes.
- `multi_boxes.obj` is a combined mesh only for Holosoma's object point sampling.
- MuJoCo and URDF collision use one separate mesh/link/geom per CRISP piece.

This per-piece collision representation is required for terrain traversal: a
single combined mesh can behave like one large convex hull, which is wrong for
stairs and separated planar pieces.

## Prepare Terrain And Motion

Run from the `real2sim2real` repo root:

```bash
source /home/ubuntu/miniconda3/bin/activate gmr

PYTHONPATH=src/holosoma_retargeting python -m holosoma_retargeting.crisp.convert_zup_scene \
  --crisp-zup-root /tmp/crisp_stairs_same75_post_visualizer_all115_zup/v2 \
  --crisp-hmr-root /tmp/crisp_stairs_legacy_stair75_112 \
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain_zup_motion_aligned \
  --overwrite \
  --validate
```

Expected local status for the current data:

- 115 terrain folders are generated.
- 112 `stair_0` through `stair_111` folders have SMPL-X motion `.npy`.
- 3 outdoor folders currently have terrain only unless matching HMR motion is
  provided separately.

Generated terrain/motion data is ignored by git:

```text
src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain_zup_motion_aligned/
```

## Smoke Test One Sequence

```bash
cd src/holosoma_retargeting/holosoma_retargeting

python examples/robot_retarget.py \
  --data_path demo_data/crisp_terrain_zup_motion_aligned \
  --task-type climbing \
  --task-name stair_75 \
  --data_format smplx \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results/g1/climbing/crisp_terrain_zup_motion_aligned_smoke \
  --retargeter.no-activate-foot-sticking \
  --retargeter.allow-infeasible-fallback
```

For this CRISP terrain pass we keep terrain non-penetration and joint limits on,
disable foot sticking, and allow infeasible frames to reuse the previous qpos.
Fallback frames are recorded in the output `.npz` as `failed_frames` and
`failed_frame_errors`.

CRISP climbing motion is already video-frame aligned with the z-up scene, so the
climbing loader keeps every frame (`downsample = 1`). Do not reuse the legacy
Holosoma mocap `::4` downsample for these inputs.

## Run All Available Stair Cases

The batch script has been adjusted so climbing tasks find `*/*.npy` regardless
of whether the motion format is `mocap` or `smplx`. For CRISP SMPL-X folders it
must use the canonical `<sequence>/<sequence>.npy` motion file and ignore
sidecar arrays such as `world_rotation.npy`.

```bash
cd src/holosoma_retargeting/holosoma_retargeting

python examples/parallel_robot_retarget.py \
  --data-dir demo_data/crisp_terrain_zup_motion_aligned \
  --task-type climbing \
  --data_format smplx \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results_parallel/g1/climbing/crisp_terrain_zup_motion_aligned \
  --max-workers 4 \
  --retargeter.no-activate-foot-sticking \
  --retargeter.allow-infeasible-fallback
```

The output files are written as:

```text
demo_results_parallel/g1/climbing/crisp_terrain_zup_motion_aligned/<sequence>_original.npz
```

Use a smaller `--max-workers` value if MuJoCo/CVX memory pressure is high.

## Current Batch Result

The current local batch was run with the aligned command above.

- Input motions: 112
- Retarget outputs: 112
- Missing outputs: 0
- Job-level failures: 0

The retargeter is configured to keep terrain non-penetration and joint limits
enabled, disable foot sticking, and write fallback metadata when CVXPY reports
an infeasible frame or solver error. Check `failed_frames` before using an
output for training.

Current fallback summary:

- The loader now uses `downsample = 1` for CRISP climbing. The verified
  full-frame smoke output is `stair_75`, with `354` input frames and
  `qpos.shape == (354, 36)`.
- Re-run the full 112-sequence batch after this change before treating the full
  folder as a consistent full-frame training set.
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

## Visualize A Result

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

## Scale Note

Holosoma's retargeting code may create `_scaled_*.urdf` and `_scaled_*.xml`
files during climbing setup. That is Holosoma's retargeting normalization. The
CRISP converter itself remains faithful to the post-scene z-up frame: input
z-up geometry is copied, HMR joints receive the same saved scene transform, and
no additional viewer-side transform or scale is applied.
