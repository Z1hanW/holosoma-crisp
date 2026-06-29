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

`hps_track_smplx.npz["global_joint_positions"]` is copied faithfully into the
Holosoma sequence folder as `<sequence>.npy`. The converter does not rotate,
translate, or scale this motion.

## Important Geometry Contract

- CRISP terrain is assumed to already be z-up.
- The converter writes terrain mesh assets with `scale="1 1 1"`.
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
  --output-root src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain \
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
src/holosoma_retargeting/holosoma_retargeting/demo_data/crisp_terrain/
```

## Smoke Test One Sequence

```bash
cd src/holosoma_retargeting/holosoma_retargeting

python examples/robot_retarget.py \
  --data_path demo_data/crisp_terrain \
  --task-type climbing \
  --task-name stair_75 \
  --data_format smplx \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results/g1/climbing/crisp_terrain_smoke \
  --retargeter.no-activate-foot-sticking \
  --retargeter.allow-infeasible-fallback
```

For this CRISP terrain pass we keep terrain non-penetration and joint limits on,
disable foot sticking, and allow infeasible frames to reuse the previous qpos.
Fallback frames are recorded in the output `.npz` as `failed_frames` and
`failed_frame_errors`.

## Run All Available Stair Cases

The batch script has been adjusted so climbing tasks find `*/*.npy` regardless
of whether the motion format is `mocap` or `smplx`.

```bash
cd src/holosoma_retargeting/holosoma_retargeting

python examples/parallel_robot_retarget.py \
  --data-dir demo_data/crisp_terrain \
  --task-type climbing \
  --data_format smplx \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save_dir demo_results_parallel/g1/climbing/crisp_terrain \
  --max-workers 4 \
  --retargeter.no-activate-foot-sticking \
  --retargeter.allow-infeasible-fallback
```

The output files are written as:

```text
demo_results_parallel/g1/climbing/crisp_terrain/<sequence>_original.npz
```

Use a smaller `--max-workers` value if MuJoCo/CVX memory pressure is high.

## Current Batch Result

The current local batch was run with the command above.

- Input motions: 112
- Retarget outputs: 112
- Missing outputs: 0
- Job-level failures: 0

The retargeter is configured to keep terrain non-penetration and joint limits
enabled, disable foot sticking, and write fallback metadata when CVXPY reports
an infeasible frame. Check `failed_frames` before using an output for training.

Current fallback summary:

- `stair_15`: 43 / 43 frames fallback. This output exists, but should be
  treated as an infeasible retarget result until inspected or re-run with a
  different retargeting setup.
- `stair_75`: frames `[73, 77]` fallback.
- `stair_80`: frame `[89]` fallback.
- All other generated stair outputs solved without fallback frames.

## Visualize A Result

```bash
python viser_player.py \
  --robot_urdf models/g1/g1_29dof_spherehand.urdf \
  --object_urdf demo_data/crisp_terrain/stair_75/multi_boxes.urdf \
  --qpos_npz demo_results_parallel/g1/climbing/crisp_terrain/stair_75_original.npz
```

## Scale Note

Holosoma's retargeting code may create `_scaled_*.urdf` and `_scaled_*.xml`
files during climbing setup. That is Holosoma's retargeting normalization. The
CRISP converter itself remains faithful: input z-up geometry and HMR joints are
written without additional transform or scale.
