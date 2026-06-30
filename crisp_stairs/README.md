# CRISP Stairs Subset

This is a ds_crisp_data-style subset copied from:

`/nfs/zzzihanw/ds_crisp_data_vggtomega_crisp_terrain_g1`

Requested sequences: 16
Copied sequences: 16
Missing sequences: 0

Use with Holosoma terrain traversal training via:

```bash
cd /home/ubuntu/FAR/holosoma
PYTHON_BIN=/home/ubuntu/miniconda3/envs/crisp/bin/python \
PAIRED_DS_CRISP_DATA_ROOT=/nfs/zzzihanw/crisp_stairs \
bash train_terrain_generalist.sh heightmap
```

Notes:

- Numeric requested ids are mapped to `stair_<id>`.
- `56_outdoor` is mapped to `56_outdoor_stairs_up_down`.
- Terrain scale is already baked into the OBJ files. Do not apply extra terrain scale.
- Treat this subset and its `/nfs` source as read-only input data. Do not mutate the original motion, terrain, or log files; write staged, fused, cached, or debug outputs only to generated/output directories.
- Viser must show the Isaac Sim world state faithfully. Do not add viewer-only offsets, z fixes, recentering, scaling, retiming, or terrain/motion adjustments to hide a mismatch.
