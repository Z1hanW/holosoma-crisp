"""
Utility functions for the kinematic retargeting system.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Literal

import numpy as np
import smplx  # type: ignore[import-not-found]
import torch
import trimesh
from jinja2 import Template
from scipy.spatial import Delaunay  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation as R  # type: ignore[import-untyped]  # noqa: N817


RaycastZMode = Literal["per_frame", "sequence", "global"]
ObjectPointMode = Literal["auto", "primitive_vertices_center", "surface_sample", "oriented_bbox"]


def _natural_path_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.as_posix())]


def _unique_points(points: np.ndarray, decimals: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return points
    _, unique_indices = np.unique(np.round(points, decimals=decimals), axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def _find_primitive_piece_files(object_file: str | os.PathLike) -> list[Path]:
    object_path = Path(object_file)
    object_dir = object_path.parent if object_path.parent != Path("") else Path(".")
    piece_files: list[Path] = []
    for piece_dir_name in ("pieces", "box_models"):
        piece_dir = object_dir / piece_dir_name
        if piece_dir.is_dir():
            piece_files.extend(piece_dir.glob("*.obj"))
    return sorted(piece_files, key=_natural_path_key)


def _load_primitive_vertices_center_points(
    object_file: str | os.PathLike,
    max_vertices_per_piece: int = 64,
) -> np.ndarray | None:
    piece_files = _find_primitive_piece_files(object_file)
    if not piece_files:
        return None

    points_per_piece: list[np.ndarray] = []
    for piece_file in piece_files:
        piece_mesh = trimesh.load(piece_file, force="mesh")
        vertices = np.asarray(piece_mesh.vertices, dtype=float).reshape(-1, 3)
        if len(vertices) == 0:
            continue
        vertices = _unique_points(vertices)
        if len(vertices) > max_vertices_per_piece:
            vertices = np.asarray(piece_mesh.bounding_box.vertices, dtype=float)
        center = np.asarray(piece_mesh.bounds, dtype=float).mean(axis=0, keepdims=True)
        points_per_piece.append(_unique_points(np.vstack([vertices, center])))

    if not points_per_piece:
        return None
    return np.vstack(points_per_piece)


def load_intermimic_data(file_path):
    """
    Load and preprocess InterMimic data.

    Args:
        file_path (str): Path to the .pt file.

    Returns:
        tuple: (human_joints, object_poses) - processed data.
    """
    intermimic_data = torch.load(file_path, map_location="cpu").detach().numpy()
    human_joints = intermimic_data[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)
    # Reorder quaternion from [qx, qy, qz, qw] to [qw, qx, qy, qz]
    object_poses = intermimic_data[:, 318:325][:, [6, 3, 4, 5, 0, 1, 2]]
    return human_joints, object_poses


def calculate_scale_factor(task_name, robot_height):
    """Calculate scale factor based on human height."""
    with open("demo_data/height_dict.pkl", "rb") as f:
        height_dict = pickle.load(f)
    sub_name = task_name.split("_")[0]
    human_height = height_dict[sub_name]
    return robot_height / human_height


def load_object_data(
    object_file,
    smpl_scale=0.714,
    bounding_box_oriented=False,
    sample_count=50,
    seed=42,
    surface_weights=None,
    use_face_normals=False,
    point_mode: ObjectPointMode = "auto",
    max_vertices_per_piece: int = 64,
):
    """
    Loads object points used to build the interaction Laplacian.

    Args:
        object_file (str): Path to the object mesh file.
        smpl_scale (float): Scale factor for SMPL compatibility.
        bounding_box_oriented (bool): Whether to use oriented bounding box vertices.
        sample_count (int): Number of points to sample from the surface.
        seed (int): Random seed for sampling.
        surface_weights: Weight function for sampling. If use_face_normals=True,
                        should take (face_normal, face_center) as arguments.
        use_face_normals (bool): Whether to use face-normal-based sampling.
        point_mode: Point source. "auto" prefers per-piece primitive vertices and
                    centers when piece meshes are available, then falls back to
                    legacy surface sampling.
        max_vertices_per_piece: Use bbox corners instead of all vertices for a
                                piece with more than this many unique vertices.

    Returns:
        tuple: (points, points_scaled) - original and scaled point arrays.
    """
    points = None
    if point_mode in {"auto", "primitive_vertices_center"}:
        points = _load_primitive_vertices_center_points(
            object_file,
            max_vertices_per_piece=max_vertices_per_piece,
        )
        if points is not None:
            print(f"Loaded primitive object points: {len(points)} points from per-piece vertices + centers.")
        elif point_mode == "primitive_vertices_center":
            raise ValueError(f"No primitive piece meshes found next to object mesh: {object_file}")

    if points is None:
        print("Loading and sampling object mesh...")
        obj_mesh = trimesh.load(object_file, force="mesh")

        if bounding_box_oriented or point_mode == "oriented_bbox":
            points = obj_mesh.bounding_box_oriented.vertices
        elif surface_weights is not None:
            if use_face_normals:
                # Use face-normal-based weighted sampling
                points = weighted_surface_sampling_by_face_normal(obj_mesh, sample_count, surface_weights, seed)
            else:
                # Use center-based weighted sampling
                points = weighted_surface_sampling(obj_mesh, sample_count, surface_weights, seed)
        else:
            points, _ = trimesh.sample.sample_surface_even(obj_mesh, sample_count, seed=seed)

    points = np.array(points)
    points_scaled = points * smpl_scale
    return points, points_scaled


def weighted_surface_sampling(mesh, sample_count, weight_func, seed=42):
    """
    Sample points from mesh surface with custom weighting.

    Args:
        mesh: Trimesh object
        sample_count: Number of points to sample
        weight_func: Function that takes (x,y,z) and returns weight
        seed: Random seed

    Returns:
        np.ndarray: Sampled points
    """
    np.random.seed(seed)

    faces = mesh.faces
    vertices = mesh.vertices

    face_areas = []
    face_centers = []

    for face in faces:
        v1, v2, v3 = vertices[face]
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        face_areas.append(area)

        center = (v1 + v2 + v3) / 3.0
        face_centers.append(center)

    face_areas = np.array(face_areas)
    face_centers = np.array(face_centers)

    weights = np.array([weight_func(center) for center in face_centers])
    weighted_areas = face_areas * weights

    total_weighted_area = np.sum(weighted_areas)
    face_probs = weighted_areas / total_weighted_area

    sampled_face_indices = np.random.choice(len(faces), size=sample_count, p=face_probs)

    sampled_points = []
    for face_idx in sampled_face_indices:
        face = faces[face_idx]
        v1, v2, v3 = vertices[face]

        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        point = v1 + r1 * (v2 - v1) + r2 * (v3 - v1)
        sampled_points.append(point)

    return np.array(sampled_points)


def weighted_surface_sampling_by_face_normal(mesh, sample_count, weight_func, seed=42):
    """
    Sample points from mesh surface with weighting based on face normals.

    Args:
        mesh: Trimesh object
        sample_count: Number of points to sample
        weight_func: Function that takes face_normal and face_center and returns weight
        seed: Random seed

    Returns:
        np.ndarray: Sampled points
    """
    np.random.seed(seed)

    faces = mesh.faces
    vertices = mesh.vertices

    face_areas = []
    face_centers = []
    face_normals = []

    for face in faces:
        v1, v2, v3 = vertices[face]
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        face_areas.append(area)

        center = (v1 + v2 + v3) / 3.0
        face_centers.append(center)

        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)
        face_normals.append(normal)

    face_areas = np.array(face_areas)
    face_centers = np.array(face_centers)
    face_normals = np.array(face_normals)

    weights = np.array([weight_func(normal, center) for normal, center in zip(face_normals, face_centers)])
    weighted_areas = face_areas * weights

    total_weighted_area = np.sum(weighted_areas)
    face_probs = weighted_areas / total_weighted_area

    sampled_face_indices = np.random.choice(len(faces), size=sample_count, p=face_probs)

    sampled_points = []
    for face_idx in sampled_face_indices:
        face = faces[face_idx]
        v1, v2, v3 = vertices[face]

        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        point = v1 + r1 * (v2 - v1) + r2 * (v3 - v1)
        sampled_points.append(point)

    return np.array(sampled_points)


def _compute_legacy_toe_z_offset(human_joints: np.ndarray, toe_indices: list[int], mat_height: float) -> float:
    z_min = float(human_joints[:, toe_indices, 2].min())
    if z_min >= mat_height:
        # On a mat.
        z_min -= mat_height
    return -z_min


def _raycast_down_to_mesh_z(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    batch_size: int = 8192,
    origin_z_epsilon: float = 1e-4,
) -> np.ndarray:
    """Return the closest scene z hit below each origin; NaN means no hit."""
    hit_z = np.full(origins.shape[0], np.nan, dtype=float)
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (batch_size, 1))

    for start in range(0, origins.shape[0], batch_size):
        end = min(start + batch_size, origins.shape[0])
        batch_origins = np.asarray(origins[start:end], dtype=float).copy()
        batch_origins[:, 2] += origin_z_epsilon
        batch_dirs = directions[: end - start]

        locations, index_ray, _ = mesh.ray.intersects_location(batch_origins, batch_dirs, multiple_hits=True)
        if len(locations) == 0:
            continue

        for location, ray_idx in zip(locations, index_ray):
            z_hit = float(location[2])
            global_idx = start + int(ray_idx)
            # With a -z ray, the largest z below the origin is the first surface touched.
            if z_hit <= batch_origins[int(ray_idx), 2] + 1e-8 and (
                np.isnan(hit_z[global_idx]) or z_hit > hit_z[global_idx]
            ):
                hit_z[global_idx] = z_hit

    return hit_z


def _raycast_local_surface_z(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    batch_size: int = 8192,
    origin_z_margin: float = 0.5,
) -> np.ndarray:
    """Return local terrain z by casting from just above each point toward -z."""
    origins = np.asarray(points, dtype=float).reshape(-1, 3).copy()
    origins[:, 2] += origin_z_margin
    return _raycast_down_to_mesh_z(mesh, origins, batch_size=batch_size, origin_z_epsilon=0.0)


def _scene_raycast_reference_indices(demo_joints: list[str], toe_indices: list[int]) -> list[int]:
    reference_keywords = ("foot", "toe", "ankle")
    reference_indices = [
        idx for idx, name in enumerate(demo_joints) if any(keyword in name.lower() for keyword in reference_keywords)
    ]
    reference_indices.extend(toe_indices)
    return sorted(set(reference_indices))


def _apply_scene_raycast_z_alignment(
    human_joints: np.ndarray,
    toe_indices: list[int],
    reference_indices: list[int],
    object_mesh_file: str | Path | None,
    legacy_offset: float,
    mode: RaycastZMode,
    clearance: float,
    global_percentile: float,
) -> bool:
    if object_mesh_file is None:
        return False

    mesh_path = Path(object_mesh_file)
    if not mesh_path.is_file():
        print(f"[preprocess_motion_data] Scene raycast z alignment skipped: missing mesh {mesh_path}")
        return False

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print(f"[preprocess_motion_data] Scene raycast z alignment skipped: empty mesh {mesh_path}")
        return False

    num_frames = human_joints.shape[0]
    reference_joints = human_joints[:, reference_indices, :]
    reference_points = reference_joints.reshape(-1, 3)
    hit_z = _raycast_local_surface_z(mesh, reference_points).reshape(num_frames, len(reference_indices))
    hit_mask = np.isfinite(hit_z)
    if not hit_mask.any():
        print(f"[preprocess_motion_data] Scene raycast z alignment found no hits; using legacy offset {legacy_offset:.4f}")
        return False

    offsets = hit_z + float(clearance) - reference_joints[:, :, 2]
    toe_local_indices = [reference_indices.index(index) for index in toe_indices if index in reference_indices]
    toe_hit_mask = hit_mask[:, toe_local_indices] if toe_local_indices else hit_mask

    if mode in {"sequence", "global"}:
        offsets_for_max = np.where(hit_mask, offsets, -np.inf)
        frame_max_offsets = np.max(offsets_for_max, axis=1)
        valid_frames = np.isfinite(frame_max_offsets)
        if not np.any(valid_frames):
            print(
                "[preprocess_motion_data] Scene raycast z alignment found no valid frames; "
                f"using legacy offset {legacy_offset:.4f}"
            )
            return False

        percentile = float(np.clip(global_percentile, 0.0, 100.0))
        valid_frame_offsets = frame_max_offsets[valid_frames]
        strict_nonpenetration_offset = float(np.max(valid_frame_offsets))
        z_offset = float(np.percentile(valid_frame_offsets, percentile))
        human_joints[:, :, 2] += z_offset
        post_clearance = z_offset - offsets
        valid_clearance = post_clearance[hit_mask]
        penetration_refs = int(np.count_nonzero(valid_clearance < -1e-6))
        penetration_frames = int(np.count_nonzero(np.any((post_clearance < -1e-6) & hit_mask, axis=1)))
        print(
            "[preprocess_motion_data] Scene raycast z alignment: "
            f"mode=sequence references={len(reference_indices)} "
            f"hit_frames={int(valid_frames.sum())}/{num_frames} "
            f"hits={int(hit_mask.sum())}/{hit_mask.size} "
            f"global_percentile={percentile:.1f} "
            f"z_offset={z_offset:.4f} strict_nonpenetration_offset={strict_nonpenetration_offset:.4f} "
            f"min_clearance={float(np.min(valid_clearance)):.4f} "
            f"penetration_refs={penetration_refs} penetration_frames={penetration_frames}"
        )
        return True

    if mode != "per_frame":
        raise ValueError(f"Unsupported scene raycast z mode: {mode}")

    valid_frames = hit_mask.any(axis=1)
    toe_hit_frames = toe_hit_mask.any(axis=1)
    frame_offsets = np.full(num_frames, legacy_offset, dtype=float)
    offsets_for_max = np.where(hit_mask, offsets, -np.inf)
    max_ref_offsets = np.max(offsets_for_max[valid_frames], axis=1)
    frame_offsets[valid_frames] = np.maximum(0.0, max_ref_offsets)
    frame_offsets[toe_hit_frames] = np.max(offsets_for_max[toe_hit_frames], axis=1)
    human_joints[:, :, 2] += frame_offsets[:, None]
    post_clearance = reference_joints[:, :, 2] + frame_offsets[:, None] - hit_z
    min_clearance = float(np.nanmin(post_clearance[hit_mask]))
    penetration_refs = int(np.count_nonzero((post_clearance < float(clearance) - 1e-6) & hit_mask))
    print(
        "[preprocess_motion_data] Scene raycast z alignment: "
        f"mode=per_frame references={len(reference_indices)} "
        f"hit_frames={int(valid_frames.sum())}/{num_frames} "
        f"toe_hit_frames={int(toe_hit_frames.sum())}/{num_frames} "
        f"hits={int(hit_mask.sum())}/{hit_mask.size} "
        f"offset_range=({float(frame_offsets.min()):.4f}, {float(frame_offsets.max()):.4f}) "
        f"fallback_frames={int((~valid_frames).sum())} "
        f"min_clearance={min_clearance:.4f} penetration_refs={penetration_refs}"
    )
    return True


def preprocess_motion_data(
    human_joints,
    retargeter,
    foot_names,
    scale=0.714,
    mat_height=0.1,
    object_poses=None,
    object_mesh_file: str | Path | None = None,
    scene_raycast_z_alignment: bool = False,
    scene_raycast_z_mode: RaycastZMode = "per_frame",
    scene_raycast_z_clearance: float = 0.0,
    scene_raycast_z_global_percentile: float = 95.0,
):
    """
    Preprocess human joints and object poses for retargeting.

    Args:
        human_joints (np.ndarray): Human joint positions.
        object_poses (np.ndarray): Object poses.
        retargeter: Retargeting object with smplh_joint2idx attribute.
        scale (float): Scaling factor.
        normalize_height (bool): Whether to normalize human joint heights.

    Returns:
        tuple: (human_joints_scaled, object_poses_scaled, object_moving_frame_idx).
    """
    toe_indices = [
        retargeter.demo_joints.index(foot_names[0]),
        retargeter.demo_joints.index(foot_names[1]),
    ]
    legacy_offset = _compute_legacy_toe_z_offset(human_joints, toe_indices, mat_height)
    aligned_to_scene = False
    if scene_raycast_z_alignment:
        reference_indices = _scene_raycast_reference_indices(retargeter.demo_joints, toe_indices)
        aligned_to_scene = _apply_scene_raycast_z_alignment(
            human_joints,
            toe_indices,
            reference_indices,
            object_mesh_file,
            legacy_offset,
            scene_raycast_z_mode,
            scene_raycast_z_clearance,
            scene_raycast_z_global_percentile,
        )
    if not aligned_to_scene:
        human_joints[:, :, 2] += legacy_offset

    # Scale human joints
    human_joints = human_joints * scale

    if object_poses is not None:
        object_poses[:, -3:-1] = object_poses[:, -3:-1] * scale
        object_z0 = object_poses[0, -1]
        dz_scale = (object_poses[:, -1] - object_z0) * scale
        object_poses[:, -1] = object_z0 + dz_scale

        object_moving_frame_idx = extract_object_first_moving_frame(object_poses)

        return human_joints, object_poses, object_moving_frame_idx

    return human_joints


def extract_object_first_moving_frame(object_poses, vel_threshold=0.0025):
    """Extract the first frame where the object starts moving."""
    object_vel = np.diff(object_poses, axis=0)
    object_vel_norm = np.linalg.norm(object_vel, axis=1)
    return np.argmax(object_vel_norm > vel_threshold)


def extract_foot_sticking_sequence(smpl_joints, demo_joints, foot_names, smpl_contact_threshold_relative=0.01):
    """
    Extract contact sequence from SMPL joint data.

    Args:
        smpl_joints (np.ndarray): SMPL joint positions.
        smplh_joint2idx (dict): Mapping from joint names to indices.
        smpl_contact_threshold_relative (float): The foot is in the air if z is
        larger than z_min + smpl_contact_threshold_relative.

    Returns:
        list: List of contact dictionaries for each frame.
    """
    z_L_min = smpl_joints[:, demo_joints.index(foot_names[0]), 2].min()
    z_R_min = smpl_joints[:, demo_joints.index(foot_names[1]), 2].min()

    return [
        {
            foot_names[0]: smpl_joints_i[demo_joints.index(foot_names[0]), 2]
            <= z_L_min + smpl_contact_threshold_relative,
            foot_names[1]: smpl_joints_i[demo_joints.index(foot_names[1]), 2]
            <= z_R_min + smpl_contact_threshold_relative,
        }
        for smpl_joints_i in smpl_joints
    ]


def augment_object_poses(
    object_poses,
    object_moving_frame_idx,
    human_initial_root,
    local_translation=None,
    rotation_initial=0,
    translation_tau=50,
    rotation_tau=25,
):
    """
    Augment object poses with translation and rotation.

    Args:
        object_poses (np.ndarray): Original object poses array.
        object_moving_frame_idx (int): Index of first moving frame.
        human_initial_root (np.ndarray): Initial human root position.
        local_translation (np.ndarray): Translation vector in human frame.
        rotation_initial (float): Initial rotation angle.

    Returns:
        np.ndarray: Augmented object poses.
    """

    if local_translation is None:
        local_translation = np.array([0, 0, 0])

    N = len(object_poses)
    object_poses_augmented = object_poses.copy()

    if (local_translation != 0).any():
        world_translation, _ = transform_from_human_to_world(human_initial_root, object_poses[0], local_translation)
        object_poses_augmented[:object_moving_frame_idx, -3:] += world_translation
        object_poses_augmented[object_moving_frame_idx:, -3:] += (
            world_translation
            * np.exp(
                (object_moving_frame_idx - np.arange(object_moving_frame_idx, len(object_poses))) / translation_tau
            )[:, None]
        )

    if rotation_initial != 0:
        rotation_list = np.zeros(N)
        rotation_list[:] = rotation_initial
        rotation_list[object_moving_frame_idx:] = rotation_initial * np.exp(
            (object_moving_frame_idx - np.arange(object_moving_frame_idx, N)) / rotation_tau
        )
        rotation = R.from_euler("z", rotation_list)
        object_quat = R.from_quat(object_poses[:, :4], scalar_first=True)
        object_quat_rotated = (rotation * object_quat).as_quat(scalar_first=True)
        object_poses_augmented[:, :4] = object_quat_rotated

    return object_poses_augmented


def transform_from_human_to_world(human_initial_root, object_initial_pose, local_translation):
    """
    Transform translation into a world frame coordinate system.

    Human frame definition:
    - Origin: human_initial_root
    - X-axis: Vector from human_initial_root[:2] to object_initial_pose[:2]
    - Z-axis: Pointing upwards [0, 0, 1]
    - Y-axis: Cross product of Z and X (right-handed coordinate system)

    Args:
        human_initial_root (np.ndarray): Human joint positions with shape (3).
        object_initial_pose (np.ndarray): Object poses with shape (7) [x, y, z, qw, qx, qy, qz].
        local_translation (np.ndarray): Local translation with shape (3).

    Returns:
        tuple: (world_translation, quaternion) - transformed translation and rotation.
    """
    human_to_object_2d = object_initial_pose[-3:-1] - human_initial_root[:2]
    x_axis_2d = human_to_object_2d / np.linalg.norm(human_to_object_2d)
    x_axis = np.array([x_axis_2d[0], x_axis_2d[1], 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    return rotation_matrix @ local_translation, quat


def transform_points_world_to_local(quat, trans, points_world):
    """
    Transform points from world frame to local frame.

    Args:
        quat (np.ndarray): Object quaternion [qw, qx, qy, qz] (scalar-last format).
        trans (np.ndarray): Object translation [x, y, z] in world frame.
        points_world (np.ndarray): Points in world frame, shape (N, 3).

    Returns:
        np.ndarray: Points in local frame, shape (N, 3).
    """
    transform_matrix = trimesh.transformations.quaternion_matrix(quat)
    transform_matrix[:3, 3] = trans
    inverse_transform_matrix = np.linalg.inv(transform_matrix)

    hom_points = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    transformed_points_hom = (inverse_transform_matrix @ hom_points.T).T
    return transformed_points_hom[:, :3]


def create_interaction_mesh(vertices: np.ndarray):
    """
    Creates a tetrahedral mesh from human and object points using Delaunay triangulation.

    Args:
        vertices (np.ndarray): (num_vertices, 3) array.

    Returns:
        tuple: (vertices, tetrahedra) - combined points and generated tetrahedra.
    """
    tri = Delaunay(vertices)
    return vertices, tri.simplices


def transform_points_local_to_world(quat, trans, points_local):
    """Transform points from local frame to world frame."""
    transform_matrix = trimesh.transformations.quaternion_matrix(quat)
    transform_matrix[:3, 3] = trans
    hom_points = np.hstack([points_local, np.ones((points_local.shape[0], 1))])
    transformed_points_hom = (transform_matrix @ hom_points.T).T
    return transformed_points_hom[:, :3]


def get_adjacency_list(tetrahedra, num_vertices):
    """Creates an adjacency list from the tetrahedra."""
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def calculate_laplacian_coordinates(vertices, adj_list, epsilon=1e-6, uniform_weight=True):
    """
    Calculates the Laplacian coordinates for each vertex in the mesh.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): Whether to use uniform weights.

    Returns:
        np.ndarray: (N, 3) array of Laplacian coordinates.
    """
    laplacian = np.zeros_like(vertices)

    for i in range(len(vertices)):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            vi = vertices[i]
            neighbor_positions = vertices[neighbors_indices]
            distances = np.linalg.norm(vi - neighbor_positions, axis=1)

            if uniform_weight:
                weights = np.ones_like(distances)
            else:
                weights = 1.0 / (1.5 * distances + epsilon)

            sum_of_weights = np.sum(weights)
            weighted_sum_of_neighbors = np.sum(weights[:, np.newaxis] * neighbor_positions, axis=0)
            center_of_neighbors = weighted_sum_of_neighbors / sum_of_weights
            laplacian[i] = vi - center_of_neighbors

    return laplacian


def calculate_laplacian_matrix(vertices, adj_list, epsilon=1e-6, uniform_weight=True):
    """
    Calculates the Laplacian matrix for the mesh with optional weight schemes.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): If True, use uniform weights; if False, use distance-based weights.

    Returns:
        np.ndarray: (N, N) Laplacian matrix.
    """
    N = len(vertices)
    laplacian_matrix = np.zeros((N, N))

    for i in range(N):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            if uniform_weight:
                weights = np.ones(len(neighbors_indices)) / len(neighbors_indices)
            else:
                vi = vertices[i]
                neighbor_positions = vertices[neighbors_indices]
                distances = np.linalg.norm(vi - neighbor_positions, axis=1)
                weights = 1.0 / (distances + epsilon)
                sum_weights = np.sum(weights)
                weights = weights / sum_weights

            laplacian_matrix[i, i] = 1.0

            for j, neighbor_idx in enumerate(neighbors_indices):
                laplacian_matrix[i, neighbor_idx] = -weights[j]

    return laplacian_matrix


def find_standing_pose(q: np.ndarray):
    """Find standing pose from current configuration q."""
    q_standing = np.copy(q)
    # rpy_vector = RollPitchYaw(Quaternion(q[:4])).vector()
    # standing_quat = RollPitchYaw(0, 0, rpy_vector[2]).ToQuaternion()
    # quat = standing_quat.wxyz()
    # if np.dot(quat, q[:4]) < 0:
    #     quat = -quat
    # q_standing[:4] = quat
    # q_standing[6] = 0.76  # slightly shorter than the height of G1 pelvis due to bending
    # q_standing[7 : 7 + 29] = Q_A_STANDING
    q_standing[19:22] = 0.0
    return q_standing


def load_smpl_motion(model_path, motion_file):
    """
    Loads SMPL model and motion data, then computes joint positions.

    Args:
        model_path (str): Path to the SMPL model directory.
        motion_file (str): Path to the .npz motion file (AMASS format).

    Returns:
        numpy.ndarray: A (num_frames, num_joints, 3) array of 3D joint positions.
        smplx.SMPL: The loaded SMPL model object.
    """
    print("Loading SMPL model and motion...")
    model = smplx.SMPL(model_path=model_path, gender="neutral", ext="pkl").to("cpu")
    motion_data = np.load(motion_file)

    num_frames = motion_data["poses"].shape[0]
    body_pose = torch.from_numpy(motion_data["poses"][:, 3:]).float()
    global_orient = torch.from_numpy(motion_data["poses"][:, :3]).float()
    betas = torch.from_numpy(motion_data["betas"][:1, :]).float().repeat(num_frames, 1)
    trans = torch.from_numpy(motion_data["trans"]).float()

    output = model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=trans)
    return output.joints.detach().numpy(), model


def create_top_surface_weight_function(up_direction=None, angle_threshold=30):
    """
    Create a weight function that prioritizes top-facing surfaces.

    Args:
        up_direction: Vector pointing upward (default: [0, 0, 1])
        angle_threshold: Maximum angle in degrees from up_direction to be considered "top"

    Returns:
        Function that takes (face_normal, face_center) and returns weight
    """
    if up_direction is None:
        up_direction = np.array([0, 0, 1])
    else:
        up_direction = up_direction / np.linalg.norm(up_direction)
    cos_threshold = np.cos(np.radians(angle_threshold))

    def top_surface_weight(face_normal, face_center):
        cos_angle = np.dot(face_normal, up_direction)

        if cos_angle >= cos_threshold:
            if face_center[2] >= 0.9:
                return 20.0
            return 1.0
        if cos_angle >= 0:
            return 1.0
        return 0.1

    return top_surface_weight


def scale_points_in_object_axes_frame(points, scale_factors, object_axes):
    """Scale points in the object axes frame."""
    return (points @ object_axes.T * scale_factors) @ object_axes


def create_scaled_object_mesh_and_urdf(
    scale_factors, object_vertices, object_faces, object_axes, object_urdf, save_dir="generated_objects/"
):
    """
    Create a scaled object mesh and URDF file.

    Args:
        scale_factors (tuple): Scale factors for x, y, z dimensions.
        save_dir (str): Directory to save the files.

    Returns:
        str: Path to the created URDF file.
    """
    object_file_name = f"largebox_scaled_{scale_factors[0]}_{scale_factors[1]}_{scale_factors[2]}"
    scaled_vertices = scale_points_in_object_axes_frame(object_vertices, scale_factors, object_axes)
    mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=object_faces)

    mesh_subdir = f"{save_dir}/meshes"
    os.makedirs(mesh_subdir, exist_ok=True)

    mesh_file_name = f"{mesh_subdir}/{object_file_name}.obj"
    if not Path(mesh_file_name).exists():
        mesh.export(mesh_file_name)

    urdf_file_name = f"{save_dir}/{object_file_name}.urdf"
    if not Path(urdf_file_name).exists():
        with open(object_urdf) as f:
            template = Template(f.read(), autoescape=True)
        rendered_urdf = template.render(scale_x=scale_factors[0], scale_y=scale_factors[1], scale_z=scale_factors[2])
        with open(urdf_file_name, "w") as f:
            f.write(rendered_urdf)

    return urdf_file_name


def create_scaled_multi_boxes_urdf(
    urdf_path: str,
    new_scale: tuple,
    output_path: str | None = None,
):
    """Read multi_boxes.urdf and generate scaled version."""
    if output_path is None:
        sx, sy, sz = new_scale
        output_path = urdf_path.replace(".urdf", f"_scaled_{sx:.2f}_{sy:.2f}_{sz:.2f}.urdf")

    if Path(output_path).exists():
        return output_path

    with open(urdf_path) as f:
        content = f.read()

    pattern = r'scale="[^"]*"'
    replacement = f'scale="{new_scale[0]} {new_scale[1]} {new_scale[2]}"'
    content = re.sub(pattern, replacement, content)

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def create_scaled_multi_boxes_xml(
    xml_path: str,
    new_scale: tuple,
    output_path: str | None = None,
):
    """Read multi_boxes.urdf and generate scaled version."""
    if output_path is None:
        sx, sy, sz = new_scale
        output_path = xml_path.replace(".xml", f"_scaled_{sx:.2f}_{sy:.2f}_{sz:.2f}.xml")

    with open(xml_path) as f:
        content = f.read()

    pattern = r'scale="[^"]*"'
    replacement = f'scale="{new_scale[0]} {new_scale[1]} {new_scale[2]}"'
    content = re.sub(pattern, replacement, content)

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def create_new_scene_xml_file(
    ori_scene_xml_path: str,
    new_scale: tuple,
    new_object_asset_xml_path: str,
    output_path: str | None = None,
):
    if output_path is None:
        sx, sy, sz = new_scale
        output_path = ori_scene_xml_path.replace(".xml", f"_scaled_{sx:.2f}_{sy:.2f}_{sz:.2f}.xml")

    with open(ori_scene_xml_path) as f:
        content = f.read()

    new_asset = new_object_asset_xml_path.split("/")[-1]
    pattern = r'file="box_assets\.xml"'
    replacement = f'file="{new_asset}"'
    content = re.sub(pattern, replacement, content)

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def extract_foot_sticking_sequence_velocity(smpl_joints, demo_joints, foot_names, velocity_threshold=0.01):
    """
    Extract contact sequence from SMPL joint data based on x,y velocity of toe joints.

    Args:
        smpl_joints (np.ndarray): SMPL joint positions of shape (T, N, 3).
        demo_joints (list): List of joint names.
        foot_names (list): List of foot joint names [left_foot, right_foot].
        velocity_threshold (float): Threshold for xy velocity to determine contact.

    Returns:
        list: List of contact dictionaries for each frame.
    """

    left_toe_idx = demo_joints.index(foot_names[0])
    right_toe_idx = demo_joints.index(foot_names[1])

    # Check xy velocities
    left_toe_positions = smpl_joints[:, left_toe_idx, :2]
    right_toe_positions = smpl_joints[:, right_toe_idx, :2]

    left_toe_velocity = np.linalg.norm(np.diff(left_toe_positions, axis=0), axis=1)
    right_toe_velocity = np.linalg.norm(np.diff(right_toe_positions, axis=0), axis=1)

    left_toe_velocity = np.concatenate([[velocity_threshold + 1], left_toe_velocity])
    right_toe_velocity = np.concatenate([[velocity_threshold + 1], right_toe_velocity])

    return [
        {"L_Toe": left_toe_velocity[i] <= velocity_threshold, "R_Toe": right_toe_velocity[i] <= velocity_threshold}
        for i in range(len(smpl_joints))
    ]


def transform_y_up_to_z_up(points):
    """
    Transform points from y-up to z-up coordinate system.

    Transformation:
    - Y-axis (up) becomes Z-axis (up)
    - Z-axis (forward) becomes Y-axis (forward)
    - X-axis (right) stays X-axis (right)

    Args:
        points (np.ndarray): Points with shape (..., 3) where last dimension is [x, y, z]

    Returns:
        np.ndarray: Transformed points with shape (..., 3) where last dimension is [x, y, z]
    """
    # Create transformation matrix
    # [x, y, z] -> [x, z, y]
    transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    # Apply transformation
    if points.ndim == 1:
        return transform_matrix @ points
    if points.ndim == 2:
        return (transform_matrix @ points.T).T
    if points.ndim == 3:
        original_shape = points.shape
        points_reshaped = points.reshape(-1, 3)
        transformed = (transform_matrix @ points_reshaped.T).T
        return transformed.reshape(original_shape)
    raise ValueError(f"Unsupported number of dimensions: {points.ndim}")


def estimate_human_orientation(human_joints, joint_names, frame_idx=0):
    """
    Estimate the human's global orientation quaternion based on joint positions.

    This function estimates the human's orientation by looking at the direction
    from the pelvis (Hips) to the spine/chest, and the direction from left to right hip.

    Args:
        human_joints (np.ndarray): Human joint positions with shape (frames, joints, 3)
        joint_names (list): List of joint names corresponding to the joint positions
        frame_idx (int): Frame index to estimate orientation from (default: 0)

    Returns:
        np.ndarray: Quaternion [w, x, y, z] representing the human's global orientation
    """
    # For LAFAN
    if "Hips" in joint_names:
        hips_idx = joint_names.index("Hips")
        spine_idx = joint_names.index("Spine")
        left_hip_idx = joint_names.index("LeftUpLeg")
        right_hip_idx = joint_names.index("RightUpLeg")
    else:
        # For SMPLH (OMOMO_new)
        hips_idx = joint_names.index("Pelvis")
        spine_idx = joint_names.index("Spine")
        left_hip_idx = joint_names.index("L_Hip")
        right_hip_idx = joint_names.index("R_Hip")

    hips_pos = human_joints[frame_idx, hips_idx]
    spine_pos = human_joints[frame_idx, spine_idx]
    left_hip_pos = human_joints[frame_idx, left_hip_idx]
    right_hip_pos = human_joints[frame_idx, right_hip_idx]

    # Calculate forward direction (from hips to spine)
    forward_vec = hips_pos - spine_pos
    forward_vec[2] = 0  # Project to horizontal plane (ignore vertical component)
    if np.linalg.norm(forward_vec) > 1e-6:
        forward_vec = forward_vec / np.linalg.norm(forward_vec)
    else:
        # If spine is directly above hips, use a default forward direction
        forward_vec = np.array([0, 1, 0])

    # Calculate right direction (from left hip to right hip)
    left_vec = left_hip_pos - right_hip_pos
    left_vec[2] = 0  # Project to horizontal plane
    if np.linalg.norm(left_vec) > 1e-6:
        left_vec = left_vec / np.linalg.norm(left_vec)
    else:
        # If hips are aligned vertically, use a default right direction
        left_vec = np.array([1, 0, 0])

    # Ensure left_vec is perpendicular to forward_vec
    left_vec = left_vec - np.dot(left_vec, forward_vec) * forward_vec
    if np.linalg.norm(left_vec) > 1e-6:
        left_vec = left_vec / np.linalg.norm(left_vec)
    else:
        # Fallback if vectors are parallel
        left_vec = np.array([1, 0, 0])

    # Calculate up direction (cross product to ensure orthogonality)
    up_vec = np.cross(forward_vec, left_vec)
    up_vec = up_vec / np.linalg.norm(up_vec)
    forward_vec = np.cross(left_vec, up_vec)
    forward_vec = forward_vec / np.linalg.norm(forward_vec)

    rotation_matrix = np.column_stack([forward_vec, left_vec, up_vec])
    assert np.linalg.det(rotation_matrix) > 0
    rotation = R.from_matrix(rotation_matrix)
    return rotation.as_quat(scalar_first=True)
