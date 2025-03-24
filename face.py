"""
Core processing functions for 3D face landmark detection and alignment
"""
import mediapipe as mp
import open3d as o3d
import numpy as np
import cv2
from utils import create_pyvista_mesh, smooth_line_points
from config import DEFAULT_FACE_CONTOUR_LANDMARKS
import scipy.spatial.transform

def extract_face_landmarks(mesh_path, contour_landmark_ids=None):
    """
    Core processing function to extract 3D landmarks and face contour.
    1. Loads the 3D face mesh.
    2. Aligns the mesh to a front view.
    3. Extracts landmarks from the aligned mesh.
    4. Projects a contour line onto the mesh.

    Parameters:
    -----------
    mesh_path : str
        Path to the 3D mesh file
    contour_landmark_ids : list, optional
        List of landmark IDs to use for the face contour. If None, uses default landmarks.
        
    Returns:
    --------
    mesh : o3d.geometry.TriangleMesh
        The aligned 3D mesh
    valid_points_3d : np.ndarray
        Extracted 3D landmarks
    valid_indices : np.ndarray
        Indices of the valid landmarks
    projected_line_points : np.ndarray
        The projected face contour line onto the mesh
    viz_image : np.ndarray
        Visualization image with 2D landmarks
    """
    # Load 3D model and compute normals
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Align face mesh
    aligned_mesh = align_face_mesh_3d(mesh)

    # Extract landmarks from the aligned mesh
    width, height = 800, 800
    landmarks_2d, landmarks_3d, valid_indices, _, color_image = extract_landmarks_from_mesh(aligned_mesh, width, height)

    # Generate face contour
    if contour_landmark_ids is None:
        contour_landmark_ids = DEFAULT_FACE_CONTOUR_LANDMARKS
    projected_line_points = extract_face_contour(
        aligned_mesh, 
        landmarks_3d, 
        valid_indices, 
        contour_landmark_ids
    )

    # Create visualization image with landmarks
    viz_image = color_image.copy()
    for x, y in landmarks_2d:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(viz_image, (int(x), int(y)), 1, (0, 255, 0), -1)

    return aligned_mesh, landmarks_3d, valid_indices, projected_line_points, viz_image

def align_face_mesh_3d(mesh):
    """
    Align a face mesh in 3D space using a two-stage process:
    1. First rotation (around X/Y) to make the nose axis vertical
    2. Second rotation (around Z or Y) to balance the eyes

    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        The original 3D face mesh
    
    Returns:
    --------
    aligned_mesh : o3d.geometry.TriangleMesh
        The aligned 3D face mesh
    """
    width, height = 800, 800

    # Extract initial landmarks
    try:
        landmarks_2d, landmarks_3d, valid_indices, _, _ = extract_landmarks_from_mesh(mesh, width, height)
        # Identify key nose landmarks
        nose_top_idx = np.where(valid_indices == 168)[0][0]
        nose_bottom_idx = np.where(valid_indices == 200)[0][0]

        nose_top_3d = landmarks_3d[nose_top_idx]
        nose_bottom_3d = landmarks_3d[nose_bottom_idx]
        
        # Stage 1: Vertical nose alignment
        nose_vector = nose_bottom_3d - nose_top_3d
        target_vector = np.array([0, -1, 0])  # Align nose to negative Y axis, adjust if needed
        rotation_1 = compute_rotation_between_vectors(nose_vector, target_vector)
        
        mesh_aligned_stage1 = o3d.geometry.TriangleMesh(mesh)
        pivot = mesh_aligned_stage1.get_center()
        mesh_aligned_stage1.rotate(rotation_1, center=pivot)

        # Stage 2: Eye balance
        # Re-extract landmarks after the first rotation
        landmarks_2d_2, landmarks_3d_2, valid_indices_2, _, _ = extract_landmarks_from_mesh(mesh_aligned_stage1, width, height)
        left_eye_idx = np.where(valid_indices_2 == 33)[0][0]
        right_eye_idx = np.where(valid_indices_2 == 263)[0][0]

        left_eye_3d = landmarks_3d_2[left_eye_idx]
        right_eye_3d = landmarks_3d_2[right_eye_idx]
        eye_vector = right_eye_3d - left_eye_3d
        
        # Align eye direction with X axis
        eye_vector_xz = np.array([eye_vector[0], 0, eye_vector[2]])
        target_vector_xz = np.array([1, 0, 0])
        rotation_2 = compute_rotation_between_vectors(eye_vector_xz, target_vector_xz)
        
        mesh_aligned_stage2 = o3d.geometry.TriangleMesh(mesh_aligned_stage1)
        mesh_aligned_stage2.rotate(rotation_2, center=pivot)

        return mesh_aligned_stage2

    except (IndexError, ValueError) as e:
        print(f"Alignment failed: {e}")
        return mesh

def compute_rotation_between_vectors(source_vector, target_vector):
    """
    Compute the rotation matrix that aligns source_vector with target_vector.
    """
    source = source_vector / np.linalg.norm(source_vector) if np.linalg.norm(source_vector) > 0 else source_vector
    target = target_vector / np.linalg.norm(target_vector) if np.linalg.norm(target_vector) > 0 else target_vector
    
    # If vectors are nearly identical, return identity
    if np.allclose(source, target, atol=1e-6):
        return np.eye(3)

    # If vectors are nearly opposite, we need 180-degree rotation around any perpendicular axis
    if np.allclose(source, -target, atol=1e-6):
        if abs(source[0]) < abs(source[1]):
            perp = np.array([0, source[2], -source[1]])
        else:
            perp = np.array([source[2], 0, -source[0]])
        perp = perp / np.linalg.norm(perp)
        rotation = scipy.spatial.transform.Rotation.from_rotvec(np.pi * perp)
        return rotation.as_matrix()
    
    # In the general case, compute the rotation matrix directly
    rotation_axis = np.cross(source, target)
    rotation_axis /= np.linalg.norm(rotation_axis)
    cos_angle = np.dot(source, target)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    rotation = scipy.spatial.transform.Rotation.from_rotvec(angle * rotation_axis)
    return rotation.as_matrix()

def extract_landmarks_from_mesh(mesh, width=800, height=800):
    """
    Extract 2D and 3D landmarks from a mesh using MediaPipe.
    
    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        The 3D mesh
    width : int
        Image width
    height : int
        Image height
        
    Returns:
    --------
    landmarks_2d : list
        2D landmark coordinates
    landmarks_3d : np.ndarray
        3D landmark coordinates
    valid_indices : np.ndarray
        Indices of valid landmarks
    depth_image : np.ndarray
        Depth image
    color_image : np.ndarray
        Color image
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.poll_events()
    vis.update_renderer()
    
    depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
    image_color = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    intrinsic = camera_params.intrinsic.intrinsic_matrix
    extrinsic = camera_params.extrinsic
    vis.destroy_window()
    
    image_color_cv = (image_color * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_color_cv, cv2.COLOR_RGB2BGR)
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError("No face detected!")
        landmarks = results.multi_face_landmarks[0]
    
    # 2D landmarks
    points_2d = []
    for lm in landmarks.landmark:
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        points_2d.append((x_px, y_px))
    
    # 3D projection
    points_3d = []
    for (x_px, y_px) in points_2d:
        if 0 <= x_px < width and 0 <= y_px < height:
            z = depth[y_px, x_px]
            if z > 0 and not np.isnan(z) and not np.isinf(z):
                x_norm = (x_px - intrinsic[0, 2]) / intrinsic[0, 0]
                y_norm = (y_px - intrinsic[1, 2]) / intrinsic[1, 1]
                x3d = x_norm * z
                y3d = y_norm * z
                z3d = z
                point_camera = np.array([x3d, y3d, z3d, 1.0])
                point_world = np.linalg.inv(extrinsic) @ point_camera
                points_3d.append(point_world[:3])
            else:
                points_3d.append([np.nan, np.nan, np.nan])
        else:
            points_3d.append([np.nan, np.nan, np.nan])
    
    points_3d = np.array(points_3d, dtype=np.float64)
    valid_mask = ~np.isnan(points_3d).any(axis=1)
    valid_points_3d = points_3d[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    return points_2d, valid_points_3d, valid_indices, depth, image_color_cv

def extract_face_contour(mesh, landmarks, landmark_indices, contour_landmark_ids=None):
    """
    Extract face contour line from landmarks and project it onto the face mesh.
    """
    pv_mesh = create_pyvista_mesh(mesh)
    if contour_landmark_ids is None:
        contour_landmark_ids = DEFAULT_FACE_CONTOUR_LANDMARKS
    
    valid_ids = [np.where(landmark_indices == lid)[0][0] 
                 for lid in contour_landmark_ids if lid in landmark_indices]
    
    line_points = landmarks[valid_ids]
    line_points = smooth_line_points(line_points, smoothing=0.1, num_samples=300)
    
    bounds = pv_mesh.bounds
    dz = 100
    projected_line_points = []
    for pt in line_points:
        origin = (pt[0], pt[1], bounds[5] + dz)
        end = (pt[0], pt[1], bounds[4] - dz)
        pts, _ = pv_mesh.ray_trace(origin, end, first_point=True)
        if pts.size:
            projected_line_points.append(pts)
        else:
            projected_line_points.append(pt)
    
    return np.array(projected_line_points)
