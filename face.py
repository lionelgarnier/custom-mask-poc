"""
Core processing functions for 3D face landmark detection and alignment
"""
import mediapipe as mp
import open3d as o3d
import numpy as np
import cv2
from utils import create_pyvista_mesh, smooth_line_points, compute_rotation_between_vectors
from config import DEFAULT_FACE_CONTOUR_LANDMARKS
import pyvista as pv
import matplotlib.pyplot as plt


def extract_face_landmarks(mesh_path):
    """
    Core processing function to extract 3D landmarks and face contour.
    1. Loads the 3D face mesh.
    2. Aligns the mesh to a front view.
    3. Extracts landmarks from the aligned mesh.

    Parameters:
    -----------
    mesh_path : str
        Path to the 3D mesh file
     
    Returns:
    --------
    valid_points_3d : np.ndarray
        Extracted 3D landmarks
    valid_indices : np.ndarray
        Indices of the valid landmarks
    """
    # Load 3D model and compute normals
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Align face mesh
    aligned_mesh = align_face_to_front_view(mesh)
    pivot = aligned_mesh.get_center()
    

    # Extract landmarks from the aligned mesh
    landmarks_3d, valid_indices = extract_landmarks_from_view(aligned_mesh)

    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(20), 0])
    # rotated_mesh1 = rotate_face(aligned_mesh, rotation_matrix, pivot)
    # landmarks_3d_rotated1, landmarks_valid_rotated1 = extract_landmarks_from_view(rotated_mesh1)
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(-20), 0])
    # landmarks_3d_rotated1 = rotate_landmarks(landmarks_3d_rotated1, rotation_matrix, pivot)

    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(-20), 0])
    # rotated_mesh2 = rotate_face(aligned_mesh, rotation_matrix, pivot)
    # landmarks_3d_rotated2, landmarks_valid_rotated2 = extract_landmarks_from_view(rotated_mesh2)
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(20), 0])
    # landmarks_3d_rotated2 = rotate_landmarks(landmarks_3d_rotated2, rotation_matrix, pivot)


    # # Display the extracted image color on the screen
    # keep_indices = DEFAULT_FACE_CONTOUR_LANDMARKS
    # plotter = pv.Plotter()
    # plotter.add_title("Face Mesh with Landmarks")

    # aligned_mesh_pyvista = create_pyvista_mesh(aligned_mesh)
    # plotter.add_mesh(aligned_mesh_pyvista, color='white', opacity=0.5)

    # # Front landmarks
    # landmarks = np.array([landmarks_3d[i] for i in keep_indices if i in valid_indices])
    # plotter.add_points(landmarks, color='red', point_size=6, render_points_as_spheres=True)
    # for idx, point in zip(keep_indices, landmarks):
    #     plotter.add_point_labels([point], [str(idx)], font_size=15, text_color='red', shape_opacity=0.0)


    # # Rotated landmarks
    # landmarks = np.array([landmarks_3d_rotated1[i] for i in keep_indices if i in landmarks_valid_rotated1])
    # plotter.add_points(landmarks, color='blue', point_size=6, render_points_as_spheres=True)
    # for idx, point in zip(keep_indices, landmarks):
    #     plotter.add_point_labels([point], [str(idx)], font_size=15, text_color='blue', shape_opacity=0.0)


    # # Rotated landmarks 2
    # landmarks = np.array([landmarks_3d_rotated2[i] for i in keep_indices if i in landmarks_valid_rotated2])
    # plotter.add_points(landmarks, color='green', point_size=6, render_points_as_spheres=True)
    # for idx, point in zip(keep_indices, landmarks):
    #     plotter.add_point_labels([point], [str(idx)], font_size=15, text_color='green', shape_opacity=0.0)




    # plotter.view_xy()  # Set to front view (looking at XY plane)
    # plotter.enable_zoom_style()
    # plotter.enable_trackball_style()
    # plotter.show()

    return aligned_mesh, landmarks_3d, valid_indices


def rotate_face(mesh, rotation_matrix, pivot = None):
    rotated_mesh = o3d.geometry.TriangleMesh(mesh)
    if pivot is None:
        pivot = rotated_mesh.get_center()
    rotated_mesh.rotate(rotation_matrix, center=pivot)

    return rotated_mesh


def rotate_landmarks(landmarks, rotation_matrix, pivot=None):
    if pivot is None:
        pivot = np.mean(landmarks, axis=0)

    landmarks_centered = landmarks - pivot
    rotated_landmarks = np.dot(landmarks_centered, rotation_matrix.T)
    rotated_landmarks += pivot
    
    return rotated_landmarks


def align_face_to_front_view(mesh, width = 800, height = 800):
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

    # Extract initial landmarks
    try:
        landmarks_3d, valid_indices = extract_landmarks_from_view(mesh, width, height)
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
        landmarks_3d_2, valid_indices_2 = extract_landmarks_from_view(mesh_aligned_stage1, width, height)
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

def extract_landmarks_from_view(mesh, width=800, height=800):
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
    
    # # Plot the color image and 2D landmarks
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(image_color_cv, cv2.COLOR_BGR2RGB))
    # points_2d_np = np.array(points_2d)
    # plt.scatter(points_2d_np[:, 0], points_2d_np[:, 1], c='red', s=10)
    # plt.title("2D Landmarks on Color Image")
    # plt.axis('off')
    # plt.show()


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

    # # Plot the mesh and 3D points using PyVista
    # plotter = pv.Plotter()
    # plotter.add_title("3D Mesh and Landmarks")

    # # Add the mesh
    # mesh_pyvista = create_pyvista_mesh(mesh)
    # plotter.add_mesh(mesh_pyvista, color='white', opacity=0.5)

    # # Add the valid 3D points
    # plotter.add_points(valid_points_3d, color='red', point_size=6, render_points_as_spheres=True)

    # plotter.view_xy()  # Set to front view (looking at XY plane)
    # plotter.enable_zoom_style()
    # plotter.enable_trackball_style()
    # plotter.show()

    return valid_points_3d, valid_indices
