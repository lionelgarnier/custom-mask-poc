"""
Core processing functions for 3D face landmark detection
"""
import mediapipe as mp
import open3d as o3d
import numpy as np
import cv2
from utils import create_pyvista_mesh, smooth_line_points
from config import DEFAULT_FACE_CONTOUR_LANDMARKS

def extract_face_landmarks(mesh_path, contour_landmark_ids=None):
    """
    Core processing function to extract 3D landmarks and face contour
    Returns valid 3D landmarks and projected line points
    
    Parameters:
    -----------
    mesh_path : str
        Path to the 3D mesh file
    contour_landmark_ids : list, optional
        List of landmark IDs to use for the face contour. If None, uses default landmarks.
    """
    # Load 3D model
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Initial setup with a single view
    width, height = 800, 800
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.poll_events()
    vis.update_renderer()

    # Capture depth and color image from the same view
    depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
    image_color = np.asarray(vis.capture_screen_float_buffer(do_render=True))

    # Get camera parameters
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    intrinsic = camera_params.intrinsic.intrinsic_matrix
    extrinsic = camera_params.extrinsic
    vis.destroy_window()

    # Prepare image for MediaPipe
    image_color_cv = (image_color * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_color_cv, cv2.COLOR_RGB2BGR)
    
    # Save projection as an image
    cv2.imwrite("projection.png", image_bgr)

    # MediaPipe face detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        raise ValueError("No face detected!")

    # Extract 2D landmarks
    landmarks = results.multi_face_landmarks[0]
    points_2d = []
    viz_image = image_bgr.copy()  # For visualization if needed

    for lm in landmarks.landmark:
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        points_2d.append((x_px, y_px))
        cv2.circle(viz_image, (x_px, y_px), 1, (0, 255, 0), -1)  # For visualization

    # Extract 3D landmarks
    points_3d = []
    for (x_px, y_px) in points_2d:
        # Check if point is within image boundaries
        if 0 <= x_px < width and 0 <= y_px < height:
            z = depth[y_px, x_px]
            
            if z > 0 and not np.isnan(z) and not np.isinf(z):
                # Use intrinsic parameters for deprojection
                x_normalized = (x_px - intrinsic[0, 2]) / intrinsic[0, 0]
                y_normalized = (y_px - intrinsic[1, 2]) / intrinsic[1, 1]
                
                x_3d = x_normalized * z
                y_3d = y_normalized * z
                z_3d = z
                
                # Transform to world space using extrinsic matrix
                point_camera = np.array([x_3d, y_3d, z_3d, 1.0])
                point_world = np.linalg.inv(extrinsic) @ point_camera
                points_3d.append(point_world[:3])
            else:
                points_3d.append([np.nan, np.nan, np.nan])
        else:
            points_3d.append([np.nan, np.nan, np.nan])

    # Filter out invalid points
    points_3d = np.array(points_3d, dtype=np.float64)
    valid_mask = ~np.isnan(points_3d).any(axis=1)
    valid_points_3d = points_3d[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Save 3D landmarks
    np.save("face_landmarks_3d.npy", valid_points_3d)
    print(f"Valid 3D landmarks extracted: {len(valid_points_3d)} / {len(points_2d)}")

    # Generate contour line from specific landmarks
    if contour_landmark_ids is None:
        contour_landmark_ids = DEFAULT_FACE_CONTOUR_LANDMARKS
    
    projected_line_points = extract_face_contour(mesh, valid_points_3d, valid_indices, contour_landmark_ids)
    
    return mesh, valid_points_3d, valid_indices, projected_line_points, viz_image

def extract_face_contour(mesh, landmarks, landmark_indices, contour_landmark_ids=None):
    """
    Extract face contour line from landmarks and project it onto the face mesh
    Returns the projected line points
    """
    # Convert Open3D mesh to PyVista for ray tracing
    pv_mesh = create_pyvista_mesh(mesh)
    
    # Define specific landmark IDs for face contour
    if contour_landmark_ids is None:
        contour_landmark_ids = DEFAULT_FACE_CONTOUR_LANDMARKS
    
    valid_ids = [np.where(landmark_indices == id)[0][0] for id in contour_landmark_ids if id in landmark_indices]
    
    # Create smooth line from landmarks
    line_points = landmarks[valid_ids]
    line_points = smooth_line_points(line_points, smoothing=0.1, num_samples=300)

    # Project line onto the face using ray casting
    bounds = pv_mesh.bounds
    dz = 100  # offset for ray tracing
    projected_line_points = []
    for pt in line_points:
        origin = (pt[0], pt[1], bounds[5] + dz)
        end = (pt[0], pt[1], bounds[4] - dz)
        pts, ind = pv_mesh.ray_trace(origin, end, first_point=True)
        if pts.size:
            projected_line_points.append(pts)
        else:
            projected_line_points.append(pt)  # fallback if no intersection
    
    return np.array(projected_line_points)
