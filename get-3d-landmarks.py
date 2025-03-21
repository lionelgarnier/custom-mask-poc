import mediapipe as mp
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyvista as pv
from scipy.interpolate import splprep, splev
import os

# -------------------- CONFIGURATION --------------------

# Default landmarks for face contour
DEFAULT_FACE_CONTOUR_LANDMARKS = [168, 417, 465, 429, 423, 391, 393, 164, 167, 165, 203, 209, 245, 193, 168]

# -------------------- CORE PROCESSING FUNCTIONS --------------------

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
    
    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        The input mesh
    landmarks : array
        3D landmark points
    landmark_indices : array
        Indices of valid landmarks
    contour_landmark_ids : list, optional
        List of landmark IDs to use for the face contour. If None, uses default landmarks.
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

def smooth_line_points(points, smoothing=0.1, num_samples=300):
    """Smooth 3D points using spline interpolation"""
    x, y, z = points[:,0], points[:,1], points[:,2]
    tck, u = splprep([x, y, z], s=smoothing, k=2, per=True)
    u_new = np.linspace(0, 1, num_samples)
    x_new, y_new, z_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new, z_new))

def create_pyvista_mesh(mesh):
    """Convert Open3D mesh to PyVista mesh"""
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.triangles)
    faces_pyvista = np.hstack((np.full((len(mesh_faces), 1), 3), mesh_faces)).astype(np.int64)
    pv_mesh = pv.PolyData(mesh_vertices, faces_pyvista)
    
    # Transfer vertex colors if available
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        # Convert RGB colors from [0,1] to [0,255] for PyVista
        colors = (colors * 255).astype(np.uint8)
        pv_mesh.point_data["RGB"] = colors
        
    return pv_mesh

def create_3d_printable_extrusion(line_points, radius=0.1, resolution=30, output_path="extruded_line.stl"):
    """
    Create a 3D printable object by extruding the line with a circular profile
    """
    # Ensure points are in numpy array format
    line_points = np.array(line_points)
    
    # Create a tube (cylinder) along the line path
    tube = pv.Spline(line_points).tube(radius=radius, n_sides=resolution)
    
    # Close the loop if needed
    if np.allclose(line_points[0], line_points[-1], atol=1e-4):
        print("Closed loop detected, ensuring watertight model")
    
    # Save the tube as STL
    tube.save(output_path)
    print(f"3D printable object saved to: {os.path.abspath(output_path)}")
    
    return tube

# -------------------- VISUALIZATION FUNCTIONS --------------------

def set_front_view(plotter):
    """Set camera to front view with head upright"""
    plotter.view_xy()  # Set to front view (looking at XY plane)
    plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]  # Position, focus, up-vector
    plotter.camera.zoom(0.8)  # Slight zoom for better framing
    plotter.enable_trackball_style()
    return plotter

def visualize_2d_landmarks(image, title="Landmarks 2D"):
    """Visualize 2D landmarks on image"""
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    return fig

def visualize_3d_landmarks(points_3d, title="Landmarks 3D"):
    """Visualize 3D landmarks in a scatter plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='b', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Adjust axes for better visualization
    max_range = np.max([
        np.max(points_3d[:,0]) - np.min(points_3d[:,0]),
        np.max(points_3d[:,1]) - np.min(points_3d[:,1]),
        np.max(points_3d[:,2]) - np.min(points_3d[:,2])
    ])
    mid_x = (np.max(points_3d[:,0]) + np.min(points_3d[:,0])) * 0.5
    mid_y = (np.max(points_3d[:,1]) + np.min(points_3d[:,1])) * 0.5
    mid_z = (np.max(points_3d[:,2]) + np.min(points_3d[:,2])) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    
    return fig

def visualize_mesh_with_landmarks(mesh, landmarks, landmark_indices=None, plotter=None, contour_landmark_ids=None):
    """Visualize 3D mesh with landmarks and IDs"""
    if plotter is None:
        plotter = pv.Plotter()
        return_plotter = True
    else:
        return_plotter = False
    
    pv_mesh = create_pyvista_mesh(mesh)
    plotter.add_mesh(pv_mesh, color='white', opacity=0.5)

    # Add landmarks with their IDs
    landmarks = np.array(landmarks)
    plotter.add_points(landmarks, color='red', point_size=6, render_points_as_spheres=True)

    if landmark_indices is None:
        landmark_indices = np.arange(len(landmarks))

    for idx, point in zip(landmark_indices, landmarks):
        plotter.add_point_labels([point], [str(idx)], font_size=10, text_color='blue', shape_opacity=0.0)

    # Draw green line joining specified landmark IDs
    if contour_landmark_ids is None:
        contour_landmark_ids = DEFAULT_FACE_CONTOUR_LANDMARKS
        
    valid_ids = [np.where(landmark_indices == id)[0][0] for id in contour_landmark_ids if id in landmark_indices]
    line_points = landmarks[valid_ids]
    line_points = smooth_line_points(line_points, smoothing=0.1, num_samples=300)
    line = pv.lines_from_points(line_points, close=True)
    plotter.add_mesh(line, color='green', line_width=2)
    
    # Project green line onto the face using a front-view projection (along Z)
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
            projected_line_points.append(pt)
            
    if len(projected_line_points) > 1:
        proj_line = pv.lines_from_points(np.array(projected_line_points), close=False)
        plotter.add_mesh(proj_line, color='orange', line_width=4)
    
    # Set to front view
    set_front_view(plotter)
    
    if return_plotter:
        plotter.show_grid()
        return plotter, projected_line_points
    else:
        return projected_line_points

def visualize_mesh_with_yellow_lines(mesh, line_points, plotter=None):
    """Visualize 3D mesh with yellow lines"""
    if plotter is None:
        plotter = pv.Plotter()
        return_plotter = True
    else:
        return_plotter = False
    
    pv_mesh = create_pyvista_mesh(mesh)
    
    # Add mesh with RGB colors if available, otherwise white
    if mesh.has_vertex_colors():
        plotter.add_mesh(pv_mesh, scalars="RGB", rgb=True, opacity=1.0)
    else:
        plotter.add_mesh(pv_mesh, color='white', opacity=1.0)

    # Add the yellow lines
    proj_line = pv.lines_from_points(np.array(line_points), close=True)
    plotter.add_mesh(proj_line, color='yellow', line_width=4)
    
    # Set to front view
    set_front_view(plotter)
    
    if return_plotter:
        plotter.show_grid()
        return plotter
    else:
        return None

# -------------------- MAIN FUNCTION --------------------

def process_face(mesh_path, show_landmarks_2d=False, show_landmarks_3d=False, show_mesh_and_landmarks=False,
                create_3d_print=False, extrusion_radius=0.1, contour_landmark_ids=None):
    """
    Main function to process a face model and extract landmarks
    
    Parameters:
    -----------
    mesh_path : str
        Path to the 3D mesh file
    show_landmarks_2d : bool
        Whether to show 2D landmarks visualization
    show_landmarks_3d : bool
        Whether to show 3D landmarks visualization
    show_mesh_and_landmarks : bool
        Whether to show the mesh with landmarks
    create_3d_print : bool
        Whether to create a 3D printable STL file
    extrusion_radius : float
        Radius of the extruded tube for 3D printing
    contour_landmark_ids : list, optional
        List of landmark IDs to use for the face contour. If None, uses default landmarks.
    """
    
    # Process the face to extract landmarks and contour
    mesh, valid_points_3d, valid_indices, projected_line_points, viz_image = extract_face_landmarks(
        mesh_path, contour_landmark_ids)
    
    # Visualizations (if requested)
    # Create a multi-panel PyVista plotter if needed
    num_pyvista_views = sum([show_mesh_and_landmarks, True])  # Count PlotterIDs (+1 for final view)
    if num_pyvista_views > 1:
        pv_plotter = pv.Plotter(shape=(1, num_pyvista_views))
        current_subplot = 0
    else:
        pv_plotter = None

    # Store visualization figures for later display
    figures = []
    if show_landmarks_2d:
        figures.append(visualize_2d_landmarks(viz_image, "MediaPipe Face Landmarks"))

    if show_landmarks_3d:
        figures.append(visualize_3d_landmarks(valid_points_3d, "3D Landmarks (Optimal Approach)"))

    # Create PyVista visualizations
    if show_mesh_and_landmarks:
        if pv_plotter is not None:
            pv_plotter.subplot(0, 0)
            pv_plotter.add_title("Face Mesh with Landmarks")
            visualize_mesh_with_landmarks(mesh, valid_points_3d, valid_indices, pv_plotter, contour_landmark_ids)
            current_subplot += 1
        else:
            # Standalone plotter if only showing this view
            landmarks_plotter = pv.Plotter()
            landmarks_plotter.add_title("Face Mesh with Landmarks")
            visualize_mesh_with_landmarks(mesh, valid_points_3d, valid_indices, landmarks_plotter, contour_landmark_ids)
            landmarks_plotter.show_grid()
            landmarks_plotter.show()

    # Final visualization - always shown
    if pv_plotter is not None:
        pv_plotter.subplot(0, current_subplot)
        pv_plotter.add_title("Face Mesh with Projected Path")
        visualize_mesh_with_yellow_lines(mesh, projected_line_points, pv_plotter)
        
        # Enable trackball style for all subplots and link them
        pv_plotter.enable_trackball_style()
        pv_plotter.link_views()  # Link camera movement between subplots
        pv_plotter.show()
    else:
        # Standalone final plotter
        final_plotter = pv.Plotter()
        final_plotter.add_title("Face Mesh with Projected Path")
        visualize_mesh_with_yellow_lines(mesh, projected_line_points, final_plotter)
        final_plotter.show_grid()
        final_plotter.enable_trackball_style()  # Enable trackball style
        final_plotter.show()

    # Create 3D printable object if requested
    if create_3d_print:
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_path = f"extruded_{base_name}.stl"
        
        # Create and save the 3D printable extrusion
        extruded_tube = create_3d_printable_extrusion(projected_line_points, 
                                                     radius=extrusion_radius,
                                                     output_path=output_path)
        
        print(f"Created 3D printable extrusion with radius {extrusion_radius}")
        print(f"Tube has {extruded_tube.n_points} points and {extruded_tube.n_cells} faces")

    # Display all matplotlib figures (if any)
    if len(figures) > 0:
        plt.show()
    
    return valid_points_3d, projected_line_points

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    # Set parameters for which visualizations to show
    SHOW_LANDMARKS_2D = False  # Show 2D landmarks on projected image
    SHOW_LANDMARKS_3D = False  # Show 3D landmarks scatter plot
    SHOW_MESH_AND_LANDMARKS = False  # Show mesh with landmarks and IDs
    
    # 3D printing options
    CREATE_3D_PRINT = True  # Create a 3D printable STL file
    EXTRUSION_RADIUS = 0.2  # Radius of the extruded tube (adjust based on model scale)
    
    # Face contour landmarks (uncomment and modify to use custom landmarks)
    # CONTOUR_LANDMARKS = [168, 417, 465, 429, 423, 391, 393, 164, 167, 165, 203, 209, 245, 193, 168]
    
    # Process the face model
    mesh_path = "D:/OneDrive/Desktop/masque/face_lionel_long_capture.obj"
    process_face(mesh_path, 
                 show_landmarks_2d=SHOW_LANDMARKS_2D,
                 show_landmarks_3d=SHOW_LANDMARKS_3D,
                 show_mesh_and_landmarks=SHOW_MESH_AND_LANDMARKS,
                 create_3d_print=CREATE_3D_PRINT,
                 extrusion_radius=EXTRUSION_RADIUS,
                 # contour_landmark_ids=CONTOUR_LANDMARKS  # Uncomment to use custom landmarks
                )
