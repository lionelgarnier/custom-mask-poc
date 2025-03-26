"""
Utility functions for 3D face processing
"""
import numpy as np
import trimesh
import pyvista as pv
import open3d as o3d
from pyvista import PolyData
from scipy.interpolate import splprep, splev
import scipy.spatial.transform

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

def set_front_view(plotter):
    """Set camera to front view with head upright"""
    plotter.view_xy()  # Set to front view (looking at XY plane)
    plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]  # Position, focus, up-vector
    plotter.camera.zoom(0.8)  # Slight zoom for better framing
    plotter.enable_trackball_style()
    return plotter

def extrude_mesh_with_normals(surface, thickness):
    """
    Extrude a surface mesh along its normals with a specified thickness.
    
    Parameters:
    -----------
    surface : pyvista.PolyData
        The input surface mesh
    thickness : float
        The thickness of the extrusion
    line_points_3d : numpy.ndarray, optional
        Optional boundary points. If not provided, they will be extracted from the surface.
    
    Returns:
    --------
    pyvista.PolyData
        The extruded volume
    """
    # Ensure the surface is triangulated and has normals
    if surface.n_cells == 0:
        surface = surface.triangulate()
    if 'Normals' not in surface.array_names:
        surface.compute_normals(inplace=True)

    # Uniform extrusion for the main surface
    offset_points = surface.points + surface.point_normals * thickness
    num_points = surface.n_points
    combined_points = np.vstack([surface.points, offset_points])

    # Duplicate faces for offset surface
    faces = surface.faces.reshape(-1, 4)
    offset_faces = faces.copy()
    offset_faces[:, 1:4] += num_points
    edges = surface.extract_feature_edges(
        boundary_edges=True, 
        feature_edges=False, 
        manifold_edges=False
    )
    boundary_points = np.array(edges.points)

    # Generate boundary side faces
    boundary_normals = np.zeros_like(boundary_points)
    for i, point in enumerate(boundary_points):
        dists = np.sum((surface.points - point)**2, axis=1)
        closest_idx = np.argmin(dists)
        boundary_normals[i] = surface.point_normals[closest_idx]
    
    offset_boundary_points = boundary_points + boundary_normals * thickness
    combined_points = np.vstack([combined_points, boundary_points, offset_boundary_points])

    # Create side faces for the boundary
    side_faces = []
    n_boundary = len(boundary_points)
    base_idx = num_points * 2  # boundary points start after original and offset surface points
    
    # Create ordered pairs of boundary points for side faces
    # This assumes the boundary points form a proper loop
    for i in range(n_boundary):
        j = (i + 1) % n_boundary
        i_orig = base_idx + i
        j_orig = base_idx + j
        i_offset = base_idx + n_boundary + i
        j_offset = base_idx + n_boundary + j
        # Create a quad face: i_orig -> j_orig -> j_offset -> i_offset
        side_faces.extend([4, i_orig, j_orig, j_offset, i_offset])

    all_faces = np.hstack([faces.flatten(), offset_faces.flatten(), np.array(side_faces, dtype=np.int32)])
    return pv.PolyData(combined_points, all_faces)

def get_surface_within_area(face_mesh, line_points_3d):
    """
    Generate a surface by extracting the portion of the face mesh inside the polygon defined by line_points_3d.
    
    Args:
        face_mesh: Face mesh as PyVista PolyData, Open3D TriangleMesh, or dict with vertices and faces
        line_points_3d: 3D points defining the boundary outline
        
    Returns:
        PyVista PolyData representing the clipped surface
    """
    # Create a polyline from the points
    polyline = pv.PolyData(line_points_3d)
    n_points = len(line_points_3d)
    lines = np.hstack((n_points, np.arange(n_points)))
    polyline.lines = np.array([lines])
    
    # Extract the 2D polygon outline in the XY plane
    polygon_points = line_points_3d[:, :2]  # Keep only X and Y coordinates
    
    face_mesh_pv = create_pyvista_mesh(face_mesh)

    # Extract points and faces inside the polygon using boolean operations
    # Create a 2D polygon and extrude it to get a selection volume
    polygon = pv.PolyData(np.hstack([polygon_points, np.zeros((len(polygon_points), 1))]))
    polygon.lines = polyline.lines
    
    # Create a surface from the polygon
    surf = polygon.delaunay_2d()
    
    # Create a selection volume by extruding far in z-direction (both ways)
    points = face_mesh_pv.points
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_range = z_max - z_min
    extrusion = surf.extrude((0, 0, 2*z_range), capping=True)
    extrusion.translate((0, 0, z_min - z_range/2), inplace=True)
    
    # Select the part of the face mesh inside the extrusion
    surface = face_mesh_pv.clip_surface(extrusion)
    
    return surface

def remove_surface_within_area(mesh, line_points_3d, height=None):
    """
    Remove from mesh any region enclosed by line_points_3d.
    """
    mesh_pv = create_pyvista_mesh(mesh)
    
    # Create a 2D polygon from line_points_3d
    polyline = pv.PolyData(line_points_3d)
    n_points = len(line_points_3d)
    lines = np.hstack((n_points, np.arange(n_points)))
    polyline.lines = np.array([lines])

    # Calculate the average Z coordinate of the line_points_3d
    z_avg = np.mean(line_points_3d[:, 2])

    # Triangulate the polygon
    polygon_2d = polyline.delaunay_2d()

    # Extrude the 2D polygon to form a cutting volume
    centered_points = line_points_3d - line_points_3d.mean(axis=0)
    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    plane_normal = vh[-1]
    points = mesh_pv.points
    plane_normal /= np.linalg.norm(plane_normal)

    if height is None:
        projections = np.dot(points, plane_normal)
        proj_min, proj_max = np.min(projections), np.max(projections)
        height = abs(proj_max - proj_min) * 2

    extrusion_vector = plane_normal * height
    polygon_2d.translate(-np.array(extrusion_vector) / 2, inplace=True)
    extrusion = polygon_2d.extrude(extrusion_vector, capping=True)
    
    # Clip the mesh (invert=False to remove the inside region)
    clipped_mesh = mesh_pv.clip_surface(extrusion, invert=False)

    # Add the surface to the plotter for visualization
    # Optionally plot for debugging
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh_pv, color='lightgreen', opacity=0.2, show_edges=True)
    # plotter.add_mesh(extrusion, color='lightblue', opacity=0.5, show_edges=True)
    # plotter.add_mesh(polygon_2d, color='red', opacity=0.2, show_edges=True)
    # plotter.add_mesh(pv.PolyData(line_points_3d), color='yellow', point_size=10, render_points_as_spheres=True)
    # plotter.show()

    return clipped_mesh

def create_pyvista_mesh(mesh):
    # Convert face mesh to PyVista if it's not already
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        # Handle Open3D mesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        # Convert triangles to the format PyVista expects
        faces = np.column_stack((np.ones(len(triangles), dtype=np.int64) * 3, triangles)).flatten()
        mesh_pv = pv.PolyData(vertices, faces)
    elif not isinstance(mesh, PolyData):
        # Handle dictionary-like mesh
        try:
            vertices = np.array(mesh['vertices'])
            faces = np.array(mesh['faces'])
            mesh_pv = pv.PolyData(vertices, faces)
        except (TypeError, KeyError):
            raise ValueError("face_mesh must be a PyVista PolyData, Open3D TriangleMesh, or a dictionary with 'vertices' and 'faces' keys")
    else:
        mesh_pv = mesh

    return mesh_pv

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

def rotate_shape_and_landmarks(shape, rotation_matrix, pivot=None):
    """
    Apply a rotation to a shape and its landmarks.
    
    Parameters:
    -----------
    shape : pyvista.PolyData or similar
        The shape to rotate
    rotation_matrix : numpy.ndarray
        3x3 rotation matrix
    pivot : numpy.ndarray, optional
        Point around which to rotate, defaults to shape.center
    
    Returns:
    --------
    shape : pyvista.PolyData or similar
        The rotated shape
    """
    if pivot is None:
        pivot = shape.center
    
    # Apply rotation to shape points
    shape.points[:] = (shape.points - pivot) @ rotation_matrix.T + pivot
    
    # Apply rotation to landmarks if they exist
    if hasattr(shape, 'landmarks'):
        for key, point in shape.landmarks.items():
            shape.landmarks[key] = (np.array(point) - pivot) @ rotation_matrix.T + pivot
    
    return shape

def translate_shape_and_landmarks(shape, translation_vector):
    """
    Apply a translation to a shape and its landmarks.
    
    Parameters:
    -----------
    shape : pyvista.PolyData or similar
        The shape to translate
    translation_vector : numpy.ndarray
        3D translation vector
    
    Returns:
    --------
    shape : pyvista.PolyData or similar
        The translated shape
    """
    # Apply translation to shape points
    shape.points[:] = shape.points + translation_vector
    
    # Apply translation to landmarks if they exist
    if hasattr(shape, 'landmarks'):
        for key, point in shape.landmarks.items():
            shape.landmarks[key] = np.array(point) + translation_vector
    
    return shape

def get_landmark(landmarks, landmark_ids=None, target_id=None):
    """
    Get a landmark by its ID from either an array of landmarks or a dictionary of landmarks.
    
    Parameters:
    -----------
    landmarks : numpy.ndarray or object with landmarks attribute
        Either an array of landmark coordinates or an object with landmarks dictionary attribute
    landmark_ids : numpy.ndarray, optional
        Array of landmark IDs corresponding to the landmarks array
    target_id : int
        The ID of the landmark to retrieve
    
    Returns:
    --------
    numpy.ndarray
        The coordinates of the requested landmark
    """
    # Handle dictionary-style landmarks (e.g., shape_3d.landmarks)
    if hasattr(landmarks, 'landmarks') and isinstance(landmarks.landmarks, dict):
        if target_id in landmarks.landmarks:
            return np.array(landmarks.landmarks[target_id])
        else:
            raise ValueError(f"Landmark {target_id} not found in landmarks dictionary.")
    
    # Handle array-style landmarks with separate IDs (e.g., face_landmarks and face_landmarks_ids)
    elif landmark_ids is not None and target_id is not None:
        idx = np.where(landmark_ids == target_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Landmark {target_id} not found in landmark_ids.")
        return landmarks[idx[0]]
    
    # If the landmarks is already a dictionary
    elif isinstance(landmarks, dict) and target_id in landmarks:
        return np.array(landmarks[target_id])
    
    else:
        raise ValueError("Invalid landmarks format. Must be either an array with landmark_ids or an object with landmarks dictionary.")


