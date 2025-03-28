"""
Utility functions for 3D face processing
"""
import numpy as np
import pyvista as pv
import open3d as o3d
from pyvista import PolyData
from scipy.interpolate import splprep, splev
import scipy.spatial.transform
# from scipy.spatial import Delaunay
import meshlib.mrmeshpy as mr
import triangle
import tempfile
import os
import trimesh
from functools import singledispatch
from scipy.spatial import Delaunay

def smooth_line_points(points, smoothing=0.1, num_samples=300):
    """Smooth 3D points using spline interpolation"""
    x, y, z = points[:,0], points[:,1], points[:,2]
    tck, u = splprep([x, y, z], s=smoothing, k=2, per=True)
    u_new = np.linspace(0, 1, num_samples)
    x_new, y_new, z_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new, z_new))

def set_front_view(plotter):
    """Set camera to front view with head upright"""
    plotter.view_xy()  # Set to front view (looking at XY plane)
    plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]  # Position, focus, up-vector
    plotter.camera.zoom(0.8)  # Slight zoom for better framing
    plotter.enable_trackball_style()
    return plotter

def convert_o3d_to_pv(o3d_mesh):
    """Convert Open3D mesh to PyVista mesh"""
    mesh_vertices = np.asarray(o3d_mesh.vertices)
    mesh_faces = np.asarray(o3d_mesh.triangles)
    faces_pyvista = np.hstack((np.full((len(mesh_faces), 1), 3), mesh_faces)).astype(np.int64)
    pv_mesh = pv.PolyData(mesh_vertices, faces_pyvista)
    
    # Transfer vertex colors if available
    if o3d_mesh.has_vertex_colors():
        colors = np.asarray(o3d_mesh.vertex_colors)
        # Convert RGB colors from [0,1] to [0,255] for PyVista
        colors = (colors * 255).astype(np.uint8)
        pv_mesh.point_data["RGB"] = colors
        
    return pv_mesh

def convert_pv_to_mr(pv_mesh: pv.PolyData) -> mr.Mesh:

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input_mesh.stl")

        # Save the surface as an STL file
        pv_mesh.save(input_path)
        mr_mesh = mr.loadMesh(input_path)
    
        pv_center = pv_mesh.center
        mr_center_3f = centroid_mr(mr_mesh)
        mr_center = np.array([mr_center_3f.x, mr_center_3f.y, mr_center_3f.z])
        translation_vector = mr_center - pv_center 
        mr_mesh = translate_mesh_mr(mr_mesh, translation_vector)

    return mr_mesh

def convert_mr_to_pv(mr_mesh: mr.Mesh) -> pv.PolyData:

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = os.path.join(temp_dir, "input_mesh.stl")
        mr.saveMesh(mr_mesh, input_path)
        pv_mesh = pv.read(input_path)
        
        pv_center = pv_mesh.center
        mr_center_3f = centroid_mr(mr_mesh)
        mr_center = np.array([mr_center_3f.x, mr_center_3f.y, mr_center_3f.z])
        translation_vector = pv_center - mr_center  
        pv_mesh = pv_mesh.translate(translation_vector, inplace=True)

    return pv_mesh


def extrude_mesh(surface, thickness = None, vector = None):
    
    if isinstance(surface, mr.Mesh):
        type = "mr"
        surface = convert_mr_to_pv(surface)
    else:
        type = "pv"
    
    if vector is None:
        vector = compute_normals(surface)

    if thickness is None:
        thickness = 5.0

    vector = vector * thickness

    extruded = surface.extrude(vector, capping=True).triangulate()
    extruded_closed = extruded.fill_holes(1e5)

    # Convert back to MRMeshPy if needed
    if type == "mr":
        extruded_closed = convert_pv_to_mr(extruded_closed)

    return extruded_closed

@singledispatch
def compute_normals(mesh) -> np.ndarray:
    raise TypeError("Unsupported mesh type.")

@compute_normals.register
def _(mesh: pv.PolyData) -> np.ndarray:
    # Compute the average surface normal
    simplified_mesh = mesh.decimate(target_reduction=0.9)  # Retain 10% of the faces

    num_faces = simplified_mesh.n_faces
    normals = []
    for i in range(num_faces):
        face = simplified_mesh.extract_feature_edges(i)
        face_points = face.points
        v0 = face_points[0]
        v1 = face_points[1]
        v2 = face_points[2]
        
        # Compute face normal
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        normals.append(normal)
    
    # Calculate average normal
    avg_normal = np.mean(normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)  # Normalize the vector

    return avg_normal

@compute_normals.register
def _(mesh: mr.Mesh) -> np.ndarray:
    # Compute the average surface normal
    num_faces = mesh.topology.numValidFaces()
    normals = []
    for i in range(num_faces):
        face_id = mr.FaceId(i)
        if not mesh.topology.hasFace(face_id):
            continue
        
        # Get vertex indices as integers
        vert_ids = mesh.topology.getTriVerts(face_id)
        # Convert VertId objects to integers for indexing
        v0_idx = vert_ids[0].get()
        v1_idx = vert_ids[1].get()
        v2_idx = vert_ids[2].get()
        
        # Get vertex coordinates
        v0 = np.array([mesh.points.vec[v0_idx].x, mesh.points.vec[v0_idx].y, mesh.points.vec[v0_idx].z])
        v1 = np.array([mesh.points.vec[v1_idx].x, mesh.points.vec[v1_idx].y, mesh.points.vec[v1_idx].z])
        v2 = np.array([mesh.points.vec[v2_idx].x, mesh.points.vec[v2_idx].y, mesh.points.vec[v2_idx].z])
        
        # Compute face normal
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        normals.append(normal)
    
    # Calculate average normal
    avg_normal = np.mean(normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)  # Normalize the vector

    return avg_normal

def translate_mesh_mr(mesh: mr.Mesh, translation_vector) -> mr.Mesh:
    """
    Translate a mesh by adding the translation vector to each vertex.
    
    Parameters:
        mesh (mr.Mesh): The mesh to translate.
        translation_vector (tuple or list of float): (tx, ty, tz)
    
    Returns:
        mr.Mesh: The translated mesh.
    """
    tx, ty, tz = translation_vector
    # Iterate over all vertices in the mesh
    for i in range(mesh.points.size()):
        # Get the current point (assuming mr.Vector3f supports .x, .y, .z)
        p = mesh.points.vec[i]
        # Create the new point
        new_point = mr.Vector3f(p.x + tx, p.y + ty, p.z + tz)
        # Update the vertex position
        mesh.points.vec[i] = new_point

    # Invalidate any cached topology/geometry if needed
    mesh.invalidateCaches()
    return mesh

def thicken_mesh(surface, thickness, vector = None):
    # Create temporary directory for mesh files
    if isinstance(surface, pv.PolyData):
        type = "pv"
        surface = convert_pv_to_mr(surface)
    else:
        type = "mr"      

    if vector is None:
        vector = compute_normals(surface)

    # Setup parameters for thickening in both directions
    params = mr.GeneralOffsetParameters()
    params.voxelSize = mr.suggestVoxelSize(surface, 1e6) 
    params.signDetectionMode = mr.SignDetectionMode.HoleWindingRule
    params.mode = mr.GeneralOffsetParametersMode.Smooth
    
    # 1. Thicken in both directions
    thickened_mesh_mr = mr.thickenMesh(surface, thickness, params)
    
    # 2. Thicken only in the interior direction
    extruded_mesh_mr = extrude_mesh(surface, thickness * 10, vector)

    # Perform boolean intersection with MRMeshPy
    result = mr.voxelBooleanIntersect(thickened_mesh_mr, extruded_mesh_mr, params.voxelSize)

    # plotter = pv.Plotter()
    # plotter.add_mesh(convert_mr_to_pv(result), color='green', opacity=0.7, show_edges=True)
    # plotter.show()
    
    # Convert back to Pyvista if needed
    if type == "pv":
        result = convert_mr_to_pv(result)


    return result
        
    
def centroid_mr(mesh: mr.Mesh) -> mr.Vector3f:
    # Get bitset of all valid vertices in the mesh
    all_verts = mesh.topology.getValidVerts()
    total = mr.Vector3f(0.0, 0.0, 0.0)    # accumulator for sum of coordinates
    count = 0 

    # Iterate through all vertex indices, summing their coordinates
    for i in range(all_verts.size()):               # iterate over bitset range
        if all_verts.test(mr.VertId(i)):            # check if vertex i is valid
            total += mesh.points.vec[i]             # add vertex i's coordinate&#8203;:contentReference[oaicite:5]{index=5}
            count += 1

    # Compute average (centroid) if there was at least one vertex
    if count > 0:
        centroid = mr.Vector3f(total.x / count, total.y / count, total.z / count)
        return centroid
    else:
        raise ValueError("Mesh has no vertices.")

def get_surface_from_points_2d(line_points_2d) -> pv.PolyData:
    # Perform Delaunay triangulation
    delaunay = Delaunay(line_points_2d)

    # Extract the simplices (triangles) from the triangulation
    simplices = delaunay.simplices

    # Add a Z-coordinate (0) to the 2D points to make them 3D
    line_points_3d = np.column_stack((line_points_2d, np.zeros(line_points_2d.shape[0])))
    
    # Create a PyVista PolyData object from the triangulation
    surface = pv.PolyData(line_points_3d, np.hstack((np.full((len(simplices), 1), 3), simplices)).astype(int))

    return surface


def get_surface_within_area(mesh: pv.PolyData, line_points_3d) -> pv.PolyData:
    #TODO: extrude in normal direction instead of Z
    # vector = compute_normals(mesh)
    try:
        # Extract the 2D polygon outline in the XY plane
        polygon_points = line_points_3d[:, :2]  # Keep only X and Y coordinates
        surface = get_surface_from_points_2d(polygon_points)

        # Create a selection volume by extruding far in z-direction (both ways)
        points = mesh.points
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        z_range = z_max - z_min
        extrusion = surface.extrude((0, 0, 2*z_range), capping=True)
        extrusion.translate((0, 0, z_min - z_range/2), inplace=True)
        
        # Select the part of the face mesh inside the extrusion
        surface = mesh.clip_surface(extrusion)

        return surface, ""
    
    except Exception as e:
        return None, f"Error in get_surface_within_area: {e}"


def remove_surface_within_area(mesh, line_points_3d):
    """
    Remove from mesh any region enclosed by line_points_3d.
    """
    # vector = compute_normals(mesh)
    vector = np.array([0, 0, 1])  # Default extrusion direction

    # Project line_points_3d onto the XY plane
    polygon_points = line_points_3d[:, :2]  # Keep only X and Y coordinates
        
    surface = get_surface_from_points_2d(polygon_points)
    extrusion = extrude_mesh(surface, 50, vector=vector)
    mesh_center = mesh.center
    extrusion_center = extrusion.center
    translation_vector = np.array(mesh_center) - np.array(extrusion_center)
    translation_vector = np.array([0, 0, translation_vector[2]])  

    # Move the extrusion to the center of the surface
    extrusion = extrusion.translate(translation_vector, inplace=True)

    clipped_mesh = mesh.clip_surface(extrusion, invert=False)

    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, color='gray', opacity=0.2, show_edges=True)
    # plotter.add_mesh(extrusion, color='lightblue', opacity=0.5, show_edges=True)
    # plotter.add_mesh(clipped_mesh, color='red', opacity=0.7, show_edges=True)
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

def clean_and_smooth(mesh, smooth_iter=50, clean_tol=1e-3):
    # Remove small isolated components
    # mesh = mesh.connectivity(largest=True)
    
    # Clean and remove degenerate features
    mesh = mesh.clean(tolerance=clean_tol)
    
    # Smooth mesh (taubin smoothing retains volume better)
    mesh = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.1)

    return mesh

