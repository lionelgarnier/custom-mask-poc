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
from scipy.spatial import (Delaunay, ConvexHull)

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter
import vtk


def smooth_line_points(points, smoothing=0.1, num_samples=300, closed=True):
    """Smooth 3D points using spline interpolation"""
    x, y, z = points[:,0], points[:,1], points[:,2]
    tck, u = splprep([x, y, z], s=smoothing, k=2, per=closed)
    u_new = np.linspace(0, 1, num_samples)
    x_new, y_new, z_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new, z_new))

def set_front_view(plotter):
    """Set camera to front view with head upright"""
    plotter.view_xy()  # Set to front view (looking at XY plane)
    plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]  # Position, focus, up-vector
    # plotter.camera.zoom(0.8)  # Slight zoom for better framing
    # plotter.enable_trackball_style()
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
    
        pv_center = centroid_pv(pv_mesh)
        mr_center_3f = centroid_mr(mr_mesh)
        mr_center = np.array([mr_center_3f.x, mr_center_3f.y, mr_center_3f.z])
        translation_vector = pv_center - mr_center
        mr_mesh = translate_mesh_mr(mr_mesh, translation_vector)

    return mr_mesh

def convert_mr_to_pv(mr_mesh: mr.Mesh) -> pv.PolyData:

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = os.path.join(temp_dir, "input_mesh.stl")
        mr.saveMesh(mr_mesh, input_path)
        pv_mesh = pv.read(input_path)
        
        pv_center = centroid_pv(pv_mesh)
        mr_center_3f = centroid_mr(mr_mesh)
        mr_center = np.array([mr_center_3f.x, mr_center_3f.y, mr_center_3f.z])
        translation_vector = mr_center - pv_center
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
    # Compute the average surface normal by sampling face normals
    simplified_mesh = mesh.decimate(target_reduction=0.9)  # Retain 10% of the faces
    
    # Ensure mesh has normals computed
    if "Normals" not in simplified_mesh.point_data:
        simplified_mesh = simplified_mesh.compute_normals()
    
    # Get face normals by accessing cells directly
    normals = []
    faces = simplified_mesh.faces
    points = simplified_mesh.points
    
    # Parse faces array (PyVista format: [n_pts, pt1, pt2, pt3, n_pts, pt4, pt5, pt6, ...])
    i = 0
    while i < len(faces):
        n_pts = faces[i]
        if n_pts >= 3:  # Only process triangular faces or larger
            # Get first 3 points to compute normal
            pt_indices = faces[i+1:i+1+3]
            v0 = points[pt_indices[0]]
            v1 = points[pt_indices[1]]
            v2 = points[pt_indices[2]]
            
            # Compute face normal
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:  # Avoid zero-length normals
                normal /= norm
                normals.append(normal)
        
        i += n_pts + 1
    
    if not normals:
        # Fallback: use average of point normals
        return np.mean(simplified_mesh.point_data["Normals"], axis=0)
    
    # Calculate average normal
    avg_normal = np.mean(normals, axis=0)
    norm = np.linalg.norm(avg_normal)
    if norm > 1e-8:
        avg_normal /= norm
    else:
        # Fallback to Z-up if calculation fails
        avg_normal = np.array([0.0, 0.0, 1.0])

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

def thicken_combined_surface_with_zones(tube_surface: pv.PolyData, tangent_surface: pv.PolyData, 
                                        tube_thickness_start: float, tube_thickness_end: float, 
                                        tube_top_points=None, circle_points=None) -> pv.PolyData:
    """
    Thickens a combined surface with different thickness zones:
    - tangent_surface area: fixed thickness
    - tube_surface area: variable thickness from start to end
    Uses unified normals to avoid interface discontinuities.
    
    Parameters
    ----------
    tube_surface : pv.PolyData
        The tube surface mesh.
    tangent_surface : pv.PolyData
        The tangent surface mesh.
    tube_thickness_start : float
        Thickness at tube outer edge (contact with tangent) = 2mm.
    tube_thickness_end : float
        Thickness at tube inner edge = 0.5mm.
    tube_top_points : np.ndarray, optional
        Points defining the interior edge of the tube.
    circle_points : np.ndarray, optional
        Points defining the exterior edge of the tube.
    
    Returns
    -------
    pv.PolyData
        A volume with zone-based variable thickness.
    """
    tangent_thickness = tube_thickness_start
    
    # Use fix_surface_normals_at_interface to get unified normals
    combined_surface = fix_surface_normals_at_interface(tube_surface, tangent_surface)
    
    points = combined_surface.points
    normals = combined_surface.point_data["Normals"]
    n_pts = points.shape[0]
    
    # Initialize thickness values with tangent thickness as default
    thickness_values = np.full(n_pts, tangent_thickness)
    
    # Identify which points belong to tube vs tangent surface
    tube_points = tube_surface.points
    tangent_points = tangent_surface.points
    
    # For each combined surface point, determine if it's closer to tube or tangent surface
    for i, point in enumerate(points):
        # Distance to nearest tube surface point
        distances_to_tube = np.linalg.norm(tube_points - point, axis=1)
        min_dist_tube = np.min(distances_to_tube)
        
        # Distance to nearest tangent surface point  
        distances_to_tangent = np.linalg.norm(tangent_points - point, axis=1)
        min_dist_tangent = np.min(distances_to_tangent)
        
        # If closer to tube surface, apply variable thickness
        if min_dist_tube < min_dist_tangent and tube_top_points is not None and circle_points is not None:
            # Distance to interior curve (tube_top_points)
            distances_to_interior = np.linalg.norm(tube_top_points - point, axis=1)
            min_dist_interior = np.min(distances_to_interior)
            
            # Distance to exterior curve (circle_points)
            distances_to_exterior = np.linalg.norm(circle_points - point, axis=1)
            min_dist_exterior = np.min(distances_to_exterior)
            
            # Normalize: closer to interior = 0, closer to exterior = 1
            total_dist = min_dist_interior + min_dist_exterior
            if total_dist > 1e-6:
                normalized_position = min_dist_interior / total_dist
            else:
                normalized_position = 0.5
            
            # Interpolate thickness for tube area: interior = tube_thickness_end, exterior = tube_thickness_start
            thickness_values[i] = tube_thickness_end + (tube_thickness_start - tube_thickness_end) * normalized_position
        
        # If closer to tangent surface, keep fixed tangent_thickness (already set as default)
        
    # Create extruded points with variable thickness using UNIFIED normals
    extruded_points = vtk.vtkPoints()
    extruded_points.SetNumberOfPoints(n_pts * 2)
    
    # Original points
    for i in range(n_pts):
        x, y, z = points[i]
        extruded_points.SetPoint(i, x, y, z)
    
    # Extruded points with variable thickness using unified normals
    for i in range(n_pts):
        x, y, z = points[i]
        nx, ny, nz = normals[i]  # These are now the UNIFIED normals
        thickness = thickness_values[i]
        extruded_points.SetPoint(n_pts + i, 
                                x + thickness * nx, 
                                y + thickness * ny, 
                                z + thickness * nz)
    
    # Parse faces and create volumetric cells
    faces_flat = combined_surface.faces
    cells_list = []
    i = 0
    while i < len(faces_flat):
        n = faces_flat[i]
        cell = faces_flat[i+1 : i+1+n]
        cells_list.append(cell)
        i += n + 1
    
    # Create unstructured grid
    extruded_ug = vtk.vtkUnstructuredGrid()
    extruded_ug.SetPoints(extruded_points)
    extruded_ug.Allocate()
    
    # Create volumetric cells
    for cell in cells_list:
        num_vertices = len(cell)
        if num_vertices == 3:
            # Triangle: create wedge
            id_list = vtk.vtkIdList()
            id_list.SetNumberOfIds(6)
            id_list.SetId(0, cell[0])
            id_list.SetId(1, cell[1])
            id_list.SetId(2, cell[2])
            id_list.SetId(3, cell[0] + n_pts)
            id_list.SetId(4, cell[1] + n_pts)
            id_list.SetId(5, cell[2] + n_pts)
            extruded_ug.InsertNextCell(vtk.VTK_WEDGE, id_list)
        elif num_vertices == 4:
            # Quad: create hexahedron
            id_list = vtk.vtkIdList()
            id_list.SetNumberOfIds(8)
            id_list.SetId(0, cell[0])
            id_list.SetId(1, cell[1])
            id_list.SetId(2, cell[2])
            id_list.SetId(3, cell[3])
            id_list.SetId(4, cell[0] + n_pts)
            id_list.SetId(5, cell[1] + n_pts)
            id_list.SetId(6, cell[2] + n_pts)
            id_list.SetId(7, cell[3] + n_pts)
            extruded_ug.InsertNextCell(vtk.VTK_HEXAHEDRON, id_list)
    
    return pv.wrap(extruded_ug)

def thicken_mesh_with_variable_thickness(surface: pv.PolyData, thickness_start: float, thickness_end: float, tube_top_points=None, circle_points=None) -> pv.PolyData:
    """
    Thickens a surface with variable thickness based on proximity to reference curves.
    
    Parameters
    ----------
    surface : pv.PolyData
        The input surface mesh.
    thickness_start : float
        Thickness at the outer edge (contact with tangent surface) = 1.5mm.
    thickness_end : float
        Thickness at the inner edge (center side) = 0.5mm.
    tube_top_points : np.ndarray, optional
        Points defining the interior edge of the tube.
    circle_points : np.ndarray, optional
        Points defining the exterior edge of the tube.
    
    Returns
    -------
    pv.PolyData
        A volume with radial variable thickness.
    """
    import vtk
    
    # Ensure normals exist
    if "Normals" not in surface.point_data:
        surface = surface.compute_normals(auto_orient_normals=True)
    
    points = surface.points
    normals = surface.point_data["Normals"]
    n_pts = points.shape[0]
    
    # Initialize thickness values
    thickness_values = np.full(n_pts, (thickness_start + thickness_end) / 2.0)  # Default to average
    
    if tube_top_points is not None and circle_points is not None:
        # Method 1: Distance to reference curves
        for i, point in enumerate(points):
            # Distance to interior curve (tube_top_points)
            distances_to_interior = np.linalg.norm(tube_top_points - point, axis=1)
            min_dist_interior = np.min(distances_to_interior)
            
            # Distance to exterior curve (circle_points)
            distances_to_exterior = np.linalg.norm(circle_points - point, axis=1)
            min_dist_exterior = np.min(distances_to_exterior)
            
            # Normalize: closer to interior = 0, closer to exterior = 1
            total_dist = min_dist_interior + min_dist_exterior
            if total_dist > 1e-6:
                normalized_position = min_dist_interior / total_dist
            else:
                normalized_position = 0.5
            
            # Interpolate thickness: interior = thickness_end, exterior = thickness_start
            thickness_values[i] = thickness_end + (thickness_start - thickness_end) * normalized_position
    else:
        # Method 2: Fallback to centerline-based approach
        z_coords = points[:, 2]
        z_unique = np.unique(z_coords)
        centerline_points = []
        
        for z in z_unique:
            mask = np.abs(z_coords - z) < 1e-6
            if np.any(mask):
                z_points = points[mask]
                centroid = np.mean(z_points, axis=0)
                centerline_points.append(centroid)
        
        if len(centerline_points) > 0:
            centerline_points = np.array(centerline_points)
            
            for i, point in enumerate(points):
                distances_to_centerline = np.linalg.norm(centerline_points - point, axis=1)
                nearest_centerline_idx = np.argmin(distances_to_centerline)
                nearest_centerline_point = centerline_points[nearest_centerline_idx]
                
                radial_distance = np.linalg.norm(point[:2] - nearest_centerline_point[:2])
                
                z_level = point[2]
                same_z_mask = np.abs(z_coords - z_level) < 1e-3
                if np.any(same_z_mask):
                    same_z_points = points[same_z_mask]
                    max_radial_dist = 0
                    for p in same_z_points:
                        dist = np.linalg.norm(p[:2] - nearest_centerline_point[:2])
                        max_radial_dist = max(max_radial_dist, dist)
                    
                    if max_radial_dist > 1e-6:
                        normalized_radial = radial_distance / max_radial_dist
                    else:
                        normalized_radial = 0.0
                else:
                    normalized_radial = 0.0
                
                thickness_values[i] = thickness_end + (thickness_start - thickness_end) * normalized_radial
    
    # Create extruded points with variable thickness
    extruded_points = vtk.vtkPoints()
    extruded_points.SetNumberOfPoints(n_pts * 2)
    
    # Original points
    for i in range(n_pts):
        x, y, z = points[i]
        extruded_points.SetPoint(i, x, y, z)
    
    # Extruded points with variable thickness
    for i in range(n_pts):
        x, y, z = points[i]
        nx, ny, nz = normals[i]
        thickness = thickness_values[i]
        extruded_points.SetPoint(n_pts + i, 
                                x + thickness * nx, 
                                y + thickness * ny, 
                                z + thickness * nz)
    
    # Parse faces and create volumetric cells (reuse logic from thicken_mesh_vtk)
    faces_flat = surface.faces
    cells_list = []
    i = 0
    while i < len(faces_flat):
        n = faces_flat[i]
        cell = faces_flat[i+1 : i+1+n]
        cells_list.append(cell)
        i += n + 1
    
    # Create unstructured grid
    extruded_ug = vtk.vtkUnstructuredGrid()
    extruded_ug.SetPoints(extruded_points)
    extruded_ug.Allocate()
    
    # Create volumetric cells
    for cell in cells_list:
        num_vertices = len(cell)
        if num_vertices == 3:
            # Triangle: create wedge
            id_list = vtk.vtkIdList()
            id_list.SetNumberOfIds(6)
            id_list.SetId(0, cell[0])
            id_list.SetId(1, cell[1])
            id_list.SetId(2, cell[2])
            id_list.SetId(3, cell[0] + n_pts)
            id_list.SetId(4, cell[1] + n_pts)
            id_list.SetId(5, cell[2] + n_pts)
            extruded_ug.InsertNextCell(vtk.VTK_WEDGE, id_list)
        elif num_vertices == 4:
            # Quad: create hexahedron
            id_list = vtk.vtkIdList()
            id_list.SetNumberOfIds(8)
            id_list.SetId(0, cell[0])
            id_list.SetId(1, cell[1])
            id_list.SetId(2, cell[2])
            id_list.SetId(3, cell[3])
            id_list.SetId(4, cell[0] + n_pts)
            id_list.SetId(5, cell[1] + n_pts)
            id_list.SetId(6, cell[2] + n_pts)
            id_list.SetId(7, cell[3] + n_pts)
            extruded_ug.InsertNextCell(vtk.VTK_HEXAHEDRON, id_list)
    
    return pv.wrap(extruded_ug)

def thicken_mesh_vtk(surface: pv.PolyData, thickness: float, reverse: bool = False) -> pv.PolyData:
    """
    Thickens a surface using VTK's native linear extrusion filter with automatic capping.
    This approach handles hole filling and surface closure automatically.
    
    Parameters
    ----------
    surface : pv.PolyData
        The input surface mesh.
    thickness : float
        The extrusion distance.
    reverse : bool, optional
        If True, extrudes in the opposite direction of the normals.
    
    Returns
    -------
    pv.PolyData
        A closed volumetric mesh.
    """
    import vtk
    
    # Ensure normals exist and are properly oriented
    if "Normals" not in surface.point_data:
        surface = surface.compute_normals(auto_orient_normals=True)
    
    # Create VTK linear extrusion filter
    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputData(surface)
    extrude.SetScaleFactor(thickness if not reverse else -thickness)
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetCapping(True)  # Automatically close the ends
    extrude.Update()
    
    # Return as PyVista mesh
    return pv.wrap(extrude.GetOutput())

def thicken_mesh(surface, thickness, vector = None):
    surface_copy = surface.copy()
    # Create temporary directory for mesh files
    if isinstance(surface, pv.PolyData):
        type = "pv"
        # print("pv center before : ", surface.center)   
        # print("pv centroid before : ", centroid_pv(surface))   
        surface = convert_pv_to_mr(surface)
        # print("mr center before : ", centroid_mr(surface))   
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

    # Convert back to Pyvista if needed
    if type == "pv":
        result = convert_mr_to_pv(result)  

        # plotter = pv.Plotter()
        # # plotter.add_mesh(convert_mr_to_pv(thickened_mesh_mr), color='green', opacity=0.4, show_edges=True)
        # # plotter.add_mesh(convert_mr_to_pv(surface), color='blue', opacity=0.6, show_edges=True)
        # plotter.add_mesh(surface_copy, color='blue', opacity=0.6, show_edges=True)
        # plotter.add_mesh(result, color='red', opacity=0.4, show_edges=True)
        # plotter.show()
    return result


def thicken_mesh_robust(surface, thickness, vector=None, method="topological_offsets", fallback=True):
    """
    Robust mesh thickening with multiple algorithms.
    
    Parameters:
    -----------
    surface : pv.PolyData or mr.Mesh
        The surface to thicken
    thickness : float
        Thickness in mm
    vector : np.ndarray, optional
        Direction vector for directional thickening
    method : str
        "topological_offsets", "mrmeshpy", "vtk_extrude"
    fallback : bool
        If True, fallback to MRMeshPy if primary method fails
    
    Returns:
    --------
    Thickened mesh in same format as input
    """
    import subprocess
    import tempfile
    import os
    
    surface_copy = surface.copy() if hasattr(surface, 'copy') else surface
    
    # Determine input type
    if isinstance(surface, pv.PolyData):
        input_type = "pv"
    else:
        input_type = "mr"
        surface = convert_mr_to_pv(surface)
    
    if vector is None:
        vector = compute_normals(surface)

    # Method 1: Topological Offsets (most robust)
    if method == "topological_offsets":
        try:
            result = _thicken_with_topological_offsets(surface, thickness, vector)
            if input_type == "mr":
                result = convert_pv_to_mr(result)
            return result
        except Exception as e:
            print(f"Topological Offsets failed: {e}")
            if not fallback:
                raise
            print("Falling back to MRMeshPy...")
            method = "mrmeshpy"
    
    # Method 2: Original MRMeshPy approach
    if method == "mrmeshpy":
        try:
            return thicken_mesh(surface_copy, thickness, vector)
        except Exception as e:
            print(f"MRMeshPy failed: {e}")
            if not fallback:
                raise
            print("Falling back to VTK extrude...")
            method = "vtk_extrude"
    
    # Method 3: Simple VTK extrusion (least robust)
    if method == "vtk_extrude":
        return thicken_mesh_vtk(surface, thickness, reverse=False)


def _thicken_with_topological_offsets(surface, thickness, vector):
    """
    Internal function to perform thickening using robust offset methods.
    """
    import subprocess
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.obj")  # .obj format widely supported
        output_path = os.path.join(temp_dir, "output.obj")
        
        # Save surface as OBJ (widely supported)
        surface.save(input_path)
        
        # Check if robust offset executable exists
        offset_exe = _find_topological_offsets_executable()
        
        if offset_exe is None:
            raise RuntimeError("❌ Aucun outil d'offset robuste trouvé. Utilisez install_topological_offsets_guide() pour l'installation.")
        
        # Detect which tool we found and build appropriate command
        exe_name = os.path.basename(offset_exe).lower()
        
        if "voxeloffset" in exe_name:
            # Feature-Preserving Offsets (daniel-zint)
            cmd = [
                offset_exe,
                "-p", input_path,
                "-o", output_path,
                "-j", str(thickness),
                "-d",  # Enable dual contouring
                "--d1", "10",  # octree subdivision
                "-n",  # normalize input
                "-q"   # quiet mode
            ]
        elif "offset3d" in exe_name:
            # VorOffset (geometryprocessing)
            cmd = [
                offset_exe,
                input_path,
                output_path,
                "-r", str(int(thickness * 2)),  # radius in voxels 
                "-x", "dilation",
                "-n", "128"  # number of dexels
            ]
        else:
            # Legacy fallback (will likely fail)
            cmd = [
                offset_exe,
                "--input", input_path,
                "--output", output_path,
                "--offset", str(thickness)
            ]
        
        # Run offset tool
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"Offset tool failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Offset tool timed out (>5min)")
        except FileNotFoundError:
            raise RuntimeError(f"Offset executable not found: {offset_exe}")
        
        # Load result
        if os.path.exists(output_path):
            thickened = pv.read(output_path)
            return thickened.triangulate()
        else:
            raise RuntimeError("Offset tool didn't produce output file")


def _find_topological_offsets_executable():
    """
    Find robust offset executable in common locations.
    """
    import shutil
    
    # Common names for different offset tools
    possible_names = [
        # Feature-Preserving Offsets (daniel-zint)
        "Voxeloffset_cli",
        "voxeloffset_cli",
        # VorOffset (geometryprocessing)
        "offset3d",
        "offset3d.exe",
        # Legacy search (will fail but keeping for compatibility)
        "topological_offsets",
        "topological_offsets.exe"
    ]
    
    # Check in PATH
    for name in possible_names:
        exe_path = shutil.which(name)
        if exe_path:
            return exe_path
    
    # Check common install locations
    common_paths = [
        # Feature-Preserving Offsets paths
        "./offsets-and-remeshing/build/src/voxeloffset_cli/Voxeloffset_cli",
        "../offsets-and-remeshing/build/src/voxeloffset_cli/Voxeloffset_cli",
        os.path.expanduser("~/offsets-and-remeshing/build/src/voxeloffset_cli/Voxeloffset_cli"),
        # VorOffset paths
        "./voroffset/build/offset3d",
        "../voroffset/build/offset3d", 
        os.path.expanduser("~/voroffset/build/offset3d"),
        # Legacy paths (will not exist)
        "./topological-offsets/build/topological_offsets",
        "/usr/local/bin/topological_offsets"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def install_topological_offsets_guide():
    """
    Print installation guide for robust thickening alternatives.
    """
    guide = """
    🚨 CORRECTION: wildmeshing/topological-offsets N'EXISTE PAS
    
    🔧 ALTERNATIVES ROBUSTES DISPONIBLES:
    
    Option 1 - Feature-Preserving Offsets (Recommandée):
    =====================================================
    
    1. Clone le repository:
       git clone https://github.com/daniel-zint/offsets-and-remeshing
       cd offsets-and-remeshing
    
    2. Build avec CMake:
       mkdir build && cd build
       cmake .. -DCMAKE_BUILD_TYPE=Release
       make
    
    3. Usage:
       ./build/src/voxeloffset_cli/Voxeloffset_cli -p input.off -o output.off -j 2.0 -d
    
    Option 2 - VorOffset (Plus simple):
    ==================================
    
    1. Clone le repository:
       git clone https://github.com/geometryprocessing/voroffset
       cd voroffset
    
    2. Build avec CMake:
       mkdir build && cd build
       cmake .. && cmake --build . -j8 --config Release
    
    3. Usage:
       ./offset3d input.obj -o output.obj -r 5 -x dilation
    
    ⚠️  MISE À JOUR DE VOTRE CODE:
    ==============================
    
    La fonction thicken_mesh_robust() fonctionne toujours avec fallback
    vers MRMeshPy et VTK même sans ces outils externes.
    
    📁 Placez l'exécutable dans votre PATH ou ajustez les chemins dans _find_topological_offsets_executable()
    """
    print(guide)
        
    
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

def centroid_pv(mesh: pv.PolyData) -> np.ndarray:
    # Get the center of the mesh
    # Calculate centroid by averaging all vertex coordinates
    points = mesh.points
    total = np.zeros(3)
    count = 0
    
    # Sum all valid vertex coordinates
    for i in range(len(points)):
        total += points[i]
        count += 1
    
    # Compute average position
    if count > 0:
        center = total / count
    else:
        raise ValueError("Mesh has no vertices.")
    return np.array([center[0], center[1], center[2]])


def delaunay2d_surface(points_array: np.ndarray) -> pv.PolyData:
    # If input points are 2D, add a z=0 coordinate.
    if points_array.shape[1] == 2:
        points_array = np.hstack([points_array, np.zeros((points_array.shape[0], 1))])
    
    # Create vtkPoints and insert the input points.
    vtk_pts = vtk.vtkPoints()
    num_points = points_array.shape[0]
    for i in range(num_points):
        x, y, z = points_array[i]
        vtk_pts.InsertNextPoint(x, y, z)
    
    # Create a vtkPolyData to hold the points.
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_pts)

    boundary_polydata = create_boundary_polygon(points_array)


    boundary = vtk.vtkPolyData()
    boundary.SetPoints(polydata.GetPoints())
    # boundary.SetPolys(cellArray)
    
    # Create and run the Delaunay2D filter.
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(polydata)
    delaunay.SetSourceData(boundary_polydata)
    delaunay.Update()
    
    # Wrap the resulting VTK output in a PyVista PolyData and return it.
    return pv.wrap(delaunay.GetOutput())

def create_boundary_polygon(points_2d_or_3d: np.ndarray) -> vtk.vtkPolyData:
    """
    Takes a closed loop of points (concave or convex) and creates a vtkPolyData
    with a single polygon cell representing that boundary.
    """
    # points_2d_or_3d = ensure_counterclockwise(points_2d_or_3d)

    # Ensure shape is (N,3) by adding z=0 if needed
    if points_2d_or_3d.shape[1] == 2:
        points_2d_or_3d = np.hstack([points_2d_or_3d, np.zeros((points_2d_or_3d.shape[0], 1))])

    # points_2d_or_3d = np.flip(points_2d_or_3d, axis=0)

    # Build vtkPoints
    vtk_points = vtk.vtkPoints()
    for i in range(len(points_2d_or_3d)):
        x, y, z = points_2d_or_3d[i]
        vtk_points.InsertNextPoint(x, y, z)

    # Create a single polygon cell referencing these points
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(len(points_2d_or_3d))
    for i in range(len(points_2d_or_3d)):
        polygon.GetPointIds().SetId(i, i)

    # Add the polygon cell to a vtkCellArray
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polygon)

    # Create polydata
    boundary = vtk.vtkPolyData()
    boundary.SetPoints(vtk_points)
    boundary.SetPolys(cells)

    return boundary
    
def ensure_counterclockwise(points_2d: np.ndarray, counterclockwise=True) -> np.ndarray:
    """
    Ensures a polygon defined by 2D points has the specified orientation.
    
    Parameters
    ----------
    points_2d : np.ndarray
        A 2D NumPy array of shape (N, 2), representing a polygon in the xy-plane.
        The polygon can be either open or closed (first point = last point).
    counterclockwise : bool, default=True
        If True, ensures points are in counterclockwise order.
        If False, ensures points are in clockwise order.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of the same shape, with the requested orientation.
    """

    # If the polygon is closed (first point == last point), ignore the last point for area calculation
    closed = np.allclose(points_2d[0], points_2d[-1])
    if closed:
        pts = points_2d[:-1]  # skip the repeated last point for area calc
    else:
        pts = points_2d

    # Shoelace formula for signed area
    # area > 0 => counterclockwise, area < 0 => clockwise
    x = pts[:, 0]
    y = pts[:, 1]
    # roll arrays by 1 for the "next" vertex
    area = np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)) / 2.0

    # Determine if we need to reverse the points based on current orientation and desired orientation
    if (counterclockwise and area < 0) or (not counterclockwise and area >= 0):
        return np.flip(points_2d, axis=0)
    else:
        # Already in the desired orientation
        return points_2d
    
# def get_surface_from_points_2d(line_points_2d) -> pv.PolyData:
#     # Perform Delaunay triangulation
#     delaunay = Delaunay(line_points_2d)

#     # Extract the simplices (triangles) from the triangulation
#     simplices = delaunay.simplices

#     # Add a Z-coordinate (0) to the 2D points to make them 3D
#     line_points_3d = np.column_stack((line_points_2d, np.zeros(line_points_2d.shape[0])))
    
#     # Create a PyVista PolyData object from the triangulation
#     surface = pv.PolyData(line_points_3d, np.hstack((np.full((len(simplices), 1), 3), simplices)).astype(int))

#     return surface

def unique_points(line_points_3d) -> np.ndarray:
    """
    Remove duplicate points from line_points_3d.
    """
    unique_points = []
    for i, point in enumerate(line_points_3d):
        # Skip if this point is identical to the previous one
        if i > 0 and np.allclose(point, line_points_3d[i-1], rtol=1e-5, atol=1e-5):
            continue
        unique_points.append(point)
    
    # Check if the last point is the same as the first (closed loop)
    if len(unique_points) > 1 and np.allclose(unique_points[0], unique_points[-1], rtol=1e-5, atol=1e-5):
        unique_points = unique_points[:-1]  # Remove the last point if it's the same as first

    return np.array(unique_points)


def get_surface_within_area(mesh: pv.PolyData, line_points_3d) -> pv.PolyData:
    #TODO: extrude in normal direction instead of Z
    # vector = compute_normals(mesh)
    try:
        # Remove duplicate points from line_points_3d
        line_points_3d = unique_points(line_points_3d)

        # Ensure we have enough points for triangulation
        if len(line_points_3d) < 3:
            return None, "Error: Not enough unique points to create a surface"
        
        # Extract the 2D polygon outline in the XY plane
        polygon_points = line_points_3d[:, :2]  # Keep only X and Y coordinates
        polygon_points = ensure_counterclockwise(polygon_points)
        surface = delaunay2d_surface(polygon_points)

        # Create a selection volume by extruding far in z-direction (both ways)
        points = mesh.points
        z_min, z_max = np.min(polygon_points), np.max(points[:, 2])
        z_range = z_max - z_min
        extrusion = surface.extrude((0, 0, 2*z_range), capping=True)
        extrusion.translate((0, 0, z_min - z_range/2), inplace=True)
        
        # Select the part of the face mesh inside the extrusion
        surface = mesh.clip_surface(extrusion,invert=False)


        return surface, ""
    
    except Exception as e:
        return None, f"Error in get_surface_within_area: {e}"


def remove_surface_within_area(mesh, line_points_3d, ):
    """
    Remove from mesh any region enclosed by line_points_3d.
    """
    # vector = compute_normals(mesh)
    vector = np.array([0, 0, 1])  # Default extrusion direction
    line_points_3d = unique_points(line_points_3d)

    # Project line_points_3d onto the XY plane
    polygon_points = line_points_3d[:, :2]  # Keep only X and Y coordinates
    polygon_points = ensure_counterclockwise(polygon_points)
    surface = delaunay2d_surface(polygon_points)
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
    # plotter.add_mesh(extrusion, color='lightblue', opacity=0.2, show_edges=True)
    # plotter.add_mesh(surface, color='blue', opacity=0.5, show_edges=True)
    # plotter.add_mesh(clipped_mesh, color='red', opacity=0.7, show_edges=True)
    # plotter.add_mesh(pv.PolyData(line_points_3d), color='yellow', point_size=10, render_points_as_spheres=True)
    # for i, point in enumerate(line_points_3d):
    #     plotter.add_point_labels(
    #         point, 
    #         [f"{i}"], 
    #         font_size=12, 
    #         text_color='black',
    #         point_color='red', 
    #         point_size=10, 
    #         render_points_as_spheres=True,
    #         shape_opacity=0.7
    #     )
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
    # mesh = mesh.connectivity(extraction_mode='largest')
    
    # Clean and remove degenerate features
    mesh = mesh.clean(tolerance=clean_tol)
    
    # Smooth mesh (taubin smoothing retains volume better)
    mesh = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.1)

    return mesh

def advanced_smooth_mesh(mesh: pv.PolyData, intensity: str = "medium") -> pv.PolyData:
    """
    Lissage avancé du mesh avec différents niveaux d'intensité pour réduire les facettes.
    CONSERVATEUR: préserve la géométrie et les épaisseurs.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Le mesh à lisser
    intensity : str
        Niveau de lissage: "light", "medium", "strong", "very_strong"
    
    Returns
    -------
    pv.PolyData
        Mesh lissé
    """
    # Configuration selon l'intensité - BEAUCOUP PLUS CONSERVATRICE
    configs = {
        "light": {
            "taubin_iter": 10,      # Très léger
            "laplacian_iter": 5,    # Très léger
            "feature_angle": 30.0,
            "pass_band": 0.05,      # Plus conservateur
            "clean_tol": 0.01       # Plus conservateur
        },
        "medium": {
            "taubin_iter": 25,      # Réduit
            "laplacian_iter": 15,   # Réduit
            "feature_angle": 45.0,
            "pass_band": 0.08,      # Plus conservateur
            "clean_tol": 0.05       # Plus conservateur
        },
        "strong": {
            "taubin_iter": 40,      # Réduit
            "laplacian_iter": 25,   # Réduit
            "feature_angle": 60.0,
            "pass_band": 0.1,       # Plus conservateur
            "clean_tol": 0.08       # Plus conservateur
        },
        "very_strong": {
            "taubin_iter": 60,      # Réduit
            "laplacian_iter": 35,   # Réduit
            "feature_angle": 75.0,
            "pass_band": 0.12,      # Plus conservateur
            "clean_tol": 0.1        # Plus conservateur
        }
    }
    
    if intensity not in configs:
        intensity = "light"  # Par défaut très conservateur
    
    config = configs[intensity]
    smoothed = mesh.copy()
    
    # 1. Nettoyage très léger
    smoothed = smoothed.clean(tolerance=config["clean_tol"])
    
    # 2. Lissage Taubin SEULEMENT (préserve mieux le volume et la géométrie)
    smoothed = smoothed.smooth_taubin(
        n_iter=config["taubin_iter"], 
        pass_band=config["pass_band"]
    )
    
    # 3. Lissage Laplacian SEULEMENT pour medium et plus
    if intensity in ["medium", "strong", "very_strong"]:
        smoothed = smoothed.smooth(
            n_iter=config["laplacian_iter"],
            feature_smoothing=True,
            boundary_smoothing=True,
            feature_angle=config["feature_angle"]
        )
    
    # 4. Recalcul des normales pour un rendu plus lisse
    smoothed = smoothed.compute_normals(
        auto_orient_normals=True,
        feature_angle=config["feature_angle"]
    )
    
    return smoothed

def loft_between_line_points(source_points1, source_points2, close = True):
    if len(source_points1) != len(source_points2):
        raise ValueError("Both curves must have the same number of points.")
    
    if close:
        source_points1 = np.vstack([source_points1, source_points1[0]])
        source_points2 = np.vstack([source_points2, source_points2[0]])

    num_points = len(source_points1)
    
    # Create vtkPoints and insert all points
    points = vtk.vtkPoints()
    for pt in source_points1:
        points.InsertNextPoint(pt)
    for pt in source_points2:
        points.InsertNextPoint(pt)

    # Create quad cells explicitly connecting corresponding points
    quads = vtk.vtkCellArray()
    for i in range(num_points-1):
        quad = vtk.vtkQuad()
        quad.GetPointIds().SetId(0, i)                  # current point in line 1
        quad.GetPointIds().SetId(1, i+1)                # next point in line 1
        quad.GetPointIds().SetId(2, num_points + i+1)   # next point in line 2
        quad.GetPointIds().SetId(3, num_points + i)     # current point in line 2
        quads.InsertNextCell(quad)
    
    # Create polydata with explicit quad connectivity
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(quads)
    
    # Return PyVista mesh
    return pv.wrap(polydata)




def extrude_tube_on_face_along_line(line_points: np.ndarray,
                                           face_normals: np.ndarray,
                                           radius: float,
                                           n_cs_points: int = 20,
                                           y_ratio: float = 1.0) :


    N = line_points.shape[0]
    if face_normals.shape[0] != N:
        raise ValueError("line_points and face_normals must have the same number of points.")
    
    # Define half circle in local coordinates.
    # Using theta in [0, π/2] produces a half circle.
    # (For a full half tube, you might use the range [0, π], but here we assume
    # that the half cross-section is defined over [0, π/2] for your specific case.)
    angles = np.linspace(0, np.pi*2, n_cs_points)
    local_cs = np.column_stack((radius * np.cos(angles),
                                radius * np.sin(angles) * y_ratio))  # shape: (n_cs_points, 2)
        
    # Check if the centerline is closed (first and last points coincide within a tolerance)
    closed_loop = np.allclose(line_points[0], line_points[-1], atol=1e-6)
    if closed_loop:
        # Remove the duplicate last point to avoid duplicate cross-section.
        line_points = line_points[:-1]
        face_normals = face_normals[:-1]
        N = line_points.shape[0]
    
    cross_sections = []  # will store each cross-section (n_cs_points, 3)
    top_points_list = []  # will store the "top" point from each cross-section

    for i in range(N):
        p = line_points[i]

        # Estimate tangent T at point p:
        if i == 0:
            T = line_points[i+1] - p
        elif i == N - 1:
            T = p - line_points[i-1]
        else:
            T = (line_points[i+1] - line_points[i-1]) * 0.5
        T_norm = np.linalg.norm(T)
        if T_norm < 1e-6:
            T = np.array([1, 0, 0])
        else:
            T = T / T_norm

        # Get face normal F at p (normalize if needed)
        F = face_normals[i]
        F_norm = np.linalg.norm(F)
        if F_norm < 1e-6:
            F = np.array([0, 0, 1])
        else:
            F = F / F_norm

        # Compute U as the projection of F onto the plane perpendicular to T.
        U = F - np.dot(F, T) * T
        U_norm = np.linalg.norm(U)
        if U_norm < 1e-6:
            U = np.cross(T, np.array([0, 0, 1]))
            if np.linalg.norm(U) < 1e-6:
                U = np.cross(T, np.array([0, 1, 0]))
            U = U / np.linalg.norm(U)
        else:
            U = U / U_norm

        # Ensure U points in the "outward" direction.
        # If the dot product with F is positive, reverse U.
        if np.dot(U, F) > 0:
            U = -U

        # Compute V as the normalized cross product: V = cross(T, U)
        V = np.cross(T, U)
        V = V / np.linalg.norm(V)
        
        # Map local cross-section points (x,y) to global.
        # To have the bottom of the tube (flat edge) touch the centerline,
        # shift the local cross-section by -radius*U.
        cs_global = np.array([ p - radius * U + (pt[0] * U) + (pt[1] * V)
                               for pt in local_cs ])
        cross_sections.append(cs_global)
                
        # The "top" of the half circle corresponds to local coordinates (0, radius) (theta = π/2).
        angle = 7 * np.pi / 4
        top_point = (p - radius * U) + radius * V * np.sin(angle) * y_ratio + radius * U * np.cos(angle)
        top_points_list.append(top_point)
    
    # Assemble all cross-section points into a single vtkPoints object.
    total_cs_pts = N * n_cs_points
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetNumberOfPoints(total_cs_pts)
    for i in range(N):
        for j in range(n_cs_points):
            idx = i * n_cs_points + j
            pt = cross_sections[i][j]
            vtk_pts.SetPoint(idx, pt[0], pt[1], pt[2])
    
    # Build connectivity: create quad cells connecting adjacent cross-sections.
    cells = vtk.vtkCellArray()

    if closed_loop:
        # Wrap-around connectivity: i goes from 0 to N-1, with next = (i+1) mod N.
        for i in range(N):
            next_i = (i + 1) % N
            for j in range(n_cs_points - 1):
                quad = vtk.vtkQuad()
                quad.GetPointIds().SetId(0, i * n_cs_points + j)
                quad.GetPointIds().SetId(1, i * n_cs_points + j + 1)
                quad.GetPointIds().SetId(2, next_i * n_cs_points + j + 1)
                quad.GetPointIds().SetId(3, next_i * n_cs_points + j)
                cells.InsertNextCell(quad)
    else:
        # Non-closed centerline: i goes from 0 to N-2.
        for i in range(N - 1):
            for j in range(n_cs_points - 1):
                quad = vtk.vtkQuad()
                quad.GetPointIds().SetId(0, i * n_cs_points + j)
                quad.GetPointIds().SetId(1, i * n_cs_points + j + 1)
                quad.GetPointIds().SetId(2, (i + 1) * n_cs_points + j + 1)
                quad.GetPointIds().SetId(3, (i + 1) * n_cs_points + j)
                cells.InsertNextCell(quad)
    
    # Create the output vtkPolyData.
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_pts)
    polydata.SetPolys(cells)
    
    # Convert top points list to numpy array.
    top_points = np.array(top_points_list)

    # Get first and last cross-section points
    first_cs = cross_sections[0]
    last_cs = cross_sections[-1]

    # Convert the resulting vtkPolyData to a PyVista PolyData object.
    pv_polydata = pv.wrap(polydata)
    
    return pv_polydata, top_points, cross_sections

def reorder_line_points(line_points: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """
    Reorder the line points so that the first point is the one closest to 'pt'
    while keeping the original order of the points.
    
    If the line is not closed (first point ≠ last point), the first point is
    appended at the end to close the loop.
    
    Parameters
    ----------
    line_points : np.ndarray
        An (N,3) array representing the points of the line.
    pt : np.ndarray
        A 3D point.
    
    Returns
    -------
    np.ndarray
        The reordered (and closed) array of line points.
    """
    # Compute distances from each point to the given pt
    distances = np.linalg.norm(line_points - pt, axis=1)

    # Find index of the closest point
    min_distance = np.min(distances)
    closest_indices = np.where(distances == min_distance)[0]
    # If there are multiple points with the same minimum distance, take the one with the lowest index
    closest_idx = np.min(closest_indices)

    # Roll the array so that the closest point becomes the first element
    reordered = np.roll(line_points, -closest_idx, axis=0)
    
    # Ensure the line is closed by appending the first point if necessary
    if not np.allclose(reordered[0], reordered[-1]):
        reordered = np.vstack([reordered, reordered[0]])
    
    return reordered

def deform_surface_at_point(surface, center_point, radius, strength, inside_direction=False):
    """
    Apply a smooth deformation to a surface around a center point.
    
    Parameters:
    -----------
    surface : pv.PolyData
        The surface to deform
    center_point : array-like
        The center point of deformation (x, y, z)
    radius : float
        The radius of influence for deformation (in mm)
    strength : float
        The maximum displacement strength
    inside_direction : boolean, optional
        Direction of displacement. 
    
    Returns:
    --------
    pv.PolyData
        Deformed surface
    """
    deformed = surface.copy()
    points = deformed.points
    
    # Find the closest point on the surface to the center_point
    center_idx = deformed.find_closest_point(center_point)
    center_on_surface = deformed.points[center_idx]
    
    # Compute displacement direction
    if "Normals" not in deformed.point_data:
        deformed.compute_normals(inplace=True)
    direction = deformed.point_data["Normals"][center_idx]
    
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    
    if inside_direction == True:
        direction = -direction

    # Compute distances from all points to center
    distances = np.linalg.norm(points - center_on_surface, axis=1)
    
    # Apply smooth falloff (using cosine falloff)
    mask = distances < radius
    normalized_distances = distances[mask] / radius
    falloff = np.cos(np.pi * normalized_distances / 2) ** 2  # Cosine squared falloff
    
    # Apply displacement
    displacement = strength * falloff[:, np.newaxis] * direction
    points[mask] += displacement
    
    return deformed


def extract_line_from_landmarks(mesh, landmarks, landmark_indices, contour_landmark_ids):
    """
    Extract face contour line from landmarks and project it onto the face mesh.
    """
    valid_ids = [np.where(landmark_indices == lid)[0][0] 
                 for lid in contour_landmark_ids if lid in landmark_indices]
    
    closed = contour_landmark_ids[0] == contour_landmark_ids[-1]
    line_points = landmarks[valid_ids]
    line_points = smooth_line_points(line_points, smoothing=0.05, num_samples=100, closed=closed)   
    
    # if hasattr(mesh, 'compute_triangle_normals'):
    #     mesh.compute_triangle_normals()

    if "Normals" not in mesh.point_data:
        mesh = mesh.compute_normals()


    bounds = mesh.bounds
    dz = 100
    projected_line_points = []
    projected_normals = []
    for pt in line_points:
        origin = (pt[0], pt[1], bounds[5] + dz)
        end = (pt[0], pt[1], bounds[4] - dz)
        pts, ids = mesh.ray_trace(origin, end, first_point=True)
        if pts.size:
            projected_line_points.append(pts)
            
            idx = mesh.find_closest_point(pt)
            n = mesh.point_data["Normals"][idx]
            n = n / np.linalg.norm(n)
            projected_normals.append(n)
        else:
            projected_line_points.append(pt)
            projected_normals.append(np.array([0.0, 0.0, 1.0]))  # Default normal
    
    return np.array(projected_line_points), np.array(projected_normals)
    # return np.array(projected_line_points)

def get_tangent_points(cross_sections, circle_points):
    
    tangent_points = []
    p0 = np.mean(circle_points, axis=0)

    # loop over each section
    for cnt, cross_section_points in enumerate(cross_sections, start = 0):
        connector_point = circle_points[cnt]
        
        outside_vec = connector_point - p0
        outside_vec /= np.linalg.norm(outside_vec)
        
        
        # Create convex hull from intersection points
        hull = ConvexHull(cross_section_points)
        hull_points = cross_section_points[hull.vertices]

        # Define the normal vector to the plane defined by cross_section_points
        plane_normal = np.cross(cross_section_points[1] - cross_section_points[0], 
                                    cross_section_points[2] - cross_section_points[0])
        plane_normal /= np.linalg.norm(plane_normal)

        # Find tangent points
        tangent_candidates = []

        for i, hull_point in enumerate(hull_points):
            # Vector from point to hull point
            vec_to_hull = hull_point - connector_point
            
            # Create a normal vector to this line in the plane
            tangent_normal = np.cross(vec_to_hull, plane_normal)
            tangent_normal /= np.linalg.norm(tangent_normal)
            
            # Check if all other hull points are on one side of this line
            is_tangent = True
            sign = None
            
            for j, other_hull_point in enumerate(hull_points):
                if j != i:
                    # Vector from hull_point to other_hull_point
                    vec_along_hull = other_hull_point - hull_point
                    
                    # Dot product with normal tells us which side it's on
                    side = np.dot(tangent_normal, vec_along_hull)
                    
                    if sign is None:
                        sign = np.sign(side)
                    elif side * sign < 0:  # Points on different sides
                        is_tangent = False
                        break
            
            if is_tangent:
                tangent_candidates.append(hull_point)

        # Choose the tangent point that is the farthest from the circle center (p0)
        if tangent_candidates:
            tangent_point = max(
                tangent_candidates, 
                key=lambda p: np.linalg.norm((p - p0)[:2])  # Consider projection on x, y plane
            )


            tangent_points.append(tangent_point)

    return tangent_points

def repair_mesh(pv_mesh: pv.PolyData, smooth_intensity: str = "medium") -> pv.PolyData:
    """
    Répare et lisse un mesh avec options de lissage avancé.
    
    Parameters
    ----------
    pv_mesh : pv.PolyData
        Le mesh à réparer
    smooth_intensity : str
        Niveau de lissage: "light", "medium", "strong", "very_strong", "none"
    
    Returns
    -------
    pv.PolyData
        Mesh réparé et lissé
    """
    # Ensure the mesh is triangulated
    pv_mesh = pv_mesh.fill_holes(hole_size=1.0)
    
    # Apply advanced smoothing if requested
    if smooth_intensity != "none":
        pv_mesh = advanced_smooth_mesh(pv_mesh, intensity=smooth_intensity)
    else:
        # Minimal smoothing for repair only
        pv_mesh = pv_mesh.smooth(n_iter=20,
                                feature_smoothing=True,
                                boundary_smoothing=True,
                                feature_angle=45.0)

    pv_mesh = pv_mesh.triangulate()
    # Convert PyVista to Trimesh
    tm_mesh = trimesh.Trimesh(vertices=pv_mesh.points, faces=pv_mesh.faces.reshape(-1, 4)[:, 1:])
    
    # Repair: fix watertight, remove degenerate
    tm_mesh.fill_holes()
    tm_mesh.remove_degenerate_faces()
    tm_mesh.fix_normals()
    tm_mesh.merge_vertices()
    
    # Convert back to PyVista
    faces = np.hstack([np.full((len(tm_mesh.faces), 1), 3), tm_mesh.faces]).flatten()
    return pv.PolyData(tm_mesh.vertices, faces)

def fix_surface_normals_at_interface(tube_surface: pv.PolyData, tangent_surface: pv.PolyData) -> pv.PolyData:
    """
    Fix normal discontinuities at the interface between tube_surface and tangent_surface
    by recalculating smooth normals across the combined surface.
    
    Parameters
    ----------
    tube_surface : pv.PolyData
        The tube surface mesh.
    tangent_surface : pv.PolyData
        The tangent surface mesh.
    
    Returns
    -------
    pv.PolyData
        Combined surface with smooth normals at the interface.
    """
    # Combine the two surfaces
    combined_surface = tube_surface + tangent_surface
    
    # Clean and merge close vertices at the interface
    combined_surface = combined_surface.clean(tolerance=0.05)
    
    # Recalculate normals with feature angle to smooth across interfaces
    # feature_angle=60 allows smoothing across moderate angle changes
    combined_surface = combined_surface.compute_normals(
        auto_orient_normals=True,
        feature_angle=60.0  # Smooth across angles up to 60 degrees
        # Note: PyVista doesn't have 'splitting' parameter - removed
    )
    
    return combined_surface