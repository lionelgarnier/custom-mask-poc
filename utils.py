"""
Utility functions for 3D face processing
"""
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
from shapely.ops import unary_union
import os
from models.base_model import Face3DObjectModel

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

# def create_3d_printable_shape(
#     line_points_3d, 
#     model: Face3DObjectModel,
#     thickness=3.0,   # total wall thickness in millimeters
#     height=10.0,     # extrusion height in millimeters
#     output_path="tube.stl"
# ):
#     """
#     1. Project a 3D closed line onto XY plane.
#     2. Create outer buffer (+thickness/2) => outer polygon.
#     3. Create inner buffer (-thickness/2) => inner polygon (the "hole").
#     4. Extrude each polygon with PyVista => 3D solids.
#     5. Boolean difference => hollow 'tube' with correct hole.
#     6. Save final shape to STL.
#     """
#     model.create_3d_object(line_points_3d, output_path, thickness=thickness, height=height)

def set_front_view(plotter):
    """Set camera to front view with head upright"""
    plotter.view_xy()  # Set to front view (looking at XY plane)
    plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]  # Position, focus, up-vector
    plotter.camera.zoom(0.8)  # Slight zoom for better framing
    plotter.enable_trackball_style()
    return plotter
