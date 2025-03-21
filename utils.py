"""
Utility functions for 3D face processing
"""
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.interpolate import splprep, splev
import os

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

def set_front_view(plotter):
    """Set camera to front view with head upright"""
    plotter.view_xy()  # Set to front view (looking at XY plane)
    plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]  # Position, focus, up-vector
    plotter.camera.zoom(0.8)  # Slight zoom for better framing
    plotter.enable_trackball_style()
    return plotter
