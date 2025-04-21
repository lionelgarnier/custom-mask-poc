"""
Visualization functions for 3D face models and landmarks
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyvista as pv
import vtk
from utils import create_pyvista_mesh, smooth_line_points, set_front_view
from config import DEFAULT_FACE_CONTOUR_LANDMARKS
from face import extract_face_landmarks

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

def visualize_mesh_with_landmarks(plotter, mesh, landmarks, landmark_indices, keep_indices = None):
    """Visualize 3D mesh with landmarks and IDs"""

    pv_mesh = create_pyvista_mesh(mesh)
    plotter.add_mesh(pv_mesh, color='white', opacity=0.5)

    # Add landmarks with their IDs
    if keep_indices is not None:
        # Keep only landmarks corresponding to given IDs
        landmarks = np.array([landmarks[i] for i in keep_indices])
    else:
        landmarks = np.array(landmarks)

    plotter.add_points(landmarks, color='green', point_size=4, render_points_as_spheres=True)
    # for idx, point in zip(landmark_indices, landmarks):
    #     plotter.add_point_labels([point], [str(idx)], font_size=10, text_color='blue', shape_opacity=0.0)

    
    return plotter

def visualize_contact_line(mesh, line_points, plotter=None):
    """Visualize 3D mesh with contact line"""
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
    
    # Enable trackball actor style for better navigation
    plotter.enable_trackball_actor_style()
    
    if return_plotter:
        plotter.show_grid()
        return plotter
    else:
        return None
