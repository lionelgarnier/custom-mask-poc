"""
Main entry point for face landmark detection and contour generation
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from config import DEFAULT_FACE_CONTOUR_LANDMARKS, DEFAULT_VISUALIZATION_SETTINGS, DEFAULT_PRINTING_SETTINGS
from core_processing import extract_face_landmarks
from visualization import (visualize_2d_landmarks, visualize_3d_landmarks, 
                          visualize_mesh_with_landmarks, visualize_mesh_with_yellow_lines)
from utils import create_3d_printable_extrusion

def process_face(mesh_path, 
                show_landmarks_2d=False, 
                show_landmarks_3d=False, 
                show_mesh_and_landmarks=False,
                create_3d_print=False, 
                extrusion_radius=0.1, 
                contour_landmark_ids=None):
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

if __name__ == "__main__":
    # Use default settings from config
    vis_settings = DEFAULT_VISUALIZATION_SETTINGS
    print_settings = DEFAULT_PRINTING_SETTINGS
    
    # Face contour landmarks (uncomment and modify to use custom landmarks)
    # CONTOUR_LANDMARKS = [168, 417, 465, 429, 423, 391, 393, 164, 167, 165, 203, 209, 245, 193, 168]
    
    # Process the face model
    mesh_path = "D:/OneDrive/Desktop/masque/face_lionel_long_capture.obj"
    process_face(mesh_path, 
                 show_landmarks_2d=vis_settings['SHOW_LANDMARKS_2D'],
                 show_landmarks_3d=vis_settings['SHOW_LANDMARKS_3D'],
                 show_mesh_and_landmarks=vis_settings['SHOW_MESH_AND_LANDMARKS'],
                 create_3d_print=print_settings['CREATE_3D_PRINT'],
                 extrusion_radius=print_settings['EXTRUSION_RADIUS'],
                 # contour_landmark_ids=CONTOUR_LANDMARKS  # Uncomment to use custom landmarks
                )
