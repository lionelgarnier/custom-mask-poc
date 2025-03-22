"""
Main entry point for face landmark detection and contour generation
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import tkinter as tk
from tkinter import filedialog
import importlib

from config import DEFAULT_FACE_CONTOUR_LANDMARKS, DEFAULT_VISUALIZATION_SETTINGS, DEFAULT_PRINTING_SETTINGS, DEFAULT_FILE_SETTINGS, DEFAULT_MODEL, OUTPUT_FOLDER
from core_processing import extract_face_landmarks
from visualization import (visualize_2d_landmarks, visualize_3d_landmarks, 
                          visualize_mesh_with_landmarks, visualize_contact_line)
# from utils import create_3d_printable_shape

def select_mesh_file(default_folder=None):
    """
    Open a file dialog to select a mesh file
    
    Parameters:
    -----------
    default_folder : str, optional
        Default folder path to open the dialog in
    
    Returns:
    --------
    str or None
        Selected file path or None if canceled
    """
    try:
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()
        
        # Set default file types for 3D meshes
        filetypes = [
            ("3D Mesh Files", "*.obj *.ply *.stl"),
            ("OBJ Files", "*.obj"),
            ("PLY Files", "*.ply"),
            ("STL Files", "*.stl"),
            ("All Files", "*.*")
        ]
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select 3D Mesh File",
            initialdir=default_folder,
            filetypes=filetypes
        )
        
        # Close the hidden root window
        root.destroy()
        
        return file_path if file_path else None
    
    except ImportError:
        print("Tkinter not available. Please provide the file path manually.")
        return input("Enter the path to the 3D mesh file: ")

def load_model(model_name):
    module_name = f"models.{model_name}_model"
    class_name = f"{model_name.capitalize()}Model"
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class()

def process_face(mesh_path, model, show_landmarks_2d=False, show_landmarks_3d=False, show_mesh_and_landmarks=False, show_contact_line=False, show_3d_print=False, create_3d_print=True, extrusion_width=3.0, contour_landmark_ids=None):
    """
    Main function to process a face model and extract landmarks
    
    Parameters:
    -----------
    mesh_path : str
        Path to the 3D mesh file
    model : object
        Model object to create 3D printable object
    show_landmarks_2d : bool
        Whether to show 2D landmarks visualization
    show_landmarks_3d : bool
        Whether to show 3D landmarks visualization
    show_mesh_and_landmarks : bool
        Whether to show the mesh with landmarks
    show_contact_line : bool
        Whether to show the contact line visualization
    show_3d_print : bool
        Whether to show the 3D print visualization
    create_3d_print : bool
        Whether to create a 3D printable STL file
    extrusion_width : float
        Width of the extruded tube for 3D printing
    contour_landmark_ids : list, optional
        List of landmark IDs to use for the face contour. If None, uses default landmarks.
    """
    # Process the face to extract landmarks and contour
    mesh, valid_points_3d, valid_indices, projected_line_points, viz_image = extract_face_landmarks(
        mesh_path, contour_landmark_ids)
    
    # Visualizations (if requested)
    # Create a multi-panel PyVista plotter if needed
    num_pyvista_views = sum([show_mesh_and_landmarks, show_contact_line, show_3d_print])  # Count PlotterIDs (+1 for final view)
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

    # Final visualization - only if requested
    if show_contact_line:
        if pv_plotter is not None:
            pv_plotter.subplot(0, current_subplot)
            pv_plotter.add_title("Face Mesh with Projected Path")
            visualize_contact_line(mesh, projected_line_points, pv_plotter)
            
            # Enable trackball style for all subplots and link them
            pv_plotter.enable_trackball_style()
            pv_plotter.link_views()  # Link camera movement between subplots
            pv_plotter.show()
        else:
            # Standalone final plotter
            final_plotter = pv.Plotter()
            final_plotter.add_title("Face Mesh with Projected Path")
            visualize_contact_line(mesh, projected_line_points, final_plotter)
            final_plotter.show_grid()
            final_plotter.enable_trackball_style()  # Enable trackball style
            final_plotter.show()

    # Create 3D printable object if requested
    if create_3d_print:
        # Ensure the output folder exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"extruded_{base_name}.stl")
        
        # Create and save the 3D printable extrusion
        extruded = model.create_3d_object(projected_line_points, output_path, thickness=extrusion_width)

        # Visualize the 3D print if requested
        if show_3d_print:
            if pv_plotter is not None:
                pv_plotter.subplot(0, current_subplot)
                pv_plotter.add_title("3D Printable Object")
                pv_plotter.add_mesh(extruded, color='white')
                pv_plotter.show()
            else:
                # Standalone plotter for 3D print visualization
                print_plotter = pv.Plotter()
                print_plotter.add_title("3D Printable Object")
                print_plotter.add_mesh(extruded, color='white')
                print_plotter.show_grid()
                print_plotter.show()

    # Display all matplotlib figures (if any)
    if len(figures) > 0:
        plt.show()
    
    return valid_points_3d, projected_line_points

if __name__ == "__main__":
    # Use default settings from config
    vis_settings = DEFAULT_VISUALIZATION_SETTINGS
    print_settings = DEFAULT_PRINTING_SETTINGS
    file_settings = DEFAULT_FILE_SETTINGS
    
    # Face contour landmarks (uncomment and modify to use custom landmarks)
    # CONTOUR_LANDMARKS = [168, 417, 465, 429, 423, 391, 393, 164, 167, 165, 203, 209, 245, 193, 168]
    
    # Let user select mesh file
    mesh_path = select_mesh_file(file_settings['DEFAULT_MESH_FOLDER'])
    
    if mesh_path:
        model = load_model(DEFAULT_MODEL)
        process_face(mesh_path, 
                    model,
                    show_landmarks_2d=vis_settings['SHOW_LANDMARKS_2D'],
                    show_landmarks_3d=vis_settings['SHOW_LANDMARKS_3D'],
                    show_mesh_and_landmarks=vis_settings['SHOW_MESH_AND_LANDMARKS'],
                    show_contact_line=vis_settings['SHOW_CONTACT_LINE'],
                    show_3d_print=vis_settings['SHOW_3D_PRINT'],
                    create_3d_print=print_settings['CREATE_3D_PRINT'],
                    extrusion_width=print_settings['EXTRUSION_WIDTH'],
                    # contour_landmark_ids=CONTOUR_LANDMARKS  # Uncomment to use custom landmarks
                    )
    else:
        print("No mesh file selected. Exiting.")
