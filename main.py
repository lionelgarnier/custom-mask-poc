"""
Main entry point for face landmark detection and contour generation
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import tkinter as tk
from tkinter import filedialog
import importlib
import cv2

from config import DEFAULT_FACE_CONTOUR_LANDMARKS, DEFAULT_VISUALIZATION_SETTINGS, DEFAULT_PRINTING_SETTINGS, DEFAULT_FILE_SETTINGS, DEFAULT_MODEL, OUTPUT_FOLDER
from face import extract_face_landmarks
from visualization import (visualize_2d_landmarks, visualize_3d_landmarks, 
                          visualize_mesh_with_landmarks, visualize_contact_line)

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

def process_face(mesh_path, 
                 model_name, 
                 show_3d_print=False, 
                 create_3d_print=True, 
                 save_3d_print=True, 
                 show_landmarks_2d=False, 
                 show_landmarks_3d=False):
    """
    Main function to process a face model and extract landmarks
    
    Parameters:
    -----------
    mesh_path : str
        Path to the 3D mesh file
    model_name : str
        Model object to create 3D printable object
    show_3d_print : bool
        Whether to show the 3D print visualization
    create_3d_print : bool
        Whether to create a 3D printable STL file
    extrusion_width : float
        Width of the extruded tube for 3D printing
    """
    # Process the face to extract landmarks and contour
    face_mesh, face_landmarks, valid_indices = extract_face_landmarks(mesh_path)
    
    # # Display the extracted image color on the screen
    # keep_indices = None
    # landmarks_plotter = pv.Plotter()
    # landmarks_plotter.add_title("Face Mesh with Landmarks")
    # landmarks_plotter = visualize_mesh_with_landmarks(landmarks_plotter, face_mesh, face_landmarks, valid_indices, keep_indices)
    
    # landmarks_plotter.view_xy()  # Set to front view (looking at XY plane)
    # landmarks_plotter.enable_zoom_style()
    # landmarks_plotter.enable_trackball_style()
    # landmarks_plotter.show()
    
    # Create 3D printable object if requested
    if create_3d_print:
        # Ensure the output folder exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"3d_print_{base_name}.stl")

        model = load_model(model_name)

        # Create and save the 3D printable extrusion
        model_3d_object, error = model.create_3d_object(
            output_path=output_path, 
            face_mesh=face_mesh,
            face_landmarks=face_landmarks,
            face_landmarks_ids=valid_indices
        )

        if error:
            print(f"Error creating 3D printable object: {error}")
            return False

        # Visualize the 3D print if requested
        if show_3d_print :
            print_plotter = pv.Plotter()
            print_plotter.add_title("3D Printable Object")
            print_plotter.add_mesh(model_3d_object, color='white')
            print_plotter.show_grid()
            print_plotter.show()

        if save_3d_print:
            model_3d_object.save(output_path)



    return True

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face landmark detection and contour generation")
    parser.add_argument("-p", "--path", type=str, help="Path to the 3D mesh file")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help="Model name to use for 3D object creation")
    args = parser.parse_args()

    # Use default settings from config
    vis_settings = DEFAULT_VISUALIZATION_SETTINGS
    print_settings = DEFAULT_PRINTING_SETTINGS
    file_settings = DEFAULT_FILE_SETTINGS
    
    # Check if mesh file path is provided as a command-line argument
    mesh_path = args.path if args.path else select_mesh_file(file_settings['DEFAULT_MESH_FOLDER'])
    
    if mesh_path:
        process_face(mesh_path, 
                    args.model,
                    show_3d_print=vis_settings['SHOW_3D_PRINT'],
                    create_3d_print=print_settings['CREATE_3D_PRINT'],
                    save_3d_print=print_settings['SAVE_3D_PRINT'],
                    show_landmarks_2d=vis_settings['SHOW_LANDMARKS_2D'],
                    show_landmarks_3d=vis_settings['SHOW_LANDMARKS_3D'],   
                    )
    else:
        print("No mesh file selected. Exiting.")
