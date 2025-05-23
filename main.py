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
import open3d as o3d

from config import DEFAULT_FACE_CONTOUR_LANDMARKS, DEFAULT_VISUALIZATION_SETTINGS, DEFAULT_PRINTING_SETTINGS, DEFAULT_FILE_SETTINGS, DEFAULT_MODEL, OUTPUT_FOLDER
from face import extract_face_landmarks, extract_average_face_landmarks_from_video, compute_rigid_transform_parameters, compute_similarity_transform_parameters
from visualization import (visualize_2d_landmarks, visualize_3d_landmarks, 
                          visualize_mesh_with_landmarks, visualize_contact_line)
from utils import create_pyvista_mesh

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
    module_name = f"models.{model_name.lower()}_model"
    class_name = f"{model_name}Model"
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
        output_path = os.path.join(
            OUTPUT_FOLDER,
            f"{base_name}_{model_name}.stl"
        )

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
    parser.add_argument("-v", "--video", type=str, help="Path to the input video file for landmark averaging")
    parser.add_argument("-s", "--scan", type=str, help="Path to the 3D scan mesh to match against video landmarks")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization for video landmark extraction")
    parser.add_argument("--feature-scale", action="store_true", help="Use eye-distance feature to compute landmark scaling")
    parser.add_argument("--axis-adjust", action="store_true", help="Adjust scaling per axis based on scan/video extents")
    args = parser.parse_args()

    # Video demo: compute and display average 3D landmarks
    if args.video:
        avg_landmarks = extract_average_face_landmarks_from_video(args.video, debug=args.debug)
        # If a scan mesh is provided, align and visualize both
        if args.scan:
            # extract landmarks from the 3D scan
            scan_mesh_o3d, scan_landmarks, scan_indices = extract_face_landmarks(args.scan)
            # filter video average landmarks to only scan-detected indices
            avg_subset = avg_landmarks[scan_indices]
        
            # compute transform with optional feature-based scaling
            if not args.feature_scale:
                # similarity transform (uniform scale + rotation + translation)
                s, R, t = compute_similarity_transform_parameters(avg_subset, scan_landmarks)
            else:
                # rigid rotation + translation
                R, t = compute_rigid_transform_parameters(avg_subset, scan_landmarks)
                # compute scale from eye-distance (indices 33 and 263)
                idx_arr = np.array(scan_indices)
                if 33 in scan_indices and 263 in scan_indices:
                    vid_pts = avg_landmarks[[33, 263]]
                    pos_i = np.where(idx_arr == 33)[0][0]
                    pos_j = np.where(idx_arr == 263)[0][0]
                    scan_pts = scan_landmarks[[pos_i, pos_j]]
                    d_scan = np.linalg.norm(scan_pts[1] - scan_pts[0])
                    d_video = np.linalg.norm(vid_pts[1] - vid_pts[0])
                    if d_video <= 0:
                        raise ValueError("Invalid eye-distance in video landmarks for scaling")
                    s = d_scan / d_video
                else:
                    raise ValueError("Required landmark indices (33, 263) not found in scan for feature scaling")

            # apply transform to all video landmarks
            video_pts = (R @ avg_landmarks.T)
            # apply uniform or per-axis scaling
            if args.axis_adjust:
                # compute video and scan axis ranges
                vid_np = video_pts.T  # (N,3)
                vid_min = vid_np.min(axis=0)
                vid_max = vid_np.max(axis=0)
                scan_min = scan_landmarks.min(axis=0)
                scan_max = scan_landmarks.max(axis=0)
                # avoid zero range
                vid_range = np.where((vid_max-vid_min)==0, 1.0, vid_max-vid_min)
                scan_range = scan_max - scan_min
                axis_scale = scan_range / vid_range
                if args.debug:
                    print(f"Debug: axis_scale = {axis_scale}")
                    print(f"Debug: video range = {vid_range}, scan range = {scan_range}")
                # apply scales axis-wise
                pts_scaled = video_pts * axis_scale[:, np.newaxis]
            else:
                if 's' not in locals():  # ensure s is defined when not feature-scale
                    s = 1.0
                pts_scaled = s * video_pts
            transformed_avg = pts_scaled.T + t

            # prepare visualization
            plotter = pv.Plotter()
            mesh_pv = create_pyvista_mesh(scan_mesh_o3d)
            plotter.add_mesh(mesh_pv, color='white', opacity=0.5)
            plotter.add_points(scan_landmarks, color='red', point_size=6, render_points_as_spheres=True)
            plotter.add_points(transformed_avg, color='blue', point_size=6, render_points_as_spheres=True)
            plotter.add_legend([('Scan landmarks','red'), ('Video avg landmarks','blue')])
            plotter.show()
            sys.exit(0)
        else:
            # only visualize average landmarks from video
            fig = visualize_3d_landmarks(avg_landmarks, title="Average 3D Landmarks from Video")
            plt.show()
            sys.exit(0)

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
