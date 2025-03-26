from models.base_model import Face3DObjectModel
import pyvista as pv
import numpy as np
from utils import (extrude_mesh_with_normals, get_surface_within_area, remove_surface_within_area,
                  create_pyvista_mesh, compute_rotation_between_vectors,
                  rotate_shape_and_landmarks, translate_shape_and_landmarks,
                  get_landmark)
from shapes.n5_connector import N5Connector
from face import extract_line_from_landmarks
from scipy.spatial.transform import Rotation as R

class MoldModel(Face3DObjectModel):    
    def create_3d_object(self, output_path, **kwargs):
        # Get required parameters
        face_mesh = kwargs.get('face_mesh', None)
        face_landmarks = np.array(kwargs.get('face_landmarks'))
        face_landmarks_ids = np.array(kwargs.get('face_landmarks_ids'))
        thickness = kwargs.get('thickness', 3.0)
        
        if face_mesh is None:
            raise ValueError("face_mesh is required for MoldModel")
        
        if face_landmarks is None:
            raise ValueError("face_landmarks are required for MoldModel")
        
        if face_landmarks_ids is None:
            raise ValueError("face_landmarks_ids are required for MoldModel")
        
        # shape_landmarks = [168, 465, 419, 363, 134, 196, 245, 168] # Small nose top
        shape_landmarks = [168, 464, 423, 391, 393, 164, 167, 165, 203, 193, 168] # Large Nose
        shape_hole_landmarks = [6, 437, 429, 279, 294, 327, 326, 164, 97, 98, 64, 49, 209, 217, 6] 

        line_points_3d = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_landmarks)  
        shape_hole_points_3d = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_hole_landmarks) 

        # Generate the surface using the extracted function
        nose_surface = get_surface_within_area(face_mesh, line_points_3d)
        nose_surface = remove_surface_within_area(nose_surface, shape_hole_points_3d)


        # Use helper function for extrusion
        nose_volume = extrude_mesh_with_normals(nose_surface, thickness)


        # Create an instance of N5Connector and generate the 3D object
        shape_builder = N5Connector()
        shape_3d = shape_builder.create_3d_object()

        # Locate landmarks defining horizontal and vertical alignments
        landmark_x1 = get_landmark(face_landmarks, face_landmarks_ids, 168)
        landmark_x2 = get_landmark(face_landmarks, face_landmarks_ids, 94)
        landmark_y1 = get_landmark(face_landmarks, face_landmarks_ids, 117)
        landmark_y2 = get_landmark(face_landmarks, face_landmarks_ids, 347)

        # Compute the direction vectors for the lines
        vector_x = landmark_x2 - landmark_x1
        vector_y = landmark_y2 - landmark_y1

        vector_shape_z1 = get_landmark(shape_3d, target_id=1)
        vector_shape_z2 = get_landmark(shape_3d, target_id=3)

        vector_z = vector_shape_z2 - vector_shape_z1

        # Normalize the vectors
        vector_x /= np.linalg.norm(vector_x)
        vector_y /= np.linalg.norm(vector_y)
        vector_z /= np.linalg.norm(vector_z)


        # Compute the rotation matrix to align the direction vectors
        rotation_1 = compute_rotation_between_vectors(vector_z, vector_x)
        shape_3d = rotate_shape_and_landmarks(shape_3d, rotation_1)

        # Compute the second rotation to align the second vector
        vector_shape_z1 = get_landmark(shape_3d, target_id=1)
        vector_shape_z2 = get_landmark(shape_3d, target_id=5)
        vector_z = vector_shape_z2 - vector_shape_z1
        vector_z /= np.linalg.norm(vector_z)
        
        rotation_2 = compute_rotation_between_vectors(vector_z, vector_y)
        shape_3d = rotate_shape_and_landmarks(shape_3d, rotation_2)


        # Translate above the nose tip
        landmark_z = get_landmark(face_landmarks, face_landmarks_ids, 5)
        center_shape_3d = get_landmark(shape_3d, target_id=1)
        translation = landmark_z - center_shape_3d
        shape_3d = translate_shape_and_landmarks(shape_3d, translation)
   

        p1 = shape_3d.landmarks[1]
        p3 = shape_3d.landmarks[3]
        p5 = shape_3d.landmarks[5]
        p7 = shape_3d.landmarks[7]

        center = p1
        radius = np.linalg.norm(p7 - p1)

        normal = np.cross(p3 - p1, p5 - p1)
        normal /= np.linalg.norm(normal)

        ref_vec = p3 - p1
        ref_vec /= np.linalg.norm(ref_vec)
        side_vec = np.cross(normal, ref_vec)
        side_vec /= np.linalg.norm(side_vec)

        angles = np.linspace(0, 2*np.pi, 50)
        circle_points = []
        for angle in angles:
            circle_points.append(center + radius*(ref_vec*np.cos(angle) + side_vec*np.sin(angle)))
        circle_points = np.array(circle_points)

        combined_points = np.vstack((circle_points, shape_hole_points_3d))

        # Create a surface from the combined points
        join_surface = pv.PolyData(combined_points)
        join_surface = join_surface.delaunay_2d()

        join_volume = extrude_mesh_with_normals(join_surface, thickness)

        # Add the surface to the plotter for visualization
        # Optionally plot for debugging
        plotter = pv.Plotter()
        # pv_face_mesh = create_pyvista_mesh(face_mesh)
        plotter.add_mesh(nose_surface, color='red', opacity=0.3, show_edges=True)
        plotter.add_mesh(nose_volume, color='lightblue', opacity=0.3, show_edges=True)
        # plotter.add_mesh(shape_3d, color='blue', opacity=0.5, show_edges=True)
        # plotter.add_mesh(pv.PolyData(line_points_3d), color='red', point_size=10, render_points_as_spheres=True)
        pv_join_volume = create_pyvista_mesh(join_volume)
        plotter.add_mesh(pv_join_volume, color='yellow', opacity=0.5, show_edges=True)
        # plotter.add_mesh(pv.Sphere(center=landmark_x1, radius=2.0), color='orange', render_points_as_spheres=True)
        # plotter.add_mesh(pv.Sphere(center=landmark_x2, radius=2.0), color='orange', render_points_as_spheres=True)
        # plotter.add_mesh(pv.Sphere(center=landmark_y1, radius=2.0), color='orange', render_points_as_spheres=True)
        # plotter.add_mesh(pv.Sphere(center=landmark_y2, radius=2.0), color='orange', render_points_as_spheres=True)
        # plotter.add_mesh(pv.Sphere(center=landmark_z, radius=2.0), color='red', render_points_as_spheres=True)
        # plotter.add_mesh(pv.Sphere(center=shape_3d.landmarks[1], radius=2.0), color='green', render_points_as_spheres=True)
        # plotter.add_mesh(pv.Sphere(center=shape_3d.landmarks[3], radius=2.0), color='green', render_points_as_spheres=True)

        plotter.show()
        model_shape = (nose_volume, shape_3d)

        return model_shape
