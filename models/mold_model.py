from models.base_model import Face3DObjectModel
import pyvista as pv
import numpy as np
from utils import (create_pyvista_mesh, thicken_mesh, clean_and_smooth, thicken_mesh_vtk,
                   get_surface_within_area, remove_surface_within_area, deform_surface_at_point,
                  compute_rotation_between_vectors, extrude_mesh, reorder_line_points, 
                  rotate_shape_and_landmarks, translate_shape_and_landmarks,
                  get_landmark, loft_between_line_points, extrude_half_tube_on_face_along_line)
from shapes.n5_connector import N5Connector
from shapes.fsa_connector import FSAConnector
from face import extract_line_from_landmarks
from scipy.spatial.transform import Rotation as R


class MoldModel(Face3DObjectModel):    
    def create_3d_object(self, output_path, **kwargs):
        
        # Get required parameters
        face_mesh = kwargs.get('face_mesh', None)
        face_landmarks = np.array(kwargs.get('face_landmarks'))
        face_landmarks_ids = np.array(kwargs.get('face_landmarks_ids'))
        thickness = 1.0
        
        if face_mesh is None:
            raise ValueError("face_mesh is required for MoldModel")
        
        if face_landmarks is None:
            raise ValueError("face_landmarks are required for MoldModel")
        
        if face_landmarks_ids is None:
            raise ValueError("face_landmarks_ids are required for MoldModel")
        
        # Nose landmarks 1
        # shape_landmarks = [168, 465, 419, 363, 134, 196, 245, 168] # Small nose top
        # shape_landmarks = [8, 417, 464, 453, 350, 266, 426, 391, 164, 165, 206, 36, 121, 233, 244, 193, 8] # Large Nose
        # shape_hole_landmarks = [195, 437, 371, 423, 326, 2, 97, 203, 142, 217, 195] 
        # shape_support_landmarks = [197, 437, 371, 423, 391, 164, 165, 203, 142, 217, 197]

        # Nose landmarks 2
        shape_landmarks = [6, 417, 464, 453, 350, 266, 322, 267, 37, 92, 206, 36, 121, 233, 244, 193, 6]
        shape_hole_landmarks = [195, 429, 279, 423, 326, 2, 97, 203, 49, 209, 195] 
        shape_support_landmarks = [197, 437, 371, 423, 391, 164, 165, 203, 142, 217, 197]

        # Mouth landmarks
        # shape_landmarks = [8, 417, 464, 453, 350, 266, 426, 436, 432, 422, 424, 418, 421, 200, 201, 194, 204, 202, 212, 216, 206, 36, 121, 233, 244, 193, 8] # Large Nose
        # shape_hole_landmarks = [197, 437, 371, 423, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57 , 186, 203, 142, 217, 197]
        # shape_support_landmarks = [197, 437, 371, 423, 410, 432, 422, 424, 418, 421, 200, 201, 194, 204, 202, 212, 186, 203, 142, 217, 197]



        # shape_hole_landmarks = [4, 281, 371, 423, 326, 2, 97, 203, 142, 51, 4] 
        # shape_strong_landmarks = [168, 417, 453, 266, 371, 437, 197, 217, 142, 36, 233, 193, 168] # Large Nose
        shape_strong_landmarks = [168, 417, 453, 266, 371, 437, 197, 217, 142, 36, 233, 193, 168] # Large Nose
        shape_tube_landmarks = [423, 393, 164, 167, 203] 
        # shape_support_landmarks = [142, 217, 197, 437, 371] 
        

        try:        
            face_mesh = create_pyvista_mesh(face_mesh)
            face_mesh = face_mesh.connectivity(largest=True)

            
            # Extract the nose surface and the hole surface
            shape_points_3d, _ = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_landmarks)  
            shape_hole_points_3d, _ = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_hole_landmarks) 
            shape_strong_points_3d, _ = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_strong_landmarks) 
            shape_support_points_3d, _ = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_support_landmarks) 
            shape_tube_points_3d, shape_tube_normals = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_tube_landmarks) 
            
            # Generate the surfaces using the extracted function
            nose_surface, _ = get_surface_within_area(face_mesh, shape_points_3d)
            nose_surface = remove_surface_within_area(nose_surface, shape_hole_points_3d)


            # Apply local deformation to nose_surface
            target_point = get_landmark(face_landmarks, face_landmarks_ids, 464) 
            nose_surface = deform_surface_at_point(nose_surface, center_point=target_point, radius=10.0, strength=4.0, inside_direction=True)

            target_point = get_landmark(face_landmarks, face_landmarks_ids, 244) 
            nose_surface = deform_surface_at_point(nose_surface, center_point=target_point, radius=10.0, strength=4.0, inside_direction=True)

            target_point = get_landmark(face_landmarks, face_landmarks_ids, 453) 
            nose_surface = deform_surface_at_point(nose_surface, center_point=target_point, radius=9.0, strength=4.0, inside_direction=True)

            target_point = get_landmark(face_landmarks, face_landmarks_ids, 233) 
            nose_surface = deform_surface_at_point(nose_surface, center_point=target_point, radius=9.0, strength=5.0, inside_direction=True)

            target_point = get_landmark(face_landmarks, face_landmarks_ids, 350) 
            nose_surface = deform_surface_at_point(nose_surface, center_point=target_point, radius=8.0, strength=5.0, inside_direction=True)

            target_point = get_landmark(face_landmarks, face_landmarks_ids, 121) 
            nose_surface = deform_surface_at_point(nose_surface, center_point=target_point, radius=8.0, strength=4.0, inside_direction=True)


            # nose_volume_mesh = thicken_mesh(nose_surface, 1, np.array([0, 0, 1]))
            nose_volume_mesh = thicken_mesh_vtk(nose_surface, 1.5)
            
            # nose_strong_surface, _ = get_surface_within_area(face_mesh, shape_strong_points_3d)
            # nose_strong_volume_mesh = thicken_mesh_vtk(nose_strong_surface, 3.0)

            # Create a flexible volume to interface around the mouth
            tube_surface, tube_top_points, shape_tube_start_section, shape_tube_end_section = extrude_half_tube_on_face_along_line(shape_tube_points_3d, shape_tube_normals, 5.0)
            # Interface at volume start
            pt = get_landmark(face_landmarks, face_landmarks_ids, 371)
            landmark_array = np.tile(pt, (shape_tube_start_section.shape[0], 1))
            shape_tube_start_surface = loft_between_line_points(shape_tube_start_section, landmark_array)
            # Interface at volume end
            pt = get_landmark(face_landmarks, face_landmarks_ids, 142)
            landmark_array = np.tile(pt, (shape_tube_end_section.shape[0], 1))
            shape_tube_end_surface = loft_between_line_points(shape_tube_end_section, landmark_array)
            
            tube_surface = tube_surface + shape_tube_start_surface + shape_tube_end_surface
            tube_volume = thicken_mesh_vtk(tube_surface, 1.5)

            # Create a continuous line with points from shape_support_points_3d and tube_top_points
            join_contact_line_points = np.vstack([shape_support_points_3d, tube_top_points])
            join_contact_line_points = np.vstack([join_contact_line_points, join_contact_line_points[0]])


            join_contact_line_points = shape_support_points_3d

            # Ensure points in join_contact_line_points are evenly distributed
            evenly_distributed_points = [join_contact_line_points[0]]
            for i in range(1, len(join_contact_line_points)):
                start_point = join_contact_line_points[i - 1]
                end_point = join_contact_line_points[i]
                distance = np.linalg.norm(end_point - start_point)
                num_segments = max(1, int(distance // 0.5))  # Adjust 0.5 for desired spacing
                for j in range(1, num_segments + 1):
                    new_point = start_point + (end_point - start_point) * (j / num_segments)
                    evenly_distributed_points.append(new_point)
            join_contact_line_points = np.array(evenly_distributed_points)

            

            pt = get_landmark(face_landmarks, face_landmarks_ids, 197)
           
            # join_contact_line_points = reorder_line_points(join_contact_line_points, pt)
            join_contact_line_points = reorder_line_points(join_contact_line_points, pt)
            # join_contact_line_points = np.vstack([join_contact_line_points, join_contact_line_points[0]])

            # Create an instance of N5Connector and generate the 3D object
            shape_builder = N5Connector()
            # shape_builder = FSAConnector()
            connector = shape_builder.create_3d_object()

            # Locate face landmarks defining horizontal and vertical alignments (x= vertical, y=horizontal)
            landmark_x1 = get_landmark(face_landmarks, face_landmarks_ids, 94)
            landmark_x2 = get_landmark(face_landmarks, face_landmarks_ids, 6)
            vector_x = landmark_x2 - landmark_x1
            vector_x /= np.linalg.norm(vector_x)

            landmark_y1 = get_landmark(face_landmarks, face_landmarks_ids, 117) # Coin paumette gauche
            landmark_y2 = get_landmark(face_landmarks, face_landmarks_ids, 346) # Coin paumette droite
            vector_y = landmark_y2 - landmark_y1
            vector_y /= np.linalg.norm(vector_y)

            # Locate connector landmarks defining vertical axis
            vector_shape_x1 = get_landmark(connector, target_id=1) # Center
            vector_shape_x2 = get_landmark(connector, target_id=5) # Top
            vector_shape_x = vector_shape_x2 - vector_shape_x1
            vector_shape_x /= np.linalg.norm(vector_shape_x)

            # Compute the rotation matrix to align the direction vectors
            rotation_1 = compute_rotation_between_vectors(vector_shape_x, vector_x)
            connector = rotate_shape_and_landmarks(connector, rotation_1)


            # Locate connector landmarks defining horizontal axis
            vector_shape_y1 = get_landmark(connector, target_id=1) # Center
            vector_shape_y2 = get_landmark(connector, target_id=3) # Right
            vector_shape_y = vector_shape_y2 - vector_shape_y1
            vector_shape_y /= np.linalg.norm(vector_shape_y)
            
            rotation_2 = compute_rotation_between_vectors(vector_shape_y, vector_y)
            connector = rotate_shape_and_landmarks(connector, rotation_2)


            center_connector = get_landmark(connector, target_id=1) # Center

            # Translate above the nose tip
            landmark_z = get_landmark(face_landmarks, face_landmarks_ids, 4)
            translation = landmark_z - center_connector
            connector = translate_shape_and_landmarks(connector, translation)

            # Translate above the mouth tip
            # landmark_z1 = get_landmark(face_landmarks, face_landmarks_ids, 164)
            # translation = landmark_z1 - center_connector

            # landmark_z21 = get_landmark(face_landmarks, face_landmarks_ids, 5)
            # landmark_z22 = get_landmark(face_landmarks, face_landmarks_ids, 4)
            # z_height = max(landmark_z21[2], landmark_z22[2])
            # translation_z = z_height - landmark_z1[2] + 20
            # translation += np.array([0, 0, translation_z])

            # connector = translate_shape_and_landmarks(connector, translation)

            # Build the shape to join nose surface with connector
            p0 = connector.landmarks[1] # Center Bottom
            px = connector.landmarks[7] # Inside Right Bottom
            py = connector.landmarks[9] # Inside Top Bottom
            # pxi = connector.landmarks[9] # Inside Top

            # radius = np.linalg.norm(px - p0)

            center = p0
            x_vec = px - p0
            y_vec = py - p0

            # normal = np.cross(px - p0, py - p0)
            # normal /= np.linalg.norm(normal)

            # ref_vec = px - p0
            # ref_vec /= np.linalg.norm(ref_vec)
            # side_vec = np.cross(normal, ref_vec)
            # side_vec /= np.linalg.norm(side_vec)

            
            # Generate circle points with the same number of points as shape_support_points_3d
            num_points = len(join_contact_line_points)-1
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            circle_points = []
            for angle in angles:
                # circle_points.append(center + radius*(ref_vec*np.cos(angle) + side_vec*np.sin(-angle)))
                circle_points.append(center + x_vec*np.cos(angle+np.pi/2) + y_vec*np.sin(angle+np.pi/2))
            circle_points = np.array(circle_points)
            # Reverse the order of points to go in the opposite direction
            circle_points = circle_points[::-1]
            # Append first point to the end to close the loop
            circle_points = np.vstack([circle_points, circle_points[0]])

            join_surface = loft_between_line_points(circle_points, join_contact_line_points)

            join_volume = thicken_mesh_vtk(join_surface, 1.5)



            # Add the surface to the plotter for visualization
            # Optionally plot for debugging
            plotter = pv.Plotter()
            plotter.add_mesh(face_mesh, color='lightgrey', opacity=0.2, show_edges=True)
            plotter.add_mesh(nose_volume_mesh, color='blue', opacity=0.4, show_edges=True)
            # plotter.add_mesh(nose_strong_volume_mesh, color='blue', opacity=0.4, show_edges=True)
            plotter.add_mesh(connector, color='green', opacity=0.4, show_edges=True)
            plotter.add_mesh(join_volume, color='yellow', opacity=0.4, show_edges=True)
            # plotter.add_mesh(tube_volume, color='orange', opacity=0.5, show_edges=True)
            # plotter.add_mesh(pv.PolyData(circle_points), color='red', point_size=3, render_points_as_spheres=True)
            # # Add point IDs for join_contact_line_points
            # for i, point in enumerate(circle_points):
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
            # plotter.add_mesh(pv.PolyData(shape_support_points_3d), color='blue', point_size=7, render_points_as_spheres=True)
            # plotter.add_mesh(pv.PolyData(tube_top_points), color='green', point_size=7, render_points_as_spheres=True)
            # plotter.add_mesh(pv.PolyData(shape_tube_points_3d), color='orange', point_size=5, render_points_as_spheres=True)
            # plotter.add_mesh(pv.PolyData(shape_tube_landmarks), color='red', point_size=5, render_points_as_spheres=True)
            # plotter.add_mesh(pv.Sphere(center=connector_point_1, radius=3.0), color='orange', render_points_as_spheres=True)
            # plotter.add_mesh(pv.Sphere(center=connector_point_3, radius=3.0), color='yellow', render_points_as_spheres=True)
            # plotter.add_mesh(pv.Sphere(center=connector_point_5, radius=3.0), color='red', render_points_as_spheres=True)
            plotter.show()

            # Merge nose_volume_mesh and nose_strong_volume_mesh
            combined_volume_mesh = (
                nose_volume_mesh  
                # + nose_strong_volume_mesh
                + join_volume
                # + tube_volume
                + connector
            )

            # Convert UnstructuredGrid to PolyData by extracting the surface
            combined_volume_mesh = combined_volume_mesh.extract_surface()

            # Clean the form
            # face_volume = extrude_mesh(face_mesh, 10, np.array([0, 0, -1]))
            # combined_volume_mesh = combined_volume_mesh.triangulate()
            # face_volume = face_volume.triangulate()

            # combined_volume_mesh = combined_volume_mesh.boolean_difference(face_volume)
           
            # plotter = pv.Plotter()
            # plotter.add_mesh(face_volume, color='lightgrey', opacity=0.2, show_edges=True)
            # plotter.add_mesh(combined_volume_mesh, color='blue', opacity=0.4, show_edges=True)
            # plotter.show()

            # combined_volume_mesh = clean_and_smooth(combined_volume_mesh, smooth_iter=100, clean_tol=1e-3)


            return combined_volume_mesh, ""

        except Exception as e:
            return None, f"An error occurred during mold moddeling: {e}"
