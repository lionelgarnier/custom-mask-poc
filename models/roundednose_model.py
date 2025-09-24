import numpy as np
import pyvista as pv
from models.base_model import Face3DObjectModel
from utils import (create_pyvista_mesh, repair_mesh, clean_and_smooth, thicken_mesh_vtk,
                   thicken_mesh_with_variable_thickness, thicken_combined_surface_with_zones, 
                   fix_surface_normals_at_interface, remove_surface_within_area, deform_surface_at_point,
                   compute_rotation_between_vectors, extrude_mesh, reorder_line_points, 
                   rotate_shape_and_landmarks, translate_shape_and_landmarks,
                   get_landmark, loft_between_line_points, extrude_tube_on_face_along_line,
                  extract_line_from_landmarks, get_tangent_points, advanced_smooth_mesh, smooth_line_points,
                  get_surface_within_area, orient_normals_z_up, apply_preload_to_contact_surface)
from shapes.n5_connector import N5Connector

class RoundedNoseModel(Face3DObjectModel):
    def create_3d_object(self, output_path, **kwargs):        
        # Get required parameters
        face_mesh = kwargs.get('face_mesh', None)
        face_landmarks = np.array(kwargs.get('face_landmarks'))
        face_landmarks_ids = np.array(kwargs.get('face_landmarks_ids'))
        smooth_intensity = kwargs.get('smooth_intensity', 'light')  # Plus conservateur par défaut

        
        if face_mesh is None:
            raise ValueError("face_mesh is required for Extrusion Model")
        
        if face_landmarks is None:
            raise ValueError("face_landmarks are required for Extrusion Model")
        
        if face_landmarks_ids is None:
            raise ValueError("face_landmarks_ids are required for Extrusion Model")
        
        face_mesh = create_pyvista_mesh(face_mesh)
        face_mesh = face_mesh.connectivity(extraction_mode='largest')
        
        # Nose model
        # shape_landmarks = [197, 437, 371, 423, 391, 393, 164, 167, 165, 203, 142, 217, 197] 
        # shape_landmarks = [6, 437, 371, 423, 391, 393, 164, 167, 165, 203, 142, 217, 6] # Test 1 Face Mika
        shape_landmarks = [6, 351, 412, 277, 371, 423, 391, 393, 164, 167, 165, 203, 142, 47, 188, 122, 6] 
        shape_hole_landmarks = [195, 456, 429, 279, 331, 327, 326, 2, 97, 98, 129, 49, 209, 236, 195]


        # Mouth model
        # shape_landmarks = [197, 437, 371, 266, 425, 427, 434, 430, 431, 418, 421, 200, 201, 194, 211, 210, 214, 207, 205, 36, 142, 217, 197] 
        # shape_landmarks = [8, 417, 464, 453, 350, 266, 426, 436, 432, 422, 424, 418, 421, 200, 201, 194, 204, 202, 212, 216, 206, 36, 121, 233, 244, 193, 8] # Large Nose
        # shape_hole_landmarks = [197, 437, 371, 423, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57 , 186, 203, 142, 217, 197]
        

        # air_landmarks = [47, 114, 188, 122, 6, 351, 412, 343, 277] 
        
        
        # Tubular contact with face
        shape_points_3d, shape_normals = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_landmarks) 

        shape_hole_points_3d, shape_hole_normals = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_hole_landmarks) 


        contact_surface, _ = get_surface_within_area(face_mesh, shape_points_3d)
        contact_surface = remove_surface_within_area(contact_surface, shape_hole_points_3d)
        # Apply preload by offsetting the inner region toward the face
        contact_surface = apply_preload_to_contact_surface(
            contact_surface,
            face_mesh,
            shape_hole_points_3d,
            shape_points_3d,
            preload_mm=1
        )
        # Force normals to +Z before thickening for consistent extrusion direction
        contact_surface = orient_normals_z_up(contact_surface)


        tube_surface, tube_top_points, cross_sections = extrude_tube_on_face_along_line(shape_points_3d, shape_normals, 3.0)
        
        
        # Create an instance of N5Connector and generate the 3D object
        shape_builder = N5Connector()
        # shape_builder = FSAConnector()
        connector = shape_builder.create_3d_object()

        # Locate face landmarks defining horizontal and vertical alignments (x= vertical, y=horizontal)
        landmark_x1 = get_landmark(face_landmarks, face_landmarks_ids, 164)
        landmark_x2 = get_landmark(face_landmarks, face_landmarks_ids, 6)
        vector_x = landmark_x2 - landmark_x1
        vector_x /= np.linalg.norm(vector_x)

        landmark_y1 = get_landmark(face_landmarks, face_landmarks_ids, 142) # Coin paumette gauche
        landmark_y2 = get_landmark(face_landmarks, face_landmarks_ids, 371) # Coin paumette droite
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

        # # Translate above the nose tip
        landmark_z = get_landmark(face_landmarks, face_landmarks_ids, 4)
        translation = landmark_z - center_connector
        translation += np.array([0, 0, 5]) # Add a small offset to avoid intersection with the face and inrease comfort
        connector = translate_shape_and_landmarks(connector, translation)

        # Translate above the mouth tip
        # landmark_z1 = get_landmark(face_landmarks, face_landmarks_ids, 164)
        # translation = landmark_z1 - center_connector

        # landmark_z21 = get_landmark(face_landmarks, face_landmarks_ids, 5)
        # landmark_z22 = get_landmark(face_landmarks, face_landmarks_ids, 4)
        # z_height = max(landmark_z21[2], landmark_z22[2])
        # translation_z = z_height - landmark_z1[2] + 15
        # translation += np.array([0, 0, translation_z])
        # connector = translate_shape_and_landmarks(connector, translation)

        
        # Build the shape to join nose surface with connector
        p0 = connector.landmarks[1] # Center Bottom
        px = connector.landmarks[3] # Outside Right Bottom
        py = connector.landmarks[5] # Outside Top Bottom
        pz = connector.landmarks[2] # Center Top

        # radius = np.linalg.norm(px - p0)

        center = p0
        x_vec = px - p0
        y_vec = py - p0

        # Generate circle points with the same number of points as shape_points_3d
        num_points = len(shape_points_3d)-1
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        circle_points = []
        for angle in angles:
            # circle_points.append(center + radius*(ref_vec*np.cos(angle) + side_vec*np.sin(-angle)))
            circle_points.append(center + x_vec*np.cos(angle+np.pi/2) + y_vec*np.sin(angle+np.pi/2))
        circle_points = np.array(circle_points)
        # Reverse the order of points to go in the opposite direction
        circle_points = circle_points[::-1]


        tangent_points = get_tangent_points(cross_sections, circle_points)

        # Create a surface using the points
        tangent_surface = loft_between_line_points(circle_points, tangent_points)

        merged_surface = loft_between_line_points(tube_top_points, tangent_points)
        # tube_surface = tube_surface.clip_surface(merged_surface, invert=True)

        # Thicken the surface with variable thickness
        # volume = thicken_combined_surface_with_zones(
        #     tube_surface=tube_surface, 
        #     tangent_surface=tangent_surface,
        #     tube_thickness_start=2.0,        # Thickness of the tangent surface
        #     tube_thickness_end=2.0,       # Thickness of the tube surface  
        #     # tube_thickness_end=0.5,       # Thickness of the tube surface  
        #     tube_top_points=tube_top_points, 
        #     circle_points=circle_points
        # )
        
        tube_surface = thicken_mesh_vtk(tube_surface, 2.0, True)   
        tangent_surface = thicken_mesh_vtk(tangent_surface, 3.0, True)  
        contact_surface = thicken_mesh_vtk(contact_surface, 0.5)  
        # contact_surface.translate((0, 0, -2), inplace=True)
        volume = tube_surface + tangent_surface + contact_surface
        # Add the connector
        volume = volume + connector

        # Extract surface and repair holes/gaps between volumes
        volume = volume.extract_surface()
        
        # Clean and merge nearby vertices to close small gaps
        volume = volume.clean(tolerance=0.1)
        
        # Fill any remaining holes (especially at volume junctions)
        # Use a larger hole_size to catch the connection gap
        volume = volume.fill_holes(hole_size=5.0)
        
        # Final repair avec lissage conservateur pour ne pas dégrader la géométrie
        volume = repair_mesh(volume, smooth_intensity="none")  # Pas de lissage agressif dans repair
        
        # Lissage final optionnel et léger pour réduire les facettes SEULEMENT si demandé
        if smooth_intensity == "medium":
            # Lissage équilibré
            volume = advanced_smooth_mesh(volume, intensity="light")
        elif smooth_intensity == "strong":
            # Lissage plus fort mais encore conservateur
            volume = advanced_smooth_mesh(volume, intensity="medium")
        elif smooth_intensity == "very_strong":
            # Lissage maximum mais encore contrôlé
            volume = advanced_smooth_mesh(volume, intensity="strong")
        # Si smooth_intensity == "none" ou "light", on ne fait rien d'autre

        # surface = tube_surface + tangent_surface
        # volume = thicken_mesh_vtk(surface, 2)
        # tangent_volume = thicken_mesh_vtk(tangent_surface, 1.5)

        # volume = volume + connector #+ tangent_volume

        # volume = volume.extract_surface()
        # volume = repair_mesh(volume)

        # Add the surface to the plotter for visualization
        # Optionally plot for debugging
        # plotter = pv.Plotter()
        # plotter.add_mesh(face_mesh, color='lightgrey', opacity=0.2, show_edges=True)
        # plotter.add_mesh(tube_surface, color='blue', opacity=0.4, show_edges=True)
        # plotter.add_mesh(tangent_surface, color='orange', opacity=0.4, show_edges=True)
        # plotter.add_mesh(connector, color='green', opacity=0.4, show_edges=True)
        # plotter.add_mesh(air_surface, color='red', opacity=0.3, show_edges=True)
        # plotter.add_mesh(merged_polydata, color='red', line_width=2)
        # plotter.add_mesh(join_surface, color='yellow', opacity=0.4, show_edges=True)
        # plotter.add_mesh(tangent_volume, color='yellow', opacity=0.4, show_edges=True)
        # plotter.add_mesh(tube_volume, color='orange', opacity=0.5, show_edges=True)
        # plotter.add_mesh(contact_surface, color='red', opacity=0.5, show_edges=True)
        # plotter.add_mesh(volume, color='orange', opacity=0.5, show_edges=True)
        # plotter.add_mesh(pv.PolyData(shape_hole_points_3d), color='red', point_size=3, render_points_as_spheres=True)
        # Add point IDs for join_contact_line_points
        # for i, point in enumerate(shape_hole_landmarks):
        #     plotter.add_point_labels(
        #         point, 
        #         [f"{i}"], 
        #         font_size=10, 
        #         text_color='black',
        #         point_color='red', 
        #         point_size=5, 
        #         render_points_as_spheres=True,
        #         shape_opacity=0.7
        #     )    
        # plotter.show()


        return volume, ""