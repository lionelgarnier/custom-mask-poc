import numpy as np
import pyvista as pv
from models.base_model import Face3DObjectModel
from utils import (create_pyvista_mesh, thicken_mesh, clean_and_smooth, thicken_mesh_vtk,
                   get_surface_within_area, remove_surface_within_area, deform_surface_at_point,
                  compute_rotation_between_vectors, extrude_mesh, reorder_line_points, 
                  rotate_shape_and_landmarks, translate_shape_and_landmarks,
                  get_landmark, loft_between_line_points, extrude_tube_on_face_along_line,
                  extract_line_from_landmarks, get_tangent_points)
from shapes.n5_connector import N5Connector
from scipy.interpolate import interp1d
import math

class AlongNoseModel(Face3DObjectModel):
    def create_3d_object(self, output_path, **kwargs):        
        # Get required parameters
        face_mesh = kwargs.get('face_mesh', None)
        face_landmarks = np.array(kwargs.get('face_landmarks'))
        face_landmarks_ids = np.array(kwargs.get('face_landmarks_ids'))

        
        if face_mesh is None:
            raise ValueError("face_mesh is required for Extrusuin Model")
        
        if face_landmarks is None:
            raise ValueError("face_landmarks are required for Extrusuin Model")
        
        if face_landmarks_ids is None:
            raise ValueError("face_landmarks_ids are required for Extrusuin Model")
        
        face_mesh = create_pyvista_mesh(face_mesh)
        face_mesh = face_mesh.connectivity(extraction_mode='largest')
        
        # Along nose model
        shape_landmarks = [10,338,336,417,465,399,363,279,423,391,164,165,203,115,131,134,174,245,193,107,109,10] 
        shape_hole_landmarks = [1,274,438,327,326,2,97,98,218,44,1] 
        shape_support_landmarks = [1,274,438,327,393,164,167,98,218,44,1] 
        shape_tube_landmarks = [151,168, 1] 
                
        # Tubular contact with face
        shape_points_3d, _ = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_landmarks) 
        shape_hole_points_3d, _ = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_hole_landmarks) 
        shape_support_points_3d, _ = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_support_landmarks) 
        shape_tube_points_3d, shape_tube_normals = extract_line_from_landmarks(face_mesh, face_landmarks, face_landmarks_ids, shape_tube_landmarks) 
        
        n_support  = len(shape_support_points_3d)

        # Project all shape_tube_normals on the z-axis
        shape_tube_normals = np.array(shape_tube_normals)
        shape_tube_normals[:, 0] = 0  # Set x-component to 0
        shape_tube_normals[:, 1] = 0  # Set y-component to 0
        shape_tube_normals = shape_tube_normals / np.linalg.norm(shape_tube_normals, axis=1, keepdims=True)  # Normalize
        
        nose_surface, _ = get_surface_within_area(face_mesh, shape_points_3d)
        nose_surface = remove_surface_within_area(nose_surface, shape_hole_points_3d)
                
        tube_surface, _, cross_sections = extrude_tube_on_face_along_line(shape_tube_points_3d, shape_tube_normals, 6.0, n_support, y_ratio=2)
        
        tube_surface2, _, cross_secti_ons2 = extrude_tube_on_face_along_line(shape_tube_points_3d, shape_tube_normals, 6.0, y_ratio=0.9)
        


        last_cross_section = cross_sections[-1]
        # Calculate center of the last cross section
        # center = np.mean(last_cross_section, axis=0)

        # # Find average radius
        # cross_section_centered = last_cross_section - center
        # radius = np.mean(np.linalg.norm(cross_section_centered, axis=1))

        # # Use PCA to find the principal axes
        # cov = np.cov(cross_section_centered.T)
        # evals, evecs = np.linalg.eigh(cov)

        # # Sort eigenvalues and eigenvectors in descending order
        # idx = evals.argsort()[::-1]
        # evecs = evecs[:, idx]

        # # x_vec and y_vec are the principal directions, scaled by radius
        # x_vec = evecs[:, 0] * radius
        # y_vec = evecs[:, 1] * radius
        # Nombre de points initiaux
        n_cross = len(last_cross_section)
        n_hole  = len(shape_support_points_3d)

        # Plus grand commun diviseur
        g = math.gcd(n_cross, n_hole)

        # Combien de subdivisions par segment
        sub_div_cross = n_hole // g
        sub_div_hole  = n_cross // g

        def subdivide_loop(pts, sub_div):
            new_pts = []
            n = len(pts)
            for i in range(n):
                p1 = pts[i]
                p2 = pts[(i+1) % n]  # boucle fermée
                delta = p2 - p1
                for k in range(sub_div):
                    new_pts.append(p1 + (k / sub_div) * delta)
            return np.array(new_pts)

        # On remplace les deux boucles par leurs versions subdivisées
        circle_points         = subdivide_loop(last_cross_section, sub_div_cross)
        shape_support_points_3d  = subdivide_loop(shape_support_points_3d,  sub_div_hole)




        # # Find the 3D position of landmark ID 1
        # landmark_1_position = get_landmark(face_landmarks, face_landmarks_ids, 1)

        # # Calculate distances to landmark 1 and find closest point
        # distances = np.array([np.linalg.norm(point - landmark_1_position) for point in circle_points])
        # closest_idx = np.argmin(distances)

        # # Rotate the circle points so the closest point to landmark 1 is first
        # circle_points = np.roll(circle_points, -closest_idx, axis=0)

        # Reverse the order of points to go in the opposite direction
        # circle_points = circle_points[::-1]
        # Create a surface using the points
        tangent_surface = loft_between_line_points(circle_points, shape_support_points_3d)



        tube_surface = tube_surface + tangent_surface 
        tube_surface = tube_surface.extract_surface()

        tube_surface2 = tube_surface2.extract_surface()
        tube_volume2 = thicken_mesh_vtk(tube_surface2, 1.5, True)

        volume = thicken_mesh_vtk(nose_surface, 1.5)
        tube_volume = thicken_mesh_vtk(tube_surface, 1.5, True)

        # Remove from tube_volume anything that is below face_mesh
        tube_volume = tube_volume.clip_surface(face_mesh, invert=False)

        # Add the surface to the plotter for visualization
        # Optionally plot for debugging
        plotter = pv.Plotter()
        plotter.add_mesh(face_mesh, color='lightgrey', opacity=0.2, show_edges=True)
        plotter.add_mesh(volume, color='blue', opacity=0.4, show_edges=True)
        plotter.add_mesh(tube_volume, color='red', opacity=0.4, show_edges=True)
        plotter.add_mesh(tube_volume2, color='orange', opacity=0.4, show_edges=True)
        # for i, point in enumerate(circle_points):
        #     plotter.add_point_labels(
        #         point, 
        #         [f"{i}"], 
        #         font_size=10, 
        #         text_color='black',
        #         point_color='orange', 
        #         point_size=5, 
        #         render_points_as_spheres=True,
        #         shape_opacity=0.7
        #     )    
        plotter.show()


        volume = volume + tube_volume + tube_volume2
        volume = volume.extract_surface()

        return volume, ""