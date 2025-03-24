from models.base_model import Face3DObjectModel
import pyvista as pv
import numpy as np
from pyvista.core.pointset import PolyData
import open3d as o3d

class MoldModel(Face3DObjectModel):
    def create_3d_object(self, output_path, **kwargs):
        # Get required parameters
        face_mesh = kwargs.get('face_mesh', None)
        line_points_3d = np.array(kwargs.get('line_points_3d'))
        thickness = kwargs.get('thickness', 3.0)
        
        if face_mesh is None:
            raise ValueError("face_mesh is required for MoldModel")
            
        # Create a polyline from the points
        polyline = pv.PolyData(line_points_3d)
        n_points = len(line_points_3d)
        lines = np.hstack((n_points, np.arange(n_points)))
        polyline.lines = np.array([lines])
        
        # Extract the 2D polygon outline in the XY plane
        polygon_points = line_points_3d[:, :2]  # Keep only X and Y coordinates
        
        # Extract the portion of the face mesh that is inside the polygon
        # Convert face mesh to PyVista if it's not already
        if isinstance(face_mesh, o3d.geometry.TriangleMesh):
            # Handle Open3D mesh
            vertices = np.asarray(face_mesh.vertices)
            triangles = np.asarray(face_mesh.triangles)
            # Convert triangles to the format PyVista expects
            faces = np.column_stack((np.ones(len(triangles), dtype=np.int64) * 3, triangles)).flatten()
            face_mesh_pv = pv.PolyData(vertices, faces)
        elif not isinstance(face_mesh, PolyData):
            # Handle dictionary-like mesh
            try:
                vertices = np.array(face_mesh['vertices'])
                faces = np.array(face_mesh['faces'])
                face_mesh_pv = pv.PolyData(vertices, faces)
            except (TypeError, KeyError):
                raise ValueError("face_mesh must be a PyVista PolyData, Open3D TriangleMesh, or a dictionary with 'vertices' and 'faces' keys")
        else:
            face_mesh_pv = face_mesh
            
        # Create a selection based on points inside the 2D polygon
        points = face_mesh_pv.points
        
        # Extract points and faces inside the polygon using boolean operations
        # Create a 2D polygon and extrude it to get a selection volume
        polygon = pv.PolyData(np.hstack([polygon_points, np.zeros((len(polygon_points), 1))]))
        polygon.lines = polyline.lines
        
        # Create a surface from the polygon
        surf = polygon.delaunay_2d()
        
        # Create a selection volume by extruding far in z-direction (both ways)
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        z_range = z_max - z_min
        extrusion = surf.extrude((0, 0, 2*z_range), capping=True)
        extrusion.translate((0, 0, z_min - z_range/2), inplace=True)
        
        # Select the part of the face mesh inside the extrusion
        surface = face_mesh_pv.clip_surface(extrusion)
        
        # Ensure the surface is triangulated and has normals computed
        if surface.n_cells == 0:  
            surface = surface.triangulate()
        if 'Normals' not in surface.array_names:
            surface.compute_normals(inplace=True)

        # Extract the outer surface
        surface = surface.extract_surface()

        # Clean the surface to remove small disconnected components
        surface = surface.clean(tolerance=0.01, inplace=True)

        # Uniform extrusion: create an offset copy by moving each point along its normal by 'thickness'
        offset_points = surface.points + surface.point_normals * thickness

        num_points = surface.n_points
        # Combine original and offset points (first half: original; second half: offset surface)
        combined_points = np.vstack([surface.points, offset_points])

        # Duplicate the original faces for the offset surface (update indices by adding num_points)
        faces = surface.faces.reshape(-1, 4)  # assuming triangles (3 vertices + count per face)
        offset_faces = faces.copy()
        offset_faces[:, 1:4] += num_points

        # First, incorporate the boundary points into the mesh
        boundary_points = line_points_3d.copy()
        # Create offset boundary points by moving along normals
        # We need to compute normals for boundary points
        # This is a simplified approach - you might need a more robust method
        boundary_normals = np.zeros_like(boundary_points)
        for i, point in enumerate(boundary_points):
            # Find closest point in surface and use its normal
            dists = np.sum((surface.points - point)**2, axis=1)
            closest_idx = np.argmin(dists)
            boundary_normals[i] = surface.point_normals[closest_idx]

        # Generate offset boundary points
        offset_boundary_points = boundary_points + boundary_normals * thickness

        # Track where these points will be in the combined array
        base_idx = num_points  # Starting index after surface points
        # Update combined_points to include both sets of boundary points
        combined_points = np.vstack([surface.points, offset_points, 
                                    boundary_points, offset_boundary_points])

        # Now create side faces with the correct indices
        side_faces = []
        n_boundary = len(boundary_points)
        for i in range(n_boundary):
            j = (i + 1) % n_boundary
            # Original boundary points start at base_idx
            i_orig = base_idx*2 + i
            j_orig = base_idx*2 + j
            # Offset boundary points start at base_idx + n_boundary
            i_offset = base_idx*2 + n_boundary + i
            j_offset = base_idx*2 + n_boundary + j
            # Define quad: [count, point1, point2, point3, point4]
            side_faces.extend([4, i_orig, j_orig, j_offset, i_offset])

        # Combine all faces: original surface, offset surface, and side faces
        all_faces = np.hstack([faces.flatten(), offset_faces.flatten(), np.array(side_faces)])
        extruded = pv.PolyData(combined_points, all_faces)

        # Visualize the extruded mold and the original line points for debugging
        plotter = pv.Plotter()
        # plotter.add_mesh(extruded, color='lightblue', opacity=0.7, show_edges=True)
        # plotter.add_points(line_points_3d, color='red', point_size=10)
        plotter.add_points(solid, color='green', opacity=0.7, show_edges=True)
        plotter.show()

        # Save the mold to an STL file
        extruded.save(output_path)
        return extruded