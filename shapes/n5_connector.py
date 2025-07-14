import pyvista as pv
import numpy as np
import vtk
from vtkmodules.vtkFiltersGeneral import vtkBooleanOperationPolyDataFilter
from pymeshfix import MeshFix
import trimesh
from utils import get_landmark

class N5Connector:
    def create_3d_object(self, landmarks=None, radius=20, height=10, output_path=None, **kwargs):
        if landmarks:
            # Position based on landmarks (ex. center at nose tip)
            center = get_landmark(landmarks, target_id=4)  # Example: nose tip
            connector = trimesh.creation.cylinder(radius=radius, height=height)
            connector.apply_translation(center + [0, 0, 5])  # Offset
        else:
            # Fallback to original
            # Tube parameters
            height_mm = 5.0
            outside_radius = 34.0 / 2.0
            inside_radius = 29.0 / 2.0

            ring = (
                pv.Disc(inner=inside_radius,
                        outer=outside_radius,
                        c_res=128, 
                        r_res=1)
                .triangulate()
                .clean(tolerance=1e-6)
            )

            cylinder = (
                ring.extrude([0, 0, height_mm], capping=True)
                .triangulate()
                .clean(tolerance=1e-6)
            )

            # Add a half torus to the connector to avoid supports during 3d printing
            major_R = (outside_radius + inside_radius) / 2.0
            minor_r = (outside_radius - inside_radius) / 2.0
            half_torus = (
                pv.ParametricTorus(ringradius=major_R,
                                crosssectionradius=minor_r,
                                u_res=128, v_res=64)
                .triangulate()
                .clean(tolerance=1e-6)
                .clip_closed_surface('-z')      # bottom half, already capped
            )


            ring_height = 0
            notch_height = height_mm + ring_height
            cylinder = self.create_notch(cylinder, outside_radius, notch_height,  70, 2.0, 1.0, -ring_height)
            cylinder = self.create_notch(cylinder, outside_radius, notch_height, 115, 2.0, 1.0, -ring_height)
            cylinder = self.create_notch(cylinder, outside_radius, notch_height, 245, 2.0, 1.0, -ring_height)
            cylinder = self.create_notch(cylinder, outside_radius, notch_height, 290, 2.0, 1.0, -ring_height)

            ring_height = outside_radius - inside_radius
            notch_height = ring_height
            half_torus = self.create_notch(half_torus, outside_radius, notch_height,  70, 2.0, 1.0, -ring_height)
            half_torus = self.create_notch(half_torus, outside_radius, notch_height, 115, 2.0, 1.0, -ring_height)
            half_torus = self.create_notch(half_torus, outside_radius, notch_height, 245, 2.0, 1.0, -ring_height)
            half_torus = self.create_notch(half_torus, outside_radius, notch_height, 290, 2.0, 1.0, -ring_height)

            shape = cylinder + half_torus

            landmarks = {
                1: (0, 0, 0), # Center Bottom
                2: (0, 0, height_mm), # Center Top
                3: (outside_radius, 0, 0), # Outside Right Bottom
                4: (outside_radius, 0, height_mm), # Outside Right Top
                5: (0, outside_radius, 0),  # Outside Up Bottom
                6: (0, outside_radius, height_mm), # Outside Up Top
                7: (inside_radius, 0, 0), # Inside Right Bottom
                8: (inside_radius, 0, height_mm), # Inside Right Top
                9: (0, inside_radius, 0), # Inside Up Bottom
                10: (0, inside_radius, height_mm) # Inside Up Top
            }
            shape.landmarks = landmarks
            
            # Save the 3D connector to an STL file if path is provided
            if output_path:
                shape.save(output_path)
                
            return shape

    def create_notch(self, mesh, mesh_outside_radius, mesh_height, notch_angle, notch_width, notch_depth, offset=0.0):
        
        # Adjust for the coordinate system with clockwise rotation
        notch_angle = - notch_angle + 90

        # Calculate the angle for an arc with length notch_width on a circle with radius outside_radius
        notch_radius = mesh_outside_radius - notch_depth
        notch_width_angle = notch_width * 360 / (2 * np.pi * mesh_outside_radius)

        # Create the notch as a shape between mesh_outside_radius and notch_radius
        # Create points for the 2D notch shape (in x-y plane)
        arc_resolution = 20
        theta_start = (notch_angle - notch_width_angle) * np.pi/180
        theta_end = (notch_angle + notch_width_angle) * np.pi/180
        theta_range = np.linspace(theta_start, theta_end, arc_resolution)

        # Inner arc points
        inner_arc = np.zeros((arc_resolution, 3))
        for i, theta in enumerate(theta_range):
            inner_arc[i] = [notch_radius * np.cos(theta), notch_radius * np.sin(theta), -1]

        # Outer arc points (in reverse order to create a proper polygon)
        outer_arc = np.zeros((arc_resolution, 3))
        for i, theta in enumerate(reversed(theta_range)):
            outer_arc[i] = [mesh_outside_radius*1.1 * np.cos(theta), mesh_outside_radius*1.1 * np.sin(theta), -1]

        # Combine points to form a closed polygon
        notch_points = np.vstack([inner_arc, outer_arc])

        # Create a 2D polygon
        notch_polygon = pv.PolyData(notch_points).delaunay_2d()

        # Extrude to create the 3D notch shape
        notch_box = notch_polygon.extrude((0, 0, mesh_height+2), capping=True).extract_surface().triangulate().clean(tolerance=1e-6)
        notch_box = notch_box.translate((0, 0, offset))

        # Cut the notch from the outer cylinder
        mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-6)

        fix = MeshFix(mesh)
        fix.repair()
        mesh = pv.wrap(fix.mesh)

        fix = MeshFix(notch_box)
        fix.repair(joincomp=True)
        notch_box = pv.wrap(fix.mesh)

        # Perform boolean difference using VTK
        boolean_filter = vtkBooleanOperationPolyDataFilter()
        boolean_filter.SetOperationToDifference()
        boolean_filter.SetInputData(0, mesh)
        boolean_filter.SetInputData(1, notch_box)
        boolean_filter.Update()

        # Wrap the result back into PyVista and clean up
        mesh = pv.wrap(boolean_filter.GetOutput()).triangulate().clean(tolerance=1e-6)


        return mesh
    
