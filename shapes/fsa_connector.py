import pyvista as pv
import numpy as np

class FSAConnector:
    def create_3d_object(self, output_path=None, **kwargs):
        # Tube parameters
        height_mm = 7.0
        outside_radius = 44.0 / 2.0
        inside_radius = 36.0 / 2.0
        
        # Create the outer cylinder
        outer_cylinder = pv.Cylinder(
            center=(0, 0, height_mm / 2.0),
            direction=(0, 0, 1),
            radius=outside_radius,
            height=height_mm
        ).triangulate()
        

        # Create the inner cylinder
        inner_cylinder = pv.Cylinder(
            center=(0, 0, height_mm / 2.0),
            direction=(0, 0, 1),
            radius=inside_radius,
            height=height_mm + 2.0
        ).triangulate()
        
        # Boolean difference to hollow out the inside
        shape = outer_cylinder.boolean_difference(inner_cylinder)
        
        # shape = self.create_notch(shape, outside_radius, height_mm, 70, 2.0, 1.0)
        # shape = self.create_notch(shape, outside_radius, height_mm, 115, 2.0, 1.0)
        # shape = self.create_notch(shape, outside_radius, height_mm, 245, 2.0, 1.0)
        # shape = self.create_notch(shape, outside_radius, height_mm, 290, 2.0, 1.0)

        # plotter = pv.Plotter()
        # plotter.add_mesh(shape, color='green', opacity=0.4, show_edges=True)
        # plotter.show()

        landmarks = {
            1: (0, 0, height_mm), # Center Bottom
            2: (0, 0, 0), # Center Top
            3: (inside_radius, 0, height_mm), # Outside Right Bottom
            4: (inside_radius, 0, 0), # Outside Right Top
            5: (0, inside_radius, height_mm),  # Outside Up Bottom
            6: (0, inside_radius, 0), # Outside Up Top
            7: (outside_radius, 0, height_mm), # Inside Right Bottom
            8: (outside_radius, 0, 0), # Inside Right Top
            9: (0, outside_radius, height_mm), # Inside Up Bottom
            10: (0, outside_radius, 0) # Inside Up Top
        }
        shape.landmarks = landmarks
        
        # Save the 3D connector to an STL file if path is provided
        if output_path:
            shape.save(output_path)
            
        return shape

    def create_notch(self, mesh, mesh_outside_radius, mesh_height, notch_angle, notch_width, notch_depth):
        
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
        notch_box = notch_polygon.extrude((0, 0, mesh_height+2), capping=True).triangulate()

        # Cut the notch from the outer cylinder
        mesh = mesh.boolean_difference(notch_box)

        return mesh