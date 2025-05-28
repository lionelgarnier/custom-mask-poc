import pyvista as pv
import numpy as np

class F20Connector:
    def create_3d_object(self, output_path=None, **kwargs):
        # Tube parameters
        height_mm = 5.0
        outside_radius = 40.0 / 2.0
        inside_radius = 35.0 / 2.0
        notch_depth = 3.0
        notch_radius = 34.0 / 2.0
        
        # Create the outer cylinder
        outer_cylinder = pv.Cylinder(
            center=(0, 0, height_mm / 2.0),
            direction=(0, 0, 1),
            radius=outside_radius,
            height=height_mm
        ).triangulate()
        

        # Create the inner cylinder
        inner_cylinder = pv.Cylinder(
            center=(0, 0, height_mm / 2.0 + (height_mm - notch_depth)),
            direction=(0, 0, 1),
            radius=inside_radius,
            height=height_mm
        ).triangulate()
        
        # Boolean difference to hollow out the inside
        shape = outer_cylinder.boolean_difference(inner_cylinder)
        

        # Create the notch cylinder
        notch_cylinder = pv.Cylinder(
            center=(0, 0, (height_mm - notch_depth) / 2.0),
            direction=(0, 0, 1),
            radius=notch_radius,
            height=height_mm
        ).triangulate()
        
        # Boolean difference to hollow out the inside
        shape = shape.boolean_difference(notch_cylinder)
        
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
