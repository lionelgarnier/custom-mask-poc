import pyvista as pv

class N5Connector:
    def create_3d_object(self, output_path=None, **kwargs):
        # Tube parameters
        height_mm = 7.0
        outside_radius = 34.0 / 2.0
        inside_radius = 29.0 / 2.0
        
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
            height=height_mm
        ).triangulate()
        
        # Boolean difference to hollow out the inside
        shape = outer_cylinder.boolean_difference(inner_cylinder)
        
        landmarks = {
            1: (0, 0, 0),
            2: (0, 0, height_mm),
            3: (outside_radius, 0, 0),
            4: (outside_radius, 0, height_mm),
            5: (0, outside_radius, 0),
            6: (0, outside_radius, height_mm),
            7: (inside_radius, 0, 0),
            8: (inside_radius, 0, height_mm),
            9: (0, inside_radius, 0),
            10: (0, inside_radius, height_mm)
        }
        shape.landmarks = landmarks
        
        # Save the 3D connector to an STL file if path is provided
        if output_path:
            shape.save(output_path)
            
        return shape