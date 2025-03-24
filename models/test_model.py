from models.base_model import Face3DObjectModel
import pyvista as pv

class TestModel(Face3DObjectModel):
    def create_3d_object(self, output_path, **kwargs):
        
        # Create a tube with 3cm length, 2mm inner radius and 3mm outer radius
        inner_radius = 2.0  # 2mm
        outer_radius = 2.5  # 3mm
        length = 20.0  # 3cm = 30mm

        # Create a cylinder representing the outer surface
        outer_cylinder = pv.Cylinder(
            center=(0, 0, length/2),
            direction=(0, 0, 1),
            radius=outer_radius,
            height=length
        )
        
        # Triangulate the outer cylinder
        outer_cylinder = outer_cylinder.triangulate()

        # Create a cylinder representing the inner surface (to be subtracted)
        inner_cylinder = pv.Cylinder(
            center=(0, 0, length/2),
            direction=(0, 0, 1),
            radius=inner_radius,
            height=length
        )
        
        # Triangulate the inner cylinder
        inner_cylinder = inner_cylinder.triangulate()

        # Ensure both cylinders are closed surfaces
        outer_cylinder = outer_cylinder.extract_surface().triangulate()
        inner_cylinder = inner_cylinder.extract_surface().triangulate()

        # Create the hollow tube by subtracting the inner cylinder from the outer
        extruded = outer_cylinder.boolean_difference(inner_cylinder)

        # Save the mold to an STL file
        extruded.save(output_path)
        return extruded