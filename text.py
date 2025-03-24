import vtk
import pyvista as pv

def run_one_shot_test():
    # Create 3D text using vtkVectorText
    text_source = vtk.vtkVectorText()
    text_source.SetText("ID: TEST")
    
    # Extrude the text along the Z axis
    extruder = vtk.vtkLinearExtrusionFilter()
    extruder.SetInputConnection(text_source.GetOutputPort())
    extruder.SetExtrusionTypeToNormalExtrusion()
    extruder.SetVector(0, 0, 0.5)  # Adjust extrusion vector as needed
    extruder.SetScaleFactor(1)
    extruder.Update()

    # Wrap extruded text into a PyVista mesh
    text_mesh = pv.wrap(extruder.GetOutput())
    # Optionally position the text for better visibility (e.g., raise it slightly)
    text_mesh.translate([0, 0, 0.5])

    # Create a semi-transparent sphere for context
    sphere = pv.Sphere(radius=2.0, theta_resolution=30, phi_resolution=30)

    # Initialize PyVista plotter and add the meshes
    plotter = pv.Plotter()
    plotter.add_title("One-Shot 3D Text Test")
    # Add the sphere as a background element
    plotter.add_mesh(sphere, color='lightblue', opacity=0.2, show_edges=True)
    # Add the extruded text
    plotter.add_mesh(text_mesh, color='red', specular=1)

    plotter.show_grid()
    plotter.enable_trackball_style()
    plotter.show()

if __name__ == "__main__":
    run_one_shot_test()