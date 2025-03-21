"""
Configuration settings for face landmark detection and processing
"""

# Default landmarks for face contour
DEFAULT_FACE_CONTOUR_LANDMARKS = [168, 417, 465, 429, 423, 391, 393, 164, 167, 165, 203, 209, 245, 193, 168]

# Visualization settings
DEFAULT_VISUALIZATION_SETTINGS = {
    'SHOW_LANDMARKS_2D': False,  # Show 2D landmarks on projected image
    'SHOW_LANDMARKS_3D': False,  # Show 3D landmarks scatter plot
    'SHOW_MESH_AND_LANDMARKS': False,  # Show mesh with landmarks and IDs
}

# 3D printing options
DEFAULT_PRINTING_SETTINGS = {
    'CREATE_3D_PRINT': True,  # Create a 3D printable STL file
    'EXTRUSION_RADIUS': 0.2,  # Radius of the extruded tube (adjust based on model scale)
    'EXTRUSION_RESOLUTION': 30,  # Resolution of the circular cross-section
}
