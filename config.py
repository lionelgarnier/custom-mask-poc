"""
Configuration settings for face landmark detection and processing
"""

# Default landmarks for face contour
# DEFAULT_FACE_CONTOUR_LANDMARKS = [168, 417, 465, 429, 423, 391, 393, 164, 167, 165, 203, 209, 245, 193, 168]
DEFAULT_FACE_CONTOUR_LANDMARKS = [168, 464, 423, 391, 393, 164, 167, 165, 203, 193, 168]

# File system settings
DEFAULT_FILE_SETTINGS = {
    'DEFAULT_MESH_FOLDER': "D:/OneDrive/Desktop/masque/faces",  # Default folder for mesh files
}

# Output folder for STL files
OUTPUT_FOLDER = "output"

# Visualization settings
DEFAULT_VISUALIZATION_SETTINGS = {
    'SHOW_LANDMARKS_2D': False,  # Show 2D landmarks on projected image
    'SHOW_LANDMARKS_3D': False,  # Show 3D landmarks scatter plot
    'SHOW_MESH_AND_LANDMARKS': False,  # Show mesh with landmarks and IDs
    'SHOW_CONTACT_LINE': False,  # Show contact line visualization
    'SHOW_3D_PRINT': True  # Show 3D print visualization
}

# 3D printing options
DEFAULT_PRINTING_SETTINGS = {
    'CREATE_3D_PRINT': True,  # Create a 3D printable STL file
    'EXTRUSION_WIDTH': 2,  # Width of the extruded tube (thickness of the walls in mm)
    'EXTRUSION_RESOLUTION': 30,  # Resolution of the circular cross-section
}

# Default 3D model for creating printable objects
DEFAULT_MODEL = "mold"  # Options: "extrusion", "mold"
