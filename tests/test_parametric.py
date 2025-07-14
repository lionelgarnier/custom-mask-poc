import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import trimesh
import numpy as np
from models.parametric_nose_model import ParametricNoseModel
from face import extract_face_landmarks
from utils import export_stl

# Sample scans dir
scans_dir = 'assets/samples/'
output_dir = 'output/test/'
os.makedirs(output_dir, exist_ok=True)

model = ParametricNoseModel()
for scan in os.listdir(scans_dir):
    if scan.lower().endswith(('.ply', '.obj', '.stl')):
        path = os.path.join(scans_dir, scan)
        _, landmarks, indices = extract_face_landmarks(path)  # Real extraction
        output_path = os.path.join(output_dir, f'{scan}_parametric.stl')
        result, _ = model.create_3d_object(output_path, face_landmarks=landmarks)
        if result.is_watertight:
            print(f'{scan} OK - Watertight')
            export_stl(result, output_path)
        else:
            print(f'{scan} Failed - Not watertight')
        # Optional: Visualize
        # result.show() 