# Migration Plan: Parametric + CSG Approach for Custom Mask Generation

## Overview
This plan outlines the migration from the current extrusion-based method to a parametric modeling + Constructive Solid Geometry (CSG) approach. The goal is to improve robustness (watertight/manifold meshes), simplicity (KISS), and flexibility (variable thickness/radius). We use landmarks to parameterize shapes, then CSG for solid volumes.

**Benefits:** Native watertight solids, easier variable params, fewer artifacts.

**Libs:** Python + Trimesh/Manifold (CSG), Open3D/MediaPipe (landmarks), PyVista (vis/debug).

## Key Steps

### 1. Setup Environment
- Install/update required libraries: Run `pip install trimesh manifold3d open3d mediapipe pyvista scipy`.
- Create a new Git branch for development: `git checkout -b migration-parametric-csg`.
- Backup the current codebase: Copy the project folder or commit all changes.
- Set up a virtual environment if not already done (ex. venv or conda).

### 2. Input Handling
- Retain and refine `face.py` for landmark extraction: Ensure it handles various scan formats (PLY/OBJ/STL) and outputs cleaned landmarks (ex. filter outliers).
- Add a new function in `utils.py` called `segment_face_region(mesh, landmarks, buffer=10)`: Use Trimesh to crop the mesh around landmarks with a buffer (ex. convex hull + expansion).
- Implement parametric curve fitting: Create `fit_spline_to_landmarks(landmarks, smoothing=0.5)` using scipy.interpolate.splprep/splev for smooth Bézier/spline curves.
- Sub-steps: Test on sample scans; visualize fitted curves on mesh using PyVista.

### 3. Parametric Modeling
- Create a new directory `parametric_models/` for modular shapes.
- Develop base classes (ex. `base_parametric.py`): Define abstract methods for generating surfaces from params (landmarks, radius_map, thickness_map).
- Implement specific models like `nose_param.py`: Generate tube-like surface with Trimesh.creation.extrude_polygon (polygon from cross-sections along spline).
- Add variability: Accept dicts like `radius_map = {pos: value}`; interpolate along curve (scipy.interpolate.interp1d).
- For connectors: In `shapes/`, update to parametric (ex. `cylinder_connector(landmarks, radius=20, height=10)` using Trimesh.creation.cylinder).
- Sub-steps: Generate isolated components; visualize/debug with PyVista; ensure params allow variations (ex. thinner in mobile areas).

### 4. CSG Operations
- Implement thickening: Create `thicken_parametric(surface, thickness_map)` using Manifold.offset for variable offset (loop over sections if needed).
- Handle combinations: Add `csg_union(meshes)` and `csg_difference(base, subtract)` wrapping Manifold.boolean_union/difference.
- Ensure manifold: Post-CSG, add validation `validate_mesh(mesh)` with Trimesh.is_watertight; if fails, repair with tm_mesh.fill_holes() + tm_mesh.fix_normals().
- Integrate into `models/` : New classes like `parametric_nose_model.py` that inherit from base and compose parametric + CSG steps.
- Sub-steps: Test CSG on simple shapes (cube + sphere); apply to full mask; handle errors (ex. non-manifold inputs).

### 5. Integration & Testing
- Update `main.py`: Add flags like `--mode=parametric` to switch between old/new; load new models dynamically.
- Export: Add `export_stl(mesh, path)` with validation before save.
- Testing: Create scripts for batch testing on sample scans; check STL in Cura/PrusaSlicer for printability (no holes, solid).
- Validation: Compare outputs (old vs new) via metrics (volume diff, watertight status); add pytest for unit tests (ex. test_csg_union).
- Sub-steps: Run end-to-end on 3-5 scans; fix bugs; document edge cases (ex. noisy landmarks).

### 6. Refinements & Deployment
- Expose variables: Update `config.py` with maps (ex. `thickness_map = {'nose': 2, 'mouth': 1.5}`); parse in models.
- Add lissage: Implement `smooth_mesh(mesh, iterations=50)` using PyVista.smooth_taubin or Trimesh.remesh.
- Merge: Once tested, merge branch to master; update README.md with new usage/instructions.
- Sub-steps: Optimize perf (ex. downsample large meshes); add examples in `assets/`.

## Potential Challenges
- Fitting param curves to landmarks: Handle noise with smoothing.
- CSG perf: For large meshes, downsample first.
- Variable thickness: Implement as func of position (ex. distance to mobile zones).
- If Manifold issues, fallback to PyMesh. 