from .base_parametric import BaseParametric
import trimesh
import numpy as np
from scipy.interpolate import splprep, splev, interp1d
import shapely.geometry
from trimesh import transformations

class NoseParam(BaseParametric):
    def generate_surface(self, landmarks: np.ndarray, radius_map: dict = None, thickness_map: dict = None) -> trimesh.Trimesh:
        # Fit spline to landmarks
        tck, u = splprep(landmarks.T, s=0.5)
        u_new = np.linspace(0, 1, 100)
        spline_points = np.array(splev(u_new, tck)).T
        
        # Interpolate radii
        if radius_map:
            positions = np.linspace(0, 1, len(radius_map))
            radii = np.array(list(radius_map.values()))
            interp_r = interp1d(positions, radii, kind='linear')
            radii_interp = interp_r(u_new)
        else:
            radii_interp = np.full(len(u_new), 10.0)
        
        # Generate cross-sections (2D polygons in local plane)
        sections = []
        for i, pt in enumerate(spline_points):
            # Create 2D circle polygon (XY plane, z=0 local)
            angles = np.linspace(0, 2*np.pi, 32)
            circle_2d = np.column_stack((radii_interp[i] * np.cos(angles), radii_interp[i] * np.sin(angles)))
            
            # To make 3D, transform to global (placeholder: align to tangent)
            # For simplicity, assume extrusion along z; improve with frenet frame
            circle_3d = np.hstack([circle_2d, np.zeros((32, 1))]) + pt
            sections.append(trimesh.Trimesh(vertices=circle_3d))
        
        # Connect sections with union of short extrusions (approx sweep)
        tube_parts = []
        for i in range(len(sections) - 1):
            # Create Shapely Polygon from 2D circle (assume circle_2d defined per section; adjust if needed)
            circle_2d = sections[i].vertices[:, :2]  # Use 2D proj
            poly1 = shapely.geometry.Polygon(circle_2d)
            
            height = np.linalg.norm(spline_points[i+1] - spline_points[i])
            part = trimesh.creation.extrude_polygon(poly1, height=height)
            
            # Basic alignment: Translate and rotate to match spline direction
            direction = spline_points[i+1] - spline_points[i]
            direction /= np.linalg.norm(direction)
            axis = np.cross([0,0,1], direction)
            axis /= np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else 1
            angle = np.arccos(np.dot([0,0,1], direction))
            rot = transformations.rotation_matrix(angle, axis)
            part.apply_transform(rot)
            part.apply_translation(spline_points[i])
            tube_parts.append(part)

        tube = trimesh.util.concatenate(tube_parts)
        
        if thickness_map:
            tube = self.apply_thickness(tube, thickness_map)
        return tube 