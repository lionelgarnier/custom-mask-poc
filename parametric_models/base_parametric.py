from abc import ABC, abstractmethod
import trimesh
import numpy as np

class BaseParametric(ABC):
    @abstractmethod
    def generate_surface(self, landmarks: np.ndarray, radius_map: dict = None, thickness_map: dict = None) -> trimesh.Trimesh:
        """Generate parametric surface from landmarks and maps."""
        pass

    def apply_thickness(self, surface: trimesh.Trimesh, thickness_map: dict) -> trimesh.Trimesh:
        """Apply variable thickness using offset (placeholder)."""
        # Implement variable offset logic here
        return surface

    def visualize(self, mesh: trimesh.Trimesh):
        """Optional: Visualize with PyVista or Trimesh show."""
        mesh.show() 