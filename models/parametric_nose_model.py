from models.base_model import Face3DObjectModel
from parametric_models.nose_param import NoseParam
from utils import thicken_parametric, csg_union, validate_mesh
import trimesh

class ParametricNoseModel(Face3DObjectModel):
    def create_3d_object(self, output_path, **kwargs):
        landmarks = kwargs.get('face_landmarks')
        # Example maps
        radius_map = {0: 10, 0.5: 15, 1: 10}
        thickness_map = {'nose': 2.0}
        
        param_model = NoseParam()
        surface = param_model.generate_surface(landmarks, radius_map, thickness_map)
        thickened = thicken_parametric(surface, thickness_map)
        # Example: Combine with connector (assume loaded)
        connector = trimesh.creation.cylinder(radius=20, height=10)  # Placeholder
        combined = csg_union([thickened, connector])
        final = validate_mesh(combined)
        
        # Export
        final.export(output_path, 'stl')
        return final, "" 