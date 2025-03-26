import numpy as np
import pyvista as pv
from models.base_model import Face3DObjectModel

class ExtrusionModel(Face3DObjectModel):
    def create_3d_object(self, output_path, **kwargs):
        thickness = kwargs.get('thickness', 1.0) / 2
        height = kwargs.get('height', 10.0)
        line_points_3d = np.array(kwargs.get('line_points_3d'))
        face_mesh = kwargs.get('face_mesh', None)

        min_height = np.min(line_points_3d[:, 2])
        max_height = np.max(line_points_3d[:, 2])
        height_diff = max_height - min_height

        is_closed = np.allclose(line_points_3d[0], line_points_3d[-1], atol=1e-4)
        polyline = pv.PolyData(line_points_3d)
        n_points = len(line_points_3d)
        lines = np.hstack((n_points, np.arange(n_points), 0)) if is_closed else np.hstack((n_points, np.arange(n_points)))
        polyline.lines = np.array([lines])

        polygon_3d = polyline.ribbon(width=thickness, normal=[0, 0, 1]).triangulate()
        extruded = polygon_3d.extrude((0, 0, height + height_diff), capping=True).triangulate()

        return extruded