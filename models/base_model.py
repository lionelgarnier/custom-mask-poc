from abc import ABC, abstractmethod

class Face3DObjectModel(ABC):
    @abstractmethod
    def create_3d_object(self, line_points_3d, output_path, **kwargs):
        pass