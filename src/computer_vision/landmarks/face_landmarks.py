import typing as t
from functools import cached_property

import mediapipe as mp

from src.computer_vision.landmarks import BaseLandmarkerApp


class FaceLandmarkerApp(BaseLandmarkerApp):
    landmarks_type = "face_landmarks"

    def __init__(
        self, model_path="src/computer_vision/landmarks/models/face_landmarker.task"
    ):
        super().__init__(model_path)

    @cached_property
    def landmarker(self) -> mp.tasks.vision.FaceLandmarker:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @cached_property
    def connections_list(self) -> t.List[t.FrozenSet[t.Tuple[int, int]]]:
        return [
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            mp.solutions.face_mesh.FACEMESH_IRISES,
        ]

    @cached_property
    def drawing_specs_list(
        self,
    ) -> t.List[t.Dict[str, mp.solutions.drawing_utils.DrawingSpec]]:
        return [
            {
                "connection_drawing_spec": style,
                "landmark_drawing_spec": mp.solutions.drawing_utils.DrawingSpec(thickness=1),
            }
            for style in (
                mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
        ]
