import typing as t
from functools import cached_property

import mediapipe as mp

from src.computer_vision.landmarks import BaseLandmarkerApp


class FaceLandmarkerApp(BaseLandmarkerApp):
    landmarks_type = "multi_face_landmarks"

    def __init__(self):
        super().__init__()

    @cached_property
    def landmarker(self) -> mp.solutions.face_mesh.FaceMesh:
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

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
            {"connection_drawing_spec": style, "landmark_drawing_spec": None}
            for style in (
                mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
        ]
