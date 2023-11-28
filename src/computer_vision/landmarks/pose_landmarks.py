import typing as t
from functools import cached_property

import mediapipe as mp

from src.computer_vision.landmarks import BaseLandmarkerApp


class PoseLandmarkerApp(BaseLandmarkerApp):
    landmarks_type = "pose_landmarks"

    def __init__(
        self, model_path="src/computer_vision/landmarks/models/pose_landmarker.task"
    ):
        super().__init__(model_path)

    @cached_property
    def landmarker(self) -> mp.tasks.vision.PoseLandmarker:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    @cached_property
    def connections_list(self) -> t.List[t.FrozenSet[t.Tuple[int, int]]]:
        return [mp.solutions.pose.POSE_CONNECTIONS]

    @cached_property
    def drawing_specs_list(
        self,
    ) -> t.List[t.Dict[str, mp.solutions.drawing_utils.DrawingSpec]]:
        return [
            {
                "landmark_drawing_spec": mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                "connection_drawing_spec": mp.solutions.drawing_utils.DrawingSpec(),
            }
        ]
