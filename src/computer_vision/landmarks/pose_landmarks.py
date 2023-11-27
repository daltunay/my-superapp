import typing as t
from functools import cached_property

import mediapipe as mp

from src.computer_vision.landmarks import BaseLandmarkerApp


class PoseLandmarkerApp(BaseLandmarkerApp):
    type = "pose_landmarks"

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
    def connections(self) -> t.List[t.Tuple[int, int]]:
        return [mp.solutions.pose.POSE_CONNECTIONS]

    @cached_property
    def drawing_specs(self) -> t.List[mp.solutions.drawing_utils.DrawingSpec]:
        return [mp.solutions.drawing_styles.get_default_pose_landmarks_style()]
