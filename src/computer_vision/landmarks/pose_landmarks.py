import typing as t
from functools import cached_property

import mediapipe as mp

from src.computer_vision.landmarks import BaseLandmarkerApp


class PoseLandmarkerApp(BaseLandmarkerApp):
    landmarks_type = "pose_landmarks"

    def __init__(self):
        super().__init__()

    @cached_property
    def landmarker(self) -> mp.solutions.pose.Pose:
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @cached_property
    def connections_list(self) -> t.List[t.FrozenSet[t.Tuple[int, int]]]:
        return [mp.solutions.pose.POSE_CONNECTIONS]

    @cached_property
    def drawing_specs_list(
        self,
    ) -> t.List[t.Dict[str, mp.solutions.drawing_utils.DrawingSpec]]:
        return [
            {
                "landmark_drawing_spec": mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            }
        ]
