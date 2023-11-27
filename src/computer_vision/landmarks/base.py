import typing as t
from functools import cached_property

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


class BaseLandmarkerApp:
    type = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.cap = cv2.VideoCapture(0)

    @cached_property
    def landmarker(
        self,
    ) -> mp.tasks.vision.PoseLandmarker | mp.tasks.vision.FaceLandmarker:
        raise NotImplementedError(
            "landmarker property must be implemented in subclasses"
        )

    @cached_property
    def connections(self) -> t.List[t.Tuple[int, int]]:
        raise NotImplementedError(
            "connections property must be implemented in subclasses"
        )

    @cached_property
    def drawing_specs(self) -> t.List[mp.solutions.drawing_utils.DrawingSpec]:
        raise NotImplementedError(
            "landmark_style property must be implemented in subclasses"
        )

    def run(self) -> None:
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarks = self.landmarker.detect(image)

            annotated_image = self.annotate(
                image=image.numpy_view(),
                landmarks=landmarks,
            )

            cv2.imshow("Landmarker", annotated_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def annotate(
        self,
        image: mp.Image,
        landmarks: mp.tasks.vision.PoseLandmarkerResult
        | mp.tasks.vision.FaceLandmarkerResult,
    ) -> mp.Image:
        landmark_sets = getattr(landmarks, self.type)
        if not landmark_sets:
            return image

        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                )
                for landmark_set in landmark_sets
                for landmark in landmark_set
            ]
        )

        annotated_image = np.copy(image)

        for connections, drawing_spec in zip(self.connections, self.drawing_specs):
            drawing_kwargs = {
                "landmark_drawing_spec": drawing_spec
                if self.type == "pose_landmarks"
                else mp.solutions.drawing_utils.DrawingSpec(),
                "connection_drawing_spec": drawing_spec
                if self.type == "face_landmarks"
                else mp.solutions.drawing_utils.DrawingSpec(),
                "connections": connections,
            }

            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=landmark_list,
                **drawing_kwargs,
            )

        return annotated_image
