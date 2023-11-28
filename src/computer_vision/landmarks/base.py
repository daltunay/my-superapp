import time
import typing as t
from functools import cached_property

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


class BaseLandmarkerApp:
    landmarks_type = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.cap = cv2.VideoCapture(0)
        self.history = []

    @cached_property
    def landmarker(
        self,
    ) -> mp.tasks.vision.PoseLandmarker | mp.tasks.vision.FaceLandmarker:
        raise NotImplementedError(
            "landmarker property must be implemented in subclasses"
        )

    @cached_property
    def connections_list(self) -> t.List[t.FrozenSet[t.Tuple[int, int]]]:
        raise NotImplementedError(
            "connections_list property must be implemented in subclasses"
        )

    @cached_property
    def drawing_specs_list(
        self,
    ) -> t.List[t.Dict[str, mp.solutions.drawing_utils.DrawingSpec]]:
        raise NotImplementedError(
            "drawing_specs property must be implemented in subclasses"
        )

    def run(self) -> None:
        t0 = time.time()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            detection_result = self.landmarker.detect(image)
            landmarks_list_raw = getattr(detection_result, self.landmarks_type)
            landmarks_list = landmarks_list_raw[0] if landmarks_list_raw else []

            t = time.time() - t0
            self.history.append(
                {"time": t, "landmarks": landmarks_list},
            )

            annotated_image = self.annotate_time(image=image.numpy_view(), timestamp=t)
            annotated_image = self.annotate_landmarks(
                image=annotated_image,
                connections_list=self.connections_list,
                landmarks_list=landmarks_list,
                drawing_specs_list=self.drawing_specs_list,
            )

            cv2.imshow("Landmarker", annotated_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    @classmethod
    def normalize_landmarks_list(
        cls, landmarks_list: mp.tasks.vision.PoseLandmarkerResult
    ) -> landmark_pb2.NormalizedLandmarkList:
        normalized_landmarks_list = landmark_pb2.NormalizedLandmarkList()
        normalized_landmarks_list.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                )
                for landmark in landmarks_list
            ]
        )
        return normalized_landmarks_list

    @classmethod
    def annotate_landmarks(
        cls,
        image: mp.Image,
        connections_list: t.List[t.FrozenSet[t.Tuple[int, int]]],
        landmarks_list: mp.tasks.vision.PoseLandmarkerResult
        | mp.tasks.vision.FaceLandmarkerResult,
        drawing_specs_list: t.List[t.Dict[str, mp.solutions.drawing_utils.DrawingSpec]],
    ) -> mp.Image:
        if not landmarks_list:
            return image
        normalized_landmarks_list = cls.normalize_landmarks_list(landmarks_list)

        annotated_image = np.copy(image)

        for connections, drawing_specs in zip(connections_list, drawing_specs_list):
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmarks_list=normalized_landmarks_list,
                connections=connections,
                **drawing_specs,
            )

        return annotated_image

    @classmethod
    def annotate_time(cls, image: mp.Image, timestamp: float):
        annotated_image = np.copy(image)
        cv2.putText(
            img=annotated_image,
            text=f"{timestamp:.2f}s",
            org=(10, 60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        return annotated_image
