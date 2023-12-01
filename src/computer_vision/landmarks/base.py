import os
import time
import typing as t
from functools import cached_property
from queue import Queue

import cv2
import mediapipe as mp
import streamlit_webrtc as st_webrtc
from av import VideoFrame
from mediapipe.framework.formats import landmark_pb2
from numpy import ndarray

import utils

logger = utils.CustomLogger(__file__)

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"


class BaseLandmarkerApp:
    landmarks_type = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.start_time = time.time()
        self.history = []
        self.queue: Queue[t.List[ndarray]] = Queue()

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

    def callback(self, frame: VideoFrame) -> VideoFrame:
        logger.info(f"New callback: {time.time()}")
        image = frame.to_ndarray(format="bgr24")

        detection_result = self.landmarker.detect(image)
        landmark_list_raw = getattr(detection_result, self.landmarks_type)
        landmark_list = landmark_list_raw[0] if landmark_list_raw else []

        t = time.time() - self.start_time
        self.history.append(
            {"time": t, "landmarks": landmark_list},
        )

        self.annotate_time(image=image, timestamp=t)
        self.annotate_landmarks(
            image=image,
            connections_list=self.connections_list,
            landmark_list=landmark_list,
            drawing_specs_list=self.drawing_specs_list,
        )

        self.queue.put(item=image)

        return VideoFrame.from_ndarray(image, format="bgr24")

    def stream(self) -> None:
        def callback(frame):
            return self.callback(frame)

        st_webrtc.webrtc_streamer(
            key=f"{self.landmarks_type}_streamer",
            # mode=st_webrtc.WebRtcMode.SENDRECV,
            video_frame_callback=callback,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    @classmethod
    def normalize_landmark_list(
        cls, landmark_list: mp.tasks.vision.PoseLandmarkerResult
    ) -> landmark_pb2.NormalizedLandmarkList:
        normalized_landmark_list = landmark_pb2.NormalizedLandmarkList()
        normalized_landmark_list.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                )
                for landmark in landmark_list
            ]
        )
        return normalized_landmark_list

    @classmethod
    def annotate_landmarks(
        cls,
        image: ndarray,
        connections_list: t.List[t.FrozenSet[t.Tuple[int, int]]],
        landmark_list: mp.tasks.vision.PoseLandmarkerResult
        | mp.tasks.vision.FaceLandmarkerResult,
        drawing_specs_list: t.List[t.Dict[str, mp.solutions.drawing_utils.DrawingSpec]],
    ) -> None:
        if not landmark_list:
            return image
        normalized_landmark_list = cls.normalize_landmark_list(landmark_list)

        for connections, drawing_specs in zip(connections_list, drawing_specs_list):
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=normalized_landmark_list,
                connections=connections,
                **drawing_specs,
            )

    @classmethod
    def annotate_time(cls, image: ndarray, timestamp: float):
        cv2.putText(
            img=image,
            text=f"{timestamp:.3f}s",
            org=(10, 60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
