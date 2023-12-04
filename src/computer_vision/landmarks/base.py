import time
import typing as t

import cv2
import mediapipe as mp
import streamlit_webrtc as st_webrtc
from av import VideoFrame
from mediapipe.framework.formats import landmark_pb2
from numpy import ndarray

import utils

logger = utils.CustomLogger(__file__)


class BaseLandmarkerApp:
    def __init__(self):
        self.start_time = time.time()

    def frame_callback(self, frame: VideoFrame) -> VideoFrame:
        t = time.time() - self.start_time

        image = frame.to_ndarray(format="rgb24")
        self.annotate_time(image=image, timestamp=t)

        detection_result = self.landmarker.process(image)
        landmark_list_raw = getattr(detection_result, self.landmarks_type)
        landmark_list = (
            landmark_list_raw[0]
            if isinstance(landmark_list_raw, list)
            else landmark_list_raw
        )

        self.annotate_landmarks(
            image=image,
            connections_list=self.connections_list,
            landmark_list=landmark_list,
            drawing_specs_list=self.drawing_specs_list,
        )

        return VideoFrame.from_ndarray(image, format="rgb24")

    def stream(self) -> None:
        st_webrtc.webrtc_streamer(
            video_frame_callback=self.frame_callback,
            key=f"{self.landmarks_type}_streamer",
            mode=st_webrtc.WebRtcMode.SENDRECV,
            rtc_configuration=st_webrtc.RTCConfiguration(
                {"iceServers": utils.get_ice_servers(), "iceTransportPolicy": "relay"}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    @classmethod
    def annotate_time(cls, image: ndarray, timestamp: float):
        cv2.putText(
            img=image,
            text=f"{timestamp:.3f}s",
            org=(10, 60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    @classmethod
    def annotate_landmarks(
        cls,
        image: ndarray,
        connections_list: t.List[t.FrozenSet[t.Tuple[int, int]]],
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        drawing_specs_list: t.List[t.Dict[str, mp.solutions.drawing_utils.DrawingSpec]],
    ) -> None:
        if not landmark_list:
            return image

        for connections, drawing_specs in zip(connections_list, drawing_specs_list):
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_list,
                connections=connections,
                **drawing_specs,
            )
