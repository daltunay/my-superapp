import typing as t
from functools import cached_property

import streamlit_webrtc as st_webrtc
from av import VideoFrame
from numpy import ndarray
from ultralytics import YOLO
from ultralytics.engine.results import Results

import utils

logger = utils.CustomLogger(__file__)


class MultiObjectsDetectionApp:
    def __init__(self):
        pass

    @cached_property
    def detector(self) -> YOLO:
        return YOLO(model="yolov8n.pt", task=None)

    def detect_objects(self, image: ndarray) -> t.Any:
        return self.detector.predict(
            source=image,
            stream=False,
            conf=0.5,
            line_width=1,
            show=False,
            show_labels=True,
            show_conf=True,
        )

    def video_frame_callback(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        detections = self.detect_objects(image)
        image = self.annotate_detections(detections=detections)
        utils.annotate_time(image=image)
        return VideoFrame.from_ndarray(image, format="bgr24")

    def stream(self) -> None:
        st_webrtc.webrtc_streamer(
            video_frame_callback=self.video_frame_callback,
            key="face_streamer",
            mode=st_webrtc.WebRtcMode.SENDRECV,
            rtc_configuration=st_webrtc.RTCConfiguration(
                {"iceServers": utils.get_ice_servers(), "iceTransportPolicy": "relay"}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_state=None,
        )

    @classmethod
    def annotate_detections(cls, detections: Results) -> ndarray:
        return detections[0].plot()[:, :, ::-1]
