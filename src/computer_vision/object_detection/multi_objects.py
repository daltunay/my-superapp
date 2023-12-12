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

    def detect_objects(self, image: ndarray) -> Results:
        return self.detector.predict(
            source=image,
            stream=False,
            show=False,
            show_labels=True,
            show_conf=True,
            verbose=False,
        )

    def video_frame_callback(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        detections = self.detect_objects(image)
        image = self.annotate_detections(detections)
        utils.annotate_time(image)
        return VideoFrame.from_ndarray(image, format="bgr24")

    def stream(self) -> None:
        st_webrtc.webrtc_streamer(
            video_frame_callback=self.video_frame_callback,
            key="multi_objects_streamer",
            mode=st_webrtc.WebRtcMode.SENDRECV,
            rtc_configuration=st_webrtc.RTCConfiguration(
                {"iceServers": utils.get_ice_servers(), "iceTransportPolicy": "relay"}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_state=None,
        )

    @staticmethod
    def annotate_detections(detections: Results) -> ndarray:
        return detections[0].plot()
