import os
import typing as t
from functools import cached_property

import cv2
import mediapipe as mp
import streamlit_webrtc as st_webrtc
from av import VideoFrame
from mediapipe.framework.formats import detection_pb2
from numpy import ndarray

import utils

logger = utils.CustomLogger(__file__)

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"


class FaceDetectionApp:
    def __init__(self):
        pass

    @cached_property
    def detector(self):
        return mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=0,
        )

    def detect_faces(self, image: ndarray) -> t.Any:
        return self.detector.process(image).detections

    def video_frame_callback(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="rgb24")

        detection_list = self.detect_faces(image)
        self.annotate_faces(
            image=image,
            detection_list=detection_list,
        )
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
    def annotate_faces(
        cls,
        image: ndarray,
        detection_list: t.List[detection_pb2.Detection],
    ) -> None:
        for detection in detection_list:
            bbox = detection.location_data.relative_bounding_box
            height, width, _ = image.shape
            xmin, ymin = int(bbox.xmin * width), int(bbox.ymin * height)
            xmax, ymax = int((bbox.xmin + bbox.width) * width), int(
                (bbox.ymin + bbox.height) * height
            )
            cv2.rectangle(
                img=image,
                pt1=(xmin, ymin),
                pt2=(xmax, ymax),
                color=(0, 255, 0),
                thickness=3,
            )
