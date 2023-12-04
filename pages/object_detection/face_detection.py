import streamlit as st

import utils
from src.computer_vision.object_detection.face_detection import FaceDetectionApp

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    st.caption(
        body="Using Google's Mediapipe module, this app performs face detection.",
        help="https://developers.google.com/mediapipe/solutions/vision/face_detector",
    )

    utils.show_source_code(path="src/computer_vision/landmarks/face_landmarks/")

    st_ss.setdefault("face_detection_app", FaceDetectionApp()).stream()
