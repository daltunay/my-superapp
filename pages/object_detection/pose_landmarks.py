import streamlit as st

import utils
from src.computer_vision.landmarks import PoseLandmarkerApp

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    st.caption(
        body="Using Google's Mediapipe module, this app performs landmark detection for the body pose.",
        help="https://developers.google.com/mediapipe/solutions/vision/pose_landmarker",
    )

    utils.show_source_code(path="src/computer_vision/landmarks/pose_landmarks/")

    st_ss.setdefault("pose_app", PoseLandmarkerApp()).stream()
