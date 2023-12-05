import streamlit as st

import utils
from src.computer_vision.landmarks import FaceLandmarkerApp

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.show_source_code("src/computer_vision/landmarks/face_landmarks.py")

    st_ss.setdefault("face_app", FaceLandmarkerApp()).stream()
