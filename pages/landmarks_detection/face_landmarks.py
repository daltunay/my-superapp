import time

import streamlit as st

import utils
from pages import CONFIG
from src.computer_vision.landmarks import FaceLandmarkerApp

page_config = utils.load_page_config(CONFIG, __file__)
for key, value in page_config.items():
    globals()[key] = value

logger = utils.CustomLogger(__file__)


def main():
    st.caption(
        body="Using Google's Mediapipe module, this app performs landmark detection for the face.",
        help="https://developers.google.com/mediapipe/solutions/vision/face_landmarker",
    )

    utils.show_source_code(path="src/computer_vision/landmarks/face_landmarks/")

    app = FaceLandmarkerApp()
    container = st.empty()
    start_time = time.time()
    for i, image in enumerate(app.run(streamlit_mode=True)):
        container.image(
            image=image,
            caption=f"FPS: {(i / (time.time() - start_time)):.3f} frames/sec",
            channels="BGR",
            use_column_width=True,
        )
