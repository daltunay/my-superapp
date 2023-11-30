import time

import streamlit as st

import utils
from pages import CONFIG
from src.computer_vision.landmarks import PoseLandmarkerApp

page_config = utils.load_page_config(CONFIG, __file__)
for key, value in page_config.items():
    globals()[key] = value

logger = utils.CustomLogger(__file__)


def main():
    st.caption(
        body="Using Google's Mediapipe module, this app performs landmark detection for the body pose.",
        help="https://developers.google.com/mediapipe/solutions/vision/pose_landmarker",
    )

    utils.show_source_code(path="src/computer_vision/landmarks/pose_landmarks/")

    app = PoseLandmarkerApp()
    container = st.empty()
    start_time = time.time()
    for i, image in enumerate(app.run(streamlit_mode=True)):
        container.image(
            image=image,
            caption=f"FPS: {(i / (time.time() - start_time)):.3f} frames/sec",
            channels="BGR",
            use_column_width=True,
        )
