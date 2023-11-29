import time

import streamlit as st
import streamlit_shadcn_ui as st_ui

import utils
from app.sidebar import Sidebar
from src.computer_vision.landmarks import FaceLandmarkerApp, PoseLandmarkerApp

logger = utils.set_logger(__file__)

st.set_page_config(page_title="daltunay", page_icon="🧠", layout="centered")

utils.load_secrets()


def main():
    st.title("Landmark detection", anchor=False)

    st.caption(
        body="Using Google's Mediapipe module, this app performs landmark detection for both the face and the body pose.",
        help="https://developers.google.com/mediapipe/solutions",
    )

    st_ui.link_button(
        text="Source code",
        url="https://github.com/daltunay/my-app/tree/main/src/computer_vision/landmarks/",
        variant="outline",
    )

    sidebar = st.session_state.setdefault("sidebar", Sidebar())
    sidebar.main()

    st.markdown("Select one of the two following modes:")
    app_modes = {
        "Face detection": FaceLandmarkerApp,
        "Pose detection": PoseLandmarkerApp,
    }
    selected_app = st_ui.tabs(options=app_modes.keys())

    if selected_app in app_modes:
        app = st.session_state.setdefault(selected_app, app_modes[selected_app]())
        container = st.empty()
        start_time = time.time()
        for i, image in enumerate(app.run(streamlit_mode=True)):
            elapsed_time = time.time() - start_time
            fps = i / elapsed_time
            container.image(
                image=image,
                caption=f"FPS: {fps:.3f} frames/sec",
                channels="BGR",
                use_column_width=True,
            )
    else:
        st.info(body="Please select a mode above", icon="ℹ️")

    utils.show_logos(linkedin=True, github=True)


if __name__ == "__main__":
    main()
