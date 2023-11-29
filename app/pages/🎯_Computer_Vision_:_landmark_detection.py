import time

import extra_streamlit_components as stx
import streamlit as st

import utils
from app.sidebar import Sidebar
from src.computer_vision.landmarks import FaceLandmarkerApp, PoseLandmarkerApp

logger = utils.CustomLogger(__file__)

st.set_page_config(page_title="daltunay", page_icon="🧠", layout="centered")

utils.load_secrets()


def main():
    st.title("Landmark detection", anchor=False)

    st.caption(
        body="Using Google's Mediapipe module, this app performs landmark detection for both the face and the body pose.",
        help="https://developers.google.com/mediapipe/solutions",
    )

    utils.show_source_code(path="src/computer_vision/landmarks/")

    st.session_state.setdefault("sidebar", Sidebar()).main()

    st.markdown("Select one of the two following modes:")
    app_modes = {"face": FaceLandmarkerApp, "pose": PoseLandmarkerApp}
    selected_app = stx.tab_bar(
        data=[
            stx.TabBarItemData(
                id=app_mode,
                title=f"Mode: {app_mode.capitalize()}",
                description=f"Detection of {app_mode} landmarks",
            )
            for app_mode in app_modes
        ],
        return_type=str,
        default=None,
    )

    if selected_app in app_modes:
        app = st.session_state.setdefault(selected_app, app_modes[selected_app]())
        container = st.empty()
        start_time = time.time()
        for i, image in enumerate(app.run(streamlit_mode=True)):
            elapsed_time = time.time() - start_time
            container.image(
                image=image,
                caption=f"FPS: {(i / elapsed_time):.3f} frames/sec",
                channels="BGR",
                use_column_width=True,
            )
    else:
        st.info(body="Please select a mode above", icon="ℹ️")

    utils.show_logos(linkedin=True, github=True)


if __name__ == "__main__":
    main()
