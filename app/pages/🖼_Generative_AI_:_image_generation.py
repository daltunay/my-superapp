import streamlit as st
import streamlit_shadcn_ui as st_ui

import utils
from app.sidebar import Sidebar

logger = utils.set_logger(__file__)

st.set_page_config(page_title="daltunay", page_icon="🧠", layout="centered")

utils.load_secrets()


def main():
    st.title("", anchor=False)

    st.caption(
        body="",
        help="",
    )

    st_ui.link_button(
        text="Source code",
        url="https://github.com/daltunay/my-app/tree/main/src/generative_ai/image_generation/",
        variant="outline",
    )

    sidebar = st.session_state.setdefault("sidebar", Sidebar())
    sidebar.main()

    st.markdown("Select one of the two following modes:")
    app_modes = {
        "DALL·E": None,
        "Stable Diffusion": None,
    }
    selected_app = st_ui.tabs(options=app_modes.keys())

    if selected_app in app_modes:
        app = st.session_state.setdefault(selected_app, app_modes[selected_app]())
        pass
    else:
        st.info(body="Please select a mode above", icon="ℹ️")

    utils.show_logos(linkedin=True, github=True)


if __name__ == "__main__":
    main()
