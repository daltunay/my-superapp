import streamlit as st

import utils
from app.sidebar import Sidebar

logger = utils.CustomLogger(__file__)

st.set_page_config(page_title="daltunay", page_icon="🧠", layout="centered")

utils.load_secrets()


def main():
    st.title("", anchor=False)

    utils.show_source_code(path="")

    st.caption(
        body="",
        help="",
    )

    sidebar = st.session_state.setdefault("sidebar", Sidebar())
    sidebar.main()

    utils.show_logos(linkedin=True, github=True)


if __name__ == "__main__":
    main()
