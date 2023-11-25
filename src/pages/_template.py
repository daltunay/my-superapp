import streamlit as st

from src.sidebar import Sidebar
from utils.logging import configure_logger
from utils.misc import show_logos

logger = configure_logger(__file__)

st.set_page_config(page_title="daltunay", page_icon="🧠")


def main():
    st.title("<title>", anchor=False)
    st.caption("<caption>")

    sidebar = st.session_state.setdefault("sidebar", Sidebar())
    sidebar.main()

    # ...

    show_logos(linkedin=True, github=True)


if __name__ == "__main__":
    main()
