import streamlit as st

from app.sidebar import Sidebar
from utils.logging import set_logger
from utils.misc import show_logos
from utils.secrets import load_secrets

logger = set_logger(__file__)

st.set_page_config(page_title="daltunay", page_icon="🧠")

load_secrets()


def main():
    st.title("<title>", anchor=False)
    st.caption("<caption>")

    sidebar = st.session_state.setdefault("sidebar", Sidebar())
    sidebar.main()

    # ...

    show_logos(linkedin=True, github=True)


if __name__ == "__main__":
    main()
