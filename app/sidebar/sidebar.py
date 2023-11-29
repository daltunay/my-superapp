import streamlit as st

from app.sidebar import \
    LakeraGuardAPIManager  # , LanguageManager, ModelAPIManager


class Sidebar:
    def __init__(self):
        self.lakera_guard_api_manager = st.session_state.setdefault(
            "lakera_guard_api_manager", LakeraGuardAPIManager()
        )

    def main(self):
        with st.sidebar:
            self.lakera_guard_api_manager.main()
