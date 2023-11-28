import streamlit as st

from app.sidebar import LakeraAPIManager  # , LanguageManager, ModelAPIManager


class Sidebar:
    def __init__(self):
        # self.language_manager = LanguageManager()
        # self.model_api_manager = ModelAPIManager()
        self.lakera_api_manager = LakeraAPIManager()

    def main(self):
        with st.sidebar:
            # self.language_manager.main()
            # self.model_api_manager.main()
            self.lakera_api_manager.main()
