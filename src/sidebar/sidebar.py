import streamlit as st

from .lakera_api_manager import LakeraAPIManager
from .language_manager import LanguageManager
from .model_api_manager import ModelAPIManager


class Sidebar:
    def __init__(self):
        self.language_manager = LanguageManager()
        self.model_api_manager = ModelAPIManager()
        self.lakera_api_manager = LakeraAPIManager()

    def main(self):
        with st.sidebar:
            st.header("Language", divider="gray")
            self.language_manager.main()
            st.header("Advanced settings", divider="gray")
            with st.expander("Model", expanded=False):
                self.model_api_manager.main()
            with st.expander("Lakera Guard", expanded=False):
                self.lakera_api_manager.main()
