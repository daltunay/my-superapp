import streamlit as st

from utils.logging import configure_logger

logger = configure_logger(__file__)

LANGUAGES = [
    "English",
    "French",
    "German",
    "Spanish",
]


class LanguageManager:
    def __init__(self, default_language="English"):
        self.languages = LANGUAGES
        self.selected_language = default_language

    def reset_state(self):
        st.session_state.chatbot = None
        st.session_state.current_choice = None

    def choose_language(self):
        self.selected_language = st.selectbox(
            label="Select chat language:",
            options=list(self.languages),
            key="language_manager.selected_language",
            index=list(self.languages).index(
                st.session_state.get(
                    "language_manager.selected_language", self.selected_language
                )
            ),
            help="Changes the **chat language only**, not the interface language",
            on_change=self.reset_state,
        )

    def main(self):
        self.choose_language()
