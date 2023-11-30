import streamlit as st

import utils

logger = utils.CustomLogger(__file__)

LANGUAGES = [
    "English",
    "French",
    "German",
    "Spanish",
]


class LanguageManager:
    key = "language_manager"

    def __init__(self):
        logger.info("Initializing Language Manager")

    def select_language(self):
        self.selected_language = st.selectbox(
            label="Select chat language:",
            options=list(LANGUAGES),
            key="self.key",
            index=list(LANGUAGES).index(st.session_state.get("self.key", "English")),
            help="Changes the **chat language only**, not the interface language",
        )

    def main(self):
        self.select_language()
