import streamlit as st

import utils

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


class LanguageWidget:
    key = "language_widget"
    languages = ["English", "French"]

    def __init__(self):
        logger.info("Initializing Language Widget")

    @property
    def selected_language(self):
        return st.selectbox(
            label="Chat language:",
            options=list(self.languages),
            key=f"{self.key}.selection",
            index=list(self.languages).index(
                st_ss.get(f"{self.key}.selection", "English")
            ),
            help="Changes the **chat language only**, not the interface language",
        )
