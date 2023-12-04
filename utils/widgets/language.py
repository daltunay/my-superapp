import typing as t

import streamlit as st

import utils

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


class LanguageWidget:
    widget_key = "language_widget"
    selectbox_key = f"{widget_key}.selection"

    def __init__(
        self,
        languages: t.List[str] | None = None,
        default: str | None = None,
    ):
        logger.info(f"Initializing {self.__class__.__name__}")
        self.languages = languages or ["English", "French"]
        self.default = default or "English"

    @property
    def selected_language(self):
        return st.selectbox(
            label="Language:",
            options=list(self.languages),
            index=list(self.languages).index(
                st_ss.get(self.selectbox_key, self.default)
            ),
            key=self.selectbox_key,
            help="Changes the **chat language only**, not the interface language",
        )
