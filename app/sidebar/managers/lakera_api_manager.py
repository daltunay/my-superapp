import os

import requests
import streamlit as st
import streamlit_shadcn_ui as st_ui

from utils.logging import set_logger

logger = set_logger(__file__)


class LakeraAPIManager:
    def __init__(self):
        self.key = "lakera_api_manager.activated"

    @property
    def toggle_switch(self):
        return st_ui.switch(
            default_checked=st.session_state.get(self.key, False),
            label="LLM prompt injection security",
            key=self.key,
        )

    @classmethod
    @st.cache_resource(show_spinner=False, max_entries=1)
    def authentificate(_cls, toggle_state: bool):
        if not toggle_state:
            return

        lakera_guard_api_key = os.getenv("LAKERA_GUARD_API_KEY")
        try:
            response = requests.post(
                url="https://api.lakera.ai/v1/prompt_injection",
                json={"input": "<AUTHENTICATION TEST>"},
                headers={"Authorization": f"Bearer {lakera_guard_api_key}"},
            )
        except requests.exceptions.SSLError:
            toast = {"body": "SSL CERTIFICATE VERIFY FAILED", "icon": "🚫"}
        else:
            body = "Lakera Guard API authentication"
            if response.ok:
                toast = {"body": f"{body} successful", "icon": "✅"}
            else:
                toast = {"body": f"{body} failed", "icon": "🚫"}

        st.toast(**toast)

    def main(self):
        toggle_state = self.toggle_switch
        self.authentificate(toggle_state=toggle_state)
