import os

import requests
import streamlit as st

from utils.logging import set_logger

logger = set_logger(__file__)


class LakeraAPIManager:
    key = "lakera_api_manager.activated"

    def __init__(self):
        pass

    def checkbox(self):
        return st.checkbox(
            label="LLM prompt injection security",
            value=st.session_state.get(self.key, False),
            key=self.key,
            help="Use Lakera Guard API to defend against LLM prompt injections",
            on_change=self.authentificate,
        )

    @classmethod
    def authentificate(cls):
        if not st.session_state.get(cls.key):
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
        self.checkbox()
