import os

import requests
import streamlit as st

import utils

logger = utils.CustomLogger(__file__)


class LakeraGuardAPIManager:
    key = "lakera_guard_api_manager"

    def __init__(self):
        logger.info("Initializing Lakera Guard API Manager")

    @property
    def checkbox(self):
        return st.checkbox(
            label="LLM prompt injection security",
            value=st.session_state.get(f"{self.key}.checkbox", False),
            key=f"{self.key}.checkbox",
            help="Use Lakera Guard API to defend against LLM prompt injections",
            on_change=self.authentificate,
        )

    @classmethod
    def authentificate(cls):
        if not st.session_state.get(f"{cls.key}.checkbox"):
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
        st.session_state.__setattr__("", self.checkbox)
