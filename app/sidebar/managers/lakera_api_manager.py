import os

import requests
import streamlit as st

from utils.logging import set_logger

logger = set_logger(__file__)


class LakeraAPIManager:
    def __init__(self):
        self.activated = False
        self.authentificated = False

    def activate(self):
        self.activated = st.checkbox(
            label="Activate Lakera Guard",
            key="lakera_api_manager.activated",
            value=st.session_state.get("lakera_api_manager.activated", self.activated),
            help="Protection against LLM prompt injection and jailbreak using Lakera Guard API",
            on_change=self.authentificate,
        )

    def authentificate(self):
        if not st.session_state.get("lakera_api_manager.activated"):
            return

        success = requests.post(
            url="https://api.lakera.ai/v1/prompt_injection",
            json={"input": "<AUTHENTIFICATION TEST>"},
            headers={"Authorization": f"Bearer {os.getenv('LAKERA_GUARD_API_KEY')}"},
        )

        if success.ok:
            st.toast("Lakera Guard API authentification", icon="✅")
            self.authentificated = True
        else:
            st.toast("Lakera Guard API authentification", icon="🚫")
            self.authentificated = False

    def main(self):
        self.activate()
