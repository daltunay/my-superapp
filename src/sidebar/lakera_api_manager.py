import os

import requests
import streamlit as st

from utils.logging import configure_logger

logger = configure_logger(__file__)


class LakeraAPIManager:
    def __init__(self):
        self.activated = False
        self.authentificated = False
        self.api_key = {"api_key": "", "default": True}

    def reset_state(self):
        self.authentificated = False

    def activate(self):
        self.activated = st.checkbox(
            label="Activate Lakera Guard",
            key="lakera_api_manager.activated",
            value=st.session_state.get("lakera_api_manager.activated", self.activated),
            help="Protection against prompt injection and jailbreak using Lakera API",
            on_change=self.reset_state,
        )

    def default_api_key(self):
        self.api_key["default"] = st.checkbox(
            label="Default API key",
            key="lakera_api_manager.default",
            value=st.session_state.get(
                "lakera_api_manager.default", self.api_key["default"]
            ),
            help="Use the provided default API key, if you don't have any",
            on_change=self.reset_state,
            disabled=not self.activated,
        )

        if self.api_key["default"]:
            api_key = st.secrets.get("lakera_guard_api").key
        else:
            api_key = self.api_key["api_key"]

        if self.activated:
            self.authentificate(api_key=api_key)

    def api_key_form(self):
        with st.form("lakera_gard_api"):
            self.api_key["api_key"] = st.text_input(
                label="Enter your Lakera Guard API key:",
                value=self.api_key["api_key"],
                placeholder="[default]" if self.api_key["default"] else "",
                type="password",
                help="Click [here](https://platform.lakera.ai/account/api-keys) to get your Lakera Guard API key",
                autocomplete="",
                disabled=self.api_key["default"],
            )

            api_key = (
                st.secrets.get("lakera_guard_api").key
                if self.api_key["default"]
                else self.api_key["api_key"]
            )

            st.form_submit_button(
                label="Authentificate",
                on_click=self.authentificate,
                kwargs={"api_key": api_key},
                disabled=self.api_key["default"],
                use_container_width=True,
            )

    def authentificate(self, api_key):
        success = self.authentificate_lakera_guard(api_key)

        if not self.authentificated:
            if success:
                st.toast("API Authentication successful — Lakera Guard", icon="✅")
                os.environ["LAKERA_GUARD_API_KEY"] = api_key
                self.authentificated = True
            else:
                st.toast("API Authentication failed — Lakera Guard", icon="🚫")
                os.environ.pop("LAKERA_GUARD_API_KEY", None)
                self.authentificated = False

    @staticmethod
    @st.cache_data(show_spinner=False)
    def authentificate_lakera_guard(api_key):
        response = requests.post(
            url="https://api.lakera.ai/v1/prompt_injection",
            json={"input": "<AUTHENTIFICATION TEST>"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return response.ok

    def show_status(self):
        if self.activated:
            if self.authentificated:
                st.success("Successfully authentificated to Lakera Guard API", icon="🔐")
            else:
                st.info("Please configure the Lakera Guard API above", icon="🔐")

    def main(self):
        self.activate()
        self.default_api_key()
        self.api_key_form()
        self.show_status()
