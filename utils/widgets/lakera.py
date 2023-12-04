import os
import typing as t

import requests
import streamlit as st

import utils

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


class LakeraWidget:
    widget_key = "lakera_widget"
    checkbox_key = f"{widget_key}.checkbox"

    def __init__(
        self,
        default: bool = False,
    ):
        logger.info(f"Initializing {self.__class__.__name__}")
        self.api_key = os.getenv("LAKERA_GUARD_API_KEY")
        self.default = default

    @property
    def lakera_activated(self):
        return st.checkbox(
            label="Prompt injection security",
            value=st_ss.get(self.checkbox_key, self.default),
            key=self.checkbox_key,
            help="Use Lakera Guard API to defend against LLM prompt injections",
            on_change=self.authentificate,
        )

    def request_api(self, input: str) -> requests.Response:
        return requests.post(
            url="https://api.lakera.ai/v1/prompt_injection",
            json={"input": input},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def authentificate(self):
        if not st_ss.get(self.checkbox_key):
            return
        try:
            response = self.request_api("<AUTHENTICATION TEST>")
        except requests.exceptions.SSLError:
            st.toast("SSL CERTIFICATE VERIFY FAILED", icon="🚫")
        else:
            success = response.ok
            st.toast("Lakera Guard API authentication", icon="✅" if success else "🚫")

    def flag_prompt(self, prompt: str) -> t.Tuple[bool, t.Dict]:
        response = self.request_api(prompt).json()
        flagged = response["results"][0]["flagged"]
        return flagged, response
