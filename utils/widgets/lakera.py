import os
import typing as t

import requests
import streamlit as st

import utils

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


class LakeraWidget:
    key = "lakera_widget"
    api_key = os.getenv("LAKERA_GUARD_API_KEY")

    def __init__(self):
        logger.info("Initializing Lakera Guard Widget")

    @property
    def lakera_activated(self):
        return st.checkbox(
            label="LLM prompt injection security",
            value=st_ss.get(f"{self.key}.activated", False),
            key=f"{self.key}.activated",
            help="Use Lakera Guard API to defend against LLM prompt injections",
            on_change=self.authentificate,
        )

    @classmethod
    def authentificate(cls):
        if not st_ss.get(f"{cls.key}.activated"):
            return

        try:
            response = requests.post(
                url="https://api.lakera.ai/v1/prompt_injection",
                json={"input": "<AUTHENTICATION TEST>"},
                headers={"Authorization": f"Bearer {cls.api_key}"},
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

    @classmethod
    def flag_prompt(cls, prompt: str) -> t.Tuple[bool, t.Dict]:
        response = requests.post(
            "https://api.lakera.ai/v1/prompt_injection",
            json={"input": prompt},
            headers={"Authorization": f"Bearer {cls.api_key}"},
        ).json()

        flagged = response["results"][0]["flagged"]
        return flagged, response
