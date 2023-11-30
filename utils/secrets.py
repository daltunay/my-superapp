import os

import streamlit as st

import utils

logger = utils.CustomLogger(__file__)


def load_secrets():
    for secrets in st.secrets.values():
        for secret_name, secret in secrets.items():
            masked_secret = secret[:4] + "*" * (len(secret) - 4)
            logger.info(f"Setting {secret_name}={masked_secret}")
            os.environ[secret_name] = secret
