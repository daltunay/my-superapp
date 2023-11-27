import os

import streamlit as st


def load_secrets():
    for category, secrets in st.secrets.items():
        for k, v in secrets.items():
            os.environ[k] = v
