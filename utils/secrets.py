import os

import streamlit as st


def load_secrets():
    for secrets in st.secrets.values():
        for k, v in secrets.items():
            os.environ[k] = v
