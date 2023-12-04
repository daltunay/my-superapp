import os

import streamlit as st
from twilio.rest import Client

import utils

logger = utils.CustomLogger(__file__)


@st.cache_data(show_spinner=False)
def get_ice_servers():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    client = Client(account_sid, auth_token)
    token = client.tokens.create()

    return token.ice_servers
