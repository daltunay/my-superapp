import streamlit as st
import streamlit_superapp as st_superapp

import utils

utils.load_secrets()

st.set_page_config(page_title="daltunay", page_icon="🚀", layout="centered")

st_superapp.run()
