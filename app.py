import streamlit as st
import streamlit_superapp as app

import utils

st.set_page_config(page_title="daltunay", page_icon="🧠", layout="wide")

utils.load_secrets()

app.run()
