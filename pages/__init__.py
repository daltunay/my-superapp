import utils

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

import streamlit as st

st.write(st.session_state)
