import streamlit as st


def update_slider_callback(updated: str, to_update: str):
    setattr(st.session_state, to_update, 1 - st.session_state.get(updated))
