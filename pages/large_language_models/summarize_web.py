import streamlit as st

import utils
from pages.large_language_models import LLM_CONFIG
from src.generative_ai.large_language_models import ChatbotSummary

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    chosen_model = st.selectbox(
        label="Large Language Model:",
        placeholder="Choose an option",
        options=LLM_CONFIG.keys(),
        index=0,
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot_summary"},
    )

    if chosen_model:
        chatbot = st_ss.setdefault(
            "chatbot", ChatbotSummary(**LLM_CONFIG[chosen_model])
        )
    else:
        pass

    if input_url := st.chat_input(
        placeholder=f"Summarize URL with {chosen_model}!" if chosen_model else "",
        disabled=not chosen_model,
    ):
        st.chat_message("human").write(input_url)
        with st.chat_message("ai"):
            chatbot.summarize(url=input_url)
