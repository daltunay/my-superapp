import streamlit as st
import validators

import utils
from pages.large_language_models import LLM_CONFIG
from src.generative_ai.large_language_models import ChatbotWebSummary

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.show_source_code(
        path="src/generative_ai/large_language_models/chatbots/chatbot_web.py"
    )
    chosen_model = st.selectbox(
        label="Large Language Model:",
        placeholder="Choose an option",
        options=LLM_CONFIG.keys(),
        index=0,
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot_web_summary"},
    )

    chosen_chain_type = st.selectbox(
        label="Chain type:",
        options=ChatbotWebSummary.available_chain_types,
        index=None,
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot_web_summary"},
    )

    if chosen_model and chosen_chain_type:
        chatbot = st_ss.setdefault(
            "chatbot_web_summary", ChatbotWebSummary(**LLM_CONFIG[chosen_model])
        )
    else:
        st.info("Choose a chain type for the LLM", icon="ℹ️")

    if input_url := st.text_input(
        label="URL of the page to summarize:",
        disabled=not (chosen_model and chosen_chain_type),
    ):
        if validators.url(input_url):
            st.chat_message("human").write(input_url)
            with st.chat_message("ai"):
                chatbot.summarize(url=input_url)
        else:
            st.error("Invalid URL", icon="❌")
