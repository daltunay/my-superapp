import streamlit as st

import utils
from pages.large_language_models import LLM_CONFIG
from src.generative_ai.large_language_models import ChatbotTools

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())


def main():
    chosen_model = st.selectbox(
        label="Large Language Model:",
        placeholder="Choose an option",
        options=LLM_CONFIG.keys(),
        index=None,
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot"},
    )

    if chosen_model:
        chatbot = st.session_state.setdefault(
            "chatbot", ChatbotTools(**LLM_CONFIG[chosen_model])
        )
        for message in chatbot.history:
            st.chat_message(message["role"]).write(message["content"])

    if prompt := st.chat_input(
        placeholder=f"Chat with {chosen_model}!"
        if chosen_model
        else "Select a model above first",
        disabled=not chosen_model,
    ):
        st.chat_message("human").write(prompt)
        with st.chat_message("ai"):
            chatbot.ask(query=prompt, context=None, language=None)
