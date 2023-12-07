import streamlit as st

import utils
from pages.large_language_models import LLM_CONFIG
from src.generative_ai.large_language_models import Chatbot

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

st_ss = st.session_state


def main():
    utils.show_source_code("src/generative_ai/large_language_models/chatbots/chatbot.py")
    with st.expander(label="Chat parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            selected_language = st_ss.setdefault(
                "language_widget", utils.LanguageWidget()
            ).selected_language
        with col2:
            lakera_activated = st_ss.setdefault(
                "lakera_widget", utils.LakeraWidget()
            ).lakera_activated

    chosen_model = st.selectbox(
        label="Large Language Model:",
        placeholder="Choose an option",
        options=LLM_CONFIG.keys(),
        index=0,
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot"},
    )

    provided_context = st.text_area(
        label="Context:",
        value="",
        help="This context will be passed to the chatbot.",
    )

    if chosen_model:
        chatbot = st_ss.setdefault("chatbot", Chatbot(**LLM_CONFIG[chosen_model]))
        for message in chatbot.history:
            st.chat_message(message["role"]).write(message["content"])
    else:
        pass

    if prompt := st.chat_input(
        placeholder=f"Chat with {chosen_model}!" if chosen_model else "",
        disabled=not chosen_model,
    ):
        st.chat_message("human").write(prompt)
        if lakera_activated:
            flag, response = st_ss.setdefault(
                "lakera_widget", utils.LakeraWidget()
            ).flag_prompt(prompt=prompt)
            if flag:
                st.warning(body="Prompt injection detected", icon="🚨")
                st.expander(label="LOGS").json(response)
        with st.chat_message("ai"):
            chatbot.ask(
                query=prompt,
                context=provided_context,
                language=selected_language,
            )
