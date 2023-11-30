import streamlit as st

import utils
from pages.large_language_models import LLM_CONFIG
from src.generative_ai.large_language_models import ChatbotTools

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

st_ss = st.session_state


def main():
    chosen_model = st.selectbox(
        label="Large Language Model:",
        placeholder="Choose an option",
        options=LLM_CONFIG.keys(),
        index=None,
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot"},
    )

    with st.sidebar:
        st.header(body="Chat parameters", divider="gray")
        st_ss.setdefault("language_widget", utils.LanguageWidget()).select()
        st_ss.setdefault("lakera_widget", utils.LakeraWidget()).checkbox()

    chosen_tools = st.sidebar.multiselect(
        label="Tools:",
        options=ChatbotTools.available_tools,
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot"},
    )

    if chosen_model:
        chatbot = st_ss.setdefault(
            "chatbot", ChatbotTools(**LLM_CONFIG[chosen_model], tool_names=chosen_tools)
        )
        for message in chatbot.history:
            st.chat_message(message["role"]).write(message["content"])
    else:
        st.info("Select a model above", icon="ℹ️")

    if prompt := st.chat_input(
        placeholder=f"Chat with {chosen_model}!" if chosen_model else "",
        disabled=not chosen_model,
    ):
        st.chat_message("human").write(prompt)
        if st_ss.get("lakera_widget.activated"):
            flag, response = utils.LakeraWidget.flag_prompt(prompt=prompt)
            if flag:
                st.warning(body="Prompt injection detected", icon="🚨")
                st.expander(label="LOGS").json(response)
        with st.chat_message("ai"):
            chatbot.ask(query=prompt, language=st_ss.get("language_widget.selection"))
