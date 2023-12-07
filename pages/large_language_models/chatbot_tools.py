import streamlit as st

import utils
from pages.large_language_models import LLM_CONFIG
from src.generative_ai.large_language_models import ChatbotTools

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

st_ss = st.session_state


def main():
    utils.show_source_code(
        "src/generative_ai/large_language_models/chatbots/chatbot_tools.py"
    )
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
        kwargs={"key": "chatbot_tools"},
    )

    chosen_tools = st.multiselect(
        label="Tools:",
        options=ChatbotTools.available_tools,
        default=None,
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot_tools"},
    )

    if chosen_model and chosen_tools:
        chatbot = st_ss.setdefault(
            "chatbot_tools",
            ChatbotTools(**LLM_CONFIG[chosen_model], tool_names=chosen_tools),
        )
        for message in chatbot.history:
            st.chat_message(message["role"]).write(message["content"])
    else:
        st.info("Choose tools for the LLM", icon="ℹ️")

    if prompt := st.chat_input(
        placeholder=f"Chat with {chosen_model}!"
        if (chosen_model and chosen_tools)
        else "",
        disabled=not (chosen_model and chosen_tools),
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
            st.write(chatbot.ask(
                query=prompt,
                language=selected_language,
            ))
