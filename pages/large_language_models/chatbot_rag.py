import streamlit as st

import utils
from pages.large_language_models import LLM_CONFIG
from src.generative_ai.large_language_models import (ChatbotRAG,
                                                     get_vector_store)

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

st_ss = st.session_state


def main():
    utils.show_source_code(
        path="src/generative_ai/large_language_models/chatbots/chatbot_rag.py"
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
        kwargs={"key": "chatbot_rag"},
    )

    if uploaded_file := st.file_uploader(
        "Upload a PDF file",
        type="pdf",
        accept_multiple_files=False,
        help="https://python.langchain.com/docs/use_cases/question_answering/#what-is-rag",
        on_change=utils.reset_session_state_key,
        kwargs={"key": "chatbot_rag"},
    ):
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        vector_db = get_vector_store(file=uploaded_file.name, mode="upload")

    if chosen_model and uploaded_file:
        chatbot = st_ss.setdefault(
            "chatbot_rag",
            ChatbotRAG(vector_store=vector_db, **LLM_CONFIG[chosen_model]),
        )
        for message in chatbot.history:
            st.chat_message(message["role"]).write(message["content"])
    else:
        st.info("Please upload a PDF file for the RAG", icon="ℹ️")

    if prompt := st.chat_input(
        placeholder=f"Chat with {chosen_model}!"
        if (chosen_model and uploaded_file)
        else "",
        disabled=not (chosen_model and uploaded_file),
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
                language=selected_language,
            )
