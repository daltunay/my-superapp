import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler


class StreamingChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        pass

    def on_llm_start(self, *args, **kwargs):
        self.container = st.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.text += token
        self.container.markdown(
            body=self.text,
            unsafe_allow_html=False,
        )

    def on_llm_end(self, response: str, *args, **kwargs):
        self.container.markdown(
            body=response.generations[0][0].text,
            unsafe_allow_html=False,
        )
