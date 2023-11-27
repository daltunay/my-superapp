import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler


class StreamingChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        pass

    def on_llm_start(self, *args, **kwargs):
        self.container = st.empty()
        self.text = ""

    def on_llm_new_token(self, token, *args, **kwargs):
        self.text += token
        self.container.markdown(self.text, unsafe_allow_html=True)

    def on_llm_end(self, response, *args, **kwargs):
        self.container.empty()
        self.container.markdown(response.generations[0][0].text, unsafe_allow_html=True)
