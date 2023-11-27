import os

import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = st.secrets.openai_api.key


def main() -> None:
    loader = DirectoryLoader(
        path="data/documents/",
        glob="./*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    documents = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    db.save_local("faiss_index")


if __name__ == "__main__":
    main()
