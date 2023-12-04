import os
import typing as t

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

import utils


def get_loader(
    file: str | None = None,
    mode: t.Literal["local"] | t.Literal["upload"] = "local",
) -> DirectoryLoader | PyPDFLoader:
    if mode == "local":
        return DirectoryLoader(
            path="data/documents/",
            glob="./*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
    elif mode == "upload":
        return PyPDFLoader(file)


def get_vector_store(
    file: str | None = None,
    mode: t.Literal["local"] | t.Literal["upload"] = "local",
) -> None:
    loader = get_loader(file=file, mode=mode)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    documents_chunked = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents=documents_chunked, embedding=embeddings)

    if mode == "local":
        db.save_local(
            folder_path="faiss_index",
            index_name="index" if mode == "local" else os.path.splitext(file)[0],
        )
    elif mode == "upload":
        return db


def main():
    get_vector_store(file=None, mode="local")


if __name__ == "__main__":
    utils.load_secrets()
    main()
