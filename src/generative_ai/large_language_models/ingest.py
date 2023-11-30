from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

import utils

utils.load_secrets()


def main() -> None:
    loader = DirectoryLoader(
        path="data/documents/",
        glob="./*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    documents = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    db.save_local("faiss_index")


if __name__ == "__main__":
    main()
