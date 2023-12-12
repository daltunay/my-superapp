import typing as t
from functools import cached_property

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredURLLoader
from unstructured.cleaners.core import (clean, clean_extra_whitespace,
                                        remove_punctuation)

from src.generative_ai.large_language_models.chatbots import Chatbot, ModelArgs


class ChatbotWebSummary(Chatbot):
    available_chain_types = ["stuff", "map_reduce"]

    def __init__(
        self,
        chain_type: t.Literal["stuff"] | t.Literal["map_reduce"] = "stuff",
        **model_kwargs: t.Unpack[ModelArgs],
    ) -> None:
        super().__init__(**model_kwargs)
        self.chain_type = chain_type

    @staticmethod
    def url_to_doc(source_url: str) -> Document:
        url_loader = UnstructuredURLLoader(
            urls=[source_url],
            mode="elements",
            post_processors=[clean, remove_punctuation, clean_extra_whitespace],
        )

        narrative_elements = [
            element
            for element in url_loader.load()
            if element.metadata.get("category") == "NarrativeText"
        ]
        cleaned_content = " ".join(
            element.page_content for element in narrative_elements
        )

        return Document(page_content=cleaned_content, metadata={"source": source_url})

    @cached_property
    def chain(self) -> BaseCombineDocumentsChain:
        return load_summarize_chain(self.llm, chain_type=self.chain_type, verbose=True)

    def summarize(self, url: str) -> str:
        document = self.url_to_doc(url)
        return self.chain.run(
            [document],
            callbacks=self.callbacks,
        )
