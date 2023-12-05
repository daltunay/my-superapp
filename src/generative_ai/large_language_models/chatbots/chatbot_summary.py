import typing as t
from functools import cached_property

from src.generative_ai.large_language_models.chatbots import Chatbot, ModelArgs

from langchain.document_loaders import UnstructuredURLLoader
from langchain.docstore.document import Document
from unstructured.cleaners.core import remove_punctuation, clean, clean_extra_whitespace
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain


class ChatbotSummary(Chatbot):
    def __init__(
        self,
        **model_kwargs: t.Unpack[ModelArgs],
    ) -> None:
        super().__init__(**model_kwargs)

    @classmethod
    def url_to_doc(cls, source_url: str) -> Document:
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
        return load_summarize_chain(self.llm, chain_type="stuff", verbose=True)

    def summarize(self, url: str) -> str:
        document = self.url_to_doc(url)
        return self.chain.run(
            [document],
            callbacks=self.callbacks,
        )
