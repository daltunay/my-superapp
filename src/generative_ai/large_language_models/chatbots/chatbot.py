import typing as t
from functools import cached_property

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import Together
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.generative_ai.large_language_models.callbacks import \
    StreamingChatCallbackHandler


class ModelArgs(t.TypedDict):
    provider: t.Literal["openai", "together"]
    owner: t.Literal["mistralai", "togethercomputer"] | None
    string: t.Literal["gpt-3.5-turbo", "llama-2-7b-chat", "Mistral-7B-Instruct-v0.1"]


class Chatbot:
    BASE_TEMPLATE = """
    Use the following context and chat history to answer the question:
    
    Context: {context}
    Chat history: {chat_history}
    Question: {question}
    
    Your answer (in {language}):
    """

    def __init__(self, **model_kwargs: t.Unpack[ModelArgs]) -> None:
        self.model_provider = model_kwargs.get("provider", "openai")
        self.model_owner = model_kwargs.get("owner", None)
        self.model_string = model_kwargs.get("string", "gpt-3.5-turbo")

    @cached_property
    def llm(self) -> ChatOpenAI | Together:
        if self.model_provider == "openai":
            return ChatOpenAI(
                model=self.model_string,
                streaming=True,
                model_kwargs={},
            )
        elif self.model_provider == "together":
            return Together(
                model=f"{self.model_owner}/{self.model_string}",
                max_tokens=1024,
            )

    @cached_property
    def memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
        )

    @property
    def history(self) -> t.List[t.Dict[str, str]]:
        return [
            {"role": message.type, "content": message.content}
            for message in self.memory.buffer
        ]

    @cached_property
    def template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.BASE_TEMPLATE,
            input_variables=["context", "chat_history", "question", "language"],
        )

    @cached_property
    def chain(self) -> Chain:
        return LLMChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            prompt=self.template,
        )

    @property
    def callbacks(self) -> t.List[BaseCallbackHandler]:
        return [StreamingChatCallbackHandler(), StreamingStdOutCallbackHandler()]

    def ask(
        self,
        query: str,
        context: str | None = None,
        language: str | None = None,
    ) -> str:
        return self.chain.run(
            question=query,
            context=context or "",
            language=language or "the input language",
            callbacks=self.callbacks,
        )
