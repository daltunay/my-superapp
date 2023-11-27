import typing as t
from functools import cached_property

from langchain.agents import (AgentExecutor, AgentType, initialize_agent,
                              load_tools)
from langchain.tools import BaseTool

from src.generative_ai.large_language_models.chatbots import Chatbot, ModelArgs


class ChatbotTools(Chatbot):
    def __init__(
        self,
        tool_names: t.List[
            t.Literal[
                "google-search",
                "wikipedia",
                "python_repl",
            ]
        ]
        | None = None,
        **model_kwargs: t.Unpack[ModelArgs],
    ) -> None:
        super().__init__(**model_kwargs)
        self.tool_names = tool_names or []
        self.memory.input_key = "input"

    @cached_property
    def tools(self) -> t.List[BaseTool]:
        return load_tools(tool_names=self.tool_names)

    @cached_property
    def chain(self) -> AgentExecutor:
        agent = initialize_agent(
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            agent_kwargs={
                "input_variables": [
                    "input",
                    "chat_history",
                    "agent_scratchpad",
                    "language",
                ]
            },
            tools=self.tools,
            handle_parsing_errors=True,
        )
        agent.agent.llm_chain.prompt += "Answer in {language} only"
        return agent

    def ask(
        self,
        query: str,
        language: str | None = None,
    ) -> str:
        return self.chain.run(
            input=query,
            language=language or "the input language",
            callbacks=self.callbacks,
        )
