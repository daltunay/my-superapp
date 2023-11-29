import logging
import os
import typing as t
from functools import cached_property

import streamlit as st


class CustomLogger:
    method_names = ["debug", "info", "warning", "error", "critical"]

    def __init__(self, file: str, level: str = "info"):
        self.script_name = os.path.splitext(os.path.basename(file))[0]
        self.level = getattr(logging, level.upper())
        self.cache_methods(methods_to_cache=self.method_names)

    @cached_property
    def logger(self) -> logging.Logger:
        logger = logging.getLogger(self.script_name)
        logger.setLevel(self.level)

        if not logger.hasHandlers():
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def cache_methods(self, methods_to_cache: t.List[str]) -> None:
        for method_name in methods_to_cache:
            method = getattr(self.logger, method_name)
            wrapped_method = st.cache_resource(
                func=method,
                show_spinner=False,
                max_entries=1,
            )
            setattr(self, method_name, wrapped_method)
