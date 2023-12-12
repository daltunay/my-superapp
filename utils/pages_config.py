import typing as t
from functools import cached_property

import yaml

import utils


class PageConfigLoader:
    config_path = "pages/pages_config.yaml"

    def __init__(self, file):
        self.file = file
        self.logger = utils.CustomLogger(self.file)

    @cached_property
    def pages_config(self) -> t.Dict:
        with open(self.config_path, "r") as file:
            pages_config = yaml.safe_load(file)
        return pages_config

    @cached_property
    def page_config(self) -> t.Dict:
        path_keys = self.file.split("my-superapp/pages/")[1].split("/")
        section = self.pages_config

        for path_key in path_keys:
            section = section.get(path_key, {})

        return self._set_recursive(section, path_keys)

    def _set_recursive(self, section, keys) -> t.Dict:
        return {
            key: self._set_recursive(value, keys + [key])
            if isinstance(value, dict)
            else value
            for key, value in section.items()
        }

    def set_page_config(self, _globals):
        self.logger.info(f"Setting page config: {self.page_config}")
        for key, value in self.page_config.items():
            _globals[key] = value
