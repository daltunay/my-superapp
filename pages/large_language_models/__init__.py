import typing as t

import yaml

import utils

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

with open("config/models.yaml") as f:
    LLM_CONFIG: t.Dict[str, str] = yaml.safe_load(f)["generative_ai"][
        "large_language_models"
    ]

__all__ = ["LLM_CONFIG"]
