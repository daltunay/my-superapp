import typing as t

import yaml

from src.generative_ai.image_generation.dall_e import dall_e_image
from src.generative_ai.image_generation.stable_diffusion import \
    stable_diffusion_image

with open("config/models.yaml") as f:
    IMAGE_GEN_CONFIG: t.Dict[str, str] = yaml.safe_load(f)["generative_ai"][
        "image_creation"
    ]

__all__ = ["IMAGE_GEN_CONFIG", "dall_e_image", "stable_diffusion_image"]
