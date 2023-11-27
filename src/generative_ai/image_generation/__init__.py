import typing as t

import yaml

from src.generative_ai.image_generation.dall_e import dall_e_image
from src.generative_ai.image_generation.stable_diffusion import \
    stable_diffusion_image

with open("config/models.yaml") as f:
    CONFIG: t.Dict = yaml.safe_load(f)["gen_ai"]["image_creation"]

__all__ = ["CONFIG", "dall_e_image", "stable_diffusion_image"]
