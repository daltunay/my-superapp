import together
from PIL import Image

import utils
from utils.misc import base64_to_img

logger = utils.CustomLogger(__file__)


def stable_diffusion_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
) -> Image.Image:
    from src.generative_ai.image_generation import CONFIG

    model_config = CONFIG["Stable Diffusion 2.1"]

    response = together.Image.create(
        model=f"{model_config['owner']}/{model_config['string']}",
        prompt=prompt,
        width=width,
        height=height,
    )

    base64 = response["output"]["choices"][0]["image_base64"]
    return base64_to_img(base64)
