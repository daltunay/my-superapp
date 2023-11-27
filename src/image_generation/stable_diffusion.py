import together

from src.image_generation import CONFIG
from utils.logging import set_logger
from utils.misc import base64_to_img

logger = set_logger(__file__)

model_config = CONFIG["Stable Diffusion 2.1"]


def stable_diffusion_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
):
    response = together.Image.create(
        model=f"{model_config['owner']}/{model_config['string']}",
        prompt=prompt,
        width=width,
        height=height,
    )

    base64 = response["output"]["choices"][0]["image_base64"]
    return base64_to_img(base64)
