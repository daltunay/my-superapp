import os

import streamlit as st
import together
from PIL import Image

import utils
from utils.misc import base64_to_img

logger = utils.CustomLogger(__file__)


@st.cache_data(show_spinner="Generating picture...")
def stable_diffusion_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
) -> Image.Image:
    from src.generative_ai.image_generation import IMAGE_GEN_CONFIG

    model_config = IMAGE_GEN_CONFIG["Stable Diffusion 2.1"]
    together.api_key = os.getenv("TOGETHER_API_KEY")

    response = together.Image.create(
        model=f"{model_config['owner']}/{model_config['string']}",
        prompt=prompt,
        width=width,
        height=height,
    )

    base64 = response["output"]["choices"][0]["image_base64"]
    return base64_to_img(base64)
