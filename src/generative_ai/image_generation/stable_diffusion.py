import os
import typing as t

import requests

from utils.logging import configure_logger
from utils.misc import base64_to_img

logger = configure_logger(__file__)


def stable_diffusion_image(
    prompt: str,
    together_api_key: t.Optional[str] = None,
    verify_ssl: bool = True,
):
    together_api_key = together_api_key or os.getenv("TOGETHER_API_KEY")
    response = requests.post(
        "https://api.together.xyz/inference",
        json={
            "model": "stabilityai/stable-diffusion-2-1",
            "prompt": prompt,
            "n": 1,
            "width": 1024,
            "height": 1024,
            "steps": 20,
        },
        headers={"Authorization": f"Bearer {together_api_key}"},
        verify=verify_ssl,
    ).json()

    base64 = response["output"]["choices"][0]["image_base64"]
    return base64_to_img(base64)
