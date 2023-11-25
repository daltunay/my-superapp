import os
import typing as t

import requests

from utils.logging import configure_logger
from utils.misc import base64_to_img

logger = configure_logger(__file__)


def dall_e_image(
    prompt: str,
    openai_api_key: t.Optional[str] = None,
    verify_ssl: bool = True,
):
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        json={
            "model": "dall-e-2",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "response_format": "b64_json",
        },
        headers={"Authorization": f"Bearer {openai_api_key}"},
        verify=verify_ssl,
    ).json()

    base64 = response["data"][0]["b64_json"]
    return base64_to_img(base64)
