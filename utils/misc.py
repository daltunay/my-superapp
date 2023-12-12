from base64 import b64decode
from io import BytesIO

import streamlit as st
from PIL import Image

st_ss = st.session_state


def generate_logo_link(url: str, img_url: str) -> str:
    return f'<a href="{url}"><img src="{img_url}"></a>'


def show_source_code(path: str):
    st.markdown(
        "[![source code](https://img.shields.io/badge/source_code-gray?logo=github)]"
        f"(https://github.com/daltunay/my-superapp/tree/main/{path})"
    )


def show_logos(linkedin: bool = True, github: bool = True):
    logos = []

    if linkedin:
        logos.append(
            generate_logo_link(
                url="https://linkedin.com/in/daltunay",
                img_url="https://img.icons8.com/?id=13930&format=png",
            )
        )

    if github:
        logos.append(
            generate_logo_link(
                url="https://github.com/daltunay",
                img_url="https://img.icons8.com/?id=AZOZNnY73haj&format=png",
            )
        )

    logos_html = "".join(logos)
    html_content = f"""
        <div style="text-align: center;">
            Made by Daniel Altunay<br>
            {logos_html}
        </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)


def base64_to_img(base64: str) -> Image.Image:
    return Image.open(BytesIO(b64decode(base64)))


def reset_session_state_key(key: str):
    if hasattr(st_ss, key):
        delattr(st_ss, key)
