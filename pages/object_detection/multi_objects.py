import streamlit as st

import utils
from src.computer_vision.object_detection import MultiObjectsDetectionApp

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.show_source_code(path="src/computer_vision/object_detection/multi_objects.py")

    st_ss.setdefault("multi_objects_detection_app", MultiObjectsDetectionApp()).stream()
