from datetime import datetime

import cv2
from numpy import ndarray


def annotate_time(image: ndarray) -> None:
    text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    text_args = {
        "text": text,
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": .5,
        "thickness": 1,
    }
    text_size = cv2.getTextSize(**text_args)[0]
    rect_width, rect_height = text_size[0] + 20, text_size[1] + 20
    cv2.rectangle(
        img=image,
        pt1=(0, 0),
        pt2=(rect_width, rect_height),
        color=(255, 255, 255),
        thickness=cv2.FILLED,
    )
    cv2.rectangle(
        img=image,
        pt1=(0, 0),
        pt2=(rect_width, rect_height),
        color=(0, 0, 0),
        thickness=2,
    )
    cv2.putText(
        img=image,
        org=(10, text_size[1] + 10),
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        **text_args,
    )
