import pathlib
from typing import List

import cv2


def images_to_videos(images_paths: List[pathlib.Path], target_path: pathlib.Path, codec_name: str):
    img_array = []
    for images_path in images_paths:
        img = cv2.imread(str(images_path))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(str(target_path), cv2.VideoWriter_fourcc(*codec_name), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
