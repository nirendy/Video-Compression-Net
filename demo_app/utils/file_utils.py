import os
import pathlib
import shutil
from typing import List

from PIL import Image

import numpy as np

from demo_app.utils.format_utils import get_int_from_path


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def get_file_size(p: pathlib.Path) -> int:
    return p.stat().st_size

def get_image_shape(p: pathlib.Path) -> int:
    return np.array(Image.open(p)).shape


def get_dir_size(p: pathlib.Path) -> int:
    total_size = 0
    for sub_p in p.glob('**/*'):
        total_size += get_file_size(sub_p)

    return total_size


def get_frame_paths(p) -> List[pathlib.Path]:
    images = list(p.iterdir())
    return sorted(images, key=get_int_from_path)