import os
import pathlib
from typing import List

import streamlit as st

from demo_app.consts import CHECKPOINTS_PATH
from demo_app.consts import VIDEO_DATASET_PATH, WORKING_DIR_PATH
from demo_app.utils.format_utils import get_int_from_path


def _get_dataset_dir() -> pathlib.Path:
    return pathlib.Path(os.getcwd()) / VIDEO_DATASET_PATH


def _get_demo_cache() -> pathlib.Path:
    return pathlib.Path(os.getcwd()) / WORKING_DIR_PATH


def get_exising_demos_names() -> List[str]:
    return sorted([p.name for p in _get_demo_cache().iterdir()], reverse=True)


def get_videos_names() -> List[pathlib.Path]:
    return list(_get_dataset_dir().glob('*'))


@st.cache()
def get_available_checkpoints_weights() -> List[pathlib.Path]:
    checkpoint_dir = pathlib.Path(os.getcwd()) / CHECKPOINTS_PATH
    return sorted(
        list(checkpoint_dir.glob('video*')),
        key=get_int_from_path
    )
