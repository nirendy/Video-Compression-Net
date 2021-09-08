import tensorflow.compat.v1 as tf
import math
import os
import pathlib
from typing import List

import numpy as np
from PIL import Image
import streamlit as st

from compress import compress
from decompress import decompress
from demo_app.consts import CHECKPOINTS_PATH
from demo_app.consts import FINETUNE_PREFIX
from demo_app.consts import VIDEO_DATASET_PATH, WORKING_DIR_PATH
from demo_app.utils.file_utils import copytree
from demo_app.utils.file_utils import get_frame_paths
from demo_app.utils.format_utils import get_int_from_path
from fine_tune import fine_tune
from utils import write_png


def _get_dataset_dir() -> pathlib.Path:
    return pathlib.Path(os.getcwd()) / VIDEO_DATASET_PATH


def _get_demo_cache() -> pathlib.Path:
    return pathlib.Path(os.getcwd()) / WORKING_DIR_PATH


def get_videos_names() -> List[pathlib.Path]:
    return list(_get_dataset_dir().glob('*'))

@st.cache()
def get_available_checkpoints_weights() -> List[pathlib.Path]:
    checkpoint_dir = pathlib.Path(os.getcwd()) / CHECKPOINTS_PATH
    return sorted(
        list(checkpoint_dir.glob('video*')),
        key=get_int_from_path
    )


class Demo:
    def __init__(self, demo_name):
        self.demo_path = _get_demo_cache() / demo_name
        self.demo_path.mkdir(exist_ok=True)

    @property
    def input_path(self):
        return self.demo_path / 'input'

    @property
    def compression_path(self):
        return self.demo_path / 'compressed'

    @property
    def reconstructed_path(self):
        return self.demo_path / 'reconstructed'

    @property
    def fine_tune_path(self):
        return self.demo_path / 'fine_tuned'

    @property
    def reconstructed_diff_path(self):
        return self.demo_path / 'reconstructed_diff'

    def get_finetune_weights(self) -> pathlib.Path:
        return list(self.fine_tune_path.glob(FINETUNE_PREFIX + '*'))[0]

    def get_frequency(self) -> int:
        l1 = len(list(self.input_path.iterdir()))
        l2 = len(list(self.compression_path.glob('*.png')))
        return math.ceil(l1 / l2)

    def load_input_video(self, video_path):
        if self.input_path.exists():
            return
        self.input_path.mkdir()
        copytree(video_path, self.input_path)

    def get_input_dim(self):
        first_frame = next(self.input_path.iterdir())
        return np.array(Image.open(first_frame)).shape

    def run(
            self,
            weight_file: pathlib.Path = None,
            frequency: int = None,
            learning_rate: float = None,
            fine_tune_iterations: int = None
    ):

        finetuned_weights = (
            self.get_finetune_weights()
            if weight_file is None
            else self.fine_tune_path / (FINETUNE_PREFIX + weight_file.name)
        )

        if frequency is None:
            frequency = self.get_frequency()

        if not finetuned_weights.exists():
            fine_tune(
                input_dir=self.input_path,
                weights_pklfile=weight_file,
                train_lamda=get_int_from_path(weight_file),
                weights_pklfile_out=finetuned_weights,
                learning_rate=learning_rate,
                fine_tune_iterations=fine_tune_iterations
            )

        if not self.compression_path.exists():
            compress(
                args_output=self.compression_path,
                args_input=self.input_path,
                args_model=weight_file,
                args_finetune_model=finetuned_weights,
                args_frequency=frequency
            )

        if not self.reconstructed_path.exists():
            decompress(
                args_output=self.reconstructed_path,
                args_input=self.compression_path,
                args_model=weight_file,
                args_finetune_model=finetuned_weights,
                args_frequency=frequency
            )

        if not self.reconstructed_diff_path.exists():
            input_frame_paths = get_frame_paths(self.input_path)
            reconstructed_path = get_frame_paths(self.reconstructed_path)
            self.reconstructed_diff_path.mkdir()
            for frame_i, (in_path, rec_path) in enumerate(zip(input_frame_paths, reconstructed_path)):
                diff_image = np.array(Image.open(in_path)) - np.array(Image.open(rec_path))
                with tf.Session() as sess:
                    sess.run(
                        write_png(
                            str(self.reconstructed_diff_path / f"diff{frame_i}.png"),
                            np.expand_dims(diff_image, axis=0)
                        )
                    )
