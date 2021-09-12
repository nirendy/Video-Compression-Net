import json
import math
import pathlib
from typing import List

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

from compress import compress
from decompress import decompress
from demo_app.consts import FINETUNE_PREFIX
from demo_app.utils.file_utils import get_frames_paths
from demo_app.utils.format_utils import get_int_from_path
from demo_app.utils.video_dataset import _get_demo_cache
from fine_tune import fine_tune
from utils import write_png


class Demo:
    def __init__(self, demo_name):
        self.demo_path = _get_demo_cache() / demo_name

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
    def fine_tune_hyper_params_path(self):
        return self.demo_path / 'hyper_params.json'

    def save_fine_tune_hyper_params(self, **kwargs):
        with self.fine_tune_hyper_params_path.open(mode='w') as f:
            json.dump(kwargs, f)

    def load_fine_tune_hyper_params(self, ):
        with self.fine_tune_hyper_params_path.open(mode='r') as f:
            return json.load(f)

    @property
    def reconstructed_diff_path(self):
        return self.demo_path / 'reconstructed_diff'

    def get_finetune_weights(self) -> pathlib.Path:
        return list(self.fine_tune_path.glob(FINETUNE_PREFIX + '*'))[0]

    def get_frequency(self) -> int:
        l1 = len(list(self.input_path.iterdir()))
        l2 = len(list(self.compression_path.glob('*.png')))
        return math.ceil(l1 / l2)

    def load_input_video(self, frame_paths: List[pathlib.Path]):
        self.demo_path.mkdir(exist_ok=True)
        if self.input_path.exists():
            return
        self.input_path.mkdir()
        for frame_i, frame_path in enumerate(frame_paths):
            target_frame_path = self.input_path / f"im{frame_i+1}.png"
            target_frame_path.write_bytes(frame_path.read_bytes())

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
            self.save_fine_tune_hyper_params(
                learning_rate=learning_rate,
                fine_tune_iterations=fine_tune_iterations,
            )
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
            input_frame_paths = get_frames_paths(self.input_path)
            reconstructed_path = get_frames_paths(self.reconstructed_path)
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
