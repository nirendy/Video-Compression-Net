import pathlib

import tensorflow.compat.v1 as tf

from demo_app.utils.file_utils import get_image_shape
from demo_app.utils.file_utils import get_frames_paths
from utils import VideoCompressor
import numpy as np
from PIL import Image
import os
import pickle as pkl
import argparse
import os, os.path

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chkfile", "-c", default="checkpoints/videocomp.chk",
        help="Checkpointing file\n"
             "Default=`checkpoints/videocomp.chk`"
    )

    parser.add_argument(
        "--pklfile", "-p", default="checkpoints/videocomp.pkl",
        help="Pkl file to save weights of the trained network\n"
             "Default=`checkpoints/videocomp.pkl`"
    )

    parser.add_argument(
        "--input", "-i", default="vimeo_septuplet/sequences/",
        help="Directory where training data lie. The structure of the directory should be like:\n"
             "vimeo_septuplet/sequences/00001/\n"
             "vimeo_septuplet/sequences/00002\n"
             "...............................\n"
             "For each vimeo_septuplet/sequences/x there should be subfolders like:\n"
             "00001/0001\n"
             "00001/002\n"
             ".........\n"
             "Check vimeo_septuplet folder. \n"
             "Download dataset for more information. For other dataset, you can parse the input\n"
             "in your own way\n"
             "Default=`vimeo_septuplet/sequences/`"
    )

    parser.add_argument(
        "--frequency", "-f", type=int, default=25,
        help="Number of steps to saving the checkpoints\n"
             "Default=25"
    )

    parser.add_argument(
        "--lamda", "-l", type=int, default=4096,
        help="Weight assigned to reconstruction loss compared to bitrate during training. \n"
             "Default=4096"
    )

    parser.add_argument(
        "--restore", "-r", action="store_true",
        help="Whether to restore the checkpoints to continue interrupted training, OR\n"
             "Start training from the beginning"
    )

    parseargs = parser.parse_args()
    return parseargs


def fine_tune(
        input_dir,
        weights_pklfile,
        learning_rate=1e-6,
        train_lamda=1024,
        fine_tune_iterations=1,
        weights_pklfile_out=None
):
    if weights_pklfile_out is None:
        weights_pklfile_out = weights_pklfile[:-4] + '_finetune_rescomp_weights.pkl'

    pathlib.Path(weights_pklfile_out).parent.mkdir()

    net = VideoCompressor(
        finetune=True,
        is_mse=('msssim' not in str(weights_pklfile))
    )

    print('Starting')
    h, w, d = get_image_shape(get_frames_paths(input_dir)[0])
    tfprvs = tf.placeholder(tf.float32, shape=[4, h, w, d], name="first_frame")
    tfnext = tf.placeholder(tf.float32, shape=[4, h, w, d], name="second_frame")

    l_r = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    lamda = tf.placeholder(tf.int16, shape=[], name="train_lambda")

    recon, mse, bpp = net(tfprvs, tfnext)
    train_loss = tf.cast(lamda, tf.float32) * mse + bpp
    train = tf.train.AdamOptimizer(learning_rate=l_r).minimize(train_loss)
    aux_step1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(net.ofcomp.entropy_bottleneck.losses[0])
    aux_step2 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(net.rescomp.entropy_bottleneck.losses[0])

    init = tf.global_variables_initializer()

    print('Starting Session')
    with tf.Session() as sess:
        sess.run(init)
        with open(weights_pklfile, "rb") as f:
            net.set_weights(pkl.load(f))

        lr = learning_rate
        lmda = train_lamda

        print("lr={}, lambda = {}".format(lr, lmda))

        num_of_pictures = len(
            [name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))]
        )

        for fine_tune_i in range(fine_tune_iterations):
            print(f"fine_tune_i: {fine_tune_i + 1} / {fine_tune_iterations}")

            for frame_i in range(1, num_of_pictures + 1):
                print("Picture number " + str(frame_i))
                bat = os.path.join(input_dir, f'im{frame_i}.png')
                bat = np.array(Image.open(bat)).astype(np.float32) * (1.0 / 255.0)
                bat = np.expand_dims(bat, axis=0)
                for item in range(2, 5):
                    img = os.path.join(input_dir, f'im{frame_i}.png')
                    img = np.array(Image.open(img)).astype(np.float32) * (1.0 / 255.0)
                    img = np.expand_dims(img, axis=0)
                    bat = np.concatenate((bat, img), axis=0)

                if frame_i == 1:
                    prevReconstructed = bat
                else:
                    recloss, rate, rec, _, _, _, _, _ = sess.run(
                        [mse, bpp, recon, train, aux_step1,
                         net.ofcomp.entropy_bottleneck.updates[0],
                         aux_step2,
                         net.rescomp.entropy_bottleneck.updates[0]],
                        feed_dict={
                            tfprvs: prevReconstructed,
                            tfnext: bat, l_r: lr,
                            lamda: lmda
                        }
                    )
                    prevReconstructed = rec

                    print("recon loss = {:.8f}, bpp = {:.8f}".format(recloss, rate))

            pkl.dump(net.rescomp.get_weights(), open(weights_pklfile_out, "wb"))


if __name__ == "__main__":
    args = parse_args()
    fine_tune(
        input_dir=args.input,
        weights_pklfile=args.pklfile,
        chkfile=args.chkfile,
        train_lamda=args.lamda,
    )
