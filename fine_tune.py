import tensorflow.compat.v1 as tf
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
    parser.add_argument("--chkfile", "-c", default="checkpoints/videocomp.chk",
                        help="Checkpointing file\n"
                             "Default=`checkpoints/videocomp.chk`")

    parser.add_argument("--pklfile", "-p", default="checkpoints/videocomp.pkl",
                        help="Pkl file to save weights of the trained network\n"
                             "Default=`checkpoints/videocomp.pkl`")

    parser.add_argument("--input", "-i", default="vimeo_septuplet/sequences/",
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
                             "Default=`vimeo_septuplet/sequences/`")

    parser.add_argument("--frequency", "-f", type=int, default=25,
                        help="Number of steps to saving the checkpoints\n"
                             "Default=25")

    parser.add_argument("--lamda", "-l", type=int, default=4096,
                        help="Weight assigned to reconstruction loss compared to bitrate during training. \n"
                             "Default=4096")

    parser.add_argument("--restore", "-r", action="store_true",
                        help="Whether to restore the checkpoints to continue interrupted training, OR\n"
                             "Start training from the beginning")

    parseargs = parser.parse_args()
    return parseargs


if __name__ == "__main__":
    args = parse_args()
    net = VideoCompressor(finetune=True)
    subdircount = len(os.listdir(args.input))
    print('Starting')

    tfprvs = tf.placeholder(tf.float32, shape=[4, 240, 416, 3], name="first_frame")
    tfnext = tf.placeholder(tf.float32, shape=[4, 240, 416, 3], name="second_frame")

    l_r = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    lamda = tf.placeholder(tf.int16, shape=[], name="train_lambda")

    recon, mse, bpp = net(tfprvs, tfnext)
    train_loss = tf.cast(lamda, tf.float32) * mse + bpp
    train = tf.train.AdamOptimizer(learning_rate=l_r).minimize(train_loss)
    aux_step1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(net.ofcomp.entropy_bottleneck.losses[0])
    aux_step2 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(net.rescomp.entropy_bottleneck.losses[0])

    tfvideo_batch = tf.get_variable("tfvideo_batch", initializer=tf.constant(0))
    increment_video_batch = tf.assign(tfvideo_batch, tfvideo_batch + 1)
    directory = tf.get_variable("directory", initializer=tf.constant(1))

    increment_directory = tf.assign(directory, directory + 1)
    init_video_batch_updater = tf.assign(tfvideo_batch, 0)
    init_directory_updater = tf.assign(directory, 1)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    starting = args.restore

    print('Starting Session')
    with tf.Session() as sess:
        sess.run(init)
        with open(args.pklfile, "rb") as f:
            net.set_weights(pkl.load(f))
        if starting:
            saver.restore(sess, args.chkfile)

        lr = 1e-6
        lmda = args.lamda

        print("lr={}, lambda = {}".format(lr, lmda))
        load_dir = directory.eval() if starting else 1

        num_of_pictures = len(
            [name for name in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, name))])

        for fine_tune_i in range(1):
            print(f"fine_tune_i: {fine_tune_i}")

            # for i in range(0, num_of_pictures-6, 8):
            for i in range(0, 1, 8):
                print("Picture number " + str(i + 1))
                for batch in range(1, 8):
                    print(f"batch: {batch}")
                    bat = os.path.join(args.input, 'im' + str(i + batch) + '.png')
                    bat = np.array(Image.open(bat)).astype(np.float32) * (1.0 / 255.0)
                    bat = np.expand_dims(bat, axis=0)
                    for item in range(2, 5):
                        img = os.path.join(args.input, 'im' + str(i + batch) + '.png')
                        img = np.array(Image.open(img)).astype(np.float32) * (1.0 / 255.0)
                        img = np.expand_dims(img, axis=0)
                        bat = np.concatenate((bat, img), axis=0)

                    if batch == 1:
                        prevReconstructed = bat

                    else:
                        recloss, rate, rec, _, _, _, _, _ = sess.run([mse, bpp, recon, train, aux_step1,
                                                                      net.ofcomp.entropy_bottleneck.updates[0],
                                                                      aux_step2,
                                                                      net.rescomp.entropy_bottleneck.updates[0]],
                                                                     feed_dict={tfprvs: prevReconstructed,
                                                                                tfnext: bat, l_r: lr,
                                                                                lamda: lmda})
                        prevReconstructed = rec

                increment_video_batch.op.run()
                print("recon loss = {:.8f}, bpp = {:.8f}".format(recloss, rate))

            pkl.dump(net.ofcomp.get_weights(), open(args.pklfile[:-4] + '_finetune_of_weights.pkl', "wb"))
            saver.save(sess, args.chkfile)

