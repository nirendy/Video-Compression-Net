import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
import os
import argparse

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", default="demo/input/",
        help="Folder1\n"
             "Default=`demo/input/`"
    )
    parser.add_argument(
        "--output", "-o", default="demo/input/",
        help="Folder2\n"
             "Default=`demo/input/`"
    )

    parseargs = parser.parse_args()
    return parseargs

def psnr_mssim_calc(args_input=None, args_output=None):
    if args_input[-1] is not '/':
        args_input += '/'

    if args_output[-1] is not '/':
        args_output += '/'

    if not os.path.exists(args_output):
        os.mkdir(args_output)

    w, h, _ = np.array(Image.open(os.path.join(args_input, "im1.png"))).shape
    testtfprvs = tf.placeholder(tf.float32, shape=[w, h, 3], name="testfirst_frame")
    testtfnext = tf.placeholder(tf.float32, shape=[w, h, 3], name="testsecond_frame")

    msssim = tf.squeeze(tf.image.ssim_multiscale(testtfprvs, testtfnext, 255))
    psnr = tf.squeeze(tf.image.psnr(testtfprvs, testtfnext, 255))

    testinit = tf.global_variables_initializer()

    num_frames = 0
    for _ in os.listdir(args_input):
        num_frames += 1

    with tf.Session() as sess:
        sess.run(testinit)
        totpsnr = 0
        totmsssim = 0
        count = 0
        for i in range(1, num_frames + 1):
            tenFirst = np.array(Image.open(args_input + 'im' + str(i) + '.png')).astype(np.float32)
            tenSecond = np.array(Image.open(args_output + str(i) + '.png')).astype(np.float32)
            ps, msim = sess.run([psnr, msssim], feed_dict={testtfprvs: tenFirst, testtfnext: tenSecond})
            # print(ps, msim)
            totpsnr += ps
            totmsssim += msim
            count += 1

        print("Average")
        print("psnr = {:.8f}, ms-ssim ={:.8f}".format(totpsnr / count, totmsssim / count))
        return totpsnr / count, totmsssim / count


if __name__ == "__main__":
    args = parse_args()
    psnr_mssim_calc(args.input, args.output)
