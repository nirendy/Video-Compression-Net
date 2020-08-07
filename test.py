import tensorflow.compat.v1 as tf
from utils.network import VideoCompressor
import numpy as np
from PIL import Image
import pickle as pkl
from utils.basics import write_png
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="checkpoints/tfvideocomp.pkl",
                        help="Saved model that you want to test with")
    parser.add_argument("--input", "-i", default="vimeo_septuplet/sequences/00001/0001/",
                        help="Directory where uncompressed frames lie and what you want to compress")
    parser.add_argument("--output", "-o", default="outputs/",
                        help="Directory where you want the reconstructed frames to be saved")
    parseargs = parser.parse_args()
    return parseargs


if __name__ == "__main__":
    args = parse_args()
    w, h, _ = np.array(Image.open(args.input + "im1.png")).shape
    testnet = VideoCompressor(training=False)
    testtfprvs = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testfirst_frame")
    testtfnext = tf.placeholder(tf.float32, shape=[1, w, h, 3], name="testsecond_frame")

    recon_image, _, estimated_bpp = testnet(testtfprvs, testtfnext)
    orig = tf.round(tf.convert_to_tensor(tf.reshape(testtfnext, [256, 448, 3])) * 255)
    rec = tf.round(tf.convert_to_tensor(tf.reshape(recon_image, [256, 448, 3])) * 255)

    mse = tf.reduce_mean(tf.math.squared_difference(orig, rec))
    psnr = tf.squeeze(tf.image.psnr(rec, orig, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(rec, orig, 255))

    testinit = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(testinit)
        with open(args.model, "rb") as f:
            testnet.set_weights(pkl.load(f))

        tenFirst = np.array(Image.open(args.input + 'im' + str(1) + '.png')).astype(np.float32) * (1.0 / 255.0)
        tenFirst = np.expand_dims(tenFirst, axis=0)
        sess.run(write_png(args.output + str(1) + ".png", tenFirst))

        for batch in range(2, 8):
            tenSecond = np.array(Image.open(args.input + 'im' + str(batch) + '.png')).astype(np.float32) * (1.0 / 255.0)
            tenSecond = np.expand_dims(tenSecond, axis=0)

            reconimage, recloss, ps, ms, rate = sess.run([recon_image, mse, psnr, msssim, estimated_bpp],
                                                         feed_dict={testtfprvs: tenFirst, testtfnext: tenSecond})
            tenFirst = reconimage

            print("recon loss = {:.8f}, psnr = {:.8f}, msssim = {:.8f}, bpp = {:.8f}".format(recloss, ps, ms, rate))
            sess.run(write_png(args.output + str(batch) + ".png", reconimage))