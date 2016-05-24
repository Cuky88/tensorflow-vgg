import skimage
import skimage.io
import skimage.transform

import numpy as np
import tensorflow as tf

import vgg16
import vgg16_fcn as vgg16_fcn
import utils

from tensorflow.python.framework import ops

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session() as sess:
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg_fcn = vgg16_fcn.Vgg16_fcn()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(images)

    prob = sess.run(vgg_fcn.prob, feed_dict=feed_dict)

    print(prob)
    utils.print_prob(prob[0], './synset.txt')
    utils.print_prob(prob[1], './synset.txt')