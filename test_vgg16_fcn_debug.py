import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

import vgg16_fcn as vgg16_fcn
import utils

from tensorflow.python.framework import ops

os.environ['CUDA_VISIBLE_DEVICES'] = ''

img1 = skimage.io.imread("./test_data/tabby_cat.png")
img1 = img1 / 255.0

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = vgg16_fcn.Vgg16FCN()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images)

    print('Finished building Network.')

    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())

    print('Starting to run Network.')

    pred_slice = sess.run(vgg_fcn.pred_slice, feed_dict=feed_dict)
    pred_color = utils.color_image(pred_slice[0])
    scp.misc.imsave('down.png', pred_color)

    up = sess.run(vgg_fcn.pred_up, feed_dict=feed_dict)

    pred_color = utils.color_image(up[0])
    scp.misc.imsave('up.png', pred_color)
