import skimage
import skimage.io
import skimage.transform

import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

import vgg16
import vgg16_fcn as vgg16_fcn
import utils

from tensorflow.python.framework import ops

img1 = skimage.io.imread("./test_data/tabby_cat.png")
img1 = img1 / 255.0
#img2 = utils.load_image("./test_data/puzzle.jpeg")

#batch1 = img1.reshape((1, 224, 224, 3))
#batch2 = img2.reshape((1, 224, 224, 3))

#batch = np.concatenate((batch1, batch2), 0)

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)


    vgg_fcn = vgg16_fcn.Vgg16_fcn()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images)

    pred = tf.argmax(vgg_fcn.fc8, dimension=3)

    pred = sess.run(pred, feed_dict=feed_dict)

    pred_color = utils.color_image(pred[0])
    import ipdb
    ipdb.set_trace()
    scp.misc.imsave('color.png', pred_color)




    print('Test')


    #print(prob)
    #utils.print_prob(prob[0], './synset.txt')
    #utils.print_prob(prob[1], './synset.txt')