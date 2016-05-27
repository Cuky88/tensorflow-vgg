import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16FCN:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")
            print(path)
            vgg16_npy_path = path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb, train=False):
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_2 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_2, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')

        self.fc6 = self._fc_layer(self.pool5, "fc6")
        # assert self.fc6.get_shape().as_list()[1:] == [4096]

        self.relu6 = tf.nn.relu(self.fc6)
        if train:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self._fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = self._fc_layer(self.relu7, "fc8")

        self.pred = tf.argmax(self.fc8, dimension=3)

        self.up = self._upscore_layer(self.fc8, shape=tf.shape(bgr),
                                      num_classes=1000,
                                      name='up', ksize=64, stride=32)
        # self.reshape = tf.reshape(self.fc8, [-1, 1000])

        # self.prob = tf.nn.softmax(self.reshape, name="prob")

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            # print("bottom.get_shape(): %s"% shape)
            # dim = 1
            # for d in shape[1:]:
            #     dim *= d
            # x = tf.reshape(bottom, [-1, dim])
            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'fc8':
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000])
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name,
                       ksize=4, stride=2,
                       wd=5e-4):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value
            new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.pack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape, wd)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')
        return deconv

    def get_deconv_filter(self, f_shape, wd):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        weights = np.zeros(f_shape)
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                f_shape[x, y, :, :].fill(value)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter",
                               initializer=init, shape=f_shape)

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        return tf.get_variable(name="filter", initializer=init, shape=shape)

    def get_bias(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][1],
                                       dtype=tf.float32)
        shape = self.data_dict[name][1].shape
        return tf.get_variable(name="biases", initializer=init, shape=shape)

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        return tf.get_variable(name="weights", initializer=init, shape=shape)

    def get_fc_weight_reshape(self, name, shape):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init, shape=shape)
