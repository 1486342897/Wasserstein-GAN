#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:17:01 2017

@author: liuy
"""

import tensorflow as tf
import numpy as np
import scipy.misc


def conv2d(input_, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
           scope="conv2d"):
    with tf.variable_scope(scope):
        Weights = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, Weights, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv

def conv2d_transpose(input_, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
                     scope="conv2d_transpose"):
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        Weights = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        conv_trans = tf.nn.conv2d_transpose(input_, Weights, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        conv_trans = tf.nn.bias_add(conv_trans, biases)

        return conv_trans

def linear(input_, output_size, scope="Linear", stddev=0.02):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        Weights = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, Weights) + biases
    
def batch_norm(x, phase_train, scope='bn', decay=0.9, eps=1e-5, stddev=0.02):
    return tf.contrib.layers.batch_norm(x,
                                        decay=decay,
                                        epsilon=eps,
                                        scale=True,
                                        is_training=phase_train,
                                        updates_collections=None,
                                        scope=scope)
    
def leaky_relu(x, alpha=0.2, name="leaky_relu"):
    return tf.maximum(alpha * x, x, name)

def center_crop(images, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = images.shape[:2]
    i = int(round((h - crop_h)/2.))
    j = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(images[i:i+crop_h, j:j+crop_w], [resize_w, resize_w])

def transform(images, crop_h=64, crop_w=None, is_crop=True, resize_w=64):
    # Mapping into [-1, 1]
    if crop_w is None:
        crop_w = crop_h
    if is_crop:
        cropped_images = center_crop(images, crop_h, crop_w, resize_w)
    else:
        cropped_images = images
        
    return np.array(cropped_images)/127.5 - 1

def inverse_transform(images):
    # Mapping back into [0,1]
    return (images + 1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros(shape=(h*size[0], w*size[1], 3))
    for idx, image in enumerate(images):
        if idx >= size[0]*size[1]:
            break
        
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def get_images(image_path, crop_size, is_crop=True, resize_w=64):
    return transform(scipy.misc.imread(image_path).astype(np.float), crop_size, is_crop=is_crop, resize_w=resize_w)

def save_images(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def save_single_image(images, path):
    for idx, image in enumerate(images):
        scipy.misc.imsave(path + '_{:02d}.png'.format(idx), image)
