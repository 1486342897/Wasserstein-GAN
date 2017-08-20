#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:11:57 2017

@author: liuy
"""
from __future__ import print_function
import tensorflow as tf
import utils as utils
import glob
import os
import numpy as np
import time


class DCWGAN(object):
    def __init__(self, data_dir, sample_dir, real_dir, single,
                 sample_single_dir=None, real_single_dir=None, batch_size=64, 
                 crop_size=100, is_crop=True, image_size=64,
                 z_dim=100, df_dim=64, gl_dim=64, num_epoch=25):
        self.data_dir = data_dir
        self.sample_dir = sample_dir
        self.real_dir = real_dir
        self.single = single
        self.sample_single_dir = sample_single_dir
        self.real_single_dir = real_single_dir
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.is_crop = is_crop
        self.image_size = image_size
        self.z_dim = z_dim
        self.df_dim = df_dim    # channels of the first layer of the discriminator
        self.gl_dim = gl_dim    # channels of the last but one layer of the generator
        self.epoch = num_epoch
        
    def generator(self, z, is_train=True, scope_name="generator"):
        with tf.variable_scope(scope_name):
            h0 = utils.linear(z, 4*4* self.gl_dim *8, scope='g_h0_lin')
            h0 = tf.reshape(h0, [-1, 4, 4, self.gl_dim *8])
            h0 = tf.nn.relu(utils.batch_norm(h0, is_train, scope='g_bn0'))
            
            h1 = utils.conv2d_transpose(h0, [self.batch_size, 8, 8, self.gl_dim *4], scope='g_h1_deconv')
            h1 = tf.nn.relu(utils.batch_norm(h1, is_train, scope='g_bn1'))
            
            h2 = utils.conv2d_transpose(h1, [self.batch_size, 16, 16, self.gl_dim *2], scope='g_h2_deconv')
            h2 = tf.nn.relu(utils.batch_norm(h2, is_train, scope='g_bn2'))
            
            h3 = utils.conv2d_transpose(h2, [self.batch_size, 32, 32, self.gl_dim], scope='g_h3_deconv')
            h3 = tf.nn.relu(utils.batch_norm(h3, is_train, scope='g_bn3'))
            
            h4 = utils.conv2d_transpose(h3, [self.batch_size, 64, 64, 3], scope='g_h4_deconv')
            
            return tf.nn.tanh(h4)
        
    def discriminator(self, images, is_train=True, reuse=False, scope_name="discriminator"):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
                
            h0 = utils.conv2d(images, self.df_dim, scope='d_h0_conv')
            h0 = utils.leaky_relu(utils.batch_norm(h0, is_train, scope='d_bn0'))
            
            h1 = utils.conv2d(h0, self.df_dim *2, scope='d_h1_conv')
            h1 = utils.leaky_relu(utils.batch_norm(h1, is_train, scope='d_bn1'))
            
            h2 = utils.conv2d(h1, self.df_dim *4, scope='d_h2_conv')
            h2 = utils.leaky_relu(utils.batch_norm(h2, is_train, scope='d_bn2'))
            
            h3 = utils.conv2d(h2, self.df_dim *8, scope='d_h3_conv')
            h3 = utils.leaky_relu(utils.batch_norm(h3, is_train, scope='d_bn3'))
            
            h4 = utils.linear(tf.reshape(h3, [self.batch_size, -1]), 1, scope='d_h4_lin')
            
            return h4
        
    def initialize_network(self, logs_dir):
        print("Initializing network...")
        self.logs_dir = logs_dir
        
        #Settings for GPU
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
#        config.gpu_options.per_process_gpu_memory_fraction = 0.3
	
        self.sess = tf.Session(config=config)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        #print(tf.train.latest_checkpoint(self.logs_dir))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")
            
    def create_network(self, learning_rate):
        print("Creating network...")
        self.images = tf.placeholder(tf.float32, 
                                     shape=[self.batch_size, self.image_size, self.image_size, 3], 
                                     name='real_images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        
        self.G = self.generator(self.z, self.is_train)
        self.D_logits_real = self.discriminator(self.images)
        self.D_logits_fake = self.discriminator(self.G, reuse=True)
        
        self.d_loss = tf.reduce_mean(self.D_logits_fake - self.D_logits_real)
        self.g_loss = tf.reduce_mean(-self.D_logits_fake)
        
        train_var = tf.trainable_variables()
        self.d_vars = [var for var in train_var if 'd_' in var.name]
        self.g_vars = [var for var in train_var if 'g_' in var.name]
        
        # Another approach
#        self.d_vars = [var for var in train_var if var.name.startswith("discriminator")]
#        self.g_vars = [var for var in train_var if var.name.startswith("generator")]
        
#        print(map(lambda x: x.op.name, self.g_vars))
#        print(map(lambda x: x.op.name, self.d_vars))
        
        self.d_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        
        self.d_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01, name='clip_op')) for var in self.d_vars]
        
#        self.d_clip = [var = tf.clip_by_value(var, -0.01, 0.01, name='clip_op') for var in self.d_vars]
        
    def train(self):
        print("Training a model...")
        data_list = []
        extensions = ['jpg', 'jpeg', 'webp', 'JPG', 'JPEG']
        for extension in extensions:
            data_glob = os.path.join(self.data_dir, '*.' + extension)
            data_list.extend(glob.glob(data_glob))
            
        np.random.shuffle(data_list)
        
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        
        for epoch in xrange(self.epoch):
            np.random.shuffle(data_list)
            batch_idx = len(data_list) // self.batch_size
            n_dis = 0
            
            for idx in xrange(0, batch_idx):
                batch_list = data_list[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [utils.get_images(batch_file, 
                                          self.crop_size,
                                          is_crop=self.is_crop,
                                          resize_w=self.image_size) for batch_file in batch_list]
                batch_images = np.array(batch).astype(np.float32)
                
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                
                _ = self.sess.run(self.d_optim, feed_dict={self.images: batch_images, 
                                                           self.z: batch_z,
                                                           self.is_train: True })
                _ = self.sess.run(self.d_clip)
                n_dis = n_dis + 1
                
                if n_dis == 5:
                    _ = self.sess.run(self.g_optim, feed_dict={self.z: batch_z,
                                                               self.is_train: True })
                    n_dis = 0
                
                if idx % 500 == 0:
                    stop_time = time.time()
                    duration = stop_time - start_time
                    d_loss_val, g_loss_val = self.sess.run([self.d_loss, self.g_loss], 
                                                           feed_dict={self.images: batch_images,
                                                                      self.z: batch_z,
                                                                      self.is_train: True })
                    print("Epoch: %d Step: %d/%d, Time: %g, d_loss: %g, g_loss: %g" \
                          % (epoch, idx, batch_idx, duration, d_loss_val, g_loss_val))
                    
                    sample_images = self.sess.run(self.G, feed_dict={self.z: batch_z,
                                                              self.is_train: False }) 
    
                    utils.save_images(batch_images, [6, 6],
                                      '{}/batch_{:02d}_{:04d}.png'.format(self.real_dir, epoch, idx))
                    utils.save_images(sample_images, [6, 6],
                                      '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                    
                    if self.single:
                        
                        utils.save_single_image(batch_images, 
                                                '{}/batch_{:02d}_{:04d}'.format(self.real_single_dir, epoch, idx))
                        utils.save_single_image(sample_images, 
                                                '{}/train_{:02d}_{:04d}'.format(self.sample_single_dir, epoch, idx))
                    
            self.saver.save(self.sess, os.path.join(self.logs_dir, 'model.ckpt'), global_step=epoch )
                
                
    
        