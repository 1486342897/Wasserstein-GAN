#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:37:19 2017

@author: liuy
"""
from __future__ import print_function
from DCWGAN import DCWGAN
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='path to dataset')
parser.add_argument('--sample', required=True, help='path to sample images of the generator')
parser.add_argument('--real', required=True, help='path to sample images of the original dataset')
parser.add_argument('--logs_dir', required=True, help='path to save logs and models')

parser.add_argument('--single', type=bool, default=False, help='whether to save single images')
parser.add_argument('--sample_single', default=None, help='path to sample images of the generator')
parser.add_argument('--real_single', default=None, help='path to sample images of the original dataset')
parser.add_argument('--batchsize', type=int, default=64, help='the size of a minibatch')
parser.add_argument('--cropsize', type=int, default=160, help='the height / width of the cropped image')
parser.add_argument('--imagesize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
args = parser.parse_args()


if not os.path.exists(args.sample):
    os.makedirs(args.sample)
if not os.path.exists(args.real):
    os.makedirs(args.real)
if args.single:
    if args.sample_single is None or args.real_single is None:
        print('Please provide the path to save single images')
    else:
        if not os.path.exists(args.sample_single):
            os.makedirs(args.sample_single)
        if not os.path.exists(args.real_single):
            os.makedirs(args.real_single)
    
model = DCWGAN(args.dataset, args.sample, args.real, args.single, args.sample_single, args.real_single,
               batch_size=args.batchsize, crop_size=args.cropsize, image_size=64)
model.create_network(args.learning_rate)
model.initialize_network(args.logs_dir)
model.train()