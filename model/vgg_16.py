#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 03:25:16 2020
@author: J.Lee
Model_01_VGGnet
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import tensorflow as tf

from keras.layers import Input, Activation, Conv1D, Conv2D, Conv3D, Flatten, Dense, MaxPooling1D, MaxPooling2D, MaxPooling3D,GlobalAveragePooling3D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau 

def vgg16_3D(input_image_size, model_class_num):
    
    #kernel_init = keras.initializers.glorot_uniform()
    #bias_init = keras.initializers.Constant(value=0.2)
    conv1_filt_num = 64
    conv2_filt_num = 128
    conv3_filt_num = 256
    conv4_filt_num = 512
    FC_first_second = 4096
    FC_last = model_class_num
    
    input_layer = Input(shape=input_image_size)
    
    conv1_1 = Conv3D(conv1_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1_1')(input_layer) #, kernel_initializer=kernel_init, bias_initializer=bias_init
    conv1_2 = Conv3D(conv1_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1_2')(conv1_1)
    pool1 = MaxPooling3D(pool_size=2, strides=2, name='pool1')(conv1_2)
    
    conv2_1 = Conv3D(conv2_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2_1')(pool1)
    conv2_2 = Conv3D(conv2_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2_2')(conv2_1)
    pool2 = MaxPooling3D(pool_size=2, strides=2, name='pool2')(conv2_2)
    
    conv3_1 = Conv3D(conv3_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3_1')(pool2)
    conv3_2 = Conv3D(conv3_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3_2')(conv3_1)
    conv3_3 = Conv3D(conv3_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3_3')(conv3_2)
    pool3 = MaxPooling3D(pool_size=2, strides=2, name='pool3')(conv3_3)
    
    conv4_1 = Conv3D(conv4_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4_1')(pool3)
    conv4_2 = Conv3D(conv4_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4_2')(conv4_1)
    conv4_3 = Conv3D(conv4_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4_3')(conv4_2)
    pool4 = MaxPooling3D(pool_size=2, strides=2, name='pool4')(conv4_3)
    
    conv5_1 = Conv3D(conv4_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv5_1')(pool4)
    conv5_2 = Conv3D(conv4_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv5_2')(conv5_1)
    conv5_3 = Conv3D(conv4_filt_num, kernel_size=3, strides=1, padding='same', activation='relu', name='conv5_3')(conv5_2)
    pool5 = MaxPooling3D(pool_size=2, strides=2, name='pool5')(conv5_3)

    flatten1 = GlobalAveragePooling3D(name='CAM_pool')(pool5)

    #flatten_6 = Flatten()(pool5)
    FC1 = Dense(FC_first_second, activation='relu', name='fc1')(flatten1)
    FC2 = Dense(FC_last, activation='relu', name='fc2')(FC1)
    #FC3 = Dense(FC_last, activation='relu', name='fc3')(FC2)
    
    outputs = Activation('linear')(FC2)
    
    model = Model(inputs=input_layer, outputs=outputs, name='vgg_3D')
    
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    #model.summary()
        
    return model