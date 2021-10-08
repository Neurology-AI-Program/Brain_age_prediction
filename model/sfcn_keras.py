#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jeyeon Lee
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import keras as K
import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Activation, Conv3D, Flatten, Dense,MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau 


def SFCN(input_shape,dropout):
    channel_number = [32, 64, 128, 256, 256,64] # [32, 64, 128, 256, 256, 64]
    output_dim = 1
    dropout = True
    n_layer = len(channel_number)
    img_input = Input(shape=input_shape, name='data')
    for i in range(n_layer):
        out_channel = channel_number[i]
        if i == 0:
            x = Conv3D(out_channel, kernel_size=3,strides=1, padding='same',name='conv_%d' % i)(img_input)
            x = BatchNormalization(axis=-1)(x)
            x = MaxPooling3D(pool_size=2, strides=2,name='pool_%d' % i)(x)
            x = Activation('relu')(x)
        if i>0 and i < n_layer - 1:
            x = Conv3D(out_channel, kernel_size=3,strides=1, padding='same',name='conv_%d' % i)(x)
            x = BatchNormalization(axis=-1)(x)
            x = MaxPooling3D(pool_size=2, strides=2,name='pool_%d' % i)(x)
            x = Activation('relu')(x)
        if i ==n_layer - 1:
            x = Conv3D(out_channel, kernel_size=(1,1,1),strides=(1,1,1), padding='same',name='conv_%d' % i)(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
    x = GlobalAveragePooling3D(name='GAP')(x)
    if dropout is True:
        x = Dropout(0.5)(x)
    out_channel = output_dim
    #x = Conv3D(out_channel, kernel_size=1, strides=1, padding='same', name='conv_%d' % i)(x)
    x = Dense(1, name='fc')(x)
    x = Activation('linear')(x)
    model = Model(inputs=img_input, outputs=x, name='SFCN')
    return model

#model = SFCN((121,145,121,1),'False')
#model.summary()