#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ Modified by Jeyeon Lee
author: Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten
        Densely Connected Convolutional Networks
        arXiv:1608.06993
        (See https://github.com/flyyufelix/DenseNet-Keras/blob/master/densenet161.py)
"""

from keras.models import Model
from keras.layers import (
    Input, concatenate, ZeroPadding3D,
    Conv3D, Dense, Dropout, Activation,
    AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D,
    BatchNormalization
)
import keras.backend as K

from custom_layers import Scale

def build_densenet(input_shape, densenettype,nc):
    '''Instantiate the DenseNet 161 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5
    nb_dense_block=4 #4
    growth_rate=48
    nb_filter=96
    reduction=0.0
    dropout_rate=0.0
    weight_decay=1e-4
    weights_path=None
    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    #if K.image_data_format() == 'tf':
    concat_axis = 4
    img_input = Input(shape=input_shape, name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    if densenettype == -2:
        nb_dense_block=2
        nb_filter = 64
        nb_layers = [3,6] # For DenseNet-CAM
    elif densenettype == -1:
        nb_filter = 64
        nb_layers = [3,6,6,4] # For DenseNet-CAM
    elif densenettype == 0:
        nb_filter = 64
        nb_layers = [3,6,12,8] # For DenseNet-CAM
    elif densenettype == 1:
        nb_filter = 64
        nb_layers = [6,12,24,16] # For DenseNet-121
    elif densenettype == 2:
        nb_filter = 96
        nb_layers = [6,12,36,24] # For DenseNet-161
    elif densenettype == 3:    
        nb_filter = 64
        nb_layers = [6,12,32,32] # For DenseNet-169

    # Initial convolution
    x = ZeroPadding3D((3, 3, 3), name='conv1_zeropadding')(img_input)
    if densenettype == -2 or densenettype == -1 or densenettype == 0:
        x = Conv3D(nb_filter, (5, 5, 5), strides=(2, 2, 2), name='conv1', use_bias=False)(x) #7 7 7
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding3D((1, 1, 1), name='pool1_zeropadding')(x)
    if densenettype == -1 or densenettype == 0:
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), name='pool1')(x)
    else:
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), name='pool1')(x)
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, \
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, \
                             dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, \
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)


        # add lastconv and global average pool for CAM
    CAM_conv = Conv3D(filters=x._keras_shape[4],
                                kernel_size=(3, 3, 3),
                                strides=(1, 1, 1), padding="same",
                                name='CAM_conv')(x)
    flatten1 = GlobalAveragePooling3D(name='CAM_pool')(CAM_conv)

    if nc == 1:
        x = Dense(1,name='fc')(flatten1)
        x = Activation('linear')(x)
        model = Model(img_input, x, name='densenetregression')
    else:
        x = Dense(nc, name='fc')(flatten1)
        x = Activation('softmax', name='prob')(x)
        model = Model(img_input, x, name='densenetclassification')

    if weights_path is not None:
      model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv3D(inter_channel, (1, 1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding3D((1, 1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv3D(nb_filter, (3, 3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

#    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], name='concat_'+str(stage)+'_'+str(branch))
#        concat_feat = add([concat_feat, x], name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
    
    