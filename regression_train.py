#!/usr/bin/env python
# coding: utf-8

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:36:37 2019

for 3D CAM (DenseNet, ResNet50, InceptionV3)

@author: J.Lee

"""
import inspect, os, sys, h5py, shutil
import scipy.io as sio
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from densenet3d_regression import build_densenet_forCAM
from resnet_3d import Resnet3DBuilder
from keras.utils import plot_model
from vgg_16 import vgg16_3D
from sfcn_keras import SFCN
from datetime import datetime

#################### VARIABLES ####################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
j = 4  # nth_fold 0-4
learning_rate = 0.001
modality = 'mri'
model_id = 2  # 0: densenet,   1: resnet101,    2: VGG16,   3: SFCN
datapath = '/home/m186870/data_j6/dl_data/MRI_masked_bc/regression'
datafilename = 'mri_age_optB4_v2_fold' + str(j) + '.mat'
codepath = '/home/m186870/data_j6/dl_data/dl_code'
lru, bs = 0.5, 4
fit_iter, fit_ep = 10, 15
###################################################

currentcode = inspect.getfile(inspect.currentframe())
print('\n [Info] Running code: ', currentcode)

def Data_Load(mat_file_name):
    os.chdir(datapath)
    mat_contents = h5py.File(mat_file_name, 'r')
    X_Train = mat_contents['X_Train']
    Y_Train = mat_contents['Y_Train']
    X_Train = np.transpose(X_Train)  # for transpose becauseof HDF matfile v7.3
    Y_Train = np.transpose(Y_Train)
    X_Val = mat_contents['X_Val']
    Y_Val = mat_contents['Y_Val']
    X_Val = np.transpose(X_Val)  # for transpose becauseof HDF matfile v7.3
    Y_Val = np.transpose(Y_Val)
    X_Test = mat_contents['X_Test']
    Y_Test = mat_contents['Y_Test']
    X_Test = np.transpose(X_Test)  # for transpose becauseof HDF matfile v7.3
    Y_Test = np.transpose(Y_Test)

    nanidx = np.isnan(X_Train)
    X_Train[nanidx] = 0
    nanidx = np.isnan(X_Test)
    X_Test[nanidx] = 0
    nanidx = np.isnan(X_Val)
    X_Val[nanidx] = 0

    if Y_Train.shape[0] < Y_Train.shape[1]:
        Y_Train = np.transpose(Y_Train)
    if Y_Test.shape[0] < Y_Test.shape[1]:
        Y_Test = np.transpose(Y_Test)
    if Y_Val.shape[0] < Y_Val.shape[1]:
        Y_Val = np.transpose(Y_Val)

    print('\tDatafile: ', datapath, datafilename)
    print('\tX_Train shape :', X_Train.shape)
    print('\tY_Train shape :', Y_Train.shape)
    print('\tX_Val shape :', X_Val.shape)
    print('\tY_Val shape :', Y_Val.shape)
    print('\tX_Test shape :', X_Test.shape)
    print('\tY_Test shape :', Y_Test.shape)
    return X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test

os.chdir(codepath)
print(codepath)

# %% Build Model
if model_id == 0:  # densenet3d
    modelname = 'densenet3dregression'
    model = build_densenet_forCAM((121, 145, 121, 1), 0)
    opt = 'Adam'
    outpath = datapath+'/paper_'+modality+'_age_optb4_densenet_lr' + str(
        learning_rate) + '_' + opt + '/'
if model_id == 1:  # Resnet
    modelname = 'Resenet101'
    model = Resnet3DBuilder.build_resnet_101((121, 145, 121, 1), 1)
    opt = 'Adam'
    outpath = datapath+'/paper_'+modality+'_age_optb4_resnet_lr' + str(
        learning_rate) + '_' + opt + '/'
if model_id == 2:  # VGG
    modelname = 'VGG16'
    model = vgg16_3D((121, 145, 121, 1), 1)
    opt = 'SGD'
    outpath = datapath+'/paper_'+modality+'_age_optb4_vgg_lr' + str(
        learning_rate) + '_' + opt + '/'
if model_id == 3:  # SFCN
    modelname = 'SFCN'
    model = SFCN((121, 145, 121, 1), 'False')
    opt = 'SGD'
    outpath = datapath+'/paper_'+modality+'_age_optb4_sfcn_lr' + str(
        learning_rate) + '_' + opt + '/'

##################################
if not os.path.exists(datapath):
    os.makedirs(datapath)
os.chdir(datapath)
X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test = Data_Load(datafilename)
X_Train = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1], X_Train.shape[2], X_Train.shape[3], 1))
X_Val = np.reshape(X_Val, (X_Val.shape[0], X_Val.shape[1], X_Val.shape[2], X_Val.shape[3], 1))
X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1], X_Test.shape[2], X_Test.shape[3], 1))

print(datetime.now())
print('\n [Info] Data loading done')

if not os.path.exists(outpath):
    os.makedirs(outpath)
print('\tProcessing path: ', outpath)

os.chdir(outpath)
print('\n [Info] Model set: ', modelname)
plot_model(model, to_file=modelname + '.pdf', show_shapes=True)
model.summary()
with open(modelname + '.txt', 'w') as f2:
    model.summary(print_fn=lambda x: f2.write(x + '\n'))

# %% Training
print("\n [Info] Training Start!")

for i in range(fit_iter):
    print("\t Validating setnum:", str(j), "-Training iter:", str(i + 1))
    filepath_weights_best = './weights.best_' + str(j) + 'fold.h5'
    filepath_weights_best_past = './weights.best_' + str(j) + 'fold.h5'

    if i > 0:
        model.load_weights(filepath_weights_best_past)
        learning_rate *= lru
        print('\tcurrent leraning_rate : ', learning_rate)

    if 'Adam' in opt:
        if model_id == 2:
            optm = Adam(lr=learning_rate)
        else:
            optm = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif 'SGD' in opt:
        optm = SGD(lr=learning_rate)
    elif 'RMSprop' in opt:
        optm = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_absolute_error', optimizer=optm)

    checkpoint = ModelCheckpoint(filepath_weights_best, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')
    csv_logger = CSVLogger('training.log', separator=',', append=True)
    callbacks_list = [csv_logger, checkpoint, earlystop]
    trained_model = model.fit(X_Train, Y_Train, batch_size=bs, epochs=fit_ep, verbose=1, shuffle=True,
                              callbacks=callbacks_list, validation_data=(X_Val, Y_Val))

    # list all data in history
    print(trained_model.history.keys())

    model.load_weights(filepath_weights_best)

    y_pred = model.predict(X_Test, batch_size=bs)
    mae = mean_absolute_error(y_pred, Y_Test)

    y_valpred = model.predict(X_Val, batch_size=bs)
    valmae = mean_absolute_error(y_valpred, Y_Val)

    print("[INFO] Test Mean absolute error: {:.4f}".format(mae))
    with open('Acc_' + str(j) + 'fold_iter' + str(i + 1) + '.txt', 'w') as f:
        print("\nlearning_rate:" + str(learning_rate), file=f)
        print("\nValidation mean absolute error: {:.4f}".format(valmae), file=f)
        print("\nTest mean absolute error: {:.4f}".format(mae), file=f)

    outname = 'regression_score_test_' + str(j) + 'fold.mat'
    sio.savemat(outpath + outname, {'Ypred': y_pred, 'Ydata': Y_Test})




