# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:08:03 2018

@author: Alessandro Masullo
"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input, Concatenate, Lambda
from keras.layers import Conv1D, Conv3D, MaxPooling1D, AveragePooling1D, MaxPooling3D, BatchNormalization
import keras.backend as K

def NetworkSilhouette(img_rows, img_cols, Nchannels):
    main_input = Input(shape=(img_rows, img_cols, Nchannels))

    x = Lambda(lambda x: K.expand_dims(x,4))(main_input)
    
    x = Conv3D(4, kernel_size=(3,3,3), padding='same', strides=(2,2,1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2,1), strides=(2,2,1))(x)
    
    x = Conv3D(8, kernel_size=(3,3,3), padding='same', strides=(2,2,1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2,1), strides=(2,2,1))(x)
    
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    
    x = Dense(400)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    main_output = Dense(1)(x)
    main_output = Activation('relu')(main_output)
    
    model = Model(inputs=main_input, outputs=main_output)
    
    return model

def NetworkAccelerometer(New_bufsiz, Nchannels):
    main_input = Input(shape=(New_bufsiz, Nchannels))
    channels = []
    for i in range(Nchannels):
        y = Lambda(lambda t: K.expand_dims(t[:,:,i],2))(main_input)
    
        y = Conv1D(8, kernel_size=(5), padding='same', strides=(1))(y)
        y = Activation('relu')(y)
        y = AveragePooling1D(pool_size=(4), strides=(2))(y)
    
        y = Conv1D(4, kernel_size=(5), padding='same', strides=(1))(y)
        y = Activation('relu')(y)
        y = AveragePooling1D(pool_size=(4), strides=(2))(y)
        y = Flatten()(y)
        
        channels.append(y)
    
    x = Concatenate()(channels)
    x = Dense(400)(x)
    
    main_output = Dense(1,kernel_initializer=keras.initializers.Zeros())(x)
    main_output = Activation('linear')(main_output)
    
    model = Model(inputs=main_input, outputs=main_output)
    
    return model

def NetworkZhu(New_bufsiz, Nchannels):
    main_input = Input(shape=(New_bufsiz, Nchannels))
    channels = []
    for i in range(Nchannels):
        y = Lambda(lambda t: K.expand_dims(t[:,:,i],2))(main_input)
    
        y = Conv1D(8, kernel_size=(5), padding='same', strides=(1))(y)
        y = Activation('tanh')(y)
        y = AveragePooling1D(pool_size=(4), strides=(2))(y)
    
        y = Conv1D(4, kernel_size=(5), padding='same', strides=(1))(y)
        y = Activation('tanh')(y)
        y = AveragePooling1D(pool_size=(4), strides=(2))(y)
        y = Flatten()(y)
    
        channels.append(y)
    
    x = Concatenate()(channels)
    x = Dense(400)(x)
       
    main_output = Dense(1,kernel_initializer=keras.initializers.Zeros())(x)
    main_output = Activation('linear')(main_output)
    
    model = Model(inputs=main_input, outputs=main_output)
    
    return model

def NetworkCombined(img_rows, img_cols, Nv_chan, acc_buffersiz, Na_chan):
    input_video = Input(shape=(img_rows, img_cols, Nv_chan), name='input_video')
    input_accel = Input(shape=(acc_buffersiz, Na_chan), name='input_accel')
    
    # Video network
    vid = Lambda(lambda t: K.expand_dims(t, 4))(input_video)
    
    vid = Conv3D(4, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 1))(vid)
    vid = Activation('relu')(vid)
    vid = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(vid)
    
    vid = Conv3D(8, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 1))(vid)
    vid = Activation('relu')(vid)
    vid = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(vid)
    
    vid = Dropout(0.25)(vid)
    vid = Flatten()(vid)
    
    # Accelerometer network
    channels = []
    for i in range(Na_chan):
        acc = Lambda(lambda t: K.expand_dims(t[:, :, i], 2))(input_accel)
    
        acc = Conv1D(8, kernel_size=(5), padding='same', strides=(1))(acc)
        acc = Activation('relu')(acc)
        acc = AveragePooling1D(pool_size=(4), strides=(2))(acc)
    
        acc = Conv1D(4, kernel_size=(5), padding='same', strides=(1))(acc)
        acc = Activation('relu')(acc)
        acc = AveragePooling1D(pool_size=(4), strides=(2))(acc)
        acc = Flatten()(acc)
    
        channels.append(acc)
    
    acc = Concatenate()(channels)
    
    # Combine them
    comb = Concatenate()([acc, vid])
    
    comb = Dense(400)(comb)
    comb = Activation('relu')(comb)
    comb = Dropout(0.5)(comb)
    
    main_output = Dense(1)(comb)
    main_output = Activation('relu')(main_output)
    
    model = Model(inputs=(input_video, input_accel), outputs=main_output)
    
    return model