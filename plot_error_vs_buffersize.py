# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:52:26 2018

@author: Alessandro Masullo
"""

from __future__ import division # Bluecrystal uses python 2.7
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input, Concatenate, Lambda, Maximum
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Conv1D, MaxPooling1D, AveragePooling1D
from matplotlib import pyplot as plt
import os
import CaloriesDataset
from keras.callbacks import ModelCheckpoint
from PlotLearning import PlotLearning
from SaveHistory import SaveHistory
import numpy as np
import keras.backend as K
import pickle

from NetworkArchitectures import NetworkCombined

smoothcal = 20
Nsubjects = 10
data_path = r'C:\Data\calories_sphere\processed_data'
model_path = r'C:\Data\calories_sphere\models\VariableN'
model_name = 'OurComb%d_leave_%d_out_best.h5'

buffer_tested = [250, 500, 1000, 2000]

overall_error = []

for bi in range(len(buffer_tested)):
    print('Testing buffer size %d' % buffer_tested[bi])
    for si in range(1, Nsubjects+1):
        # Read the testing data
        print('Loading data for subject %d' % si)
        (xv_test, yv_test, lab_test) = CaloriesDataset.load_range_N((si,), buffer_tested[bi], data_path)
        (xa_test, ya_test, _) = CaloriesDataset.load_acc_range((si,))
        
        # Filter gravity from accelerometers
        import scipy.signal
        for i in range(xa_test.shape[0]):
            for c in range(6):
                xa_test[i,:,c] -= scipy.signal.wiener(xa_test[i,:,c], 30)
    
        # Resize the accelerometers to match the video buffer size
        xa_test = xa_test[:, -buffer_tested[bi]:]
    
        # Get data size
        Nv_chan = xv_test.shape[1]
        img_rows = xv_test.shape[2]
        img_cols = xv_test.shape[3]
        Na_chan = xa_test.shape[2]
        acc_buffersiz = xa_test.shape[1]
        
        # Reshape into keras format
        xv_test = np.moveaxis(xv_test, 1, 3).astype('float32')/255
        xa_test = xa_test.astype('float32')/9.81
        
        # Remove Nan data from test
        nan_test = np.logical_or(np.isnan(ya_test), np.isnan(yv_test))
        xv_test = np.delete(xv_test, np.where(nan_test), 0)
        xa_test = np.delete(xa_test, np.where(nan_test), 0)
        y_test = np.delete(yv_test, np.where(nan_test))
        lab_test = np.delete(lab_test, np.where(nan_test))
        
        # Initialize the network structure
        if si == 1:
            # Initialise error per label
            Nlabels = 11
            R_per_lab = np.zeros(Nlabels)
            err_per_lab = np.zeros(Nlabels)
            N_per_lab = np.zeros(Nlabels)
            
            # Network
            model = NetworkCombined(img_rows, img_cols, Nv_chan, acc_buffersiz, Na_chan)
            
        # Read the model weights
        file_name = os.path.join(model_path, model_name % (buffer_tested[bi], si))
        
        if os.path.isfile(file_name):
            print('Loading model for leave %s out' % file_name)
            model.load_weights(file_name)
        else:
            print('FILE NOT FOUND! PREDICTING WITH RANDOM WEIGHTS! **********************************************')

        # Predict the results
        print('Predicting the result...')
        y_predict = model.predict([xv_test, xa_test])
            
        # Smoothen the prediction
        weights = np.concatenate((np.linspace(0, 1, int(smoothcal/2)),
                                np.linspace(1, 0, int(smoothcal/2))))
        weights = weights/np.sum(weights)
        y_predict[:, 0] = np.convolve(y_predict[:,0], weights, mode='same')
        
        # Results
        err = np.square(y_test - y_predict[:,0])
        for li in range(1, Nlabels+1):
            # Subject 3 skipped the exercising part completely!
            if np.any(lab_test == li):
                err_per_lab[li-1] += np.sqrt(np.mean(err[lab_test == li]))
                N_per_lab[li-1] += 1

    # Calculate the mean error
    err_per_lab /= N_per_lab
    overall_error.append(err_per_lab.mean())

# Save the results
with open('results_vs_buffer', 'wb') as file:
    pickle.dump([overall_error, buffer_tested], file, protocol=0)
    
# %% Read data and plot the results
plt.close('all')
with open('results_vs_buffer', 'rb') as file:
    data = pickle.load(file)

overall_error, buffer_tested = data

plt.figure(figsize=[5,1.65])
plt.plot(buffer_tested, overall_error, 'o-')
plt.xlabel('Buffer size')
plt.ylabel('Overall RMS')
plt.yticks([0.8, 0.85, 0.9, 0.95])
plt.tight_layout()

plt.savefig('error_vs_buffersize.png')