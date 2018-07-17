# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:25:46 2018

@author: Alessandro Masullo
"""

import keras
import time
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import csv
import os

class SaveHistory(keras.callbacks.Callback):
    '''Save history of training and validation losses'''
    
    def on_train_begin(self, logs={}):
        self.ep = 0
        self.logs = []
        self.x = []
        self.losses = []
        self.val_losses = []
        self.file_log = '%s_%s.txt' % (self.model_name,
                                       time.strftime('%d-%m-%Y %H-%M-%S', time.gmtime()))
        
    
    def on_epoch_end(self, epoch, logs={}):
        self.ep += 1
        self.logs.append(logs)
        self.x.append(self.ep)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        data = np.stack((self.x,self.losses,self.val_losses),axis=1)
        np.savetxt(self.file_log,data)
        
        # Only save the model after at least 100 epochs
        if self.ep >= self.start_saving_after:
            # Check if model has improved
            if logs.get('val_loss') < self.best_val_loss:
                # Save the model
                save_path = os.path.join(self.save_dir,'%s_best.h5' % self.model_name)
                self.model.save_weights(save_path)
                print('Validation loss improved from %.3f to %.3f' % (self.best_val_loss, logs.get('val_loss')))
                self.best_val_loss = logs.get('val_loss')
                
        
    def SetFilename(self, model_name, save_dir):
        self.model_name = model_name
        self.save_dir = save_dir
        self.start_saving_after = 30 # Only save the model after N epochs
        self.best_val_loss = float('inf')
