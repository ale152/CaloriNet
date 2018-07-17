# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:37:48 2018

@author: Alessandro Masullo
"""

import keras
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

class PlotLearning(keras.callbacks.Callback):
    '''Plot losses during training'''
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.fig_pred = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.i += 1
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        plt.figure(self.fig.number)
        plt.clf()
        
        plt.semilogy(self.x, self.losses, label="loss")
        plt.semilogy(self.x, self.val_losses, label="val_loss")
        plt.legend()
        
        plt.draw()
        plt.pause(0.01)
        
        if epoch % 10 == 0:
            y_pred = self.model.predict(self.x_test)
            plt.figure(self.fig_pred.number)
            plt.clf()
            plt.plot(self.y_test)
            plt.plot(y_pred)
            plt.draw()
            plt.pause(0.01)
        
    def StoreTesting(self,x_test,y_test):
        self.x_test = x_test
        self.y_test = y_test

