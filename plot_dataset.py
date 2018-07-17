# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:30:18 2018

@author: Alessandro Masullo
"""
import os, pickle
import CaloriesDataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
    
plt.close('all')
# %% Plot the dataset
Nsubjects = 10
data_path = os.getcwd()
plt.figure(figsize=(10, 3))

for si in range(1,Nsubjects+1):
    # Read the testing data
    print('Loading data for subject %d' % si)
    file_root = 'subj_%d_case' % si
    file_list = [bf for bf in os.listdir(data_path) if bf.startswith(file_root) and bf.endswith('.npz')]
    first = 1
    for fil in file_list:
        print('Loading %s...' % fil)
        data = np.load(os.path.join(data_path,fil))
        y_cal = data['y_train']
        label = np.array(data['label'])

        y_cal = np.expand_dims(y_cal/10.,0)
        nan = np.isnan(y_cal)
        y_cal = np.ma.array(y_cal, mask=nan)
        
        cmap = matplotlib.cm.jet
        cmap.set_bad('k')
        plt.subplot(Nsubjects,2,si*2-first)
        imh = plt.imshow(y_cal, cmap='jet')
        plt.clim([1, 8])
        plt.axis('tight')
        plt.yticks([])
        plt.xticks([])
        if first:
            plt.yticks([0.], ['Subj. %d' % si], rotation=0)
        
        if si == 1 and first:
            plt.xlabel('Session 1')
            plt.gca().xaxis.set_label_position('top')
            
        if si == 1 and not first:
            plt.xlabel('Session 2')
            plt.gca().xaxis.set_label_position('top')
        
        if si == Nsubjects and first:
            plt.xlabel('Time (normalised)')
            
        if si == Nsubjects and not first:
            plt.xlabel('Time (normalised)')
        plt.show()
        plt.pause(1)
        first = 0

plt.subplots_adjust(right=0.8)
cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(imh, cax=cbar_ax, label='Calories/minute')

plt.savefig('dataset.png')