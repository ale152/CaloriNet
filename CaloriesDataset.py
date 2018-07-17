# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:10:08 2018

@author: Alessandro Masullo
"""

import os
import numpy as np

def load_range(subj_to_load):
    data_path = os.getcwd()
    first = True
    for si in subj_to_load:
        file_root = 'silhouette_subj_%d_case' % si
        file_list = [bf for bf in os.listdir(data_path) if bf.startswith(file_root) and bf.endswith('.npz')]
        for fil in file_list:
            print('Loading %s...' % fil)
            data = np.load(os.path.join(data_path,fil))
            if first:
                x_train = np.array(data['x_train'])
                y_train = data['y_train']
                label = np.array(data['label'])
                first = False
            else:
                x_train = np.append(x_train,np.array(data['x_train']),axis=0)
                y_train = np.append(y_train,np.array(data['y_train']),axis=0)
                label = np.append(label,np.array(data['label']),axis=0)
                
    return (x_train,y_train,label)

def load_range_N(subj_to_load, N, data_path):
    first = True
    for si in subj_to_load:
        file_root = 'silhouette_%d_subj_%d_case' % (N, si)
        file_list = [bf for bf in os.listdir(data_path) if bf.startswith(file_root) and bf.endswith('.npz')]
        for fil in file_list:
            print('Loading %s...' % fil)
            data = np.load(os.path.join(data_path,fil))
            if first:
                x_train = np.array(data['x_train'])
                y_train = data['y_train']
                label = np.array(data['label'])
                first = False
            else:
                x_train = np.append(x_train,np.array(data['x_train']),axis=0)
                y_train = np.append(y_train,np.array(data['y_train']),axis=0)
                label = np.append(label,np.array(data['label']),axis=0)
                
    return (x_train,y_train,label)


def load_acc_range(subj_to_load):
    data_path = os.getcwd()
    first = True
    for si in subj_to_load:
        file_root = 'accelerometer_subj_%d_case' % si
        file_list = [bf for bf in os.listdir(data_path) if bf.startswith(file_root) and bf.endswith('.npz')]
        for fil in file_list:
            print('Loading %s...' % fil)
            data = np.load(os.path.join(data_path,fil))
            if first:
                x_train = np.array(data['x_train'])
                y_train = data['y_train']
                label = np.array(data['label'])
                first = False
            else:
                x_train = np.append(x_train,np.array(data['x_train']),axis=0)
                y_train = np.append(y_train,np.array(data['y_train']),axis=0)
                label = np.append(label,np.array(data['label']),axis=0)

    return (x_train,y_train,label)

def load_ahw_range(subj_to_load):
    data_path = os.getcwd()
    first = True
    for si in subj_to_load:
        file_root = 'accelerometer_subj_%d_case' % si
        file_list = [bf for bf in os.listdir(data_path) if bf.startswith(file_root) and bf.endswith('.npz')]
        for fil in file_list:
            print('Loading %s...' % fil)
            data = np.load(os.path.join(data_path,fil))
            if first:
                y_train = data['y_train']
                subj_ahw = np.array(data['subj_ahw'])
                label = np.array(data['label'])
                first = False
            else:
                y_train = np.append(y_train,np.array(data['y_train']),axis=0)
                subj_ahw = np.append(subj_ahw,np.array(data['subj_ahw']),axis=0)
                label = np.append(label,np.array(data['label']),axis=0)

    return (y_train,label,subj_ahw)