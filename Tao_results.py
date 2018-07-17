# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:48:09 2018

@author: Alessandro Masullo
"""

import numpy as np
import os
import CaloriesDataset
import pickle

Nsub = 10
Nlab = 11

# Normalised RMS from Tao's paper. They are normalised with the average
# calories per activity
table7_vis_rec1 = [.36, .39, .28, .38, .30, .38, .54, .41, .50, .35, .41, .42]
# Mean and std calories per activity
mean_cal_act = np.zeros(len(table7_vis_rec1))
std_cal_act = np.zeros(len(table7_vis_rec1))
# Load all the data
(_, y_test, lab_test) = CaloriesDataset.load_range(range(1, Nsub+1))

# Delete missing data
ynan = np.isnan(y_test)
y_test = np.delete(y_test, np.where(ynan))
lab_test = np.delete(lab_test, np.where(ynan))

# Find the avereage calories for each activity
labels = np.arange(1, 11+1)
Nlab = len(labels)
for li in range(Nlab):
    mean_cal_act[li] = np.mean(y_test[lab_test == labels[li]])
    std_cal_act[li] = np.std(y_test[lab_test == labels[li]])

mean_cal_act[-1] = np.mean(y_test)
std_cal_act[-1] = np.std(y_test)

RMS = table7_vis_rec1 * mean_cal_act

with open('Tao_results.dat', 'wb') as handle:
    pickle.dump(RMS, handle, protocol=0)
