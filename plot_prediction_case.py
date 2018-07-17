# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:28:36 2018

@author: Alessandro Masullo
"""

import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import CaloriesDataset

plt.close('all')

# Subject to plot    
si = 1

# Load the data
with open('zhuetal_predictions.dat','rb') as handle:
    data = pickle.load(handle)
    mdata = np.ma.array(np.zeros(len(data[0][si])), mask=data[0][si])
    mdata[np.where(np.logical_not(mdata.mask))] = data[1][si]
    zhu_predictions = mdata
    
with open('CaloriNet_predictions.dat','rb') as handle:
    data = pickle.load(handle)
    mdata = np.ma.array(np.zeros(len(data[0][si])), mask=data[0][si])
    mdata[np.where(np.logical_not(mdata.mask))] = data[1][si]
    combined_predictions = mdata
    
with open('SiluCalNet_predictions.dat','rb') as handle:
    data = pickle.load(handle)
    mdata = np.ma.array(np.zeros(len(data[0][si])), mask=data[0][si])
    mdata[np.where(np.logical_not(mdata.mask))] = data[1][si]
    silu_predictions = mdata
	
with open('AccuCalNet_predictions.dat','rb') as handle:
    data = pickle.load(handle)
    mdata = np.ma.array(np.zeros(len(data[0][si])), mask=data[0][si])
    mdata[np.where(np.logical_not(mdata.mask))] = data[1][si]
    acc_predictions = mdata
    
# %% Find RMS for METs results
# From https://sites.google.com/site/compendiumofphysicalactivities/
METs = np.array([1.3, 1.3, 2.0, 2.3, 3.3, 3.3, 1.0, 5.0, 2.5, 2.3, 1.3])

# Read the testing data
print('Loading METs data for subject %d' % si)
(y_test, label, subj_ahw) = CaloriesDataset.load_ahw_range((si+1,))

#remove = np.logical_or(np.isnan(yv_test),np.isnan(ya_test))
#y_test = np.delete(y_test,np.where(remove))
#label = np.delete(label,np.where(remove))
y_predict_lab = METs*subj_ahw[2]/60
#remove = np.logical_or(np.isnan(y_test),label==0)
#label = np.delete(label,np.where(remove))
#y_test = np.delete(y_test,np.where(remove))

Nlabels = 11
    
mets_predictions = np.zeros(len(y_test))
for li in range(1,Nlabels+1):
    mets_predictions[label == li] = y_predict_lab[li-1]

# %% Image results
plt.figure(figsize=(8,2))
color_lim = [1.5, 7.0]
colormap = 'jet'

cmap = matplotlib.cm.jet
cmap.set_bad('k')

Npl = 6
Hp = 1/Npl-0.025
Hps = 1/Npl
Ws = 0.15
Wp = 0.65

plt.axes([Ws, 1-Hps*1, Wp, Hp])
imh = plt.imshow(np.ma.expand_dims(y_test[631:], 0), cmap=colormap)
plt.clim(color_lim)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
plt.yticks([0.], ['Ground truth'], rotation=0)

plt.axes([Ws, 1-Hps*2, Wp, Hp])
imh = plt.imshow(np.ma.expand_dims(combined_predictions[631:], 0), cmap=colormap)
plt.clim(color_lim)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
plt.yticks([0.], ['CaloriNet'], rotation=0)

plt.axes([Ws, 1-Hps*3, Wp, Hp])
imh = plt.imshow(np.ma.expand_dims(silu_predictions[631:], 0), cmap=colormap)
plt.clim(color_lim)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
plt.yticks([0.], ['SiluCalNet'], rotation=0)

plt.axes([Ws, 1-Hps*4, Wp, Hp])
imh = plt.imshow(np.ma.expand_dims(acc_predictions[631:], 0), cmap=colormap)
plt.clim(color_lim)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
plt.yticks([0.], ['AccuCalNet'], rotation=0)

plt.axes([Ws, 1-Hps*5, Wp, Hp])
imh = plt.imshow(np.ma.expand_dims(zhu_predictions[631:], 0), cmap=colormap)
plt.clim(color_lim)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
plt.yticks([0.], ['Zhu et al. 2015'], rotation=0)

plt.axes([Ws, 1-Hps*6, Wp, Hp])
imh = plt.imshow(np.ma.expand_dims(mets_predictions[631:], 0), cmap=colormap)
plt.clim(color_lim)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
plt.yticks([0.], ['METs lookup'], rotation=0)

#plt.tight_layout()
#plt.subplots_adjust(right=0.8)
cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(imh, cax=cbar_ax, label='Calories/minute')

plt.savefig('prediction_case.png')