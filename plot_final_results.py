# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:35:34 2018

@author: Alessandro Masullo
"""
import os, pickle
import CaloriesDataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Plot final results
with open('Tao_results.dat','rb') as handle:
	RMSlili = pickle.load(handle)

with open('AccuCalNet_results.dat','rb') as handle:
	RMS_accel = pickle.load(handle)    
    
with open('Zhu_results.dat','rb') as handle:
	RMSzhu = pickle.load(handle)
    
with open('CaloriNet_results.dat','rb') as handle:
	RMS_combined = pickle.load(handle)    
    
with open('SiluCalNet_results.dat','rb') as handle:
	RMS_video = pickle.load(handle)
    
# %% Find RMS for METs table
# From https://sites.google.com/site/compendiumofphysicalactivities/
Nsubjects = 10
METs = np.array([1.3, 1.3, 2.0, 2.3, 3.3, 3.3, 1.0, 5.0, 2.5, 2.3, 1.3])

for si in range(1,Nsubjects+1):
    # Read the testing data
    print('Loading METs data for subject %d' % si)
    (y_test, label, subj_ahw) = CaloriesDataset.load_ahw_range((si,))
    
    y_predict = METs*subj_ahw[2]/60
    remove = np.logical_or(np.isnan(y_test),label==0)
    label = np.delete(label,np.where(remove))
    y_test = np.delete(y_test,np.where(remove))
    
    # Initialize the network structure
    if si == 1:
        # Initialise error per label
        Nlabels = 11
        RMSmets = np.zeros(Nlabels)
        
    y_fakepred = np.copy(y_test)
    for li in range(1,Nlabels+1):
        y_test_lab = y_test[label == li]
        y_fakepred[label == li] = y_predict[li-1]
        RMSmets[li-1] += np.sqrt(np.mean(np.square(y_test-y_predict[li-1])))
        
#    plt.figure(figsize=(15,3))
#    plt.plot(y_fakepred)
#    plt.plot(y_test)

RMSmets /= Nsubjects
RMSmets = np.append(RMSmets,np.mean(RMSmets))
    

# %%
plt.figure()
plt.plot(RMSmets,label='Mets lookup')
plt.plot(RMSlili,label='Tao Etal')
plt.plot(RMSzhu,label='Zhu Etal')
plt.plot(RMS_accel,label='AccuCalNet')
plt.plot(RMS_video,label='SiluCalNet')
plt.plot(RMS_combined,label='CaloriNet')
plt.legend()

# %% Bar plot
font = {'size'   : 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rc('text', usetex=True)

Nmethods = 6
x = np.arange(0,Nlabels+1)
s = 1/(Nmethods+1)
plt.figure(figsize=(16,5))

colmets = [0.313, 0.321, 0.325]
collili = [0.549, 0.549, 0.549]
colozhu = [0.709, 0.709, 0.709]
coloacc = [0.937, 0.596, 0.262]
colovid = [0.592, 0.937, 0.262]
colcomb = [0.262, 0.635, 0.937]

RMSmets = np.take(RMSmets,np.arange(-1,len(RMSmets)-1),mode='wrap')
RMSlili = np.take(RMSlili,np.arange(-1,len(RMSlili)-1),mode='wrap')
RMSzhu = np.take(RMSzhu,np.arange(-1,len(RMSzhu)-1),mode='wrap')
RMS_accel = np.take(RMS_accel,np.arange(-1,len(RMS_accel)-1),mode='wrap')
RMS_video = np.take(RMS_video,np.arange(-1,len(RMS_video)-1),mode='wrap')
RMS_combined = np.take(RMS_combined,np.arange(-1,len(RMS_combined)-1),mode='wrap')

plt.bar(x-2*s, RMSmets, width=s, color=colmets, label='METs lookup')
plt.bar(x-s, RMSlili, width=s, color=collili, label='Tao \em{et al.} 2018')
plt.bar(x, RMSzhu, width=s, color=colozhu, label='Zhu \em{et al.} 2015')
plt.bar(x+s, RMS_accel, width=s, color=coloacc, label='AccuCalNet')
plt.bar(x+2*s, RMS_video, width=s, color=colovid, label='SiluCalNet')
plt.bar(x+3*s, RMS_combined, width=s, color=colcomb, label='CaloriNet')
plt.legend(mode='expand', ncol=6)

activities = [r'\textbf{Overall}', 'Stand', 'Sit', 'Walk', 'Wipe', 'Vacuum', 'Sweep', 'Lye',
              'Exercise', 'Stretch', 'Clean\n carpet', 'Read']

plt.xticks(x, activities, rotation=0)
plt.ylabel('RMS (cal/min)')
plt.tight_layout()

plt.savefig('results_bar.png')