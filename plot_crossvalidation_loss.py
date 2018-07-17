# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 08:39:51 2018

@author: Alessandro Masullo
"""

import os
import numpy as np
from matplotlib import pyplot as plt
path = r'C:\calories\losses'

files = [bf for bf in os.listdir(path) if bf.endswith('.txt')]
plt.close('all')

for i in range(len(files)):
    data = np.loadtxt(os.path.join(path,files[i]))
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(data[:,0],data[:,1],data[:,0],data[:,2])
    plt.ylim((0,1000))
    plt.title(files[i])
    plt.subplot(212)
    plt.semilogx(data[:,0],data[:,1],data[:,0],data[:,2])
    plt.ylim((0,1000))
    
    figname = '%s.png' % files[i].split('.txt')[0]
    plt.savefig(os.path.join(path,figname))
    
    if i == 0:
        all_loss = np.zeros(len(data))
        all_vloss = np.zeros(len(data))
        siz = len(data)
        
    siz = np.min((siz,len(data)))
    all_loss[0:siz] += data[0:siz,1]
    all_vloss[0:siz] += data[0:siz,2]
        
all_loss /= len(files)
all_vloss /= len(files)

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(np.arange(0,siz),all_loss[0:siz],np.arange(0,siz),all_vloss[0:siz])
plt.ylim((0,500))
plt.title('Average losses')
plt.subplot(212)
plt.semilogx(np.arange(0,siz),all_loss[0:siz],np.arange(0,siz),all_vloss[0:siz])
plt.ylim((0,500))
plt.savefig(os.path.join(path,'average.png'))

print('Mean vloss: %.1f, min vloss: %.1f' % (np.mean(all_vloss[0:siz]),np.min(all_vloss[0:siz])))