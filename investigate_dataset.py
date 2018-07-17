# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:42:44 2018

@author: Alessandro Masullo
"""

import os
import xlrd
import numpy as np
import time, datetime
from matplotlib import pyplot as plt
import cv2
import re
from ImgCache import ImgCache

# Some settings
Set = {'smoothen_calorie': 20}
Set['cache_size']= 1100
Set['downsample'] = 2
Set['windows'] = (10,30,100,300,1000) # 0.3, 1, 10, 30 seconds
Set['n_win'] = len(Set['windows'])
Set['buf_size'] = np.max(Set['windows'])
Set['save_videos'] = True

Dataset = {'folder_allcases': 'F:\\SPHERE\\SPHERE_Calorie',
   'folder_silhouette': 'silhouette',
   'box_filename': 'userBB*.txt',
   'label_filename': 'labels.txt',
   'subj_name_format': 'Subject%d_Record%d'}

# Regexp to find numbers in file names
regex = re.compile('\d+')

# List all the cases
cases = [bf for bf in os.listdir(Dataset['folder_allcases']) if bf.startswith('Subject')]
Ncases = len(cases)

img = np.zeros((1200,Ncases))
imglab = np.zeros((700,Ncases))
ytick = []

# Loop over all the cases
plt.figure()
for ci in range(Ncases):
    print('Processing case %d of %d...' % (ci,Ncases))
    xls_dir = os.path.join(Dataset['folder_allcases'],cases[ci],'GT_calorie')
    cal_file = [bf for bf in os.listdir(xls_dir) if bf.startswith('Subject')]
    # Check that there's only one file
    if len(cal_file) > 1:
        raise Exception('There are two calorie files in folder: %s' % cal_file)
        
    # Read the calories data
    book = xlrd.open_workbook(os.path.join(xls_dir,cal_file[0]))
    book_sheet = book.sheet_by_index(0)
    cal_data = book_sheet.col_values(59,start_rowx=3)
    cal_clock = book_sheet.col_values(9,start_rowx=3)
    cal_time = [time.strptime(t,'%H:%M:%S') for t in cal_clock]
    cal_seconds = [datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds() for x in cal_time]
    cal_seconds = np.array(cal_seconds)
    cal_data = np.array(cal_data)
    cal_bio = book_sheet.col_values(1,start_rowx=4,end_rowx=7)
    id_sub = cases[ci].split('Subject')[1].split('_')[0]
    print('Subject id %s, age: %d, height: %d, weight: %d' % (id_sub,cal_bio[0],cal_bio[1],cal_bio[2]))
    Ncal = len(cal_seconds)
    
    weights = np.concatenate((np.linspace(0,1,int(Set['smoothen_calorie']/2)),
                            np.linspace(1,0,int(Set['smoothen_calorie']/2))))
    weights = weights/np.sum(weights)
    cal_data = np.convolve(cal_data, weights, mode='same')

    for iii in range(len(cal_seconds)):
        img[int(cal_seconds[iii]//2),ci] = cal_data[iii]

    # Read the labels
    label_file = os.path.join(Dataset['folder_allcases'],cases[ci],'labels.txt')
    label_data = np.genfromtxt(label_file,delimiter=',').astype('int')
    
    sub = label_data[0:-1:100,1]
    imglab[0:len(sub),ci] = sub
    
    ytick.append('sub %s' % cases[ci].split('Subject')[1].split('_')[0])

plt.close('all')
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(img)); plt.axis('tight'); plt.set_cmap('jet')
plt.yticks(np.arange(Ncases), ytick)
plt.xlabel('Time (sec)')
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(imglab)); plt.axis('tight'); plt.set_cmap('jet')
plt.yticks(np.arange(Ncases), ytick)
plt.xlabel('Frame/100')