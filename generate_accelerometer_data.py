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
import pickle

# Some settings
Set = {'smoothen_calorie': 20}
# The buffer size is 1000 elements
Set['buf_size'] = 1000

Dataset = {'folder_allcases': 'F:\\SPHERE\\SPHERE_Calorie',
   'folder_accelerometer1': 'ACC0.000000',
   'folder_accelerometer2': 'ACC1.000000',
   'subj_name_format': 'Subject%d_Record%d',
   'time_file': 'frameTSinfo0.000000_all.txt',
   'acc_file_1': 'ACC_0.000000.txt',
   'acc_file_2': 'ACC_1.000000.txt'}

# List all the cases
cases = [bf for bf in os.listdir(Dataset['folder_allcases']) if bf.startswith('Subject')]
Ncases = len(cases)

# Loop over all the cases
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
    print('Subject age: %d, height: %d, weight: %d' % (cal_bio[0],cal_bio[1],cal_bio[2]))
    Ncal = len(cal_seconds)
    
    # Read the labels
    label_file = os.path.join(Dataset['folder_allcases'],cases[ci],'labels.txt')
    label_data = np.genfromtxt(label_file,delimiter=',').astype('int')
    all_label = np.zeros(Ncal)
    
    # Read the Video time data, used to syncronise the labels
    vid_time_file = os.path.join(Dataset['folder_allcases'],cases[ci],Dataset['time_file'])
    vid_time_data = np.genfromtxt(vid_time_file,delimiter='\t')
    vid_time = vid_time_data[:,2]/1e6
        
    # Initialisation
    if ci == 0:
        # Find the average difference between calories measurements
        dt_cal = np.diff(cal_seconds).mean()
        
        # Weights for smoothening the calories
        weights = np.concatenate((np.linspace(0,1,int(Set['smoothen_calorie']/2)),
                            np.linspace(1,0,int(Set['smoothen_calorie']/2))))
        weights = weights/np.sum(weights)
        
    # Smoothen the calories data
    cal_data = np.convolve(cal_data, weights, mode='same')
    
    # Read the Accelerometer time data
    acc_file1 = os.path.join(Dataset['folder_allcases'],cases[ci],
                            Dataset['folder_accelerometer1'],Dataset['acc_file_1'])
    acc_data1 = np.genfromtxt(acc_file1,delimiter='\t')
    acc_time1 = (acc_data1[:,6]-acc_data1[0,6])/1e9
    
    acc_file2 = os.path.join(Dataset['folder_allcases'],cases[ci],
                            Dataset['folder_accelerometer2'],Dataset['acc_file_2'])
    acc_data2 = np.genfromtxt(acc_file2,delimiter='\t')
    acc_time2 = (acc_data2[:,6]-acc_data2[0,6])/1e9
    
    # Initialise the data container (Ncal~700,winsize=1000,Nchan=3)
    all_data = np.zeros((Ncal,Set['buf_size'],6))
    
    # Loop over the calories
    tic = time.time()
    pcount = 0
    for ical in range(Ncal):
        pcount += 1
        if time.time()-tic > 10:
            #print('Remaining time: %.1f min' % ((time.time()-tic)/pcount*(Ncal-ical)/60))
            tic = time.time()
            pcount = 0
            
        #tic = time.time() 
        # Find the video frames _before_ the calorie time point
        id_cen1 = (abs(acc_time1-cal_seconds[ical])).argmin()
        id_cen2 = (abs(acc_time2-cal_seconds[ical])).argmin()
        
        # Save the average frame in memory
        id_lef1 = id_cen1-Set['buf_size']
        id_lef2 = id_cen2-Set['buf_size']
            
        if id_lef1 >= 0:
            all_data[ical,:,0:3] = acc_data1[id_lef1+1:id_cen1+1,3:6]
        else:
            # If any of the buffers are not complete, the data point will be removed
            cal_data[ical] = float('NaN')
        if id_lef2 >= 0:
            all_data[ical,:,3:] = acc_data2[id_lef2+1:id_cen2+1,3:6]
        else:
            # If any of the buffers are not complete, the data point will be removed
            cal_data[ical] = float('NaN')    
            
        # Find the label using the video time
        id_cen = (abs(vid_time-cal_seconds[ical])).argmin()
        lab = label_data[label_data[:,0] == id_cen,1]
        # If the label is missing something is wrong, skip the datapoint
        if lab.size != 0:
            all_label[ical] = lab
        else:
            cal_data[ical] = float('NaN')

    fig = plt.figure(ci)
    plt.plot(cal_data)

    # Save the average frames to use them for training
    id_sub = cases[ci].split('Subject')[1].split('_')[0]
    foutname = 'accelerometer_subj_%s_case_%d.npz' % (id_sub,ci)
    np.savez_compressed(foutname,x_train=all_data,y_train=cal_data,label=all_label,subj_ahw=cal_bio)
    print('Data daved in %s' % foutname)
    
# Save the settings
with open('acc_settings.dat', 'wb') as outfile:
    pickle.dump(Set, outfile, protocol=0)
