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
import re
from ImgCache import ImgCache
import matplotlib.animation as manimation

# Some settings
Set = {'smoothen_calorie': 20,
       'buf_size': 1000, # Changes between 250, 500, 1000 and 2000
       'cache_size': 1100,
       'downsample': 2}

# The buffer sizes are M/3^k, for k in [0...4], where M is 'buf_size'
Set['windows'] = (Set['buf_size']/3**np.arange(4,-1,-1)).astype('int')
Set['n_win'] = len(Set['windows'])

Set['save_videos'] = False

# Dataset path
Dataset = {'folder_allcases': 'F:\\SPHERE\\SPHERE_Calorie',
   'folder_silhouette': 'silhouette',
   'box_filename': 'userBB*.txt',
   'label_filename': 'labels.txt',
   'subj_name_format': 'Subject%d_Record%d',
   'frame_info': 'frameTSinfo0.000000_all.txt',
   'save_folder': r'C:\Data\calories_sphere\processed_data'}

# Regexp to find numbers in file names
regex = re.compile('\d+')

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
    label_file = os.path.join(Dataset['folder_allcases'],cases[ci],Dataset['label_filename'])
    label_data = np.genfromtxt(label_file,delimiter=',').astype('int')
    all_label = np.zeros(Ncal)
        
    # Initialisation
    if ci == 0:
        # Find the average difference between calories measurements
        dt_cal = np.diff(cal_seconds).mean()
        
        # Weights for smoothening the calories
        weights = np.concatenate((np.linspace(0,1,int(Set['smoothen_calorie']/2)),
                            np.linspace(1,0,int(Set['smoothen_calorie']/2))))
        
        weights = weights/np.sum(weights)
        
        # Initialise the image cache
        Cache = ImgCache(Set['cache_size'],Set['downsample'])
        
    # Smoothen the calories data
    cal_data = np.convolve(cal_data, weights, mode='same')
    
    # Read the Video time data
    vid_time_file = os.path.join(Dataset['folder_allcases'],cases[ci],Dataset['frame_info'])
    vid_time_data = np.genfromtxt(vid_time_file,delimiter='\t')
    vid_time = vid_time_data[:,2]/1e6
    
    # Read the list of silhouettes
    sil_dir = os.path.join(Dataset['folder_allcases'],cases[ci],Dataset['folder_silhouette'])
    sil_list = [bf for bf in os.listdir(sil_dir) if bf.endswith('.png')]
    # Sort the list
    sil_id = [int(x) for x in re.findall(regex,'|'.join(sil_list))]
    id_order = np.argsort(sil_id)
    sil_list = [sil_list[i] for i in id_order]
    sil_id = np.array([sil_id[i] for i in id_order])
    # Create a dictionary that returns the id of each frame (to avoid using "find")
    sil_id_map = dict(zip(sil_id,enumerate(sil_id)))
    # Open a sample image
    imgsamp = Cache.read_image(os.path.join(sil_dir,sil_list[0]))
    
    # Video stuff
    if Set['save_videos']:
        fig = plt.figure()
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=15)
        id_sub = cases[ci].split('Subject')[1].split('_')[0]
        dbgvideo_file = 'subj_%s_case_%d.mp4' % (id_sub,ci)
        writer.setup(fig=fig,outfile=dbgvideo_file, dpi=100)
        vidframe = np.zeros((imgsamp.shape[0],imgsamp.shape[1],3))
    
    # Initialise the average frame
    all_data = np.zeros((Ncal,Set['n_win'],imgsamp.shape[0],imgsamp.shape[1]),dtype='uint8')
    # Keep track of how many missing frames there are for each calorie point
    vid_goodframes = np.zeros((Ncal,Set['n_win']))
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
        id_cen = (abs(vid_time-cal_seconds[ical])).argmin()
        id_vid = np.arange(id_cen-Set['buf_size']+1,
                           id_cen+1)
        avg_frame = np.zeros((Set['n_win'],imgsamp.shape[0],imgsamp.shape[1]))
        lab = label_data[label_data[:,0] == id_cen,1]
        # If the label is missing something is wrong, skip the datapoint
        if lab.size != 0:
            all_label[ical] = lab
        else:
            cal_data[ical] = float('NaN')
        
        # Open the video frames
        for ifram in range(Set['buf_size']):
            # Read the frame
            # Find the list id for this frame (the frames might start from 400...)
            #bf = [counter for counter,val  in enumerate(sil_id) if sil_id[counter] == id_vid[ifram]]
            #print('Time to find silhouette id: %f' % (time.time() - tic))
            if id_vid[ifram] in sil_id_map:
                bf = sil_id_map[id_vid[ifram]]
            else:
                continue
            frame_path = os.path.join(sil_dir,sil_list[bf[0]]);
            if os.path.isfile(frame_path):
                frame = Cache.read_image(frame_path)
                for layer in range(Set['n_win']):
                    if ifram >= Set['buf_size']-Set['windows'][layer]: # Check the numbers are right####################
                        avg_frame[layer] += frame.astype('float')
                        vid_goodframes[ical,layer] += 1
        
        # Save the average frame in memory
        for layer in range(Set['n_win']):
            avg_frame[layer,] /= vid_goodframes[ical,layer]
            # Check that at least 90% of the buffer was filled
            if vid_goodframes[ical,layer] >= Set['windows'][layer]*0.9:
                all_data[ical,layer,] = avg_frame[layer,]
            else:
                # If any of the buffers are not complete, the data point will be removed
                cal_data[ical] = float('NaN')
        
        # Save the video
        if Set['save_videos']:
            vidframe = np.moveaxis(avg_frame[(0,2,4),],0,2).astype('uint8')
            plt.clf()
            plt.subplot(5,1,(1,4))
            plt.imshow(vidframe)
            plt.text(5,15,'Activity: %d' % all_label[ical], color='w')
            plt.subplot(5,1,5)
            plt.plot(cal_seconds/60,cal_data)
            isnan = np.isnan(cal_data)
            plt.plot(cal_seconds[isnan]/60,np.zeros(np.sum(isnan.astype('int'))),'.r')
            plt.plot(cal_seconds[ical]/60,cal_data[ical],'ok')
            plt.pause(0.1)
            writer.grab_frame()
        
        # Save frame for debug
#        if ci == 0:
#            foutname = 'case_%d_cal_%d_(%d).png' % (ci,ical,vid_goodframes[ical,0])
#            dbg = avg_frame[(1,2,4),].transpose((1,2,0))
#            cv2.imwrite(foutname,dbg)
     
    # Finalize video
    if Set['save_videos']:
        writer.finish()
    
    fig = plt.figure(ci)
    plt.plot(cal_data)
    
    # Remove broken data
    print('%d datapoints over %d were marked as nan because of missing data' % (np.sum(np.isnan(cal_data).astype('int')),len(cal_data)))
    #all_data = all_data[np.logical_not(np.isnan(cal_data)),]
    #cal_data = cal_data[np.logical_not(np.isnan(cal_data)),]
    
    # Save the average frames to use them for training
    id_sub = cases[ci].split('Subject')[1].split('_')[0]
    foutname = os.path.join(Dataset['save_folder'], 'silhouette_%d_subj_%s_case_%d.npz' % (Set['buf_size'], id_sub, ci))
    np.savez_compressed(foutname,x_train=all_data,y_train=cal_data,label=all_label,subj_ahw=cal_bio)
    print('Data daved in %s' % foutname)