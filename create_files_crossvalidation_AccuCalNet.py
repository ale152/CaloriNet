# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:04:00 2018

@author: Alessandro Masullo
"""

import os, io
import numpy as np

template_py = 'template_validation_AccuCalNet.py'
template_sh = 'template_validation_acc.sh'
pwd = os.getcwd()

Nsubj = 10
Nov = 1 # Number overlapping subjects
subjects = np.arange(1,Nsubj+1)
subjmat = subjects
for i in range(1,Nsubj):
    subjmat = np.column_stack((subjmat,np.roll(subjects,-i)))

for i in subjects: 
    print(i)
    out_py = 'AccuCalNet_leave_%d_out.py' % i
    out_sh = 'AccuCalNet_leave_%d_out.sh' % i
    # Open the templates
    ftemp_py = io.open(os.path.join(pwd,template_py),'r',newline='\n')
    ftemp_sh = io.open(os.path.join(pwd,template_sh),'r',newline='\n')
    # Create new files
    fdest_py = io.open(os.path.join(pwd,out_py),'w',newline='\n')
    fdest_sh = io.open(os.path.join(pwd,out_sh),'w',newline='\n')
    
    # Read templates
    content_py = ftemp_py.readlines()
    content_sh = ftemp_sh.readlines()
    
    # Modify py script
    net_name = 'AccuCalNet_leave_%d_out' % i
    content_py[17] = content_py[17].replace('NETWORKNAME',net_name)
    
    train_subj = subjmat[Nov:,i-1]
    train_subj = ','.join(train_subj.astype('str'))
    train_subj = '('+train_subj+')'
    content_py[26] = content_py[26].replace('TRAINSUBJ',train_subj)
    
    test_subj = subjmat[0:Nov,i-1]
    test_subj = ','.join(test_subj.astype('str'))
    test_subj = '('+test_subj+',)'
    content_py[27] = content_py[27].replace('TESTSUBJ',test_subj)
    
    # Modify sh script
    content_sh[2] = content_sh[2].replace('SCRIPTNAME',net_name)
    content_sh[15] = content_sh[15].replace('SCRIPTNAME',out_py)
    
    # Flush content in file
    fdest_py.writelines(content_py)
    fdest_sh.writelines(content_sh)
    
    # Close the files
    ftemp_py.close()
    ftemp_sh.close()
    fdest_py.close()
    fdest_sh.close()
    

# After creating the files, run the following from bluecrystal:
# for I in {1..10}; do sbatch $(echo leave_${I}_out.sh); done
    
    