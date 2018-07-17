# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:30:21 2018

@author: Alessandro Masullo
"""
import cv2
from time import time

class ImgCache:
    def __init__(self,CacheSize,DownSample):
        self.CacheSize = CacheSize
        self.DownSample = DownSample
        self.CacheData = {}
        self.CacheTime = {}
    
    def read_image(self,filename):
        if len(self.CacheData) == 0:
            # If the dictionary is empty, read the image from file and add it to the cache
            img = cv2.imread(filename,0)
            img = img[::self.DownSample,::self.DownSample]
            now = str(time())
            self.CacheData[filename] = img
            self.CacheTime[filename] = now
        else:
            # Check if the file already exist in the cache
            if filename in self.CacheTime:
                # If it exists, take the image file from the cache
                img = self.CacheData[filename]
                # and update the the it was accessed
                now = str(time())
                self.CacheTime[filename] = now
            else:
                # If the file doen't exist, add it to the cache
                if len(self.CacheData) < self.CacheSize:
                    img = cv2.imread(filename,0)
                    img = img[::self.DownSample,::self.DownSample]
                    now = str(time())
                    self.CacheData[filename] = img
                    self.CacheTime[filename] = now
                else:
                    # If the cache is already full, select the oldest file and replace it
                    img = cv2.imread(filename,0)
                    img = img[::self.DownSample,::self.DownSample]
                    now = str(time())
                    self.CacheData[filename] = img
                    self.CacheTime[filename] = now
                    # Delete oldest file
                    oldest = min(self.CacheTime, key=self.CacheTime.get)
                    del self.CacheData[oldest]
                    del self.CacheTime[oldest]
    
        return img