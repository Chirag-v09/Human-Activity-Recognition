# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:25:10 2020

@author: Chirag
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob
import math

files = [file for file in glob.glob("UCF-101/*")]

sub_files = []
for file in files:
    print(file)
    sub_files = [ sub_file for sub_file in glob.glob(file + "/*.*")]
    sub_files = sub_files[:-1]
    counter = 0
    for sub_file in sub_files:
        counter += 1
        count = 0
        cap = cv2.VideoCapture(sub_file)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        x=1
        direc = sub_file.split("\\")[1]
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                # storing the frames in a new folder named train_1
                filename = "UCF 101 (me)\\Training\\" + direc + "\\Training_frame%d%d.jpg" %(counter,count) ;count+=1
                cv2.imwrite(filename, frame)
        cap.release()