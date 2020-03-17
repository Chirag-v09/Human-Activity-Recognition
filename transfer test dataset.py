# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:39:48 2020

@author: Chirag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import cv2

files = [file for file in glob.glob("UCF-101//*")]

sub_files = []
for file in files:
    print(file)
    sub_files = [ sub_file for sub_file in glob.glob(file + "/*.*")]
    last_file = sub_files[-1]
    
    count = 0
    cap = cv2.VideoCapture(last_file)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    direc = last_file.split("\\")[1]
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        print(frameId)
        ret, frame = cap.read()
        if (ret != True):
            break
        # if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
        filename = "UCF 101 (me)\\Video Classification\\Test\\" + direc + "\\Test_frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
    cap.release()




