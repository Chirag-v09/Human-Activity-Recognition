# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:15:16 2020

@author: Chirag
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import cv2
import os

files = [file for file in glob.glob("UCF 101 (me)\\Video Classification\\training\\*")]

empty_file = []
for file in files:
    if [sub_file for sub_file in glob.glob(file + "\*")] == []:
        empty_file.append(file.split("\\")[-1])

sub_files = []
for file in files:
    sub_files.append([sub_file for sub_file in glob.glob(file + "\*")])

a = 100000
for sub_file in sub_files:
    if a > len(sub_file):
        a = len(sub_file)

lists_names8001000 = []
for sub_file in sub_files:
    if 800 < len(sub_file) < 1000:
        lists_names8001000.append(sub_file[0].split('\\')[3])

lists8001000 = []
for sub_file in sub_files:
    if 800 < len(sub_file) < 1000:
        lists8001000.append(sub_file)

lists_names600800 = []
for sub_file in sub_files:
    if 600 < len(sub_file) < 800:
        lists_names600800.append(sub_file[0].split('\\')[3])

lists600800 = []
for sub_file in sub_files:
    if 600 < len(sub_file) < 800:
        lists600800.append(sub_file)

lists_names600 = []
for sub_file in sub_files:
    if 100 < len(sub_file) < 600:
        lists_names600.append(sub_file[0].split('\\')[3])

lists600 = []
for sub_file in sub_files:
    if 100 < len(sub_file) < 600:
        lists600.append(sub_file)

for sub_file in sub_files:
    if 800 < len(sub_file) < 1000:
        for i in range(len(sub_file)):
            os.remove(sub_file[i])

files = [file for file in glob.glob("UCF-101/*")]
sub_files1 = []
for file in files:
    print(file)
    if file.split("\\")[1] in empty_file:
        sub_files1 = [ sub_file1 for sub_file1 in glob.glob(file + "/*.*")]
        sub_files1 = sub_files1[:-1]
        counter = 0
        for sub_file1 in sub_files1:
            counter += 1
            count = 0
            cap = cv2.VideoCapture(sub_file1)   # capturing the video from the given path
            frameRate = cap.get(5) #frame rate
            x=1
            direc = sub_file1.split("\\")[1]
            while(cap.isOpened()):
                frameId = cap.get(1) #current frame number
                ret, frame = cap.read()
                if (ret != True):
                    break
                if (frameId % 5 == 0):
                    # storing the frames in a new folder named train_1
                    filename = "UCF 101 (me)\\Video Classification\\Training\\" + direc + "\\Training_frame%d%d.jpg" %(counter,count) ;count+=1
                    cv2.imwrite(filename, frame)
            cap.release()


files = [file for file in glob.glob("UCF-101/*")]
sub_files1 = []
for file in files:
    if file.split("\\")[1] in lists_names600800:
        print(file.split("\\")[1])
        sub_files1 = [ sub_file1 for sub_file1 in glob.glob(file + "/*.*")]
        sub_files1 = sub_files1[:-1]
        counter = 0
        for sub_file1 in sub_files1:
            counter += 1
            count = 0
            cap = cv2.VideoCapture(sub_file1)   # capturing the video from the given path
            frameRate = cap.get(5) #frame rate
            x=1
            direc = sub_file1.split("\\")[1]
            while(cap.isOpened()):
                frameId = cap.get(1) #current frame number
                ret, frame = cap.read()
                if (ret != True):
                    break
                if (frameId % 10 == 0):
                    # storing the frames in a new folder named train_1
                    filename = "UCF 101 (me)\\Video Classification\\Training\\" + direc + "\\Training_frame%d%d.jpg" %(counter,count) ;count+=1
                    cv2.imwrite(filename, frame)
            cap.release()


files = [file for file in glob.glob("UCF-101/*")]
sub_files1 = []
for file in files:
    print(file)
    if file.split("\\")[1] in lists_names8001000:
        sub_files1 = [ sub_file1 for sub_file1 in glob.glob(file + "/*.*")]
        sub_files1 = sub_files1[:-1]
        counter = 0
        for sub_file1 in sub_files1:
            counter += 1
            count = 0
            cap = cv2.VideoCapture(sub_file1)   # capturing the video from the given path
            frameRate = cap.get(5) #frame rate
            x=1
            direc = sub_file1.split("\\")[1]
            while(cap.isOpened()):
                frameId = cap.get(1) #current frame number
                ret, frame = cap.read()
                if (ret != True):
                    break
                if (frameId % 14 == 0):
                    # storing the frames in a new folder named train_1
                    filename = "UCF 101 (me)\\Video Classification\\Training\\" + direc + "\\Training_frame%d%d.jpg" %(counter,count) ;count+=1
                    cv2.imwrite(filename, frame)
            cap.release()



# Balancing Test dataset


files = [file for file in glob.glob("UCF 101 (me)\\Video Classification\\test\\*")]

empty_file = []
for file in files:
    if [sub_file for sub_file in glob.glob(file + "\*")] == []:
        empty_file.append(file.split("\\")[-1])

sub_files = []
for file in files:
    sub_files.append([sub_file for sub_file in glob.glob(file + "\*")])

a = 100000
for sub_file in sub_files:
    if a > len(sub_file):
        a = len(sub_file)


lists_names100200 = []
for sub_file in sub_files:
    if 100 < len(sub_file) < 200:
        lists_names100200.append(sub_file[0].split('\\')[3])

lists_names200300 = []
for sub_file in sub_files:
    if 200 < len(sub_file):
        lists_names200300.append(sub_file[0].split('\\')[3])

for sub_file in sub_files:
    if len(sub_file) >= 100:
        for i in range(len(sub_file)):
            os.remove(sub_file[i])



files = [file for file in glob.glob("UCF-101//*")]

sub_files = []
for file in files:
    print(file)
    if file.split("\\")[-1] in lists_names100200:
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
            if (frameId % 2 == 0):
                # storing the frames in a new folder named train_1
                filename = "UCF 101 (me)\\Video Classification\\Test\\" + direc + "\\Test_frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        cap.release()


files = [file for file in glob.glob("UCF-101//*")]

sub_files = []
for file in files:
    print(file)
    if file.split("\\")[-1] in lists_names200300:
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
            if (frameId % 3 == 0):
                # storing the frames in a new folder named train_1
                filename = "UCF 101 (me)\\Video Classification\\Test\\" + direc + "\\Test_frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        cap.release()




