# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:13:42 2020

@author: Chirag
"""

import glob
import os

files = [file for file in glob.glob("*")]


direc = "Training"
parent_dir = "E:\\Chirag\\project\\video classification\\UCF 101 (me)"
path = os.path.join(parent_dir, direc)
os.mkdir(path)

direc = "Test"
parent_dir = "E:\\Chirag\\project\\video classification\\UCF 101 (me)"
path = os.path.join(parent_dir, direc)
os.mkdir(path)


for file in files:
    direc = file
    parent_dir = "E:\\Chirag\\project\\video classification\\UCF 101 (me)\\Training"
    path = os.path.join(parent_dir, direc)
    os.mkdir(path)

for file in files:
    direc = file
    parent_dir = "E:\\Chirag\\project\\video classification\\UCF 101 (me)\\Test"
    path = os.path.join(parent_dir, direc)
    os.mkdir(path)
