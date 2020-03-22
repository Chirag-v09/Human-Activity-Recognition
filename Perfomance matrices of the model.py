# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:19:26 2020

@author: Chirag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import glob
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input

bring_model = tf.keras.models.load_model("1584534392_32b_20e_cp-0003-0.9770a-0.0768l-0.9739va-0.0973vl.h5")


def get_activity(val):
    activities = {'ApplyEyeMakeup': 0, 'ApplyLipstick': 1, 'Archery': 2, 'BabyCrawling': 3, 'BalanceBeam': 4, 'BandMarching': 5, 'BaseballPitch': 6, 'Basketball': 7, 'BasketballDunk': 8, 'BenchPress': 9, 'Biking': 10, 'Billiards': 11, 'BlowDryHair': 12, 'BlowingCandles': 13, 'BodyWeightSquats': 14, 'Bowling': 15, 'BoxingPunchingBag': 16, 'BoxingSpeedBag': 17, 'BreastStroke': 18, 'BrushingTeeth': 19, 'CleanAndJerk': 20, 'CliffDiving': 21, 'CricketBowling': 22, 'CricketShot': 23, 'CuttingInKitchen': 24, 'Diving': 25, 'Drumming': 26, 'Fencing': 27, 'FieldHockeyPenalty': 28, 'FloorGymnastics': 29, 'FrisbeeCatch': 30, 'FrontCrawl': 31, 'GolfSwing': 32, 'Haircut': 33, 'HammerThrow': 34, 'Hammering': 35, 'HandstandPushups': 36, 'HandstandWalking': 37, 'HeadMassage': 38, 'HighJump': 39, 'HorseRace': 40, 'HorseRiding': 41, 'HulaHoop': 42, 'IceDancing': 43, 'JavelinThrow': 44, 'JugglingBalls': 45, 'JumpRope': 46, 'JumpingJack': 47, 'Kayaking': 48, 'Knitting': 49, 'LongJump': 50, 'Lunges': 51, 'MilitaryParade': 52, 'Mixing': 53, 'MoppingFloor': 54, 'Nunchucks': 55, 'ParallelBars': 56, 'PizzaTossing': 57, 'PlayingCello': 58, 'PlayingDaf': 59, 'PlayingDhol': 60, 'PlayingFlute': 61, 'PlayingGuitar': 62, 'PlayingPiano': 63, 'PlayingSitar': 64, 'PlayingTabla': 65, 'PlayingViolin': 66, 'PoleVault': 67, 'PommelHorse': 68, 'PullUps': 69, 'Punch': 70, 'PushUps': 71, 'Rafting': 72, 'RockClimbingIndoor': 73, 'RopeClimbing': 74, 'Rowing': 75, 'SalsaSpin': 76, 'ShavingBeard': 77, 'Shotput': 78, 'SkateBoarding': 79, 'Skiing': 80, 'Skijet': 81, 'SkyDiving': 82, 'SoccerJuggling': 83, 'SoccerPenalty': 84, 'StillRings': 85, 'SumoWrestling': 86, 'Surfing': 87, 'Swing': 88, 'TableTennisShot': 89, 'TaiChi': 90, 'TennisSwing': 91, 'ThrowDiscus': 92, 'TrampolineJumping': 93, 'Typing': 94, 'UnevenBars': 95, 'VolleyballSpiking': 96, 'WalkingWithDog': 97, 'WallPushups': 98, 'WritingOnBoard': 99, 'YoYo': 100}
    for key, value in activities.items():
        if val == value:
            return key
    return "Invalid"

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_generator = test_datagen.flow_from_directory(
        "testing_set/",
        target_size=(224, 224),
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = bring_model.predict_generator(test_generator,steps = nb_samples)

y_pred = []
for val in predict:
    y_pred.append(get_activity(np.argmax(val)))

y_true = []
for file in filenames:
    y_true.append(file.split("\\")[0])
# y_true = y_true[:11]

from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,confusion_matrix
cm = confusion_matrix(y_true,y_pred)
print(precision_score(y_true,y_pred,average = 'macro'))
print(recall_score(y_true,y_pred,average = 'macro'))
print(f1_score(y_true,y_pred,average = 'macro'))

print(precision_score(y_true,y_pred,average = 'micro'))
print(recall_score(y_true,y_pred,average = 'micro'))
print(f1_score(y_true,y_pred,average = 'micro'))

print(classification_report(y_true, y_pred))

dataframe = pd.DataFrame(cm)
activities = {'ApplyEyeMakeup': 0, 'ApplyLipstick': 1, 'Archery': 2, 'BabyCrawling': 3, 'BalanceBeam': 4, 'BandMarching': 5, 'BaseballPitch': 6, 'Basketball': 7, 'BasketballDunk': 8, 'BenchPress': 9, 'Biking': 10, 'Billiards': 11, 'BlowDryHair': 12, 'BlowingCandles': 13, 'BodyWeightSquats': 14, 'Bowling': 15, 'BoxingPunchingBag': 16, 'BoxingSpeedBag': 17, 'BreastStroke': 18, 'BrushingTeeth': 19, 'CleanAndJerk': 20, 'CliffDiving': 21, 'CricketBowling': 22, 'CricketShot': 23, 'CuttingInKitchen': 24, 'Diving': 25, 'Drumming': 26, 'Fencing': 27, 'FieldHockeyPenalty': 28, 'FloorGymnastics': 29, 'FrisbeeCatch': 30, 'FrontCrawl': 31, 'GolfSwing': 32, 'Haircut': 33, 'HammerThrow': 34, 'Hammering': 35, 'HandstandPushups': 36, 'HandstandWalking': 37, 'HeadMassage': 38, 'HighJump': 39, 'HorseRace': 40, 'HorseRiding': 41, 'HulaHoop': 42, 'IceDancing': 43, 'JavelinThrow': 44, 'JugglingBalls': 45, 'JumpRope': 46, 'JumpingJack': 47, 'Kayaking': 48, 'Knitting': 49, 'LongJump': 50, 'Lunges': 51, 'MilitaryParade': 52, 'Mixing': 53, 'MoppingFloor': 54, 'Nunchucks': 55, 'ParallelBars': 56, 'PizzaTossing': 57, 'PlayingCello': 58, 'PlayingDaf': 59, 'PlayingDhol': 60, 'PlayingFlute': 61, 'PlayingGuitar': 62, 'PlayingPiano': 63, 'PlayingSitar': 64, 'PlayingTabla': 65, 'PlayingViolin': 66, 'PoleVault': 67, 'PommelHorse': 68, 'PullUps': 69, 'Punch': 70, 'PushUps': 71, 'Rafting': 72, 'RockClimbingIndoor': 73, 'RopeClimbing': 74, 'Rowing': 75, 'SalsaSpin': 76, 'ShavingBeard': 77, 'Shotput': 78, 'SkateBoarding': 79, 'Skiing': 80, 'Skijet': 81, 'SkyDiving': 82, 'SoccerJuggling': 83, 'SoccerPenalty': 84, 'StillRings': 85, 'SumoWrestling': 86, 'Surfing': 87, 'Swing': 88, 'TableTennisShot': 89, 'TaiChi': 90, 'TennisSwing': 91, 'ThrowDiscus': 92, 'TrampolineJumping': 93, 'Typing': 94, 'UnevenBars': 95, 'VolleyballSpiking': 96, 'WalkingWithDog': 97, 'WallPushups': 98, 'WritingOnBoard': 99, 'YoYo': 100}
inv_dict = {v: k for k, v in activities.items()} 
dataframe = dataframe.rename(index = inv_dict)
dataframe = dataframe.rename(columns = inv_dict)

dataframe.to_csv("Perfomance Confusion matrix.csv")
