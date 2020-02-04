
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd

from tensorflow.keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
# from tensorflow.keras.utils import np_utils
import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm



f = open("videos path.txt", "r")
temp = f.read()
videos = temp.split('\n')

videos_names = []
for i in range(len(videos)):
    temp = videos[i].split('\\')
    videos_names.append(temp[-1][:-1])

video = pd.DataFrame()
video["video_name"] = videos_names
video.head()

videos_tag = []
for i in range(len(videos_names)):
    temp = videos_names[i].split('_')
    videos_tag.append(temp[1])

video["tag"] = videos_tag


for i in tqdm(range(video.shape[0])):
    count = 0
    videoFile = video['video_name'][i]
    cap = cv2.VideoCapture('UCF/'+videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
            filename ='train_1/' + videoFile +"_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()


# getting the names of all the images
images = glob("train_1/*.jpg")
train_image = []
train_class = []
for i in tqdm(range(len(images))):
    # creating the image name
    train_image.append(images[i].split('\\')[1])
    # creating the class of image
    train_class.append(images[i].split('\\')[1].split('_')[1])

# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

# converting the dataframe into csv file 
train_data.to_csv('train_new.csv',header=True, index=False)


import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


train = pd.read_csv('train_new.csv')
train.head()

# creating an empty list
train_image = []


# for loop to read and store frames
for i in tqdm(range(train.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img('train_1/'+train['image'][i], target_size=(224,224,3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255
    # appending the image to the train_image list
    train_image.append(img)


# converting the list to numpy array
X = np.array(train_image)

# shape of the array
X.shape


# separating the target
y = train['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)

# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
# In deep learning we have to pass the model y variable as sparse matrix

# creating the base model of pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False) # To remove the top bias layer

# extracting features for training frames
X_train = base_model.predict(X_train)
X_train.shape

# extracting features for validation frames
X_test = base_model.predict(X_test)
X_test.shape

# reshaping the training as well as validation frames in single dimension
X_train = X_train.reshape(9085, 7*7*512)  # After the prediction through this model the output
X_test = X_test.reshape(2272, 7*7*512) # shape is (7, 7, 512) hence we flattern the image to 
# to train another model and hence get the best accuracy.

# normalizing the pixel values
max = X_train.max()
X_train = X_train/max
X_test = X_test/max

# shape of images
X_train.shape

#defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(25088,))) # 7*7*512 = 25088
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))


# defining a function to save the weights of best model
from keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)



# Save the entire model not only weights
model.save('video_model.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('video_model.h5')

