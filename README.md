# Video-Classification
Video Classification on Dataset UCF 11

Classification is done on 11 categories using Transfer Learning using pre-trained model VGG16 224 resolution trained on an imagenet dataset of 1000 categories.

Dataset available here:- https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php

# After this I also did Video Classification on UCF 101 dataset

Here Making directories.py is used to make directories.

Transfer test dataset.py is used to extract images from videos for the test dataset.

Transfer Training dataset.py is used to extract images from videos for the training dataset.

Recognition is done on 101 categories using Transfer Learning using pre-trained model ResNet50 224 resolution trained on an imagenet dataset of 1000 categories.

Dataset available here:- https://www.crcv.ucf.edu/data/UCF101.php

I and my friend personally extract the images from the videos and upload them to the kaggle so if you want then you can use that dataset.
Dataset available here:- https://www.kaggle.com/ash81197/vid-classification-ucf101

Here dataset has an equal amount of frames in each class i.e 1500 and 150 frames for training and testing purposes.

UCF_Preprocessing.ipynb:- This Notebook contains the codes that will help to extract the balanced dataset from the videos.

UCF_Training_Evaluating( main file ).ipynb:- This Notebook contains the code that will help to train the model on different pre-trained models as per user choice.
Here I use transfer learning using different versions of the ResNet Model.

Performance matrices of the model.py:- This python file will extract all the information about the model which we get trained from the UCF 101 dataset. This will give the Confusion Matrix, Precision_Score, F1_Score, Recall_Score all in one code.

![Output sample](https://github.com/Chirag-v09/Video-Classification/blob/master/gif_1.gif)


## Preprocessing: The old fashioned way
First, download the dataset from UCF Repository [https://www.crcv.ucf.edu/data/UCF101.php] then run the UCF_Preprocessing.py file to preprocess the UCF101 dataset.

In the preprocessing phase, we used a different technique in which we extracted exactly 1,650 frames per category meaning 1,650 x 101 = 1,66,650 frames or you can say images in the whole dataset


## About Dataset

UCF101 folder: ~100 to 150 videos per category

~13,320 total videos

after combining all the videos of each and every single category and saving them in dataset named folder dataset folder:

~only 1 video per category

~101 total videos in the dataset folder after combining all the videos

training folder:

~1500 frames per category

~1500 x 101 = 151500 total frames

How do we achieve this, you may ask, well it's not that easy as it seems

What we did was calculated the total number of frames of every video of a single category meaning: let's say ApplyEyeMakeup has 10 videos of 5 seconds long clips each and let's say that those 10 videos are of 30 FPS (Frames per second) number of frames in one category = number of videos in each category x length of a clip x frame rate of a single video = 10 x 5 x 30 = 1500 total number of frames in a SINGLE CATEGORY and let's assume we only need 750 frames so we take every second frame and write it out onto the "training_set/" folder present on your same working directory that's how we separated frame from the videos and made a BALANCED DATASET

testing folder:

~150 frames per category

~150 x 101 = 15150 total frames

To make our testing data, we randomly selected 150 frames from the training set for each category and moving them from "training_set/" and storing them onto "testing_set/" named folder

frames in training set = 151500 frames validation set = frames taken from validation set = 20% of training set (30300 frames) frames in testing set = 15150 (10% of training set)

for training purpose, we used "training_set/" directory and for testing, we used "testing_set/" directory


## Model Analysis:

models used:

*ResNet50*

*ResNet101*

*ResNet50V2*

*ResNet101V2*

*MobileNet*

*MobileNetV2*

MobileNet and MobileNetV2 are the worst model to perform video classification because they aren't made for heavy datasets in fact they are made for Mobile and Embedded Devices, hence named "Mobile" also MobileNets are giving good accuracies but have higher losses, that's why we discarded this model

ResNet50 and ResNet50v2, both are giving many impressive results than their counterparts MobileNets but took much time for training because of the fact that it contains deeper hidden layers than MobileNet models.

## Required Parameters

dataset = "UCF-101/" # Actual Dataset Path

dataset2 = "dataset/" # After Combining all videos of the Dataset, the recreated Dataset Path

train_path = "training_set/" # Training Path

test_path = "testing_set/" # Testing Path

no_of_frames = 1650 # Number of Frames to be extracted from a single category

epochs = 20 # Number of epochs to run

batch_size = 32 # Batch Size

n_classes = 101 # Number of Classes

optimizer = "Adam" # Adam (adaptive momentum) optimizer is used

loss_metric = "categorical_crossentropy" # Loss Metric used for every model is one and same

last_layer_activation_function = "softmax" # Softmax function is used for last layer

input shape of ResNet50, ResNet101, ResNet50V2, ResNet101, MobileNet and MobileNetV2 are all the same and that is: (224, 224, 3) => [image height, image width and number of channels]
