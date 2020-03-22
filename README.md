# Video-Classification
Video Classification on Dataset UCF 11

Classification is done on 11 categories using Trnsfer Learning using pretrained model VGG16 224 resolution trained on imagenet dataset of 1000 categories.

Dataset available here :- https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php

# After this I also done classification on UCF 101 dataset

Here Making directories.py is used to make directories.

Transfer test dataset.py is used to extract images from videos for test dataset.

Tranfer Training dataset.py is used to etract images from videos for training dataset.

Recognition is done on 101 categories using Transfer Learning using pre-trained model ResNet50 224 reolution trained on imagenet dataset of 1000 categories.

Dataset avaliable here :- https://www.crcv.ucf.edu/data/UCF101.php

Me and my friend personally extract the images from the videos and upload it to the kaggle so if you want the you can use that dataset.
Dataset available here :- https://www.kaggle.com/ash81197/vid-classification-ucf101

Here dataset has an equal amount of frames in each clases i.e 1500 and 150 frames for training and testing purposes.

UCF_Preprocessing.ipynb :- This Notebook contain the codes that will help to extract the balacend dataset from the videos.

UCF_Training_Evaluating( main file ).ipynb :- This Notebook contain the code that will help to train the model on different pre-trained models as per user choice.
Here I use trnsfer learning using different versions of ResNet Model.

Perfomance matrices of the model.py :- This python file will extract the all the information about the model which we get trained from the UCF 101 dataset. This will give the Confusion Matrix, Precision_Score, F1_Score, Recall_Score all in one code.

![Output sample](https://github.com/Chirag-v09/Video-Classification/blob/master/gif_1.gif)
