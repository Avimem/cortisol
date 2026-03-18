# About

This Machine Learning Based Real Time Stress and Emotion Detection System uses Computer Vision to look at the users face.

The proposed system takes video from the users webcam. It does this in real time.
It looks at each picture in the video to see if it can find a face.

When a human face is detected, the Dlib Facial Landmark Detection model can be executed to create and extract 68 pre-defined points of facial landmarks
or regions of interest on the face (i.e. eyes, eyebrows, bridge of nose, corners of lips, jawline).

It checks the movement of the users eyebrows and the openness of the users eyes and the shape of the users mouth and face.

# How to run it

It uses a machine learning model which is trained on the FER2013 dataset. To successfully run this poject on your system, you need to have FER2013 on your system.

First, compile and run the extract_features.cpp file after you have downloaded and extracted the dataset. Then, you will have a CSV file containing all the data
from the dataset.

Then run the train_model.py script to train your model based on the data extracted from the dataset. This will result in you getting 2 .pkl files, keep them in 
the same directory where all the files you've downloaded are.

After that, run the server.py script which will host your trained model and then compile and run the main.cpp file.

This project is still in it's early stages and needs refinement.
