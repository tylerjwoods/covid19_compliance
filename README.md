# COVID-19 Compliance
COVID-19 Face Mask and Social Distancing

## Table of Contents

- [Basic Overview](#basic-overview)
  - [Context](#context)
  - [Goal](#goal)
- [Dataset](#dataset)
  - [Initial Intake](#initial-intake)
  - [Cleaning Images](#cleaning-images)
- [Modeling](#modeling)
  - [Transfer Learning](#transfer-learning)
  - [Application](#application)
- [Face Capture and Video Processing](#face-capture-and-video-processing)
  - [Face Capture](#face-capture)
  - [Video Processing](#video-processing)
- [Social Distance Detector](#social-distance-detector)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Basic Overview

### Context

As of March 2020, the world changed when COVID-19 swept across the United States and the rest of the world. Stores were shutdown and interaction between people was halted. As the world begins to reopen, health officials urge people to wear face masks to help prevent the spread of COVID-19.

### Goal

Imagine that you're walking into Costco and you see this sign. There's an employee at the door cheaking for compliance.
<p align="center">
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/demo/costco.jpeg" width="400" height="400" title="costco" />
</p>

You walk into the store and only a minute later, you see a customer with their face mask off. There aren't enough employees to make sure every customer is KEEPING their mask on.

The goal of this project is to automatically detect if a person is wearing a face mask in images and videos.


## Dataset

### Initial Intake

Data was scrapped from google images by typing in 'COVID-19 face masks' and 'people faces' to get Masks and No Masks data. Approximately 700 of each category were scrapped.

### Cleaning Images

I then went through each image and deleted images that were not good to be used for the model. The cleaned images are stored in the images/ folder. More than 300 images were used for each class in building the model.

## Modeling

### Transfer Learning

Transfer learning in machine learning focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. In this situation, I used a model trained on the Imagenet images and classes with varying architectures (Xception, ResNet50, and MobileNetV2). After varying the epochs and batchsizes, some architectures had extremely high accuracy such as below.

<p align="center">
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/plots/plot_20_32_MobileNetV2.png" width="800" height="800" title="MobileNetV2" />
</p>

For comparing the various models, I built an ROC curve and displayed the AUC scores:

<p align="center">
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/plots/roc_curve.png" width="800" height="800" title="roc_curve" />
</p>

Note: Although the Xception architecture had a slightly higher AUC score, the MobileNetV2 architecture was able to run at about 3x the speed when building the new model so that one was chosen for this project.

### Application

As shown below, the model does a good job of identifying wearing a mask and not wearing a mask.

<p align="center">
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/first_tyler.jpg" width="400" height="400" >
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/first_no_mask_tyler.jpg" width="400" height="400" />
</p>


So what about if there's an image with both a person wearing a mask and a person not wearing a mask?

<p align="center">
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/first_tyler_malia.jpg" width="400" height="400" >
</p>

Since the model is binary, i.e., it will predict 'mask' if it finds a mask anywhere in the image, the model found the face mask and predicted face mask. 

The next step would be for detecting faces. The program would be then be broken down into two steps:

1. Find faces in images

2. Predict if that face is wearing a mask

## Face Capture and Video Processing

### Face Capture

Face detection is not an easy act to perform even for CNN learners. Models have been built and distrubited to assist with this task. In the 'References' section of this README, I have a link to Adrian Rosebrock's page who does an amazing job teaching and showing object detection using Python.

Now that the face capture is being performed, how do the predictions look now?

<p align="center">
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/tyler.jpg" width="400" height="400" >
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/no_mask_tyler.jpg" width="400" height="400" />
</p>

And testing for the two-person:

<p align="center">
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/tyler_malia.jpg" width="400" height="400" >
</p>

### Video Processing

Photos are great for a computer to 'see', but the real test is going to be for videos. Putting the face detector and face mask detector to work:

![demo gif](https://github.com/tylerjwoods/covid19_compliance/blob/master/demo/tyler1.gif)

And checking for the two-person test:

![demo gif2](https://github.com/tylerjwoods/covid19_compliance/blob/master/demo/tyler_malia.gif)

The two videos do a good job at predicting face masks or not. However, in the second video, you can see that there is a lot of 'flickering' when the mask is on for the individual in the background. The face detector is having a hard time 'seeing' that there is a face there, mostly because the face detector model is not trained on people wearing masks.

## Social Distance Detector

In addition to face masks, health experts also recommend that the public maintain a 6-ft distance at all times. Again, Adrian Rosebrock beat me to this and put out a great tutorial on using YOLOv3 to detect people in images and then use that detection to calculate if two person are too close together. See the link in 'References'.

The first iteration of the distance detector hard-coded in a minimum distance of 50 pixels that two people must be from each other. I decided to use the average height people in the frame. Also building upon PyImageSearch's code, I use the average height of people in the frame and the average height of humans to create a scale and then display a predicted distance between two people who were found to be too close together.

![demo gif3](https://github.com/tylerjwoods/covid19_compliance/blob/master/demo/ped_trim_2.gif)

This works well for near bird's-eye view of the people in the frame. 

I also tried to apply the code to a different view - walking around in a city. This proved to be much more difficult as people in the background are predicted as being too close.

![demo gif4](https://github.com/tylerjwoods/covid19_compliance/blob/master/demo/korea_trim_output_2.gif)

## References

"MobileNetV2." - Applications - Tensorflow Documentation https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2

“ResNet-50.” Applications - Keras Documentation https://keras.io/applications/#resnet

Rosebrock, Adrian. "Object Detection with Deep Learning and OpenCV." PyImageSearch, 11 Sep 2017, https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

Rosebrock, Adrian. "OpenCV Social Distancing Detector." PyImageSearch, 01 June 2020, https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/


## Acknowledgements

* Galvanize instructors, Phil Geurin, Andrew Nicholls, and Jack Bennetto.
* My fellow cohort-mates in the Galvanize DSI who all banded together to teach and learn from each other while social distancing.
* Family and friends for encouraging me to enroll and complete the Galvanize course.