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
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/plots/plot_20_32_MobileNetV2.png" width="400" height="400" title="MobileNetV2" />
</p>

For comparing the various models, I built an ROC curve and displayed the AUC scores:

<p align="center">
<img src="https://github.com/tylerjwoods/covid19_compliance/blob/master/plots/roc_curve.png" width="400" height="400" title="roc_curve" />
</p>

### Application

As shown below, the model does a good job of identifying wearing a mask and not wearing a mask.

<p align="center">
<img align="right" src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/first_tyler.jpg" width="400" height="400" >
<img align="center" src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/first_no_mask_tyler.jpg" width="400" height="400" />
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

### Video Processing

![demo gif](https://github.com/tylerjwoods/covid19_compliance/blob/master/demo/tyler1.gif)

## References

"MobileNetV2." - Applications - Tensorflow Documentation https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2

“ResNet-50.” Applications - Keras Documentation https://keras.io/applications/#resnet.

## Acknowledgements