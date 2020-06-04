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
- [License](#license)

## Basic Overview

### Context

### Goal

## Dataset

### Initial Intake

### Cleaning Images

## Modeling

### Transfer Learning

### Application

<img align="right" src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/first_tyler.jpg">

The model does a good job of identifying wearing a mask. But what about no mask?

<img align="right" src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/first_no_mask_tyler.jpg">

Here we see that, again, the model is doing a good job of predicting no mask. So what about if there's an image with both a person wearing a mask and a person not wearing a mask?

<img align="right" src="https://github.com/tylerjwoods/covid19_compliance/blob/master/predicted_images/first_tyler_malia.jpg">

Since the model is binary, i.e., it will predict 'mask' if it finds a mask anywhere in the image, the model found the face mask and predicted face mask. 

The next step would be for detecting faces. The program would be then be broken down into two steps:

1. Find faces in images

2. Predict if that face is wearing a mask

## Face Capture and Video Processing

### Face Capture

### Video Processing

![demo gif](https://github.com/tylerjwoods/covid19_compliance/blob/master/demo/tyler1.gif)