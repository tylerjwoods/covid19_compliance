'''
This script uses model created in train_model.py
as well as a res10 caffemodel from OpenCV to find 
faces in images, put a rectangle box around the face,
and predict whether or not the face is wearing a mask.
'''

import numpy as np 
import argparse 
import cv2 
import os 

def predict_image(model, image):
    print(image)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())

    a = 'model'

    image = cv2.imread(args['image'])

    predict_image(model, image)

