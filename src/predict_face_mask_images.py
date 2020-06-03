'''
This script uses model created in train_model.py
as well as a res10 caffemodel from OpenCV to find 
faces in images, put a rectangle box around the face,
and predict whether or not the face is wearing a mask.

To use this function, run
python src/predict_face_mask_images.py -i test_images/tyler.jpg
'''

# import necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np 
import argparse 
import cv2 
import os 

def predict_image(image):
    # load serialized face detector model from disk
    print("...Loading Face Detector Model...")
    prototxt_path = 'face_detector/deploy.prototxt'
    weights_path = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNet(prototxt_path, weights_path)

if __name__ == '__main__':
    # get image from the 
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())

    image = cv2.imread(args['image'])

    print('...Predicting Image...')
    predict_image(image)

