'''
This script uses model created in train_model.py
as well as a res10 caffemodel from OpenCV to find 
faces in videos, put a rectangle box around the face,
and predict whether or not the face is wearing a mask.

To use this function, run
python src/predict_face_mask_videos.py -i test_videos/tyler1.mp4
'''

# import necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np 
import argparse 
import imutils 
import time 
import cv2 
import os 
from collections import deque

def predict_video(video, file_name, size):
    '''
    inputs
    ------
    video: cv2.VideoCapture object
    file_name: input video file name. 
    size: int, max size of queue for rolling averaging
    '''
    # load serialized face detector model from disk
    print("...Loading Face Detector Model...")
    prototxt_path = 'face_detector/deploy.prototxt'
    weights_path = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNet(prototxt_path, weights_path)

    # load face mask detector model from disk
    print("...Loading Face Mask Detector Model...")
    model_path = 'models/face_mask_detector.model'
    model = load_model(model_path)

    # initialize the image mean for mean subtraction along with the
    # predictions queue
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    Q = deque(maxlen=size)


    # initalize the video stream, pointer to outpout video file,
    # and frame dimensions
    vs = cv2.VideoCapture(file_name)
    writer = None 
    (W, H) = (None, None)

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (success, frame) = vs.read()

        # if the frame was not grabbed, then end of the stream
        if not success:
            break 

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]





if __name__ == '__main__':

    # this can be changed to fine-tune videos as needed
    size = 128

    # get the video from the argparse
    # video MUST be in test_videos directory
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True)
    args = vars(ap.parse_args())

    video = cv2.VideoCapture(args['video'])

    file_name = args['video'][12:]


    print('...Predicting Video...')
    predict_video(video, file_name, size)