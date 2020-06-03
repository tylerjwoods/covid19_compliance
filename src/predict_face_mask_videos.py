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

def predict_frame(image, net, model):
    '''
    This function uses very similar script to 
    predict_face_mask_images.py except refactored
    to return frames to the predict_video function

    inputs
    ------
    frame: cv2 vs.read() image
    net: face detector model
    model: mask detector model
    '''
     # grab the image spatial dimensions
    (h, w) = image.shape[:2]

    # use CV2 to create a 'blob' from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the net and obtain face detections
    #print('...Computing Face Detections...')
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0 , detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x,y) coords of bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure box falls within dims of box frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face region of interest (ROI), conver it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX: endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

             # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (255, 255, 0) if label == "Mask" else (245, 66, 221)

            # include the probability in the label
            label = "{}: {}%".format(label, int(max(mask, withoutMask) * 100))

             # display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)
            cv2.circle(image, (int((startX + endX)/2), int((startY + endY)/2)), 
                       max(int((endX - startX)/2), int((endY - startY)/2)), color, 8)
    return image

def predict_video(video_path, file_name, size):
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
    #mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    #Q = deque(maxlen=size)


    # initalize the video stream, pointer to outpout video file,
    # and frame dimensions
    vs = cv2.VideoCapture(video_path)
    writer = None 
    (W, H) = (None, None) 


    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (success, frame) = vs.read()

        #print('...{}...'.format(success))

        # if the frame was not grabbed, then end of the stream
        if not success:
            break 

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # clone the output frame, then convert it from BGR to
        # RGB ordering, resize the frame to a fixed 224x224, then
        # perform mean subtraction
        #output = frame.copy()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (224, 224)).astype("float32")
        #frame -= mean 

        # make predictions on the frame and then update the predictions
        # queue
        frame_prediction = predict_frame(frame, net, model)

        # check if the video writer is None
        if writer is None:
            # initalize the video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            filepath = 'predicted_videos/{}.avi'.format(file_name)
            writer = cv2.VideoWriter(filepath, fourcc, 30,
                (W, H), True)
        
        # write the frame_prediction to disk
        writer.write(frame_prediction)
    
    # release the file pointers
    print("...Cleaning Up...")
    writer.release()
    vs.release()





if __name__ == '__main__':

    # this can be changed to fine-tune videos as needed
    size = 128

    # get the video from the argparse
    # video MUST be in test_videos directory
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True)
    args = vars(ap.parse_args())

    file_name = args['video'][12:-4]

    video_path = args['video']


    print('...Predicting Video...')
    predict_video(video_path, file_name, size)