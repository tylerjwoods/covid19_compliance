# import packages
from .social_distancing_config import nms_thresh
from .social_distancing_config import min_conf
import numpy as np 
import cv2 

def detect_people(frame, net, ln, personIdx = 0):
    '''
    inputs
    ------
    frame: frame of video file
    net: pre-initialized and pre-trained YOLO object detection model
    ln: YOLO CNN output layer names
    personIdx: The YOLO model can detect many types of objects -- this index
        is specifically for the person class.
    '''

    # grab the dims of the gram and intialize the list 
    # of results
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), 
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize lists of detected bounding boxes, centroids, and 
    # confidences, respectively
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (probability) 
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum confidence is met
            if classID == personIdx and confidence > min_conf:
                # scale the bounding box coords back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x,y) coords of 
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y) coords to derive the top
                # and left center of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height /2 ))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_thresh)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coords
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # update our results list to consist of the person predition
                    # probability, bounding box coordinates,
                    # and the centroid
                    r = (confidences[i], (x, y, x+w, y+h), centroids[i])
                    results.append(r)

            # return the list of results
            return results 




