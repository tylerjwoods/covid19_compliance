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

    # load face mask detector model from disk
    print("...Loading Face Mask Detector Model...")
    model_path = 'models/face_mask_detector.model'
    model = load_model(model_path)

    # clone the image and grab the image spatial dimensions
    orig = image.copy()
    (h, w) = image.shape[:2]

    # use CV2 to create a 'blob' from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the net and obtain face detections
    print('...Computing Face Detections...')
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
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

             # display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    # get image from the 
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())

    image = cv2.imread(args['image'])

    print('...Predicting Image...')
    predict_image(image)

