# import necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np 
import argparse 
import cv2 
import os 

def predict(image, file_name):
    # load face mask detector model from disk
    print("...Loading Face Mask Detector Model...")
    model_path = 'models/face_mask_detector.model'
    model = load_model(model_path)

    # read image using cv2
    image = cv2.imread(image)

    # convert from BGR to RGB, resize to 224x224, and preprocess it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # use model to predict on the image
    mask, without_mask = model.predict(image)[0]

    # generate a label and color using the prediction
    if mask > without_mask:
        label = "Mask"
        color = (255, 255, 0)
    else:
        label = "No Mask"
        color = (245, 66, 221)
    
    # include the probability in the label
    label = "{}: {}%".format(label, int(max(mask, without_mask) * 100))

    # display label on the image
    cv2.putText(image, label, (10, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 1)

    # show the output image
    #cv2.imshow("Mask Detector", image)
    #cv2.waitKey(0) 

    # save the image
    print('...Saving Predicted Image...')
    filepath = 'predicted_images/{}_1'.format(file_name)
    #print('The image path is {}'.format(filepath))
    cv2.imwrite(filepath, image)

if __name__ == '__main__':
    # get image from the argparse
    # image MUST be in test_images directory
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())

    image = args['image']

    file_name = args['image'][12:]

    print('...Predicting Image...')
    predict(image, file_name)
    

