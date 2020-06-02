'''
This script creates a model using transfer learning and 
images scraped from google.

In order to use this script, run the following from the command line:
python train_mask_detector.py --dataset dataset
'''

# import the necessary packages
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import os

# make the plot pretty
plt.style.use('ggplot')

def load_images_and_labels():
    '''
    Loads all images from directory 'images' with target as the folder
    name.
    '''
    images = []
    labels = []
    # get list of images from the directory
    print('...Loading images...')
    image_paths = list(paths.list_images('images/'))
    
    # loop over the image_paths
    for i, image_path in enumerate(image_paths):
        label = image_path.split(os.path.sep)[-2]
        img = load_img(image_path, target_size=(224,224))
        img = img_to_array(img)
        img = preprocess_input(img)

        # add the image and label to the respective lists
        images.append(img)
        labels.append(label)

    # convert the images and labels to numpy arrays
    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    print(labels)

if __name__ == '__main__':
    load_images_and_labels()

