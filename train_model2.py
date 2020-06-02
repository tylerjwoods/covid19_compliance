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
    data = []
    labels = []
    # get list of images from the directory
    print('...Loading images...')
    image_paths = list(paths.list_images('images/'))
    
    # loop over the image_paths
    for i, image_path in enumerate(image_paths):
        label = image_path.split(os.path.sep)[-2]
        image = load_img(image_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        if i == 0:
            break
    print(image)

if __name__ == '__main__':
    load_images_and_labels()

