'''
This script creates a model using transfer learning and 
images scraped from google.

In order to use this script, run the following from the command line:
python train_model.py

Hyperparameters in __main__ can be tuned as needed.
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
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import os

# make the plot pretty
plt.style.use('ggplot')

def load_images_and_labels():
    '''
    Loads all images from directory 'images' with target as the folder
    name. Images were scrapped from google.
    inputs
    ------
    None

    returns
    -------
    images: numpy array of images
    labels: numpy array of labels
    '''
    images = []
    labels = []
    # get list of images from the directory
    image_paths = list(paths.list_images('images/'))
    
    # loop over the image_paths
    for image_path in image_paths:
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

    return images, labels

def train_CNN(images, labels, epochs, learning_rate, bs):
    '''
    Train the CNN using transfer learning and images from 
    load_images_and_labels().
    '''

    # turn labels into binary labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # split data using 20% as testing data
    X_train, X_test, y_train, y_test = train_test_split(images, labels,
                                                        test_size=0.20, 
                                                        stratify=labels, 
                                                        random_state=17)

    # transformations
    train_transformations = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2, 
        height_shift_range=0.2,
        zoom_range=0.10,
        shear_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest" # constant, nearest, reflect, or wrap
    )

    # load the MobileNetV2 network, ensuring the head FC layer sets are 
    # left off
    base_model = MobileNetV2(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224,224,3)))

    # build model head
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7,7))(head_model)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(128, activation='relu')(head_model)
    head_model = Dropout(0.50)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    # place the head FC model on top of the base model (this will become 
    # the actual model we will train)
    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False 

    # initalize adam optimizer
    opt = Adam(lr = learning_rate, decay = learning_rate / epochs)

    # compile and fit model
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    history = model.fit(
        train_transformations.flow(X_train, y_train, batch_size=bs),
        steps_per_epoch=len(X_train) // bs,
        validation_data = (X_test, y_test),
        validation_steps = len(X_test) // bs,
        epochs=epochs
    )

    # get predictions and generate confusion matrix data
    y_pred = model.predict(X_test, batch_size=bs)
    con_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    # show a nicely formatted classification report
    # print(classification_report(y_test.argmax(axis=1), y_pred,
    #     target_names=lb.classes_))

    # plot the loss and accuracy
    N = epochs
    plt.figure()
    plt.plot(np.arange(0,N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0,N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0,N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0,N), history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('plots/plot_{}_{}.png'.format(epochs, bs))

    # return the model, history, and confusion matrix
    return model, history, con_mat 

if __name__ == '__main__':
    # this can be changed to fine-tune model as needed
    # model is greater than 95% accurate with these
    epochs = 20
    learning_rate = 0.001
    batch_size = 32

    print('...Loading images...')
    images, labels = load_images_and_labels()

    print('...Building and evaluating model...')
    model, history, con_mat = train_CNN(images, labels, epochs, 
                                        learning_rate, batch_size)
    
    # save model
    print('...Saving model...')
    model.save("face_mask_detector2.model", save_format="h5")

    # save history into dataframe
    hist_df = pd.DataFrame(history.history)
    hist_csv = 'models/history_{}_epochs.csv'.format(epochs)

    with open(hist_csv, mode='w') as f:
        hist_df.to_csv(f)

    np.savetxt('models/history_{}_epochs_conmat.csv'.format(epochs), con_mat)