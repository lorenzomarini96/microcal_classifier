# pylint: disable=W0611
# pylint: disable=E0401
"""CMEPDA Project: Image binary classification
using Convolutional Neural Network (CNN).

This python script evaluates the performance of the CNN to perform a
binary classification on a microcalcification image dataset.
The mammographic images containing either microcalcifications
or normal tissue. 

The CNN binary classifiers is evaluate by means the following parameters:

- Accuracy
- Precision
- Recall
- AUC

"""

import argparse
import os
import multiprocessing as mp
from pathlib import Path
import sys

from PIL import Image
import skimage
from skimage import img_as_float
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from cnnhelper import bar_plot, count_labels, plot_random_images, read_imgs, split_dataset
from cnnhelper import plot_correct_pred, plot_mis_pred, plot_confusion_matrix, plot_roc_curve
from cnn_model import cnn_classifier
from cross_validation import plot_cv_roc
from data_augmentation import convert_to_png, data_aug, single_image_aug

sys.path.insert(0, str(Path(os.getcwd()).parent))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="CNN classifiers analysis in digital mammography."
    )

    parser.add_argument(
        "-dp",
        "--datapath",
        metavar="",
        help="path of the data folder.",
        default="/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/"
    )

    parser.add_argument(
        "-de",
        "--dataexploration",
        metavar="",
        type=bool,
        help="Data exploration: data set partition histogram and image visualization.",
        default=True,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        metavar="",
        type=int,
        help="Number of epochs for train the CNN.",
        default="25",
    )

    parser.add_argument(
        "-ad",
        "--augdata",
        metavar="",
        type=int,
        help="Perform data augmentation procedure.",
        default="25",
    )

    parser.add_argument(
        "-k",
        "--kcrossvalidation",
        metavar="",
        type=int,
        help="Number of folder to use for the cross validation and the plot of ROC curve.",
        default="5",
    )

    args = parser.parse_args()

    #===========================================
    # STEP 1: Import data set,  data exploration
    #===========================================

    #PATH = '/home/lorenzomarini/Desktop/microcal_classifier/dataset/IMAGES/Mammography_micro/'
    PATH = args.datapath
    TRAIN_PATH = os.path.join(PATH, 'Train')
    TEST_PATH = os.path.join(PATH, 'Test')

    x_train, y_train = read_imgs(TRAIN_PATH, [0, 1])
    x_test, y_test = read_imgs(TEST_PATH, [0, 1])

    if args.dataexploration:
        bar_plot(y_train, y_test)
        dataframe = count_labels(y_train, y_test, verbose=True)
        plot_random_images(X=x_train, y=y_train, n_x=3, n_y=3)

    # Split the train dataset for train and validation set
    PERC = 0.25
    x_train_split, x_val, y_train_split, y_val = split_dataset(x_train,
                                                               y_train,
                                                               x_test,
                                                               y_test,
                                                               perc=PERC,
                                                               verbose=True
                                                               )
                                                            
    #===========================================
    # STEP 2: Build, train and evaluate the CNN
    #===========================================
    INPUT_SHAPE = (60, 60, 1)
    model = cnn_classifier(shape=INPUT_SHAPE,  verbose=True)

    # Training
    EPOCHS = args.epochs

    history = model.fit(x_train, 
                    y_train,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_freq=1,
                    )
    
    # Training history
    print(history.history.keys())
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

    # Accuracy history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()


    # Prediction
    y_val_pred = model.predict(x_val)

    # Plot correct/uncorrect classified images
    y_test_pred = model.predict(x_test)
    plot_correct_pred(y_test, y_test_pred, x_test)
    plot_mis_pred(y_test, y_test_pred, x_test)

    # Performace on test data
    test_accuracy_score = accuracy_score(y_test, y_test_pred.round())
    print(f'\nTest accuracy score: {test_accuracy_score:.2f}')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred.round())
    cm_plot_label =['Normal tissue', 'Micro  cluster']
    plot_confusion_matrix(cm, cm_plot_label, title ='Confusion Matrix')

    # ROC curve
    plot_roc_curve(y_test, y_test_pred)

    # Classification report
    print(classification_report(y_test, y_test_pred.round()))

    #if args.augdata:
        #============================
        # Data augmentation procedure
        #============================

    '''
    #if args.kcrossvalidation:
        #=============================
        # K-Cross validation procedure
        #=============================
        
        INPUT_SHAPE = (60, 60, 1)
        plot_cv_roc(X=x_train_raw,
                    y=y_train_raw,
                    X_test=x_test,
                    y_test=y_test,
                    model=cnn_model(shape=INPUT_SHAPE, verbose=False),
                    n_splits=5
        )

    '''