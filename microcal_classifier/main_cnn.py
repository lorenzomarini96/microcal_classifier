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
        type=bool,
        help="Perform data augmentation procedure.",
        default="True",
    )

    parser.add_argument(
        "-cv",
        "--crossvalidation",
        metavar="",
        type=bool,
        help="Perform the cross validation and the plot of ROC curve.",
        default="True",
    )

    parser.add_argument(
        "-k",
        "--kfolds",
        metavar="",
        type=int,
        help="Number of folds for the cross validation and the plot of ROC curve.",
        default=5,
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

    #============================
    # Data augmentation procedure
    #============================
    if args.augdata:
        for data_path in [TRAIN_PATH, TEST_PATH]:
            for path, folders, fnames in os.walk(data_path):
                for fname in fnames:
                    abs_path = os.path.join(path, fname)
                    dest_folder = path.replace('Train', 'Train_png').replace('Test', 'Test_png')
                    convert_to_png(abs_path, dest_folder)

        BATCH_SIZE = 32
        IMG_WITDH, IMG_HEIGHT = (60, 60)
        TRAIN_PATH_png = os.path.join(PATH, 'Train_png')
        train_gen, val_gen = data_aug(train_dataset_path=TRAIN_PATH_png,
                                    img_width=IMG_WITDH,
                                    img_height=IMG_HEIGHT,
                                    batch_size=BATCH_SIZE
                                    )

        # Fit model on augmented dataset

        aug_model = cnn_classifier()
        aug_model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy']
                          )

        history = aug_model.fit(train_gen,
                            batch_size=32,
                            validation_data=val_gen,
                            epochs=args.epochs,
                            shuffle=True,
                            validation_freq=1,
                            )
        # History del training
        print(history.history.keys())
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title('Data Augmented Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.show()

        # History for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Data Augmented Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.show()

        # Evaluate and compare the performances of the two models on test dataset

        # 1)
        # Trai set
        model.evaluate(x_train, y_train)

        # Validation set
        model.evaluate(x_test, y_test)

        # Test set
        model.evaluate(x_val, y_val)

        # Classification report
        y_test_pred = model.predict(x_test)
        print(classification_report(y_test, y_test_pred.round()))

        # 2)
        # Trai set
        aug_model.evaluate(x_train, y_train)

        # Validation set
        aug_model.evaluate(x_test, y_test)

        # Test set
        aug_model.evaluate(x_val, y_val)

        # Classification report
        y_test_pred = aug_model.predict(x_test)
        print(classification_report(y_test, y_test_pred.round()))

        # Confusion Matrix
        cm = confusion_matrix(y_test, model.predict(x_test).round())
        cm_plot_label =['Normal tissue', 'Micro  cluster']
        plot_confusion_matrix(cm, cm_plot_label, title ='Confusion Matrix')

        cm = confusion_matrix(y_test, aug_model.predict(x_test).round())
        cm_plot_label =['Normal tissue', 'Micro  cluster']
        plot_confusion_matrix(cm, cm_plot_label, title ='Confusion Matrix aAug data')

        # ROC curve
        plot_roc_curve(y_test, model.predict(x_test))
        plot_roc_curve(y_test, aug_model.predict(x_test))

    #=============================
    # K-Cross validation procedure
    #=============================
    if args.crossvalidation:
        print(f'\nK-Cross validation procedure (K = {args.kfolds})')
        INPUT_SHAPE = (60, 60, 1)
        plot_cv_roc(X=x_train,
                    y=y_train,
                    X_test=x_test,
                    y_test=y_test,
                    model=cnn_classifier(shape=INPUT_SHAPE, verbose=False),
                    n_splits=args.kfolds,
                    epochs=args.epochs)
