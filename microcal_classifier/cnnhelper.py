"""Helper functions for microcal_classifier project."""

import os
import glob
import logging

import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

'''
import tensorflow
tensorflow.random.set_seed(1)
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
'''

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("CNN_model.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def bar_plot(y_train, y_test):
    """Function to make a horizontal bar plot
    showing the relative partition of the two given dataset.

    Args:
        y_train (array): Numpy array with the label (0,1) for the train data set.
        y_test (array): Numpy array with the label (0,1) for the test data set.

    """

    data = {'0': [sum(map(lambda x : x == 0, y_train)),sum(map(lambda x : x == 0, y_test))],
          '1': [sum(map(lambda x : x == 1, y_train)),sum(map(lambda x : x == 1, y_test))]
          }
    df = pd.DataFrame(data,columns=['0','1'], index = ['Train','Test'])
    df.plot.barh()
    plt.title('Train and Test dataset partitions')
    plt.ylabel('Classes')
    plt.xlabel('Number of images')
    plt.show()


def barplot_split_set(y_train, y_test, val_perc=0.30):
    """Function plotting an hypothetical partition of the dataset in train, validation and test.

    Args:
        y_train (array): Numpy array with the label (0,1) for the train data set.
        y_test (array): Numpy array with the label (0,1) for the test data set.
        val_perc (float, optional): Percentual of items to use in validation.
        Defaults to 0.30.
    """
    data = {'0': [sum(map(lambda x : x == 0, y_train)) * (1 - val_perc),
            sum(map(lambda x : x == 0, y_train)) * val_perc,
            sum(map(lambda x : x == 1, y_test))],
            '1': [sum(map(lambda x : x == 1, y_train)) * (1 - val_perc),
            sum(map(lambda x : x == 1, y_train)) * val_perc,
            sum(map(lambda x : x == 1, y_test))]
            }

    df = pd.DataFrame(data,columns=['0','1'], index = ['Train', 'Validation', 'Test'])
    df.plot.barh()
    plt.title('Train, Validation and Test dataset partitions')
    plt.ylabel('Classes')
    plt.xlabel('Number of images')
    plt.show()


def count_labels(y_train, y_test, verbose=True):
    """Count the number of items in the set.

    Args:
        y_train (array): Array of labels for train.
        y_train (array): Array of labels for train.
        verbose (bool, optional): Print the number of items in the set. Defaults to True.

    Returns:
        int: DataFrame containing the total number of items for a specific label and set.
    """
    count_0_train = sum(map(lambda label : label == 0, y_train))
    count_1_train = sum(map(lambda label : label == 1, y_train))
    count_0_test = sum(map(lambda label : label == 0, y_test))
    count_1_test = sum(map(lambda label : label == 1, y_test))
 
    dict = {'Label': ['0', '1'],
        'Train': [count_0_train, count_1_train],
        'Test': [count_0_test, count_1_test]}

    data = pd.DataFrame(dict)

    if verbose:
        print(data.to_string(index=False))

    return data


def plot_random_image(X, y, n_x=3, n_y=3):
    """Plot an image with given label randomly selected from a given dataset.

    Args:
        X (array): Numpy array containg the image values.
        y (array): Numpy arrays containg the label for each image.
        n_x (int): Number of images in the rows.
        n_y (int): Number of images in the columns.
    """

    plt.figure(figsize=(10,10))

    for i in range(n_x * n_y):
        plt.subplot(n_x, n_y, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        random_index = np.random.randint(len(y))
        plt.imshow(np.squeeze(X[random_index]), cmap='gray')
        plt.title(f"label={y[random_index]}", fontsize=14)

    plt.show()


# TODO: LOGGING E LE ECCEZIONI NON FUNZIONANO BENE...
def read_imgs(dataset_path, classes):
    """Function reading all the images in a given folder which already contains
    two subfolder.

    Args:
        dataset_path (str): Path to the image folder.
        classes (list): _description_

    Returns:
        array: Numpy array containing the value of image/label.

    Examples:
        >>> train_dataset_path = '/content/gdrive/MyDrive/DATASETS_new/IMAGES/Mammography_micro/Train'
        >>> x_train, y_train = read_imgs(train_dataset_path, [0, 1])

    """
    tmp = []
    labels = []
    for cls in classes:
        try:
            fnames = glob.glob(os.path.join(dataset_path, str(cls), '*.pgm'))
            logging.info(f'Read images from class {cls}')
            tmp += [imread(fname) for fname in fnames]
            labels += len(fnames)*[cls]
        except Exception as e_error:
            raise Exception('Image or path not found') from e_error
    logging.info(f'Loaded images from {dataset_path}')

    return np.array(tmp, dtype='float32')[..., np.newaxis]/255, np.array(labels)


def split_dataset(X_train, y_train, X_test, y_test, perc=0.2, verbose=True):
    """Split the train dataset for train and validation set.

    Args:
        X_train (array): Images of train set.
        y_train (array): Labels of train set.
        X_test (array): Images of test set.
        y_test (array): Labels of test set.
        perc (float): Percentual of items for the validation set.

    Returns:
        array: Numpy arrays containing...

    """

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=perc,
        random_state=11
        )

    count_0_train = sum(map(lambda label : label == 0, y_train))
    count_1_train = sum(map(lambda label : label == 1, y_train))
    count_0_val = sum(map(lambda label : label == 0, y_val))
    count_1_val = sum(map(lambda label : label == 1, y_val))
    count_0_test = sum(map(lambda label : label == 0, y_test))
    count_1_test = sum(map(lambda label : label == 1, y_test))

    dict = {'Label': ['0', '1'],
        'Train': [count_0_train, count_1_train],
        'Validation': [count_0_val, count_1_val],
        'Test': [count_0_test, count_1_test]}

    data = pd.DataFrame(dict)

    if verbose:
        print(data.to_string(index=False))

    return X_train, X_val, y_train, y_val

if __name__ == '__main__':

    TRAIN_DATASET_PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/Train'
    X_train, y_train = read_imgs(TRAIN_DATASET_PATH, [0, 1])

    TEST_DATASET_PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/Test'
    X_test, y_test = read_imgs(TEST_DATASET_PATH, [0, 1])

    #bar_plot(y_train, y_test)
    #barplot_split_set(y_train, y_test, val_perc=0.20)
    #count_labels(y_train, y_test, verbose=True)
    #plot_random_image(X_train, y_train, nx=3, ny=3)
    split_dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, perc=0.25, verbose=True)
