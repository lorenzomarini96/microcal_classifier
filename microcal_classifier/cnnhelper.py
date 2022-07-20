"""Helper functions for microcal_classifier project."""
# pylint: disable=C0103
# pylint: disable=W0611
# pylint: disable=R0914
# pylint: disable=R0913

import os
import glob
import logging
import itertools

import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("CNN_helper.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def bar_plot(y_train, y_test):
    """Plot an horizontal bar graph
    showing the relative partition of the two given dataset.

    Parameters
    ----------
    y_train : numpy_array
        Labels for the train data set.
    y_test : numpy_array
        Labels for the train data set.

    Returns
    -------
    None

    Examples
    --------
    >>> import numpy as np
    >>> y_train = np.array([1, 0, 0, 1, 1])
    >>> y_test = np.array([1, 1, 0, 1, 1])
    >>> bar_plot(y_train, y_test)
    """

    data = {'0': [sum(map(lambda x : x == 0, y_train)), sum(map(lambda x : x == 0, y_test))],
            '1': [sum(map(lambda x : x == 1, y_train)), sum(map(lambda x : x == 1, y_test))]
           }
    df = pd.DataFrame(data,columns=['0','1'], index = ['Train','Test'])
    df.plot.barh()
    plt.title('Train and Test dataset partitions')
    plt.ylabel('Classes')
    plt.xlabel('Number of images')
    plt.show()


def barplot_split_set(y_train, y_test, val_perc=0.30):
    """Plot a graph bar showing partition of the dataset for train, validation and test.

    Parameters
    ----------
    y_train : numpy_array
        Labels for the train data set.
    y_test : numpy_array
        Labels for the train data set.
    val_perc : float, optional
        Percentual of items to use in validation.
        Defaults to 0.30.

    Returns
    -------
    None

    Examples
    --------
    >>> import numpy as np
    >>> y_train = np.array([1, 0, 0, 1, 1])
    >>> y_test = np.array([1, 1, 0, 1, 1])
    >>> PERCENTUAL = 0.30
    >>> bar_plot(y_train, y_test,  val_perc=PERCENTUAl)
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

    Parameters
    ----------
    y_train : numpy_array
        Labels for the train data set.
    y_test : numpy_array
        Labels for the test data set.
    verbose : bool, optional
        Print the number of items in the set. Defaults to True.

    Returns
    -------
    data : pandas dataframe.
        Dataframe containing the total number of items for a specific label and set.

    Examples
    --------
    >>> import numpy as np
    >>> y_train = np.array([1, 0, 0, 1, 1])
    >>> y_test = np.array([1, 1, 0, 1, 1])
            0    1
    Train   2    3
    Test    1    4
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


def plot_random_images(X, y, n_x=3, n_y=3):
    """Plot an image with given label randomly selected from a given dataset.

    Parameters
    ----------
    X : numpy_array
        Image pixel intensity values.
    y : numpy_array
        Labels for the train data set.
    n_x : int
        Number of images in the rows.
    n_y : int
        Number of images in the columns.

    Returns
    -------
    fig : Figure

    Examples
    --------

    >>> import matplotlib.image as mpimg
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> imgs = ['img1.pgm', 'img2.pgm', 'img3.pgm', 'img4.pgm']
    >>> X = np.array([mpimg.imread(i), for i in imgs])
    >>> labels = [1, 0, 0, 1]
    >>> N_X, N_Y = (2, 2)
    >>> plot_random_image(X=images, y=labels, n_x=N_X, n_y=N_Y)
    """

    fig = plt.figure(figsize=(10,10))

    for i in range(n_x * n_y):
        plt.subplot(n_x, n_y, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        random_index = np.random.randint(len(y))
        plt.imshow(np.squeeze(X[random_index]), cmap='gray')
        plt.title(f"label={y[random_index]}", fontsize=14)

    plt.show()

    return fig


def read_imgs(dataset_path, classes):
    """Function reading all the images in a given folder which already contains
    two subfolder.

    Parameters
    ----------
        dataset_path : str
            Path to the image folder.
        classes : list
            0 and 1 mean normal tissue and microcalcification clusters, respectively.

    Returns
    -------
        array: numpy_array
            Array containing the value of image/label.

    Examples
    --------
    >>> TRAIN_DATA_PATH = '/path/to/train/folder'
    >>> x_train, y_train = read_imgs(TRAIN_DATA_PATH, [0, 1])
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

    Parameters
    ----------
        X_train : numpy_array
            Images of train set.
        y_train : numpy_array
            Labels of train set.
        X_test : numpy_array
            Images of test set.
        y_test : numpy_array
            Labels of test set.
        perc : float
            Percentual of items for the validation set.

    Returns
    -------
        X_train_split : numpy_array
            Split images data set for training.
        X_val : numpy_array
            Image data set for validation.
        y_train_split : numpy_array
            Split label data set for training.
        y_val : numpy_array
            Label data set for validation.

    Examples
    --------
    >>> X_train, x_val, y_train, y_val = train_test_split(X_train, y_train,
    >>>                                                   test_size = 0.25,
    >>>                                                   random_state = 11
    >>>                                                   )
    """

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train,
                                                                  y_train,
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

    return X_train_split, X_val, y_train_split, y_val


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.

    Parameters
    ----------
    cm : sklearn object
        Confusion matrix metrics computed with sklearn
    classes : str
       Binary classification labels.
    title : str
        Title of plot.
    cmap : matplotlib object
        Colors of the matrix.

    Returns
    -------
    fig : Figure

    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return fig


def plot_roc_curve(y_test, y_test_pred):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_test : numpy array
        Original labels of test set.
    y_test_pred : numpy array
        Predicted labels of test set.

    Returns
    -------
    fig : Figure

    """

    roc_log = roc_auc_score(y_test, y_test_pred)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_test_pred)
    area_under_curve = auc(false_positive_rate, true_positive_rate)

    fig = plt.figure(figsize=(10,10))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {area_under_curve:.2f}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    plt.close()

    return fig


def plot_correct_pred(y, y_pred, x):
    """
    Plot several images correctly classified.

    Parameters
    ----------
    y : numpy array
        Original labels.
    y_pred : numpy array
        Predicted labels.
    x : numpy array
        Original images.

    Returns
    -------
    None
    """

    prop_class=[]

    for i in range(len(y)):
        if y[i] == y_pred.round()[i]:
            prop_class.append(i)

    random_idx_list = np.random.permutation(prop_class)

    plt.figure(figsize=(15,15))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(x[random_idx_list[i]]), cmap='gray')
        plt.title(f"Actual result: {y[random_idx_list[i]]};\nPredicted result: {int(y_pred.round()[random_idx_list[i]])}",
        fontsize=14,
        color='black'
        )
        plt.subplots_adjust(top=0.92,
        bottom=0.08,
        left=0.10,
        right=0.95,
        hspace=0.25,
        wspace=0.35
        )
    plt.show()


def plot_mis_pred(y, y_pred, x):
    """
    Plot several images uncorrectly classified.

    Parameters
    ----------
    y : numpy array
        Original labels.
    y_pred : numpy array
        Predicted labels.
    x : numpy array
        Original images.

    Returns
    -------
    None
    """

    mis_class=[]

    for i in range(len(y)):
        if y[i] != y_pred.round()[i]:
            mis_class.append(i)

    random_idx_list = np.random.permutation(mis_class)

    plt.figure(figsize=(15,15))

    if (len(mis_class) >= 10):
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(np.squeeze(x[random_idx_list[i]]), cmap='gray')
            plt.title(f"Actual result: {y[random_idx_list[i]]};\nPredicted result: {int(y_pred.round()[random_idx_list[i]])}",
            fontsize=14,
            color='black'
            )
    else:
        for i in range(len(mis_class)):
            plt.subplot(1, len(mis_class), i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(np.squeeze(x[random_idx_list[i]]), cmap='gray')
            plt.title(f"Actual result: {y[random_idx_list[i]]};\nPredicted result: {int(y_pred.round()[random_idx_list[i]])}",
            fontsize=10,
            color='black'
            )

    plt.subplots_adjust(top=0.92,
        bottom=0.08,
        left=0.10,
        right=0.95,
        hspace=0.25,
        wspace=0.35
    )
    plt.show()

