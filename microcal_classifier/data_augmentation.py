"""Data augmentation procedure."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("Data_aumentation.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def convert_to_png(fname, dest_folder):
    """
    Convert the extension of the given image from .pgm to .png.
    The new image is converted in gray scale and saved in the dest_folder.

    Parameters
    ----------
    fname : str
        Path to the input image with extension .pgm.
    dest_folder : str
        Destination folder in which save the new images with png extension.

    Returns
    -------
    None

    Examples
    --------
    >>> TRAIN_PATH = 'path/to/train/folder/'
    >>> TEST_PATH = 'path/to/train/folder/'
    >>> for data_path in [TRAIN_PATH, TEST_PATH]: 
    >>>     for path, folders, fnames in os.walk(data_path):
    >>>         for fname in fnames:
    >>>             abs_path = os.path.join(path, fname)
    >>>             dest_folder = path.replace('Train', 'Train_png').replace('Test', 'Test_png')
    >>>             convert_to_png(abs_path, dest_folder)

    """
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    dest_fname = os.path.basename(fname).replace('.pgm', '.png')
    dest_fname = os.path.join(dest_folder, dest_fname)
    PIL.Image.open(fname).convert('L').save(dest_fname) # L (8-bit pixels, black and white)


def data_aug(train_dataset_path, img_width=60, img_height=60, batch_size=32):
    """
    Data augmentation procedure.

    Parameters
    ----------
    train_dataset_path : str
        Path to the input train data set.
    img_width : int
        X-dimension of the image
    img_height : int
        Y-dimension of the image
    batch_size : int
        Batch size

    Returns
    -------
    train_gen : ???
        Generated train data set.
    val_gen : ???
        Generated validation set.

    Examples
    --------
    >>> TRAIN_PATH = 'path/to/train/folder/'
    >>> IMG_WIDTH = 60
    >>> IMG_HEIGHT = 60
    >>> BATCH_SIZE = 32
    >>> data_aug(train_dataset_path=TRAIN_PATH,
                 img_width=IMG_WIDTH,
                 img_height=IMG_HEIGHT,
                 batch_size=BATCH_SIZE
                 )
 
    """

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect',
        validation_split=0.3)
        
    train_gen = train_datagen.flow_from_directory(
        train_dataset_path,
        target_size=(img_width, img_height),
        color_mode='grayscale', 
        class_mode='binary',
        subset='training')

    val_gen = train_datagen.flow_from_directory(
        train_dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        subset='validation')


def single_image_aug(image_path, show=True):
    """
    Show the effect of the data aumentation procedure --based on ImageDataGenerator-- 
    showing a given single image.

    Parameters
    ----------
    image_path : str
        Path to the input train data set.
    show : bool
        If True, shows the tranformed imags.

    Returns
    -------
    None
    
    Examples
    --------
    
    >>> IMAGE_PATH = 'path/to/input/image/'
    >>> single_image_aug(image_path, show=True)
    
    """

    img = keras.preprocessing.image.load_img(image_path, target_size=(60, 60, 1))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1./255,  # è già scalato tra 0 e 1. Se lo metto, vedo tutto nero.
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect',
        validation_split=0.3)

    #Creates our batch of one image
    pic = datagen.flow(img, batch_size=1)

    if show:
        plt.figure(figsize=(15,15))
        for i in range(1,9):
            plt.subplot(2, 4, i)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            batch = pic.next()
            image_ = batch[0].astype('uint8')
            plt.imshow(image_, cmap='gray')
            plt.subplots_adjust(left=0.05,
            bottom=0.001, 
            right=0.9, 
            top=0.4, 
            wspace=0.1, 
            hspace=0.1)

        plt.show()
