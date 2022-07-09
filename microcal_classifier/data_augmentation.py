"""Data augmentation procedure."""

import logging
import os

import matplotlib.pyplot as plt
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
    """Convert the extension of the given image from .pgm to .png.
    The new image is converted in gray scale and saved in the dest_folder.

    Args:
        fname (str): Path to the input image with extension .pgm
        dest_folder (directory): Destination folder in which save the new images with png extension.
    """
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    dest_fname = os.path.basename(fname).replace('.pgm', '.png')
    dest_fname = os.path.join(dest_folder, dest_fname)
    PIL.Image.open(fname).convert('L').save(dest_fname) # L (8-bit pixels, black and white)


def data_aug(train_dataset_path, batch_size=32):
    """Data augmentation.
    """
    #train_dataset_path = '/content/gdrive/MyDrive/DATASETS_experim/IMAGES/Mammography_micro/Train_png'
    batch_size = 32
    img_width, img_height = (60, 60)

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
    """Function to show the effect of the data aumentation procedure --based on ImageDataGenerator-- 
    on the singole image given in input.

    Args:
        image_path (str): Path to the input image.
        show (bool, optional): Allow to show the multi plots. Defaults to True.
    """

    #image_path = '/content/gdrive/MyDrive/DATASETS_experim/IMAGES/Mammography_micro/Train_png/1/0042t1_1_1_1.png_2.png'

    img = keras.preprocessing.image.load_img(image_path, target_size=(60, 60, 1))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Uses ImageDataGenerator to flip the images
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