"""Convolutional neural network model for MCs classification."""

import logging

# Layers needed in a CNN
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("CNN_model.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def cnn_classifier(shape=(60, 60, 1), verbose=False):
    # pylint: disable=W0613
    """CNN for microcalcification clusters classification.

    Args:
        shape (tuple, optional): Shape of the input image. Defaults to (60, 60, 1).
        verbose (bool, optional): Enables the printing of the summary. Defaults to False.
    """

    model = Sequential()

    model.add(Input(shape=(60, 60, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1'))

    model.add(MaxPooling2D((2, 2), name='maxpool_1'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
    model.add(MaxPooling2D((2, 2), name='maxpool_2'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
    model.add(MaxPooling2D((2, 2), name='maxpool_3'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
    model.add(MaxPooling2D((2, 2), name='maxpool_4'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu', name='dense_1'))
    model.add(Dense(128, activation='relu', name='dense_2'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    if verbose:
        model.summary()

    return model
