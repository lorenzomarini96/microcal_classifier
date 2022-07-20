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
    """
    CNN for microcalcification clusters classification.

    Parameters
    ----------
    shape : tuple, optional
        The first parameter.
    verbose : bool, optional
        Enables the printing of the summary. Defaults to False.

    Returns
    -------
    model
        Return the convolutional neural network.

    Examples
    --------
    >>> H_PIXEL_SIZE = 60
    >>> V_PIXEL_SIZE = 60
    >>> CHANNEL = 1
    >>> model = cnn_classifier(shape=(H_PIXEL_SIZE, V_PIXEL_SIZE, CHANNEL), verbose=True)
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
    conv_1 (Conv2D)             (None, 60, 60, 32)        320
    maxpool_1 (MaxPooling2D)    (None, 30, 30, 32)        0                                                  
    conv_2 (Conv2D)             (None, 30, 30, 64)        18496                                         
    maxpool_2 (MaxPooling2D)    (None, 15, 15, 64)        0
    dropout_30 (Dropout)        (None, 15, 15, 64)        0
    conv_3 (Conv2D)             (None, 15, 15, 128)       73856
    maxpool_3 (MaxPooling2D)    (None, 7, 7, 128)         0
    dropout_31 (Dropout)        (None, 7, 7, 128)         0
    conv_4 (Conv2D)             (None, 7, 7, 128)         147584
    maxpool_4 (MaxPooling2D)    (None, 3, 3, 128)         0
    flatten_10 (Flatten)        (None, 1152)              0
    dropout_32 (Dropout)        (None, 1152)              0
    dense_2 (Dense)             (None, 256)               295168
    dense_3 (Dense)             (None, 128)               32896
    output (Dense)              (None, 1)                 129
    =================================================================
    Total params: 568,449
    Trainable params: 568,449
    Non-trainable params: 0
    _________________________________________________________________
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', input_shape=shape))
    model.add(MaxPooling2D((2, 2), name='maxpool_1'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
    model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
    model.add(MaxPooling2D((2, 2), name='maxpool_3'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
    model.add(MaxPooling2D((2, 2), name='maxpool_4'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu', name='dense_2'))
    model.add(Dense(128, activation='relu', name='dense_3'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    if verbose:
      model.summary()
  
    return model
