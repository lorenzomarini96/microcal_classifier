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
    shape : (tuple, optional)
        The first parameter.
    verbose : (bool, optional)
        Enables the printing of the summary. Defaults to False.

    Returns
    -------
    model
        Return the convolutional neural network.

    Examples
    --------
    Implement a CNN architecture with hidden layers.
    
    >>> model = cnn_classifier(shape=(60,60), verbose=False)
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv_1 (Conv2D)             (None, 60, 60, 32)        320       
                                                                    
    maxpool_1 (MaxPooling2D)    (None, 30, 30, 32)        0         
                                                                    
    conv_2 (Conv2D)             (None, 30, 30, 64)        18496     
                                                                    
    maxpool_2 (MaxPooling2D)    (None, 15, 15, 64)        0         
                                                                    
    conv_3 (Conv2D)             (None, 15, 15, 128)       73856     
                                                                    
    maxpool_3 (MaxPooling2D)    (None, 7, 7, 128)         0         
                                                                    
    conv_4 (Conv2D)             (None, 7, 7, 128)         147584    
                                                                    
    maxpool_4 (MaxPooling2D)    (None, 3, 3, 128)         0         
                                                                    
    flatten (Flatten)           (None, 1152)              0         
                                                                    
    dropout (Dropout)           (None, 1152)              0         
                                                                    
    dense_1 (Dense)             (None, 512)               590336    
                                                                    
    dense_2 (Dense)             (None, 128)               65664     
                                                                    
    output (Dense)              (None, 1)                 129       
                                                                    
    =================================================================
    Total params: 896,385
    Trainable params: 896,385
    Non-trainable params: 0

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
