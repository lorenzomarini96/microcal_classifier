"""Convolutional neural network model for MCs classification."""

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