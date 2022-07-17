# pylint: disable=invalid-name, redefined-outer-name, import-error
# pylint: disable=R0912
# pylint: disable=R0914

"""This module computes the wavelet coefficients of images, by means of
multilevel 2D Discrete Wavelet Transform, in order to evaluate the performance of
a machine learging model for the classification on a microcalcification image dataset."""

#import os
#import glob
#import multiprocessing as mp

#from PIL import Image
#import skimage
#from skimage import img_as_float
#from skimage.io import imread
#import matplotlib.pyplot as plt
import numpy as np
import pywt


def dwtcoefftoarray(image, wavelet, level, partial=False):
    '''
    This function collects all the coefficients of the 2DWT and converts them
    in a flat numpy array which can be passed to the binary classifier.

    Parameters
    ----------
    image :  Image object
        Image opened with PIL.Image.
    wavelet : str
        Wavelet family to use.
    level : int
        Level of the wavelet decomposition.
    partial : bool
        If True AND level = 3, take only the 2nd and 3rd levels coeffiecients
        without the 1st level ones and without those related to the approximated image;
        if True AND level = 4, take only the 2nd and 3rd and 4th levels coefficients
        levels without the 1st level ones and without those related to the approximated image;
        if False take all the coefficients obtained from the 2DWT decomposition.

    Returns
    -------
    wavecoeffs : numpy array
        Flat numpy array containing all the coefficients of the 2DWT of the given image

    Examples
    --------
    >>> IMAGE_PATH = '/path/to/the/image/'
    >>> FAMILY = 'db5'
    >>> LEVEL = 4
    >>> IMAGE = Image.open(IMAGE_PATH)
    >>> wavecoeffs = dwtcoefftoarray(image=IMAGE, wavelet=FAMILY, level=LEVEL, partial=False)
    '''

    coeffs = pywt.wavedec2(image, wavelet, level=level)
    infocoeffs = pywt.ravel_coeffs(coeffs)
    # https://pywavelets.readthedocs.io/en/latest/ref/dwt-coefficient-handling.html
    # ravel_coeffs: Wavelet transform coefficient array.
    # All coefficients have been concatenated into a single 1D array.

    if partial == False:
        wavecoeffs = infocoeffs[0]

    if partial == True:
        secondlevelcoeffs = np.concatenate((infocoeffs[0][infocoeffs[1][-2]['da']],
                                            infocoeffs[0][infocoeffs[1][-2]['ad']],
                                            infocoeffs[0][infocoeffs[1][-2]['dd']]))
        thirdlevelcoeffs = np.concatenate((infocoeffs[0][infocoeffs[1][-3]['da']],
                                           infocoeffs[0][infocoeffs[1][-3]['ad']],
                                           infocoeffs[0][infocoeffs[1][-3]['dd']]))

        if level == 3:

            wavecoeffs = np.concatenate((secondlevelcoeffs, thirdlevelcoeffs))

        elif level == 4:
            fourthlevelcoeffs = np.concatenate((infocoeffs[0][infocoeffs[1][-4]['da']],
                                                infocoeffs[0][infocoeffs[1][-4]['ad']],
                                                infocoeffs[0][infocoeffs[1][-4]['dd']]))
            wavecoeffs = np.concatenate((secondlevelcoeffs,
                                         thirdlevelcoeffs,
                                         fourthlevelcoeffs))

        else:
            pass

    return wavecoeffs


def dwt_analysis(image, wavelet, level):
    # pylint: disable=C0301
    '''
    Decompose the original image by means of the multilevel 2D discrete wavelet transform
    with the selected wavelet family up to the fifth level.
    The obtained coefficients matrices are then masked keeping only those values which
    are greater than the standard deviation calculated over the matrix values.
    Finally, the image is reconstructed using the new coefficients.

    Parameters
    ----------
    image :  Image object
        Image opened with PIL.Image.
    wavelet : str
        Wavelet family to use.
    level : int
        Level of the wavelet decomposition.

    Returns
    -------
    newimage : array like
        2D array of reconstructed image using multilevel 2D inverse discrete wavelet transform.

    Examples
    --------
    >>> IMAGE_PATH = '/path/to/the/image/'
    >>> FAMILY = 'db5'
    >>> LEVEL = 4
    >>> IMAGE = Image.open(IMAGE_PATH)
    >>> newimage = dwt_analysis(image=IMAGE, wavelet=FAMILY, level=LEVEL)
    '''

    if level == 2:
        cA, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(image,
                                                            wavelet,
                                                            level=level)
    elif level == 3:
        cA, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(image,
                                                                              wavelet,
                                                                              level=level)
    elif level == 4:
        cA, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(image,
                                                                                               wavelet,
                                                                                               level=level)
    elif level == 5:
        cA, (cH5, cV5, cD5), (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(image,
                                                                                                                wavelet,
                                                                                                                level=level)
    else:
        pass

    '''
    Compute the standard deviation for each matrix (image and coefficients).
    The std will act as a treshold: if abs(value) < 0. => value = 0, otherwise: value = value
    
    https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    Thresholds the input data depending on the mode argument.
    
    mode = 'hard': In hard thresholding, the data values where their absolute value
    is less than the value param are replaced with substitute.
    Data values with absolute value greater or equal to the thresholding value stay untouched.
    '''

    ncA = np.zeros_like(cA) # No approximate image anymore

    std10 = np.std(cH1)
    std11 = np.std(cV1)
    std12 = np.std(cD1)

    ncH1 = pywt.threshold(cH1, std10, mode='hard', substitute=0.)
    ncV1 = pywt.threshold(cV1, std11, mode='hard', substitute=0.)
    ncD1 = pywt.threshold(cD1, std12, mode='hard', substitute=0.)

    std20 = np.std(cH2)
    std21 = np.std(cV2)
    std22 = np.std(cD2)

    ncH2 = pywt.threshold(cH2, std20, mode='hard', substitute=0.)
    ncV2 = pywt.threshold(cV2, std21, mode='hard', substitute=0.)
    ncD2 = pywt.threshold(cD2, std22, mode='hard', substitute=0.)

    if level == 3 or level == 4:
        std30 = np.std(cH3)
        std31 = np.std(cV3)
        std32 = np.std(cD3)

        ncH3 = pywt.threshold(cH3, std30, mode='hard', substitute=0.)
        ncV3 = pywt.threshold(cV3, std31, mode='hard', substitute=0.)
        ncD3 = pywt.threshold(cD3, std32, mode='hard', substitute=0.)
    else:
        pass

    if level == 4 or level == 5:
        std40 = np.std(cH4)
        std41 = np.std(cV4)
        std42 = np.std(cD4)

        ncH4 = pywt.threshold(cH4, std40, mode='hard', substitute=0.)
        ncV4 = pywt.threshold(cV4, std41, mode='hard', substitute=0.)
        ncD4 = pywt.threshold(cD4, std42, mode='hard', substitute=0.)
    else:
        pass

    if level == 5:
        std50 = np.std(cH5)
        std51 = np.std(cV5)
        std52 = np.std(cD5)

        ncH5 = pywt.threshold(cH5, std50, mode='hard', substitute=0.)
        ncV5 = pywt.threshold(cV5, std51, mode='hard', substitute=0.)
        ncD5 = pywt.threshold(cD5, std52, mode='hard', substitute=0.)
    else:
        pass

    # Define new coefficient according to the requests of waverec2 method.
    if level == 2:
        new_coeff = ncA, (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1)
    elif level == 3:
        new_coeff = ncA, (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1)
    elif level == 4:
        new_coeff = ncA, (ncH4, ncV4, ncD4), (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1)
    elif level == 5:
        new_coeff = ncA, (ncH5, ncV5, ncD5), (ncH4, ncV4, ncD4), (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1)

    '''
    2D multilevel image reconstruction using the new coefficients and waverec2:
    The waverec2 function reconstructs the image from a set of given coefficient
    returning the 2D array of reconstructed data.
    '''

    # Multilevel 2D Inverse Discrete Wavelet Transform.
    newimage = pywt.waverec2(new_coeff, wavelet)
    newimage = pywt.threshold(newimage, 0., mode = 'greater', substitute = 0.)
    '''
    In greater thresholding, the data is replaced with substitute ( = 0.)
    where data is below the thresholding value.
    Greater data values pass untouched.
    '''
    return newimage
