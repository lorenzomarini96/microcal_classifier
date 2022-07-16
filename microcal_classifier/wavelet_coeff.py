# pylint: disable=invalid-name, redefined-outer-name, import-error

"""This module computes the wavelet coefficients of images, by means of
multilevel 2D Discrete Wavelet Transform, in order to evaluate the performance of 
a machine learging model for the classification on a microcalcification image dataset."""

#import os
#import glob

from PIL import Image
import multiprocessing as mp

import skimage
from skimage import img_as_float
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import pywt


def dwtcoefftoarray(myim, wavelet, level, partial=False):
    ''' 
    This function collects all the coefficients of the DWT and converts them
    in a flat array which can be passed to the Deep Neural Network.
    
    Parameters
    ----------
    myim : type
        image opened with PIL.Image
    wavelet : type
        which wavelet to use
    level : type
        level of the wavelet decomposition
    denoise : type
        wheter to prior denoise the image or not, denoise should be set to "yes" or "no"
    partial : type
        True or False <-- let's you choose wheter to take only the second and third levels coeffiecients if level is 3, 
                                      or the second, third and fourth levels coefficients if level is 4

    Returns
    -------
    coeffsarray : type
        Description

    Examples
    --------

    '''

    coeffs = pywt.wavedec2(myim, wavelet, level=level)
    infocoeffs = pywt.ravel_coeffs(coeffs)
    # https://pywavelets.readthedocs.io/en/latest/ref/dwt-coefficient-handling.html
    # ravel_coeffs: Wavelet transform coefficient array. All coefficients have been concatenated into a single 1D array.

    if partial == False:
        ''' 
        If partial is False I want to take all the coefficients obtained from the wavedec2 decomposition 
        '''
        coeffsarray = infocoeffs[0]

    if partial == True:
        ''' 
        If partial is True AND level is 3 I want to take the coefficients of 2nd and 3rd levels without the 1st level ones
        and without those related to the approximated image.
        If partial is True AND level is 4 I want to take the coefficients of 2nd, 3rd and 4th levels without the 1st level ones
        and without those related to the approximated image.
        '''

        secondlevelcoeffs = np.concatenate(( infocoeffs[0][infocoeffs[1][-2]['da']], infocoeffs[0][infocoeffs[1][-2]['ad']], 
                                            infocoeffs[0][infocoeffs[1][-2]['dd']] ))
        thirdlevelcoeffs = np.concatenate(( infocoeffs[0][infocoeffs[1][-3]['da']], infocoeffs[0][infocoeffs[1][-3]['ad']], 
                                            infocoeffs[0][infocoeffs[1][-3]['dd']] ))
        if level == 3:    
            coeffsarray = np.concatenate((secondlevelcoeffs, thirdlevelcoeffs))
        
        elif level == 4:
            fourthlevelcoeffs = np.concatenate(( infocoeffs[0][infocoeffs[1][-4]['da']], infocoeffs[0][infocoeffs[1][-4]['ad']], 
                                                infocoeffs[0][infocoeffs[1][-4]['dd']] ))
            coeffsarray = np.concatenate(( secondlevelcoeffs, thirdlevelcoeffs, fourthlevelcoeffs ))
        else:
            pass
    
    return coeffsarray



def dwtanalysis(myim, wavelet, level):
    ''' 
    This function decomposes the original image with a Discrete Wavelet Transformation
    using the desired wavelet family up to the fifth level.
    One can choose to denoise the original image prior to the DWT decomposition.
    
    The coefficients matrices obtained from the DWT are then masked keeping only those values which are greater than
    the standard deviation calculated over the matrix values.
    The image is then reconstructed using the new coefficients.
    
    Parameters
    ----------
    myim : type
        image opened with PIL.Image
    wavelet : type
        which wavelet to use
    level : type
        level of the wavelet decomposition
    
    Returns
    -------
    myim : type
        Description
    mynewim : type
        Description

    Examples
    --------

    '''

    if level == 2:
        cA, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level)
    elif level == 3:
        cA, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level)
    elif level == 4:
        cA, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level)
    elif level == 5:
        cA, (cH5, cV5, cD5), (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(myim, wavelet, level=level)
    else:
        pass

    ''' 
    Now, I get the standard deviation for each matrix (image and coefficients).
    The std will act as a treshold so that if abs(value) < 0. --> value = 0.
                                            elif abs(value) > 0. --> value = value
    
    https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    - Thresholds the input data depending on the mode argument.
    
    - mode = 'hard': In hard thresholding, the data values where their absolute value is less than the value param 
      are replaced with substitute. Data values with absolute value 
      greater or equal to the thresholding value stay untouched.

    '''
    
    ncA = np.zeros_like(cA) # I won't need the approximated image anymore, I only need to modify the n'th level coefficient.

    std10 = np.std(cH1)
    std11 = np.std(cV1)
    std12 = np.std(cD1)

    ncH1 = pywt.threshold(cH1, std10, mode = 'hard', substitute = 0.)
    ncV1 = pywt.threshold(cV1, std11, mode = 'hard', substitute = 0.)
    ncD1 = pywt.threshold(cD1, std12, mode = 'hard', substitute = 0.)

    std20 = np.std(cH2)
    std21 = np.std(cV2)
    std22 = np.std(cD2)

    ncH2 = pywt.threshold(cH2, std20, mode = 'hard', substitute = 0.)
    ncV2 = pywt.threshold(cV2, std21, mode = 'hard', substitute = 0.)
    ncD2 = pywt.threshold(cD2, std22, mode = 'hard', substitute = 0.)

    if level == 3 or level == 4:
        std30 = np.std(cH3)
        std31 = np.std(cV3)
        std32 = np.std(cD3)

        ncH3 = pywt.threshold(cH3, std30, mode = 'hard', substitute = 0.)
        ncV3 = pywt.threshold(cV3, std31, mode = 'hard', substitute = 0.)
        ncD3 = pywt.threshold(cD3, std32, mode = 'hard', substitute = 0.)
    else:
        pass

    if level == 4 or level == 5:
        std40 = np.std(cH4)
        std41 = np.std(cV4)
        std42 = np.std(cD4)

        ncH4 = pywt.threshold(cH4, std40, mode = 'hard', substitute = 0.)
        ncV4 = pywt.threshold(cV4, std41, mode = 'hard', substitute = 0.)
        ncD4 = pywt.threshold(cD4, std42, mode = 'hard', substitute = 0.)
    else:
        pass

    if level == 5:
        std50 = np.std(cH5)
        std51 = np.std(cV5)
        std52 = np.std(cD5)

        ncH5 = pywt.threshold(cH5, std50, mode = 'hard', substitute = 0.)
        ncV5 = pywt.threshold(cV5, std51, mode = 'hard', substitute = 0.)
        ncD5 = pywt.threshold(cD5, std52, mode = 'hard', substitute = 0.)
    else:
        pass

    '''
    To let things be more readable I define new_coeff,
    this is just so that waverec2 (the function needed to reconstruct
    the image from a set of given coefficient) can do what it does.
    '''
    if level == 2:
        new_coeff = ncA, (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 
    elif level == 3:
        new_coeff = ncA, (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 
    elif level == 4:
        new_coeff = ncA, (ncH4, ncV4, ncD4), (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1)     
    elif level == 5:
        new_coeff = ncA, (ncH5, ncV5, ncD5), (ncH4, ncV4, ncD4), (ncH3, ncV3, ncD3), (ncH2, ncV2, ncD2), (ncH1, ncV1, ncD1) 

    ''' Here the image is reconstructed using the new coefficients.
    '''
    
    # 2D multilevel reconstruction using waverec2
    mynewim = pywt.waverec2(new_coeff, wavelet) # Multilevel 2D Inverse Discrete Wavelet Transform.
    # Returns:	2D array of reconstructed data.


    mynewim = pywt.threshold(mynewim, 0., mode = 'greater', substitute = 0.)
    # In greater thresholding, the data is replaced with substitute where data is below the thresholding value.
    # Greater data values pass untouched.

    return myim, mynewim
