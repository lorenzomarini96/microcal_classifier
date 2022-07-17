# pylint: disable=invalid-name, redefined-outer-name, import-error
# pylint: disable=C0301
# pylint: disable=E1101

"""Module prodiving some useful function for plot and visualization
of multilevel 2D Discrete Wavelet Transform."""

import os
import glob
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import interp
from PIL import Image
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc


def plot_dcw(verbose=True, show=True):
    # pylint: disable=R0914
    """Plot discrete and continuous wavelet families.

    Parameters
    ----------
    verbose : bool
        If True, print a list of available built-in wavelet families.
    show : bool
        It True, shows the plot

    Returns
    -------
    fig : Figure
        Figure showing the discrete and continuous wavelet families.

    Examples
    --------
    >>> plot_dcw(verbose=True, show=True)
    ['Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal',
    'Reverse biorthogonal', 'Discrete Meyer (FIR Approximation)',
    'Gaussian', 'Mexican hat wavelet', 'Morlet wavelet',
    'Complex Gaussian wavelets', 'Shannon wavelets',
    'Frequency B-Spline wavelets', 'Complex Morlet wavelets']
    """

    discrete_wavelets = ['db5', 'sym5', 'coif5', 'bior2.4']
    continuous_wavelets = ['mexh', 'morl', 'cgau5', 'gaus5']

    list_list_wavelets = [discrete_wavelets, continuous_wavelets]
    list_funcs = [pywt.Wavelet, pywt.ContinuousWavelet]

    fig, axarr = plt.subplots(nrows=2, ncols=4, figsize=(10,5))
    for i, list_wavelets in enumerate(list_list_wavelets):
        func = list_funcs[i]
        row_no = i
        for col_no, waveletname in enumerate(list_wavelets):
            wavelet = func(waveletname)
            family_name = wavelet.family_name
            if i == 0:
                _ = wavelet.wavefun()
                wavelet_function = _[0]
                x_values = _[-1]
            else:
                wavelet_function, x_values = wavelet.wavefun()
            if col_no == 0 and i == 0:
                axarr[row_no, col_no].set_ylabel("Discrete Wavelets", fontsize=12)
            if col_no == 0 and i == 1:
                axarr[row_no, col_no].set_ylabel("Continuous Wavelets", fontsize=12)
            axarr[row_no, col_no].set_title(f"{family_name}", fontsize=12)
            axarr[row_no, col_no].plot(x_values, wavelet_function)
            axarr[row_no, col_no].set_yticks([])
            axarr[row_no, col_no].set_yticklabels([])

    if show:
        plt.tight_layout()
        plt.show()

    if verbose:
        print(pywt.families(short=False))

    return fig


def plot_daubechies(show=True):
    # pylint: disable=E1101
    """Plot the first 5 levels of Daubechies wavelet families.

    Parameters
    ----------
    show : bool
        It True, shows the plot

    Returns
    -------
    fig : Figure
        Figure showing the first 5 levels of Daubechies wavelet families.

    Examples
    --------
    >>> plot_dcw(verbose=True, show=True)
    """

    db_wavelets = pywt.wavelist('db')[:5]

    fig, axarr = plt.subplots(ncols=5, nrows=5, figsize=(8,8))
    fig.suptitle('Daubechies family of wavelets', fontsize=10)
    for col_no, waveletname in enumerate(db_wavelets):
        wavelet = pywt.Wavelet(waveletname)
        no_moments = wavelet.vanishing_moments_psi
        for row_no, level in enumerate(range(1,6)):
            wavelet_function, _, x_values = wavelet.wavefun(level = level)
            axarr[row_no, col_no].set_title(f'{waveletname} - level {level}'
            f'\n{no_moments} vanishing moments\n'
            f'{len(x_values)} samples',
             loc='left')
            axarr[row_no, col_no].plot(x_values, wavelet_function, '-')
            axarr[row_no, col_no].set_yticks([])
            axarr[row_no, col_no].set_yticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if show:
        plt.show()

    return fig

def plot_cv_roc(X, y, classifier, n_splits=5):
    """
    Train the classifier on X data with y labels and implement the
    k-fold-CV with k=n_splits. Plot the ROC curves for each k fold
    and their average and display the corresponding AUC values
    and the standard deviation over the k folders.

    Parameters
    ----------
    X : numpy array
        Numpy array of read image.
    y : numpy array
        Numpy array of labels.
    classifier : keras model
        Name of wavelet families.
    n_splits : int
        Number of folders for K-cross validation.

    Returns
    -------
    fig : Figure
        Plot of Receiver operating characteristic (ROC) curve.

    Examples
    --------
    >>> classifier = RandomForestClassifier()
    >>> N_FOLDS = 5
    >>> plot_cv_roc(X, y, classifier, n_splits=N_FOLDS):
    """

    try:
        y = y.to_numpy()
        X = X.to_numpy()
    except AttributeError:
        pass

    cv = StratifiedKFold(n_splits)

    tprs = [] #True positive rate
    aucs = [] #Area under the ROC Curve
    interp_fpr = np.linspace(0, 1, 100)
    plt.figure()
    i = 0
    for train, test in cv.split(X, y):

        model = classifier
        prediction = model.fit(X[train], y[train])

        y_test_pred = model.predict(X[test])

        # Compute ROC curve and area under the curve
        fpr, tpr, _ = roc_curve(y[test], y_test_pred)
        interp_tpr = interp(interp_fpr, fpr, tpr)
        tprs.append(interp_tpr)

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc:.2f})')
        i += 1

    plt.legend()
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.show()

    plt.figure()
    plt.plot([0, 1], [0, 1],
            linestyle='--',
            lw=2,
            color='r',
            label='Chance',
            alpha=.8
    )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(interp_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(interp_fpr, mean_tpr,
                        color='b',
                        label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
                        lw=2,
                        alpha=.8
                        )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(interp_fpr,
                     tprs_lower,
                     tprs_upper,
                     color='grey',
                     alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('Cross-Validation ROC of RandomForestClassifier',fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()


def wavelet_info(family):
    """Print a brief information about the wavelet family
    and some properties like orthogonality and symmetry.

    Parameters
    ----------
    family : str
        Name of wavelet families.

    Returns
    -------
    None

    Examples
    --------
    >>> wavelet_info('db5')
    Wavelet db5
    Family name:    Daubechies
    Short name:     db
    Filters length: 10
    Orthogonal:     True
    Biorthogonal:   True
    Symmetry:       asymmetric
    DWT:            True
    CWT:            False
    """

    # Create a Wavelet object
    wave = pywt.Wavelet(family)
    print(wave)


def show_img_coeff(img__path, family, level, verbose=True):
    # pylint: disable=C0103
    """Show the approximation, horizontal detail, vertical detail,
    diagonal detail coefficient of a given level obtained applying
    the multilevel 2D Discrete Wavelet Transform of a given family on the input image.

    Parameters
    ----------
    img__path : str
       Path to the input image

    family : str
        Name of wavelet families.

    level : int
        Level of decomposition.

    Returns
    -------
    fig : Figure

    Examples
    --------
    >>> IMG_PATH = "/path/to/image/img.pgm"
    >>> FAMILY = "db5"
    >>> LEVEL = 4
    >>> show_img_coeff(img__path=IMG_PATH,
    >>>                         family=FAMILY,
    >>>                         level=LEVEL,
    >>>                         verbose=True)
    cA : (12, 12)
    cH : (12, 12)
    cV : (12, 12)
    cD : (12, 12)
    """

    image = mpimg.imread(img__path)

    image_array = Image.fromarray(image , 'L')
    resize_img = image_array.resize((60 , 60))

    # Multilevel 2D Discrete Wavelet Transform
    # (https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html)
    if level==1:
        cA, (cH, cV, cD) = pywt.wavedec2(resize_img, family, level=1)

    if level==2:
        #cA, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(resize_img, family, level=2)
        cA, (cH, cV, cD), (_, _, _)  = pywt.wavedec2(resize_img, family, level=2)

    if level==3:
        #cA, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(resize_img, family, level=3)
        cA, (cH, cV, cD), (_, _, _), (_, _, _) = pywt.wavedec2(resize_img, family, level=3)

    if level==4:
        #cA, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(resize_img, family, level=4)
        cA, (cH, cV, cD), (_, _, _), (_, _, _), (_, _, _) = pywt.wavedec2(resize_img, family, level=4)

    fig = plt.figure(figsize=(8, 8))
    # Wavelet transform of image, and plot approximation and details
    titles = [f'Approximation Coef. of Level {level}',
              f'Horizontal Detail Coef. of Level {level}',
              f'Vertical Detail Coef. of Level {level}',
              f'Diagonal Detail Coef. of Level {level}'
              ]
    for i, a in enumerate([cA, cH, cV, cD]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

    if verbose:
        print(f'cA : {cA.shape}\n'
              f'cH : {cH.shape}\n'
              f'cV : {cV.shape}\n'
              f'cD : {cD.shape}\n'
              )

    return fig


def plot_multi_dec(img__path, family, level):
    # pylint: disable=C0103
    # pylint: disable=E1101
    """Show the effect of multi level decomposition on a given image.

    Parameters
    ----------
    img__path : str
       Path to the input image

    family : str
        Name of wavelet families.

    level : int
        Level of decomposition.

    Returns
    -------
    fig : Figure

    Examples
    --------
    >>> IMG_PATH = "/path/to/image/img.pgm"
    >>> FAMILY = "db5"
    >>> LEVEL = 4
    >>> plot_multi_dec(img__path=IMG_PATH,
    >>>                         family=FAMILY,
    >>>                         level=LEVEL)
    """
    # Read input image
    img = mpimg.imread(img__path)
    shape = img.shape

    max_lev = 4 # how many levels of decomposition to draw
    label_levels = 4  # how many levels to explicitly label on the plots

    fig, axes = plt.subplots(2, 5, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(img, cmap=plt.cm.gray)
            axes[1, 0].set_title('Image', fontsize=14)
            axes[1, 0].set_axis_off()
            continue

        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape,
                        wavedec2_keys(level),
                        ax=axes[0, level],
                        label_levels=label_levels
                        )
        axes[0, level].set_title(f'{level} level\ndecomposition', fontsize=14)

        # compute the 2D DWT
        c = pywt.wavedec2(img, family, mode='periodization', level=level)
        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, _ = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title(f'Coefficients\n{level} level', fontsize=14)
        axes[1, level].set_axis_off()

    plt.tight_layout()
    plt.show()

    return fig


def read_img(image_path):
    '''Takes as input the path to the image folder and
    returns the numpy array of image and label found in that folder.

    Parameters
    ----------
    image_path : str
       Path to the input image.

    Returns
    -------
    images : numpy array
        Numpy array of read image.

    y_np : numpy array
        Numpy array of read label.

    Examples
    --------
    >>> IMG_PATH = "/path/to/image/"
    >>> read_img(image_path=IMG_PATH)
    '''

    # Creating a list of all image names found in image_path
    imagefilename = glob.glob(os.path.join(image_path, '*.pgm'))

    # Defining 4 sub-processes and apply Immage.open to all the images found
    pool = mp.Pool(processes=4)
    results = pool.map_async(Image.open, imagefilename)

    # Gets the list of images
    images = results.get()

    #logger.info(f'Num images found in {image_path}: {len(images)}')

    # Creates the list of corrisponding labels and convert it to numpy array
    label = os.path.basename(image_path)
    y = [int(label)] * len(images)
    y_np = np.array(y)

    return images, y_np


if __name__=='__main__':

    plot_dcw(verbose=True, show=True)
    plot_daubechies(show=True)

    IMG_PATH = '/home/lorenzomarini/Desktop/DATASETS_wavelets/IMAGES/Mammography_micro/Train/1/0006s1_1_1_1.pgm_1.pgm'
    FAMILY = "db5"
    LEVEL = 4
    wavelet_info(family=FAMILY)
    show_img_coeff(img__path=IMG_PATH, family=FAMILY, level=LEVEL, verbose=True)
    plot_multi_dec(img__path=IMG_PATH, family=FAMILY, level=LEVEL)
