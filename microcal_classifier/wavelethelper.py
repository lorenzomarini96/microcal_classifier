"""Module prodiving some useful function for plot and visualization of wavelet transformation."""

import matplotlib.pyplot as plt
import pywt

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
        Figure

    Examples
    --------

    >>> plot_dcw(verbose=True, show=True)
    >>> ['Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal',
    >>> 'Reverse biorthogonal', 'Discrete Meyer (FIR Approximation)',
    >>> 'Gaussian', 'Mexican hat wavelet', 'Morlet wavelet',
    >>> 'Complex Gaussian wavelets', 'Shannon wavelets',
    >>> 'Frequency B-Spline wavelets', 'Complex Morlet wavelets']
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
    """Plot discrete and continuous wavelet families.

    Parameters
    ----------
    show : bool
        It True, shows the plot

    Returns
    -------
    fig : Figure
        Figure

    Examples
    --------

    >>> plot_dcw(verbose=True, show=True)

    """

    db_wavelets = pywt.wavelist('db')[:5]

    fig, axarr = plt.subplots(ncols=5, nrows=5, figsize=(16,16))
    fig.suptitle('Daubechies family of wavelets', fontsize=16)
    for col_no, waveletname in enumerate(db_wavelets):
        wavelet = pywt.Wavelet(waveletname)
        no_moments = wavelet.vanishing_moments_psi
        for row_no, level in enumerate(range(1,6)):
            wavelet_function, _, x_values = wavelet.wavefun(level = level)
            axarr[row_no, col_no].set_title(f'{waveletname} - level {level}'
            f'\n{no_moments} vanishing moments\n'
            f'{len(x_values)} samples',
             loc='left')
            axarr[row_no, col_no].plot(x_values, wavelet_function, 'bD--')
            axarr[row_no, col_no].set_yticks([])
            axarr[row_no, col_no].set_yticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if show:
        plt.show()

    return fig

if __name__=='__main__':
    plot_dcw(verbose=True, show=True)
