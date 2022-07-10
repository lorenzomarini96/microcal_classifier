"""Test documentazione modulo che utilizza numpy."""

import numpy as np
import matplotlib.pyplot as plt

def somma(a, b):
    """
    Somma due vettori a e b.

    Parameters
    ----------
    a : array numpy
        The first array
    b : array numpy
        The second array

    Returns
    -------
    sum
        Return the sum of two vectors.
    
    Examples
    --------
    Esempimo di somma tra due vettori.
    
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([1, 2, 3])
    >>> sum = somma(a, b)
    [2, 4, 6]
    
    """

    sum = a + b

    return sum


def grafico(x, y):
    """
    Plotta x in funzione di y

    Parameters
    ----------
    x : array numpy
        The first array
    y : array numpy
        The second array
    
    """

    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    a = np.array([1,2,3])
    b = np.array([1,2,3])

    sum = somma(a, b)
    print(sum)

    grafico(a, b)