"""Test documentazione modulo che utilizza numpy."""

import numpy as np

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

if __name__ == '__main__':
    a = np.array([1,2,3])
    b = np.array([1,2,3])

    sum = somma(a, b)
    print(sum)