if False:
    import numpy as np


def min_max_norm(x, **kwargs):
    """
    Scale input x to range between [x_min,..,x_max].

    :param np.ndarray x:    input to be scaled.
    :return:                scaled version of the input
    :rtype:                 np.ndarray
    """
    x_min = kwargs.get('x_min', 0)
    x_max = kwargs.get('x_max', 1)
    return (x - x.min()) / (x.max() - x.min()) * (x_max - x_min) + x_min


def standartise(x, **kwargs):
    """
    Normalise x to have mean 0 and a standard deviation of 1.

    :param np.ndarray x:    input to be normalised.
    :return:                standartised version of the input
    :rtype:                 np.ndarray
    """

    return (x - x.mean()) / x.std()