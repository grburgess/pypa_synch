import numba as nb
import numpy as np
from numpy.typing import ArrayLike


@nb.njit(fastmath=True)
def pa_distribution(
    gamma: ArrayLike,
    gamma_b: float,
    gamma_c: float,
    gamma_inj: float,
    gamma_max: float,
    p: float,
):
    """
    double broken power law describing the steady state electron distribution
    where IC cooling dominates below the synchrotron cooling energy


    :param gamma:
    :type gamma: ArrayLike
    :param gamma_b:
    :type gamma_b: float
    :param gamma_c:
    :type gamma_c: float
    :param gamma_inj:
    :type gamma_inj: float
    :param gamma_max:
    :type gamma_max: float
    :param p:
    :type p: float
    :returns:

    """

    # normalizations to make
    # distribution continuous

    A = np.power(gamma_inj, 1 - p)
    B = A * np.power(gamma_c, -1)

    # output
    out = np.zeros(gamma.shape[0])

    # gather the various regions
    idx1 = gamma < gamma_c
    idx2 = (gamma >= gamma_c) & (gamma < gamma_inj)
    idx3 = (gamma >= gamma_inj) & (gamma < gamma_max)

    # compute the regions

    out[idx1] = B * np.power(gamma[idx1], -1)

    out[idx2] = A * np.power(gamma[idx2], -2)

    out[idx3] = np.power(gamma[idx3], -p - 1)

    return out


@nb.njit(fastmath=True)
def fast_cooling_distribution(
    gamma: ArrayLike, gamma_c: float, gamma_inj: float, gamma_max: float, p: float
):
    """

    steady state fast-cooling electron distribution

    :param gamma:
    :type gamma: ArrayLike
    :param gamma_c:
    :type gamma_c: float
    :param gamma_inj:
    :type gamma_inj: float
    :param gamma_max:
    :type gamma_max: float
    :param p:
    :type p: float
    :returns:

    """

    # normalizations to make
    # distribution continuous

    A = np.power(gamma_inj, 1 - p)

    # output
    out = np.zeros(gamma.shape[0])

    idx1 = (gamma >= gamma_c) & (gamma < gamma_inj)
    idx2 = (gamma >= gamma_inj) & (gamma < gamma_max)

    # compute the regions

    out[idx1] = A * np.power(gamma[idx1], -2)

    out[idx2] = np.power(gamma[idx2], -p - 1)

    return out


@nb.njit(fastmath=True)
def slow_cooling_distribution(
    gamma: ArrayLike, gamma_inj: float, gamma_max: float, p: float
):

    out = np.power(gamma, -p)

    return out
