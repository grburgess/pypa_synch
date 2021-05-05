import numba as nb
import numpy as np
from numpy.typing import ArrayLike


@nb.njit(fastmath=True)
def power_law(gamma: ArrayLike, gamma_b: float, gamma_c: float, gamma_inj: float, gamma_max: float, p: float):

    """TODO describe function

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
    A = np.power(gamma_inj, 1-p)
    B = A * np.power(gamma_c, -1)

    out = np.zeros(len(gamma))

    idx1 = gamma < gamma_c
    idx2 = (gamma >= gamma_c) & (gamma < gamma_inj)
    idx3 = (gamma >= gamma_inj) & (gamma < gamma_max)

    out[idx1] = B * np.power(gamma[idx1], -1)

    out[idx2] = A*np.power(gamma[idx2], -2)

    out[idx3] = np.power(gamma[idx3], -p-1)

    return out
