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
    Steady state fast-cooling electron distribution.

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



# from Joonas Nattila

#maxwell-juttner distribution
@nb.njit(fastmath=True)
def model_maxwell(K, kt, x):
    

    beta = np.sqrt(1.0 - 1.0/x**2)
    uvel = x*beta
    f = uvel*uvel*np.exp(-x/kt) 
    f /= np.max(f) #NOTE: omit physical normalization and just scale max(f) = 1
    return K*f

# broken powerlaw with a smooth connection
@nb.njit(fastmath=True)
def model_smoothly_broken_plaw(amplitude, alpha_1, alpha_2, x_break, delta, x):
    # based on https://docs.astropy.org/en/stable/_modules/astropy/modeling/
    # powerlaws.html#SmoothlyBrokenPowerLaw1D

    #switch sign
    alpha_1 = -alpha_1
    alpha_2 = -alpha_2

    # Pre-calculate `x/x_b`
    xx = x / x_break

    n = len(xx)
    
    # Initialize the return value
    f = np.zeros(n)

    # The quantity `t = (x / x_b)^(1 / delta)` can become quite
    # large.  To avoid overflow errors we will start by calculating
    # its natural logarithm:
    logt = np.log(xx) / delta

    # When `t >> 1` or `t << 1` we don't actually need to compute
    # the `t` value since the main formula (see docstring) can be
    # significantly simplified by neglecting `1` or `t`
    # respectively.  In the following we will check whether `t` is
    # much greater, much smaller, or comparable to 1 by comparing
    # the `logt` value with an appropriate threshold.
    threshold = 30  # corresponding to exp(30) ~ 1e13

    i = logt > threshold

    if (i.max()):
        # In this case the main formula reduces to a simple power
        # law with index `alpha_2`.
        f[i] = amplitude * xx[i] ** (-alpha_2) \
               / (2. ** ((alpha_1 - alpha_2) * delta))

    i = logt < -threshold
    if (i.max()):
        # In this case the main formula reduces to a simple power
        # law with index `alpha_1`.
        f[i] = amplitude * xx[i] ** (-alpha_1) \
               / (2. ** ((alpha_1 - alpha_2) * delta))

    i = np.abs(logt) <= threshold
    if (i.max()):
        # In this case the `t` value is "comparable" to 1, hence we
        # we will evaluate the whole formula.
        t = np.exp(logt[i])
        r = (1. + t) / 2.
        f[i] = amplitude * xx[i] ** (-alpha_1) \
               * r ** ((alpha_1 - alpha_2) * delta)

    return f

@nb.njit(fastmath=True)
def pic_distribution(x, K1, kt, K2, p, x_inj, x_cool, alphacut=1, delta=0.5):
    """

    :param K1: 
    :type K1: 
    :param kt: 
    :type kt: 
    :param K2: 
    :type K2: 
    :param alpha2: 
    :type alpha2: 
    :param xbreak:  -> gamma inj
    :type xbreak: 
    :param xcut: -> gamma cool
    :type xcut: 
    :param alphacut: -> 1 is fermi II  
    :type alphacut: 
    :param delta: 
    :type delta: 
    :param x: 
    :type x: 
    :returns: 

    """
     

    alpha1 = 4.0

    y1 = model_maxwell(K1, kt, x)

    #broken plaw
    y2 = model_smoothly_broken_plaw(K2, alpha1, p, x_inj, delta, x)

    return y1 + y2*np.exp(-(x/x_cool)**alphacut)

