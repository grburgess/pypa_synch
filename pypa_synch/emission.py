import numba as nb
import numpy as np
from numpy.typing import ArrayLike

from .electron_distribution import (fast_cooling_distribution, pa_distribution,
                                    slow_cooling_distribution, pic_distribution)
from .synchrotron_utils import compute_synchtron_matrix


@nb.njit(fastmath=True)
def pa_emission(
    photon_energy: ArrayLike,
    B: float,
    p: float,
    gamma_min: float,
    gamma_cool: float,
    gamma_inj: float,
    gamma_max: float,
    bulk_lorentz_factor: float,
    n_grid_points: int = 100,
) -> np.ndarray:
    """TODO describe function

    :param photon_energy: 
    :type photon_energy: ArrayLike
    :param B: 
    :type B: float
    :param p: 
    :type p: float
    :param gamma_min: 
    :type gamma_min: float
    :param gamma_cool: 
    :type gamma_cool: float
    :param gamma_inj: 
    :type gamma_inj: float
    :param gamma_max: 
    :type gamma_max: float
    :param bulk_lorentz_factor: 
    :type bulk_lorentz_factor: float
    :param n_grid_points: 
    :type n_grid_points: int
    :returns: 

    """
    n_photon_points: int = photon_energy.shape[0]

    # create a log10 grid of photon energies
    # do this in linspace because numba is stupid

    electron_grid = np.power(
        10.0,
        np.linspace(np.log10(gamma_min), np.log10(gamma_max), n_grid_points))

    # compute the

    s_matrix = compute_synchtron_matrix(
        energy=photon_energy,
        gamma2=electron_grid**2,
        B=B,
        bulk_lorentz_factor=bulk_lorentz_factor,
        n_photon_energies=n_photon_points,
        n_grid_points=n_grid_points,
    )

    # convolve the electron with the synchrotron kernel

    val = (pa_distribution(
        electron_grid,
        gamma_b=gamma_min,
        gamma_c=gamma_cool,
        gamma_inj=gamma_inj,
        gamma_max=gamma_max,
        p=p,
    )[1:] * np.diff(electron_grid))

    out = np.dot(np.ascontiguousarray(s_matrix[:, 1:]),
                 val) / (2.0 * photon_energy)

    return out


@nb.njit(fastmath=True)
def fast_cooling_emission(
    photon_energy: ArrayLike,
    B: float,
    p: float,
    gamma_cool: float,
    gamma_inj: float,
    gamma_max: float,
    bulk_lorentz_factor: float,
    n_grid_points: int = 100,
) -> np.ndarray:

    n_photon_points: int = photon_energy.shape[0]

    # create a log10 grid of photon energies
    # do this in linspace because numba is stupid

    electron_grid = np.power(
        10.0,
        np.linspace(np.log10(gamma_cool), np.log10(gamma_max), n_grid_points))

    # compute the

    s_matrix = compute_synchtron_matrix(
        energy=photon_energy,
        gamma2=electron_grid ** 2,
        B=B,
        bulk_lorentz_factor=bulk_lorentz_factor,
        n_photon_energies=n_photon_points,
        n_grid_points=n_grid_points,
    )

    # convolve the electron with the synchrotron kernel

    val = (
        fast_cooling_distribution(
            electron_grid,
            gamma_c=gamma_cool,
            gamma_inj=gamma_inj,
            gamma_max=gamma_max,
            p=p,
        )[1:]
        * np.diff(electron_grid)
    )

    out = np.dot(np.ascontiguousarray(s_matrix[:, 1:]), val) / (2.0 * photon_energy)

    return out


@nb.njit(fastmath=True)
def slow_cooling_emission(
    photon_energy: ArrayLike,
    B: float,
    p: float,
    gamma_inj: float,
    gamma_max: float,
    bulk_lorentz_factor: float,
    n_grid_points: int = 100,
) -> np.ndarray:

    n_photon_points: int = photon_energy.shape[0]

    # create a log10 grid of photon energies
    # do this in linspace because numba is stupid

    electron_grid = np.power(
        10.0, np.linspace(np.log10(gamma_inj), np.log10(gamma_max), n_grid_points)
    )

    
    # compute the

    s_matrix = compute_synchtron_matrix(
        energy=photon_energy,
        gamma2=electron_grid ** 2,
        B=B,
        bulk_lorentz_factor=bulk_lorentz_factor,
        n_photon_energies=n_photon_points,
        n_grid_points=n_grid_points,
    )

    # convolve the electron with the synchrotron kernel

    val = (
        slow_cooling_distribution(
            electron_grid,
            gamma_inj=gamma_inj,
            gamma_max=gamma_max,
            p=p,
        )[1:]
        * np.diff(electron_grid)
    )

    out = np.dot(np.ascontiguousarray(s_matrix[:, 1:]), val) / (2.0 * photon_energy)

    return out


@nb.njit(fastmath=True)
def anisotropic_emission(
    photon_energy: ArrayLike,
    B: float,
    p: float,
    gamma_inj: float,
    gamma_max: float,
    bulk_lorentz_factor: float,
    amplitude=0., 
    delta=0.2,
    n_grid_points: int = 100,
) -> np.ndarray:

    n_photon_points: int = photon_energy.shape[0]

    # create a log10 grid of photon energies
    # do this in linspace because numba is stupid

    electron_grid = np.power(
        10.0, np.linspace(np.log10(gamma_inj), np.log10(gamma_max), n_grid_points)
    )

    
    # compute the

    s_matrix = compute_synchtron_matrix(
        energy=photon_energy,
        gamma2=electron_grid ** 2,
        B=B,
        bulk_lorentz_factor=bulk_lorentz_factor,
        n_photon_energies=n_photon_points,
        n_grid_points=n_grid_points,
        amplitude=amplitude,
        gamma_inj=gamma_inj, delta=delta
        
    )

    # convolve the electron with the synchrotron kernel

    val = (
        slow_cooling_distribution(
            electron_grid,
            gamma_inj=gamma_inj,
            gamma_max=gamma_max,
            p=p,
        )[1:]
        * np.diff(electron_grid)
    )

    out = np.dot(np.ascontiguousarray(s_matrix[:, 1:]), val) / (2.0 * photon_energy)

    return out



@nb.njit(fastmath=True)
def pic_emission(
    photon_energy: ArrayLike,
    B: float,
    p: float,
    gamma_inj: float,
    gamma_max: float,
    gamma_cool: float,
    kt: float,
    bulk_lorentz_factor: float,
    amplitude=0., 
    delta=0.2,
    fermi_type=1,
    n_grid_points: int = 100,
) -> np.ndarray:

    n_photon_points: int = photon_energy.shape[0]

    # create a log10 grid of photon energies
    # do this in linspace because numba is stupid

    electron_grid = np.power(
        10.0, np.linspace(0, np.log10(gamma_max), n_grid_points)
    )

    
    # compute the

    s_matrix = compute_synchtron_matrix(
        energy=photon_energy,
        gamma2=electron_grid ** 2,
        B=B,
        bulk_lorentz_factor=bulk_lorentz_factor,
        n_photon_energies=n_photon_points,
        n_grid_points=n_grid_points,
        amplitude=amplitude,
        gamma_inj=gamma_inj, delta=delta
        
    )

    # convolve the electron with the synchrotron kernel

    K1 = 1
    K2 = 2
    
    val = (
        pic_distribution(x=electron_grid,
            K1=K1,
            K2=K2,
            p=p,
            x_inj=gamma_inj,
            x_cool=gamma_cool,
            alphacut = fermi_type,
                         kt=kt
            
            
        )[1:]
        * np.diff(electron_grid)
    )

    out = np.dot(np.ascontiguousarray(s_matrix[:, 1:]), val) / (2.0 * photon_energy)

    return out

