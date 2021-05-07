import numba as nb
import numpy as np
from numpy.typing import ArrayLike

from .electron_distribution import (fast_cooling_distribution, pa_distribution,
                                    slow_cooling_distribution)
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

    n_photon_points: int = photon_energy.shape[0]

    # create a log10 grid of photon energies
    # do this in linspace because numba is stupid

    electron_grid = np.power(
        10.0, np.linspace(np.log10(gamma_min), np.log10(gamma_max), n_grid_points)
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

    out = (
        np.dot(
            np.ascontiguousarray(s_matrix),
            pa_distribution(
                electron_grid,
                gamma_b=gamma_min,
                gamma_c=gamma_cool,
                gamma_inj=gamma_inj,
                gamma_max=gamma_max,
                p=p,
            ),
        )
        / (2.0 * photon_energy)
    )

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
        10.0, np.linspace(np.log10(gamma_cool), np.log10(gamma_max), n_grid_points)
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

    out = (
        np.dot(
            np.ascontiguousarray(s_matrix),
            fast_cooling_distribution(
                electron_grid,
                gamma_c=gamma_cool,
                gamma_inj=gamma_inj,
                gamma_max=gamma_max,
                p=p,
            ),
        )
        / (2.0 * photon_energy)
    )

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

    out = (
        np.dot(
            np.ascontiguousarray(s_matrix),
            slow_cooling_distribution(
                electron_grid,
                gamma_inj=gamma_inj,
                gamma_max=gamma_max,
                p=p,
            ),
        )
        / (2.0 * photon_energy)
    )

    return out
