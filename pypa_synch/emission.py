import numba as nb
import numpy as np
from numpy.typing import ArrayLike

from .electron_distribution import power_law
from .synchrotron_utils import compute_synchtron_matrix


def emission(photon_energy: ArrayLike, B: float, p: float, gamma_min: float, gamma_cool: float, gamma_inj: float, gamma_max: float, n_grid_points: int = 100) -> np.ndarray:

    blf: float = 100.

    n_photon_points: int = photon_energy.shape[0]

    # create a log10 grid of photon energies
    # do this in linspace because numba is stupid

    electron_grid = np.power(10., np.linspace(
        np.log10(gamma_min), np.log10(gamma_max)))

    # compute the

    s_matrix = compute_synchtron_matrix(
        energy=photon_energy,
        gamma2=electron_grid**2,
        B=1e5, bulk_lorentz_factor=100,
        n_photon_energies=n_photon_points, n_grid_points=n_grid_points
    )

    # convolve the electron with the synchrotron kernel

    out = np.dot(np.ascontiguousarray(
        s_matrix), power_law(electron_grid,
                             gamma_b=gamma_min,
                             gamma_c=gamma_cool,
                             gamma_inj=gamma_inj,
                             gamma_max=gamma_max,
                             p=p))/(2.0*photon_energy)

    return out
