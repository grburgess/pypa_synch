import astropy.constants as constants
import astropy.units as u
import numpy as np
from astromodels import Function1D, FunctionMeta

from .emission import emission

__author__ = "grburgess"


class PitchAngleSynchrotron(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Synchrotron emission from cooling electrions
    latex : $  $
    parameters :
        K :
            desc : normalization
            initial value : 1
            min : 0

        B :
            desc : energy scaling
            initial value : 1E2
            min : .01


        index:
            desc : spectral index of electrons
            initial value : 2.224
            min : 2.
            max : 6


        gamma_min:
            desc : minimum electron lorentz factor
            initial value : 1e2
            min : 1
            fix: yes

        gamma_cool :
            desc: cooling time of electrons
            initial value: 1e4
            min value: 1


        gamma_inj :
             desc: cooling time of electrons
                initial value: 1e5
                min value: 1



        gamma_max:
            desc : minimum electron lorentz factor
            initial value : 1E8
            min : 1
            fix: yes

        bulk_gamma:
            desc : bulk Lorentz factor
            initial value : 100
            min : 1.
            fix: yes

    """

#    __metaclass__ = FunctionMeta

    def _set_units(self, x_unit, y_unit):

        self.K.unit = y_unit / u.gauss

        self.B.unit = u.gauss

        self.gamma_min.unit = u.dimensionless_unscaled
        self.gamma_cool.unit = u.dimensionless_unscaled
        self.gamma_inj.unit = u.dimensionless_unscaled
        self.gamma_max.unit = u.dimensionless_unscaled
        self.bulk_gamma.unit = u.dimensionless_unscaled
        self.index.unit = u.dimensionless_unscaled

    def evaluate(self, x, K, B, index, gamma_min, gamma_cool, gamma_inj, gamma_max, bulk_gamma):

        n_grid_points: int = 100

        n_photon_points: int = x.shape[0]

        if isinstance(K, u.Quantity):

            flag = True

            B_ = B.value
            gamma_min_ = gamma_min.value
            gamma_max_ = gamma_max.value
            gamma_inj_ = gamma_inj.value
            bulk_gamma_ = bulk_gamma.value
            gamma_cool_ = gamma_cool.value
            index_ = index.value
            unit_ = self.y_unit
            K_ = K.value

            try:
                flag = False
                tmp = len(x)

                x_ = x.value

            except:
                flag = True
                x_ = np.array([x.value])

        else:

            flag = False

            K_, B_, gamma_min_, gamma_cool_, gamma_max_, index_, x_, bulk_gamma_, gamma_inj_ = (
                K,
                B,
                gamma_min,
                gamma_cool,
                gamma_max,
                index,
                x,
                bulk_gamma,
                gamma_inj
            )
            unit_ = 1.0

        out = emission(x_, B_, index_, gamma_min_, gamma_cool_, gamma_inj_, gamma_max_, n_grid_points




        return out
