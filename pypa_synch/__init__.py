# -*- coding: utf-8 -*-

"""Top-level package for Pitch Angle Synchrotron Model."""

__author__ = """J. Michael Burgess"""
__email__ = "jburgess@mpe.mpg.de"

from ._version import get_versions
from .electron_distribution import (fast_cooling_distribution, pa_distribution,
                                    slow_cooling_distribution)
from .emission import fast_cooling_emission, pa_emission, slow_cooling_emission
from .threeml_model import (FastCoolingSynchrotron, PitchAngleSynchrotron,
                            SlowCoolingSynchrotron)

__version__ = get_versions()["version"]
del get_versions
