# -*- coding: utf-8 -*-

"""Top-level package for Pitch Angle Synchrotron Model."""

__author__ = """J. Michael Burgess"""
__email__ = 'jburgess@mpe.mpg.de'

from ._version import get_versions
from .electron_distribution import power_law
from .emission import emission
from .threeml_model import PitchAngleSynchrotron

__version__ = get_versions()['version']
del get_versions
