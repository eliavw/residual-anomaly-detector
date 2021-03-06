# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version

    # imports
    from . import exps
    from .exps import *
    from . import ResidualAnomalyDetector
    from .ResidualAnomalyDetector import ResidualAnomalyDetector
    
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
