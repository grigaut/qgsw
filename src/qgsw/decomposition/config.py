"""Basis config.

From
https://github.com/leguillf/MASSH/blob/81b43dbd3f658a3a61675b7a95a4afeb391c163f/mapping/src/config_default.py#L1207
"""

from math import log, sqrt

import torch

from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.vardyn.core import MIOSTBasis
from qgsw.decomposition.wavelets.core import WaveletBasis

BASIS_MIOST = {
    "type": MIOSTBasis.type,
    "name_mod_var": None,  # Name of the related model variable
    "flux": False,
    "save_wave_basis": False,  # save the basis matrix in tmp_DA_path.If False, the matrix is stored in line  # noqa: E501
    "wavelet_init": False,  # Estimate the initial state
    "facns": 1.0,  # factor for wavelet spacing= space
    "facnlt": 2.0,  # factor for wavelet spacing= time
    "npsp": 3.5,  # Defines the wavelet shape
    "facpsp": 1.5,  # factor to fix df between wavelets
    "lmin": 80,  # minimal wavelength (in km)
    "lmax": 970.0,  # maximal wavelength (in km)
    "factdec": 7.5,  # factor to be multiplied to the computed time of decorrelation  # noqa: E501
    "tdecmin": 2.0,  # minimum time of decorrelation
    "tdecmax": 40.0,  # maximum time of decorrelation
    "facQ": 1,  # factor to be multiplied to the estimated Q
    "distortion_eq": 2.0,
    "lat_distortion_eq": 5.0,
    "distortion_eq_law": 2.0,
    "file_aux": None,
    "filec_aux": None,
    "tssr": 0.5,
    "facRo": 8.0,
    "Romax": 150.0,
    "cutRo": 1.6,
    "path_background": None,  # path netcdf file of a basis vector (e.g. coming from a previous run) to use as background  # noqa: E501
    "var_background": None,  # name of the variable of the basis vector
}

GAUSSIAN_EXP = {
    "type": GaussianExpBasis.type,
    "order": 0,
    "spacing": 30000,  # meters
    "l_t": 288000,  # seconds
}

WAVELETS = {
    "type": WaveletBasis.type,
    "n_theta": 7,
    "order": 5,
    "Lxy_max": 900_000,  # meters
    "sigma_t_sigma_xy_ratio": 1.17,
    "sigma_xy_l_p_ratio": 1.75 / torch.pi * 2 * sqrt(2),
    "l_xy_sigma_xy_ratio": 2 * sqrt(log(2)),
    "l_t_sigma_t_ratio": 2 * sqrt(log(2)),
    "use_time_frontier_as_centers": False,
}

BASIS_CONFIGS = {
    MIOSTBasis.type: BASIS_MIOST,
    WaveletBasis.type: WAVELETS,
    GaussianExpBasis.type: GAUSSIAN_EXP,
}
