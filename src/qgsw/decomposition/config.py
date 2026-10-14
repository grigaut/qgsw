"""Basis config.

From
https://github.com/leguillf/MASSH/blob/81b43dbd3f658a3a61675b7a95a4afeb391c163f/mapping/src/config_default.py#L1207
"""

BASIS_MIOST = {
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
