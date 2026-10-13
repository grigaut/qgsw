"""Basis as implemented in the original repository.

Minor adjustements to remove dependencies to custom classes.

From
https://github.com/leguillf/MASSH/blob/81b43dbd3f658a3a61675b7a95a4afeb391c163f/mapping/src/basis.py#L3233
"""

# ruff: noqa
# ruff: noqa: PGH004

import os

import numpy as np
import xarray as xr
from scipy.sparse import csc_matrix

from qgsw.logging.core import getLogger

logger = getLogger(__name__)


def window(x):  # x must be between -1 and 1
    y = np.cos(x * 0.5 * np.pi) ** 2
    return y


class Basis_miost:
    def __init__(
        self,
        basis_config: dict,
        lon: np.ndarray,
        lat: np.ndarray,
        nx: int,
        ny: int,
        bounds: tuple[float, float, float, float],
        multi_mode=False,
    ):

        self.km2deg = 1.0 / 110

        self.flux = basis_config["flux"]
        self.facns = basis_config["facns"]  # factor for wavelet spacing= space
        self.facnlt = basis_config["facnlt"]
        self.npsp = basis_config[
            "npsp"
        ]  # Defines the wavelet shape (nb de pseudopériode)
        self.facpsp = basis_config[
            "facpsp"
        ]  # 1.5 # factor to fix df between wavelets
        self.lmin = basis_config["lmin"]
        self.lmax = basis_config["lmax"]
        self.tdecmin = basis_config["tdecmin"]
        self.tdecmax = basis_config["tdecmax"]
        self.factdec = basis_config["factdec"]
        self.facQ = basis_config["facQ"]
        self.save_wave_basis = basis_config["save_wave_basis"]
        self.wavelet_init = basis_config["wavelet_init"]
        self.file_aux = basis_config["file_aux"]
        self.filec_aux = basis_config["filec_aux"]
        self.Romax = basis_config["Romax"]
        self.facRo = basis_config["facRo"]
        self.cutRo = basis_config["cutRo"]
        self.tssr = basis_config["tssr"]
        self.distortion_eq = basis_config["distortion_eq"]
        self.distortion_eq_law = basis_config["distortion_eq_law"]
        self.lat_distortion_eq = basis_config["lat_distortion_eq"]
        self.name_mod_var = basis_config["name_mod_var"]
        self.path_background = basis_config["path_background"]
        self.var_background = basis_config["var_background"]

        # Grid params
        self.nphys = lon.size
        # self.shape_phys = (ny,nx)
        self.shape_phys = (nx, ny)
        lon_min, lon_max, lat_min, lat_max = bounds
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon1d = lon.flatten()
        self.lat1d = lat.flatten()

        # Longitude unit
        # self.lon_unit = lon_unit
        if np.sign(self.lon_min) == -1:
            self.lon_unit = "-180_180"
        else:
            self.lon_unit = "0_360"

        # Mask
        # if State.mask is not None and np.any(State.mask):
        #     self.mask1d = State.mask.ravel()
        # else:
        #     self.mask1d = None
        self.mask1d = None

        # Time window
        self.window = window

        self.multi_mode = multi_mode

    def set_basis(self, time, return_q=False, **kwargs):

        Mutltiple_basis_exp = False
        if Mutltiple_basis_exp:
            L_MIN = 30
            L_MAX = 1000

            TIME_MIN = time.min()
            TIME_MAX = time.max()
            LON_MIN = self.lon_min
            LON_MAX = self.lon_max
            LAT_MIN = self.lat_min
            LAT_MAX = self.lat_max
            if LON_MAX < LON_MIN:
                LON_MAX = LON_MAX + 360.0

            # Ensemble of pseudo-frequencies for the wavelets (spatial)
            logff_all = np.arange(
                np.log(1.0 / L_MIN),
                np.log(1.0 / L_MAX) - np.log(1 + self.facpsp / self.npsp),
                -np.log(1 + self.facpsp / self.npsp),
            )[::-1]
            # print('_',logff_all)
            # print('A',self.lmax)
            # print('B',(logff_all>1/self.lmax))
            # print('C',self.lmin)
            # print('D',(logff_all<1/self.lmin))
            # print('E',(logff_all>1/self.lmax) & (logff_all<1/self.lmin))
            logff = logff_all[
                (logff_all >= np.log(1 / self.lmax))
                & (logff_all <= np.log(1 / self.lmin))
            ]
            ff = np.exp(logff)
            ff = ff[1 / ff <= self.lmax]
            dff = ff[1:] - ff[:-1]

        else:
            TIME_MIN = time.min()
            TIME_MAX = time.max()
            LON_MIN = self.lon_min
            LON_MAX = self.lon_max
            LAT_MIN = self.lat_min
            LAT_MAX = self.lat_max
            if LON_MAX < LON_MIN:
                LON_MAX = LON_MAX + 360.0

            # Ensemble of pseudo-frequencies for the wavelets (spatial)
            logff = np.arange(
                np.log(1.0 / self.lmin),
                np.log(1.0 / self.lmax) - np.log(1 + self.facpsp / self.npsp),
                -np.log(1 + self.facpsp / self.npsp),
            )[::-1]

            ff = np.exp(logff)
            # ff = ff[1/ff<=self.lmax]
            dff = ff[1:] - ff[:-1]

        # Ensemble of directions for the wavelets (2D plane)
        theta = np.linspace(
            0, np.pi, int(np.pi * ff[0] / dff[0] * self.facpsp)
        )[:-1]
        ntheta = len(theta)
        nf = len(ff)
        logger.info("spatial normalized wavelengths: %s", 1.0 / np.exp(logff))
        logger.info("ntheta: %s", ntheta)

        # Global time window
        deltat = TIME_MAX - TIME_MIN

        # Read aux data
        aux = xr.open_dataset(self.file_aux)
        daPSDS = aux["PSDS"]
        auxc = xr.open_dataset(self.filec_aux)
        daC = auxc["c1"]

        # Convert longitudes
        if (
            np.sign(daPSDS["lon"].data.min()) == -1
            and self.lon_unit == "0_360"
        ):
            daPSDS = daPSDS.assign_coords(
                {"lon": (("lon", daPSDS["lon"].data % 360))}
            )
        elif (
            np.sign(daPSDS["lon"].data.min()) >= 0
            or daPSDS["lon"].data.max() > 180
        ) and self.lon_unit == "-180_180":
            daPSDS = daPSDS.assign_coords(
                {"lon": (("lon", (daPSDS["lon"].data + 180) % 360 - 180))}
            )
        if np.sign(daC["lon"].data.min()) == -1 and self.lon_unit == "0_360":
            daC = daC.sel(lon=slice(-180, 179.99))
            daC = daC.assign_coords({"lon": (("lon", daC["lon"].data % 360))})
        elif (
            np.sign(daC["lon"].data.min()) >= 0 or daC["lon"].data.max() > 180
        ) and self.lon_unit == "-180_180":
            daC = daC.sel(lon=slice(0, 359.99))
            daC = daC.assign_coords(
                {"lon": (("lon", (daC["lon"].data + 180) % 360 - 180))}
            )
        daPSDS = daPSDS.sortby(daPSDS["lon"])
        daC = daC.sortby(daC["lon"])
        daC = daC.T

        # Wavelet space-time coordinates
        ENSLON = [
            None
        ] * nf  # Ensemble of longitudes of the center of each wavelets
        ENSLAT = [
            None
        ] * nf  # Ensemble of latitudes of the center of each wavelets
        tdec_max = [None] * nf
        norm_fact = (
            list()
        )  # integral of the time component (for normalization)
        enst = list()  #  Ensemble of times of the center of each wavelets
        tdec = (
            list()
        )  # Ensemble of equivalent decorrelation times. Used to define enst.
        Cb1 = list()  # First baroclinic phase speed

        DX = 1.0 / ff * self.npsp * 0.5  # wavelet extension
        DXG = (
            DX / self.facns
        )  # distance (km) between the wavelets grid in space
        NP = np.empty(
            nf, dtype="int64"
        )  # Nomber of spatial wavelet locations for a given frequency
        nwave = 0
        self.nwavemeso = 0
        lonmax = LON_MAX
        if LON_MAX < LON_MIN:
            lonmax = LON_MAX + 360.0

        for iff in range(nf):
            logger.info(
                f"[Wavelet order {iff + 1}] {1 / ff[iff]:.0f} km, spacing: {DXG[iff]}"
            )

            ENSLON[iff] = []
            ENSLAT[iff] = []
            ENSLAT1 = np.arange(
                LAT_MIN - (DX[iff] - DXG[iff]) * self.km2deg,
                LAT_MAX + DX[iff] * self.km2deg,
                DXG[iff] * self.km2deg,
            )
            for I in range(len(ENSLAT1)):
                _ENSLON = np.arange(
                    LON_MIN
                    - (DX[iff] - DXG[iff])
                    / np.cos(ENSLAT1[I] * np.pi / 180.0)
                    * self.km2deg,
                    LON_MAX
                    + DX[iff]
                    / np.cos(ENSLAT1[I] * np.pi / 180.0)
                    * self.km2deg,
                    DXG[iff]
                    / np.cos(ENSLAT1[I] * np.pi / 180.0)
                    * self.km2deg,
                )
                _ENSLAT = np.repeat(ENSLAT1[I], len(_ENSLON))
                # if self.mask1d is None:
                _ENSLON1 = _ENSLON
                _ENSLAT1 = _ENSLAT

                ENSLAT[iff] = np.concatenate([ENSLAT[iff], _ENSLAT1])
                ENSLON[iff] = np.concatenate([ENSLON[iff], _ENSLON1])
            NP[iff] = len(ENSLON[iff])
            enst.append(list())
            tdec.append(list())
            norm_fact.append(list())
            Cb1.append(list())
            for P in range(NP[iff]):
                enst[-1].append(list())
                tdec[-1].append(list())
                norm_fact[-1].append(list())
                Cb1[-1].append(list())

                # First baroclinic phase speed
                dlon = (
                    DX[iff]
                    * self.km2deg
                    / np.cos(ENSLAT[iff][P] * np.pi / 180.0)
                )  # *finterpdist(ENSLAT[iff][P])
                dlat = DX[iff] * self.km2deg
                elon = np.linspace(
                    ENSLON[iff][P] - dlon, ENSLON[iff][P] + dlon, 10
                )
                elat = np.linspace(
                    ENSLAT[iff][P] - dlat, ENSLAT[iff][P] + dlat, 10
                )
                elon2, elat2 = np.meshgrid(elon, elat)
                Ctmp = daC.interp(
                    lat=elat2.flatten(), lon=elon2.flatten()
                ).values
                Ctmp = Ctmp[np.isnan(Ctmp) == False]
                if len(Ctmp) > 0:
                    C = np.nanmean(Ctmp)
                else:
                    C = np.nan
                if np.isnan(C):
                    C = 0.0
                Cb1[-1][-1] = C

                # Decorrelation time
                fc = (
                    2
                    * 2
                    * np.pi
                    / 86164
                    * np.sin(ENSLAT[iff][P] * np.pi / 180.0)
                )
                Ro = C / np.abs(fc) / 1000.0  # Rossby radius (km)
                Ro = min(Ro, self.Romax)
                if C > 0:
                    td1 = self.factdec / (
                        1.0 / (self.facRo * Ro) * C / 1000 * 86400
                    )
                else:
                    td1 = np.nan
                PSDS = daPSDS.interp(
                    f=ff[iff].item(),
                    lat=ENSLAT[iff][P].item(),
                    lon=ENSLON[iff][P].item(),
                ).values
                if Ro > 0:
                    PSDSR = daPSDS.interp(
                        f=1.0 / (self.facRo * Ro),
                        lat=ENSLAT[iff][P].item(),
                        lon=ENSLON[iff][P].item(),
                    ).values
                else:
                    PSDSR = np.nan
                if PSDS <= PSDSR:
                    tdec[-1][-1] = td1 * (PSDS / PSDSR) ** self.tssr
                else:
                    tdec[-1][-1] = td1
                tdec[-1][-1] = min(tdec[-1][-1], self.tdecmax)
                cp = (
                    1.0
                    / (
                        2
                        * 2
                        * np.pi
                        / 86164
                        * np.sin(
                            max(10, np.abs(ENSLAT[iff][P])) * np.pi / 180.0
                        )
                    )
                    / 300000
                )
                tdecp = (1.0 / ff[iff]) * 1000 / cp / 86400 / 4
                if tdecp < tdec[-1][-1] or np.isnan(tdec[-1][-1]):
                    tdec[-1][-1] = tdecp

                # Compute time integral for each frequency for normalization
                tt = np.linspace(-tdec[iff][P], tdec[iff][P])
                tmp = np.zeros_like(tt)
                for i in range(tt.size - 1):
                    tmp[i + 1] = tmp[i] + self.window(tt[i] / tdec[iff][P]) * (
                        tt[i + 1] - tt[i]
                    )
                norm_fact[-1][-1] = tmp.max()

                # Time coordinates
                enst[-1][-1] = np.arange(
                    -tdec[iff][P] / self.facnlt,
                    deltat + tdec[iff][P] / self.facnlt,
                    tdec[iff][P] / self.facnlt,
                )

        # Harmonize the wavelet time center dimensions for all point by adding NaN if needed
        # (we must do that for the time operator Gt to be independent from the space operator Gx)
        enst_same_dim = [None] * nf
        for iff in range(nf):
            max_number_enst_iff = np.max(
                [enst[iff][P].size for P in range(NP[iff])]
            )
            enst_same_dim[iff] = (
                np.zeros((NP[iff], max_number_enst_iff)) * np.nan
            )
            for P in range(NP[iff]):
                enst_same_dim[iff][P, : enst[iff][P].size] = enst[iff][P]

        # Loop on all wavelets of given pseudo-period
        std = [None] * nf
        self.wavetest = [None] * nf
        for iff in range(nf):
            std[iff] = np.ones(NP[iff])
            self.wavetest[iff] = np.ones((NP[iff]), dtype=bool)
            for P in range(NP[iff]):
                PSDLOC = abs(
                    daPSDS.interp(
                        f=ff[iff].item(),
                        lat=ENSLAT[iff][P].item(),
                        lon=ENSLON[iff][P].item(),
                    ).values
                )
                C = Cb1[iff][P]
                fc = (
                    2
                    * 2
                    * np.pi
                    / 86164
                    * np.sin(ENSLAT[iff][P] * np.pi / 180.0)
                )
                if fc == 0:
                    Ro = self.Romax
                else:
                    Ro = C / np.abs(fc) / 1000.0  # Rossby radius (km)
                    Ro = min(Ro, self.Romax)
                # Tests
                if 1.0 / ff[iff] < self.cutRo * Ro:
                    self.wavetest[iff][P] = False
                tdec[iff][P] = max(tdec[iff][P], self.tdecmin)
                if np.isnan(PSDLOC):
                    self.wavetest[iff][P] = False
                if (np.isnan(Cb1[iff][P])) | (Cb1[iff][P] == 0):
                    self.wavetest[iff][P] = False

                if self.wavetest[iff][P]:
                    std_tmp = (
                        PSDLOC
                        * ff[iff] ** 2
                        * self.facQ
                        * np.exp(-3 * (self.cutRo * Ro * ff[iff]) ** 3)
                    ) ** 0.5
                else:
                    std_tmp = 1e-15
                std[iff][P] = std_tmp

        # Fill the Q diagonal matrix (expected variance for each wavelet)
        Q = np.array([])
        iwave = -1
        self.iff_wavebounds = [None] * (nf + 1)
        self.P_wavebounds = [None] * (nf + 1)
        for iff in range(nf):
            Qf_list = []  # Use a list instead of np.concatenate in loops
            self.iff_wavebounds[iff] = iwave + 1
            self.P_wavebounds[iff] = [None] * (NP[iff] + 1)
            self.wavetest[iff] = np.ones((NP[iff]), dtype=bool)
            _nwavef = 0

            for it in range(enst_same_dim[iff].shape[1]):
                for P in range(NP[iff]):
                    self.P_wavebounds[iff][P] = iwave + 1
                    if np.isnan(enst_same_dim[iff][P, it]):
                        Q_tmp = 1e-15  # Not zero otherwise a ZeroDivisionError exception will be raised
                    else:
                        Q_tmp = std[iff][P]
                    Qf_list.append(Q_tmp * np.ones(2 * ntheta))
                    _nwavef += 2 * ntheta

            Qf = np.concatenate(Qf_list)
            Q = np.concatenate((Q, Qf))
            iwave += _nwavef

            msg = (
                f"lambda={1 / ff[iff]:.1E}"
                f"nlocs={P:.1E}"
                f"tdec={np.nanmean(tdec[iff]):.1E}"
                f"Q={np.nanmean(Q[self.iff_wavebounds[iff] : iwave + 1]):.1E}"
            )

            logger.detail(msg)

            self.P_wavebounds[iff][P + 1] = iwave + 1
        self.iff_wavebounds[iff + 1] = iwave + 1

        # Background
        if self.path_background is not None and os.path.exists(
            self.path_background
        ):
            with xr.open_dataset(self.path_background) as ds:
                logger.info(
                    f"Load background from file: {self.path_background}"
                )
                Xb = ds[self.var_background].values
        else:
            Xb = np.zeros_like(Q)

        self.DX = DX
        self.ENSLON = ENSLON
        self.ENSLAT = ENSLAT
        self.NP = NP
        self.enst = enst_same_dim
        self.nbasis = Q.size
        self.nf = nf
        self.theta = theta
        self.ntheta = ntheta
        self.ff = ff
        self.k = 2 * np.pi * ff
        self.tdec = tdec
        self.tdec_max = tdec_max
        self.norm_fact = norm_fact

        # Compute basis components
        self.Gx, self.Nx = self._compute_component_space()  # in space
        self.Gt, self.Nt = self._compute_component_time(time)  # in time

        logger.info(
            f"reduced order: {time.size * self.nphys} --> {self.nbasis}\n reduced factor: {int(time.size * self.nphys / self.nbasis)}"
        )

        if return_q:
            return Xb, Q

    def _compute_component_space(self):

        Gx = [None] * self.nf
        Nx = [None] * self.nf

        for iff in range(self.nf):
            data = np.empty((2 * self.ntheta * self.NP[iff] * self.nphys,))
            indices = np.empty(
                (2 * self.ntheta * self.NP[iff] * self.nphys,), dtype=int
            )
            sizes = np.zeros((2 * self.ntheta * self.NP[iff],), dtype=int)

            ind_tmp = 0
            iwave = 0

            for P in range(self.NP[iff]):
                # Obs selection around point P
                indphys = np.where(
                    (
                        np.abs(
                            (self.lon1d - self.ENSLON[iff][P])
                            / self.km2deg
                            * np.cos(self.ENSLAT[iff][P] * np.pi / 180.0)
                        )
                        <= self.DX[iff]
                    )
                    & (
                        np.abs(
                            (self.lat1d - self.ENSLAT[iff][P]) / self.km2deg
                        )
                        <= self.DX[iff]
                    )
                )[0]
                xx = (
                    (self.lon1d[indphys] - self.ENSLON[iff][P])
                    / self.km2deg
                    * np.cos(self.ENSLAT[iff][P] * np.pi / 180.0)
                )
                yy = (self.lat1d[indphys] - self.ENSLAT[iff][P]) / self.km2deg
                # Spatial tapering shape of the wavelet
                if self.mask1d is not None:
                    indmask = self.mask1d[indphys]
                    indphys = indphys[~indmask]
                    xx = xx[~indmask]
                    yy = yy[~indmask]

                facs = window(xx / self.DX[iff]) * window(yy / self.DX[iff])

                for itheta in range(self.ntheta):
                    # Wave vector components
                    kx = self.k[iff] * np.cos(self.theta[itheta])
                    ky = self.k[iff] * np.sin(self.theta[itheta])
                    # Cosine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp : ind_tmp + indphys.size] = indphys
                    data[ind_tmp : ind_tmp + indphys.size] = (
                        np.sqrt(2) * facs * np.cos(kx * (xx) + ky * (yy))
                    )
                    ind_tmp += indphys.size
                    iwave += 1
                    # Sine component
                    sizes[iwave] = indphys.size
                    indices[ind_tmp : ind_tmp + indphys.size] = indphys
                    data[ind_tmp : ind_tmp + indphys.size] = (
                        np.sqrt(2) * facs * np.sin(kx * (xx) + ky * (yy))
                    )
                    ind_tmp += indphys.size
                    iwave += 1

            nwaves = iwave
            Nx[iff] = nwaves

            sizes = sizes[:nwaves]
            indices = indices[:ind_tmp]
            data = data[:ind_tmp]

            indptr = np.zeros((nwaves + 1), dtype=int)
            indptr[1:] = np.cumsum(sizes)

            Gx[iff] = csc_matrix(
                (data, indices, indptr), shape=(self.nphys, nwaves)
            )

        return Gx, Nx

    def _compute_component_time(self, time):

        Gt = {}  # Time operator that gathers the time factors for each frequency
        Nt = {}  # Number of wave times tw such as abs(tw-t)<tdec

        for t in time:
            Gt[t] = [None] * self.nf
            Nt[t] = [0] * self.nf

            for iff in range(self.nf):
                Gt[t][iff] = (
                    np.zeros(
                        (
                            self.iff_wavebounds[iff + 1]
                            - self.iff_wavebounds[iff],
                        )
                    )
                    * np.nan
                )
                ind_tmp = 0
                for it in range(self.enst[iff].shape[1]):
                    for P in range(self.NP[iff]):
                        dt = t - self.enst[iff][P, it]
                        if abs(dt) > self.tdec[iff][P] or np.isnan(
                            self.enst[iff][P, it]
                        ):
                            fact = 0
                        else:
                            fact = self.window(dt / self.tdec[iff][P])
                            fact /= self.norm_fact[iff][P]
                        Gt[t][iff][ind_tmp : ind_tmp + 2 * self.ntheta] = fact
                        if P == 0:
                            Nt[t][iff] += 1
                        ind_tmp += 2 * self.ntheta
        return Gt, Nt

    def operg(self, t, X, State=None):
        """Project to physicial space"""
        # Projection
        phi = np.zeros(self.shape_phys).ravel()
        for iff in range(self.nf):
            Xf = X[self.iff_wavebounds[iff] : self.iff_wavebounds[iff + 1]]
            GtXf = self.Gt[t][iff] * Xf
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size > 0:
                GtXf = GtXf[indNoNan].reshape(self.Nt[t][iff], self.Nx[iff])
                phi += self.Gx[iff].dot(GtXf.sum(axis=0))
        phi = phi.reshape(self.shape_phys)

        # Update State
        if State is not None:
            State[self.name_mod_var] = phi
        else:
            return phi

    def operg_transpose(self, t, adState):
        """Project to reduced space"""
        if adState[self.name_mod_var] is None:
            adState[self.name_mod_var] = np.zeros((self.nphys,))

        adX = np.zeros(self.nbasis)
        adparams = adState[self.name_mod_var].ravel()
        for iff in range(self.nf):
            Gt = +self.Gt[t][iff]
            indNoNan = ~np.isnan(self.Gt[t][iff])
            if indNoNan.size > 0:
                Gt = Gt[indNoNan].reshape(self.Nt[t][iff], self.Nx[iff])
                adGtXf = self.Gx[iff].T.dot(adparams)
                adGtXf = np.repeat(
                    adGtXf[np.newaxis, :], self.Nt[t][iff], axis=0
                )
                adX[self.iff_wavebounds[iff] : self.iff_wavebounds[iff + 1]][
                    indNoNan
                ] += (Gt * adGtXf).ravel()

        adState[self.name_mod_var] *= 0.0

        return adX
