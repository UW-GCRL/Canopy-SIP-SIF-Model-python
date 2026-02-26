"""
Canopy-SIP-SIF Model: Main simulation engine.

Simulates canopy solar-induced chlorophyll fluorescence (SIF) anisotropy
by combining geometric-optical (GO) theory and spectral invariants theory
(p-theory).
"""

import os
import numpy as np
import pandas as pd

from .campbell import campbell
from .dladgen import dladgen
from .get_p_rho import get_p_rho
from .get_rta import get_rta
from .hemi_interceptance import get_hemi_interceptance


def _get_data_dir():
    """Return the path to the data directory."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class CanopySIPSIFModel:
    """
    Canopy-SIP-SIF Model for simulating SIF anisotropy.

    Combines Geometric-Optical (GO) theory and Spectral Invariants Theory
    (p-theory) to simulate canopy solar-induced chlorophyll fluorescence.
    """

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or _get_data_dir()
        self._load_data()

    def _load_data(self):
        """Load spectral and structural data from files."""
        d = self.data_dir

        # Load leaf optical properties (Fluspect output)
        leafopt = np.load(os.path.join(d, 'leafopt_P6.npz'))
        self.refl = leafopt['refl']    # (2101,)
        self.tran = leafopt['tran']    # (2101,)
        self.MfI_raw = leafopt['MfI']  # (211, 351)
        self.MbI_raw = leafopt['MbI']  # (211, 351)
        self.MfII_raw = leafopt['MfII']  # (211, 351)
        self.MbII_raw = leafopt['MbII']  # (211, 351)

        # Load solar irradiance
        esun = np.load(os.path.join(d, 'Esun_SIP.npz'))
        self.result_SIP = esun['result_SIP']  # (2162, 2)

        # Load soil reflectance
        rs_data = np.load(os.path.join(d, 'rs_01.npz'))
        self.rs = rs_data['rs'].ravel()  # (2162,)

        # Load structural data (gap fractions, clumping indices)
        self.gap_tot = pd.read_csv(os.path.join(d, 'gap_tot.csv')).values
        self.gap_betw = pd.read_csv(os.path.join(d, 'gap_betw.csv')).values
        self.gap_within = pd.read_csv(os.path.join(d, 'gap_within.csv')).values
        self.CI_within = pd.read_csv(os.path.join(d, 'CI_within.csv')).values

        # Load gap fraction file for hemispherical interceptance
        self.het_data_path = os.path.join(d, 'HET01_true_all.txt')

    def simulate(self, SZA=20, SAA=0, LAI=5, Crowndeepth=12.86, Height=20,
                 Height_c=6.87, dthr=0.86, bl=0.2, HotSpotPar=0.02,
                 FAVD=0.375, D=0, TypeLidf=2, LIDFa=57.3, LIDFb=0,
                 angle=25):
        """
        Run the SIF simulation.

        Parameters
        ----------
        SZA : float
            Sun Zenith Angle [degrees].
        SAA : float
            Sun Azimuth Angle [degrees].
        LAI : float
            Leaf Area Index [m2/m2].
        Crowndeepth : float
            Average crown depth [m].
        Height : float
            Canopy height [m].
        Height_c : float
            Crown center height [m].
        dthr : float
            Diameter to Height Ratio.
        bl : float
            Leaf width [m].
        HotSpotPar : float
            Hotspot parameter at leaf scale.
        FAVD : float
            Foliage Area Volume Density.
        D : float
            Ratio of diffuse to incoming irradiance.
        TypeLidf : int
            LIDF type: 1=two-parameter, 2=Campbell (single-parameter).
        LIDFa : float
            Leaf angle parameter.
        LIDFb : float
            Second leaf angle parameter (only used if TypeLidf=1).
        angle : int
            Number of simulation angles.

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'SRTE_Fs_fdir_all': Total SIF emission (nf x angle)
            - 'SRTE_Fs_fdir1': PS I SIF emission (nf x angle)
            - 'SRTE_Fs_fdir2': PS II SIF emission (nf x angle)
            - 'SRTE_RefAll': Reflectance (nb x angle)
            - 'wlf': SIF wavelength array
            - 'wls': Full wavelength array
            - 'va': View angle matrix
            - 'Qapar_all': Absorbed PAR
        """
        # 0. Initialize wavelengths
        wls = np.arange(400, 2562)   # (2162,) optical wavelength range [nm]
        wlf = np.arange(640, 851)    # (211,) SIF wavelength range [nm]
        wle = np.arange(400, 751)    # (351,) PAR wavelength range [nm]

        # Find intersection indices (0-based)
        _, iwlfi, _ = np.intersect1d(wls, wle, return_indices=True)
        _, iwlfo, _ = np.intersect1d(wls, wlf, return_indices=True)

        nb = len(wls)
        nf = len(iwlfo)
        ne = len(iwlfi)

        # 1. Setup view angles (principal plane)
        va = np.zeros((angle, 4))
        for t in range(12):
            va[t, 0] = 5.0 * (13 - (t + 1))   # VZA (60, 55, ..., 5)
            va[t, 1] = 0.0                      # VAA - Forward
        # va[12, :] = 0 (Nadir)
        for t in range(12):
            va[t + 13, 0] = 5.0 * (t + 1)      # VZA (5, 10, ..., 60)
            va[t + 13, 1] = 180.0               # VAA - Backward

        # 2. Gap fraction data
        # Index 12 (0-based) = Nadir, Index 8 = Solar angle (20 degrees)
        gap_H = self.gap_betw[12, 2]
        gap_H_within = self.gap_within[12, 2]
        gap_H_tot = self.gap_tot[12, 2]
        CI_H_within = self.CI_within[12, 2]

        gap_S = self.gap_betw[8, 2]
        gap_S_within = self.gap_within[8, 2]
        gap_S_tot = self.gap_tot[8, 2]

        # Scene structural parameters
        c1 = CI_H_within
        c2 = 1.0 - gap_H
        c = c1 * c2

        # Hemispherical interceptance from gap fraction data
        het_data = pd.read_csv(self.het_data_path, sep=r'\s+', skiprows=1, header=None).values
        iD = get_hemi_interceptance(het_data)

        # 3. LIDF setup
        if TypeLidf == 1:
            lidf, litab = dladgen(LIDFa, LIDFb)
        else:
            lidf, litab = campbell(LIDFa)

        # 4. Leaf and soil optics
        rho_l = np.zeros(nb)
        tau_l = np.zeros(nb)
        rho_l[:2101] = self.refl
        tau_l[:2101] = self.tran
        rs = self.rs

        # Structural parameters
        go_par = dthr * Crowndeepth

        # 5. Fluorescence matrices
        MfI = self.MfI_raw.copy()
        MbI = self.MbI_raw.copy()
        MfII = self.MfII_raw.copy()
        MbII = self.MbII_raw.copy()

        # Adjustment factor for Photosystem I
        MfI *= 1.29
        MbI *= 1.29

        # 6. Incident radiation
        Qins = self.result_SIP[:, 1].copy()  # Direct solar irradiance (column index 1)
        Qins = Qins * np.cos(np.deg2rad(SZA))
        Qind = np.zeros_like(Qins)            # Diffuse irradiance (set to 0)

        # 7. Core calculations
        # Calculate structural probabilities for each view angle
        data_p_rho = get_p_rho(
            LAI, SZA, SAA, va, angle, lidf,
            Height, Height_c, Crowndeepth,
            self.gap_tot, self.gap_betw, self.gap_within,
            gap_S, gap_S_tot, gap_S_within,
            go_par, c, HotSpotPar, iD, D
        )

        # RT with SIF (Photosystem I)
        Qfdir_1, Qfyld_1, Qapar_all, _ = get_rta(
            nb, nf, ne, iwlfi, iwlfo, angle, Qins, Qind,
            rho_l, tau_l, rs, data_p_rho,
            MfI=MfI, MbI=MbI, compute_sif=True
        )

        # RT with SIF (Photosystem II)
        Qfdir_2, Qfyld_2, _, _ = get_rta(
            nb, nf, ne, iwlfi, iwlfo, angle, Qins, Qind,
            rho_l, tau_l, rs, data_p_rho,
            MfI=MfII, MbI=MbII, compute_sif=True
        )

        # RT without SIF (background)
        Qpdir_all, _, Qpapar_all, Qpdir_bs = get_rta(
            nb, nf, ne, iwlfi, iwlfo, angle, Qins, Qind,
            rho_l, tau_l, rs, data_p_rho,
            compute_sif=False
        )

        # 8. Calculate SIF directional emission by subtracting background
        SRTE_Fs_fdir1 = (Qfdir_1 - Qpdir_all)[iwlfo, :] / np.pi
        SRTE_Fs_fdir2 = (Qfdir_2 - Qpdir_all)[iwlfo, :] / np.pi
        SRTE_Fs_fdir_all = SRTE_Fs_fdir1 + SRTE_Fs_fdir2

        # Reflectance
        denom = (Qins + Qind)[:, np.newaxis]
        denom[denom == 0] = 1.0  # Avoid division by zero
        SRTE_RefAll = Qpdir_all / denom

        return {
            'SRTE_Fs_fdir_all': SRTE_Fs_fdir_all,
            'SRTE_Fs_fdir1': SRTE_Fs_fdir1,
            'SRTE_Fs_fdir2': SRTE_Fs_fdir2,
            'SRTE_RefAll': SRTE_RefAll,
            'wlf': wlf,
            'wls': wls,
            'va': va,
            'Qapar_all': Qapar_all,
        }


def run_simulation(**kwargs):
    """
    Convenience function to run the Canopy-SIP-SIF simulation.

    Parameters
    ----------
    **kwargs
        All parameters accepted by CanopySIPSIFModel.simulate().
        Additionally accepts 'data_dir' for specifying the data directory.

    Returns
    -------
    result : dict
        Simulation results (see CanopySIPSIFModel.simulate).
    """
    data_dir = kwargs.pop('data_dir', None)
    model = CanopySIPSIFModel(data_dir=data_dir)
    return model.simulate(**kwargs)
