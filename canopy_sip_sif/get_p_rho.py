"""
GO + p-theory hub: calculates structural probabilities and 4-component gap fractions.

This core function combines Geometric-Optical (GO) theory and spectral invariants
(p-theory). It calculates the four scene components (sunlit/shaded crown,
sunlit/shaded background) and integrates hemispherical scattering probabilities
for each viewing angle.
"""

import sys
import numpy as np
from .phase import phase
from .cixy import cixy
from .get_hsf_go import get_hsf_go
from .sunshade_h import sunshade_h
from .sunshade_kt_he import sunshade_kt_he
from .hemi_single import hemi_single
from .hemi_dif import hemi_dif
from .hemi_hemi import hemi_hemi


def get_p_rho(LAI, SZA, SAA, va, angle, lidf, hc0, Height_c, Crowndeepth,
              gap_tot, gap_betw, gap_within, gap_S, gap_S_tot, gap_S_within,
              go_par, c, HotSpotPar, iD, D):
    """
    Calculate structural probabilities and 4-component GO gap fractions.

    Parameters
    ----------
    LAI : float
        Leaf Area Index of a single crown [m2/m2].
    SZA : float
        Sun Zenith Angle [degrees].
    SAA : float
        Sun Azimuth Angle [degrees].
    va : np.ndarray
        View angle matrix (N x 4), columns: [VZA, VAA, -, -].
    angle : int
        Number of viewing angles.
    lidf : np.ndarray
        Leaf Inclination Distribution Function vector.
    hc0 : float
        Structural parameter / canopy height indicator.
    Height_c : float
        Crown center height [m].
    Crowndeepth : float
        Crown depth [m].
    gap_tot : np.ndarray
        Total gap fraction matrix.
    gap_betw : np.ndarray
        Between-crown gap fraction matrix.
    gap_within : np.ndarray
        Within-crown gap fraction matrix.
    gap_S : float
        Between-crown gap fraction in solar direction.
    gap_S_tot : float
        Total gap fraction in solar direction.
    gap_S_within : float
        Within-crown gap fraction in solar direction.
    go_par : float
        GO theory hotspot parameter.
    c : float
        Canopy-level clumping index.
    HotSpotPar : float
        Leaf-scale hotspot parameter.
    iD : float
        Hemispherical interceptance (scene level).
    D : float
        Ratio of diffuse to incoming irradiance.

    Returns
    -------
    data_p_rho : np.ndarray
        A 27 x N matrix containing integrated scattering probabilities,
        transmittances, and 4-component gap fractions for each view angle.
    """
    data_p_rho = np.zeros((27, angle))

    # CI parameters (fixed at 1 for GO model)
    CIy1 = 1.0
    CIy2 = 1.0
    CIs = cixy(CIy1, CIy2, SZA)

    # Precompute angle-independent hemispherical integrals
    print("  Computing directional-hemispherical scattering...", flush=True)
    sob_vsla, sof_vsla, kgd_val = hemi_single(
        SZA, CIs, CIy1, CIy2, LAI, lidf, hc0
    )

    print("  Computing bi-hemispherical scattering...", flush=True)
    sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif_val = hemi_hemi(
        CIy1, CIy2, LAI, lidf, hc0
    )

    for t in range(angle):
        tts = SZA
        tto = va[t, 0]  # View zenith angle
        psi = va[t, 1]  # View azimuth angle

        # Ensure relative azimuth angle is wrapped correctly
        if psi > 180:
            psi = psi - 360
        psi = abs(psi)
        psi = abs(psi - 360.0 * round(psi / 360.0))

        CIo = cixy(CIy1, CIy2, tto)

        # 1. Directional gap fractions extraction
        gap_V_tot = gap_tot[t, 2]
        gap_V_betw = gap_betw[t, 2]
        Ps_dir_go = gap_S
        Pv_dir_go = gap_V_betw

        # 2. Calculate four GO components (Kc, Kt, Kg, Kz)
        Kg = Ps_dir_go * Pv_dir_go + get_hsf_go(
            go_par, tts, SAA, tto, psi, Ps_dir_go, Pv_dir_go, Height_c
        )
        Kz = Pv_dir_go - Kg
        Kct = 1.0 - Pv_dir_go

        delta_angle = (np.cos(np.deg2rad(tts)) * np.cos(np.deg2rad(tto)) +
                       np.sin(np.deg2rad(tts)) * np.sin(np.deg2rad(tto)) *
                       np.cos(np.deg2rad(psi - SAA)))
        phi_angle = np.rad2deg(np.arccos(np.clip(delta_angle, -1, 1)))
        delta_val = np.cos(np.deg2rad(phi_angle * (1.0 - np.sin(np.pi * c / 2.0))))

        if ((hc0 - Crowndeepth) < Crowndeepth) and (tto > tts) and (SAA == psi):
            Kc = Kct
        else:
            Kc = 0.5 * (1.0 + delta_val) * Kct
        Kt = Kct - Kc

        # 3. Crown-level scattering and transmittance parameters
        Ps_dir_inKz = gap_S_within

        Gs, Go, k, K, sob, sof = phase(tts, tto, psi, lidf)
        kc_val, kg_val = sunshade_h(tts, tto, psi, Gs, Go, CIs, CIo, LAI, HotSpotPar)
        kc_kt, kg_kt = sunshade_kt_he(tts, tto, psi, Gs, Go, CIs, CIo, LAI)

        # 4. p-theory parameters
        i0 = 1.0 - gap_S_tot
        i0 = D * iD + (1.0 - D) * i0
        iv = 1.0 - gap_V_tot
        tv = 1.0 - iv

        p = 1.0 - iD / LAI
        rho2 = iv / 2.0 / LAI
        rho_hemi2 = iD / 2.0 / LAI

        # Hemispherical-directional scattering (varies per view angle)
        sob_vsla_dif, sof_vsla_dif, kg_dif = hemi_dif(
            tto, CIo, CIy1, CIy2, LAI, lidf, hc0
        )

        print(f"  Angle {t+1}/{angle}: VZA={tto:.0f} done", flush=True)

        # 5. Pack variables into output vector
        data_p_rho[:, t] = [
            i0, iD, p, rho_hemi2, sob_vsla, sof_vsla, rho2, tv,
            sob, sof, kc_val, kc_kt, kg_val, kg_kt, K, kgd_val,
            sob_vsla_dif, sof_vsla_dif, kg_dif,
            sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif_val,
            Ps_dir_inKz, Kg, Kc, Kt, Kz
        ]

    return data_p_rho
