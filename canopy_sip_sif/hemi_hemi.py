"""
Bi-hemispherical single scattering integration.

Computes the bi-hemispherical (diffuse-diffuse) integration of scattering
and transmittance using 4D Gaussian quadrature (8 points per dimension).
"""

import numpy as np
from .phase import phase
from .cixy import cixy
from .sunshade import sunshade


# 8-point Gaussian quadrature nodes and weights
_XX = np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774,
                0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425])
_WW = np.array([0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345,
                0.3137066459, 0.3137066459, 0.3626837834, 0.3626837834])


def hemi_hemi(CIy1, CIy2, LAI, lidf, hc0):
    """
    Calculate bi-hemispherical single scattering and gap probabilities.

    Performs a 4D numerical integration over both the incident and scattered
    hemispheres using 8-point Gaussian quadrature per dimension.

    Parameters
    ----------
    CIy1 : float
        Parameter 1 for the angular dependence of the clumping index.
    CIy2 : float
        Parameter 2 for the angular dependence of the clumping index.
    LAI : float
        Leaf Area Index [m2/m2].
    lidf : np.ndarray
        Leaf Inclination Distribution Function vector.
    hc0 : float
        Hotspot / structural parameter related to canopy height.

    Returns
    -------
    sob_vsla : float
        Integrated backward single scattering contribution (bi-hemispherical).
    sof_vsla : float
        Integrated forward single scattering contribution (bi-hemispherical).
    kgd_dif : float
        Integrated bi-hemispherical gap probability.
    """
    # Integration limits
    conv1_mL = np.pi / 4.0  # Incident zenith
    conv2_mL = np.pi / 4.0
    conv1_nL = np.pi         # Incident azimuth
    conv2_nL = np.pi
    conv1_tL = np.pi / 4.0  # Scattered zenith
    conv2_tL = np.pi / 4.0
    conv1_pL = np.pi         # Scattered azimuth
    conv2_pL = np.pi

    sum_mL = 0.0
    sum_mL_f = 0.0
    sum_mL_g = 0.0

    # Loop 1: Incident zenith angle
    for m in range(8):
        neword_mL = conv1_mL * _XX[m] + conv2_mL
        mu_mL = np.cos(neword_mL)
        sin_mL = np.sin(neword_mL)

        sum_nL = 0.0
        sum_nL_f = 0.0
        sum_nL_g = 0.0

        # Loop 2: Incident azimuth angle
        for n in range(8):
            neword_nL = conv1_nL * _XX[n] + conv2_nL

            sum_tL = 0.0
            sum_tL_f = 0.0
            sum_tL_g = 0.0

            # Loop 3: Scattered zenith angle
            for i in range(8):
                neword_tL = conv1_tL * _XX[i] + conv2_tL
                mu_tL = np.cos(neword_tL)
                sin_tL = np.sin(neword_tL)

                sum_pL = 0.0
                sum_pL_f = 0.0
                sum_pL_g = 0.0

                # Loop 4: Scattered azimuth angle
                for j in range(8):
                    neword_pL = conv1_pL * _XX[j] + conv2_pL

                    tts = neword_mL * 180.0 / np.pi
                    tto = neword_tL * 180.0 / np.pi

                    # Relative azimuth angle
                    psi = abs(neword_nL * 180.0 / np.pi - neword_pL * 180.0 / np.pi)
                    psi = abs(psi - 360.0 * round(psi / 360.0))

                    Gs, Go, k, K, sob, sof = phase(tts, tto, psi, lidf)
                    CIs_val = cixy(CIy1, CIy2, tts)
                    CIo_val = cixy(CIy1, CIy2, tto)
                    kca, kga = sunshade(tts, tto, psi, Gs, Go, CIs_val, CIo_val, LAI, hc0)

                    sum_pL += _WW[j] * sob * kca / K / np.pi
                    sum_pL_f += _WW[j] * sof * kca / K / np.pi
                    sum_pL_g += _WW[j] * kga / np.pi

                sum_pL *= conv1_pL
                sum_tL += _WW[i] * mu_tL * sin_tL * sum_pL

                sum_pL_f *= conv1_pL
                sum_tL_f += _WW[i] * mu_tL * sin_tL * sum_pL_f

                sum_pL_g *= conv1_pL
                sum_tL_g += _WW[i] * mu_tL * sin_tL * sum_pL_g

            sum_tL *= conv1_tL
            sum_nL += _WW[n] * sum_tL / np.pi

            sum_tL_f *= conv1_tL
            sum_nL_f += _WW[n] * sum_tL_f / np.pi

            sum_tL_g *= conv1_tL
            sum_nL_g += _WW[n] * sum_tL_g / np.pi

        sum_nL *= conv1_nL
        sum_mL += _WW[m] * mu_mL * sin_mL * sum_nL

        sum_nL_f *= conv1_nL
        sum_mL_f += _WW[m] * mu_mL * sin_mL * sum_nL_f

        sum_nL_g *= conv1_nL
        sum_mL_g += _WW[m] * mu_mL * sin_mL * sum_nL_g

    sob_vsla = sum_mL * conv1_mL
    sof_vsla = sum_mL_f * conv1_mL
    kgd_dif = sum_mL_g * conv1_mL

    return sob_vsla, sof_vsla, kgd_dif
