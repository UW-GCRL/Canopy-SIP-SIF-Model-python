"""
Directional-hemispherical single scattering integration.

Computes hemispherical integration of scattering and transmittance
for a specific solar incidence angle using 8-point Gaussian quadrature.
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


def hemi_single(tts, CIs, CIy1, CIy2, LAI, lidf, hc0):
    """
    Calculate directional-hemispherical single scattering and gap probabilities.

    Integrates over the upper hemisphere (zenith: 0 to pi/2, azimuth: 0 to 2*pi)
    using 8-point Gaussian quadrature for a fixed solar direction.

    Parameters
    ----------
    tts : float
        Solar zenith angle [degrees].
    CIs : float
        Clumping index at the solar direction.
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
        Integrated backward single scattering contribution.
    sof_vsla : float
        Integrated forward single scattering contribution.
    kgd : float
        Integrated gap probability (directional-hemispherical transmittance).
    """
    # Zenith angle integration limits: 0 to pi/2
    conv1_tL = np.pi / 4.0
    conv2_tL = np.pi / 4.0

    # Azimuth angle integration limits: 0 to 2*pi
    conv1_pL = np.pi
    conv2_pL = np.pi

    sum_tL = 0.0
    sum_tL_f = 0.0
    sum_tL_g = 0.0

    for i in range(8):
        neword_tL = conv1_tL * _XX[i] + conv2_tL
        mu_tL = np.cos(neword_tL)
        sin_tL = np.sin(neword_tL)

        sum_pL = 0.0
        sum_pL_f = 0.0
        sum_pL_g = 0.0

        for j in range(8):
            neword_pL = conv1_pL * _XX[j] + conv2_pL

            tta = neword_tL * 180.0 / np.pi
            psia = neword_pL * 180.0 / np.pi

            Gs, Ga, k, Ka, sob, sof = phase(tts, tta, psia, lidf)
            CIa = cixy(CIy1, CIy2, tta)
            kca, kga = sunshade(tts, tta, psia, Gs, Ga, CIs, CIa, LAI, hc0)

            sum_pL += _WW[j] * sob * kca / Ka / np.pi
            sum_pL_f += _WW[j] * sof * kca / Ka / np.pi
            sum_pL_g += _WW[j] * kga / np.pi

        sum_pL *= conv1_pL
        sum_tL += _WW[i] * mu_tL * sin_tL * sum_pL

        sum_pL_f *= conv1_pL
        sum_tL_f += _WW[i] * mu_tL * sin_tL * sum_pL_f

        sum_pL_g *= conv1_pL
        sum_tL_g += _WW[i] * mu_tL * sin_tL * sum_pL_g

    sob_vsla = sum_tL * conv1_tL
    sof_vsla = sum_tL_f * conv1_tL
    kgd = sum_tL_g * conv1_tL

    return sob_vsla, sof_vsla, kgd
