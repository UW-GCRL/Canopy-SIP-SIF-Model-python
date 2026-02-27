"""
Sunlit foliage and soil probabilities with direct hotspot input.
Kuusk (1995) formulation.
"""

import numpy as np
from scipy.integrate import quad


def _pso_function_hotspot(xl, K, k, CIs, CIo, LAI, q, dso):
    """Internal: Kuusk hotspot correlation function."""
    if dso != 0:
        alf = (dso / q) * 2.0 / (k + K)
        pso = np.exp(
            (K * CIo + k * CIs) * LAI * xl +
            np.sqrt(K * CIo * k * CIs) * LAI / alf * (1.0 - np.exp(xl * alf))
        )
    else:
        pso = np.exp(
            (K * CIo + k * CIs) * LAI * xl -
            np.sqrt(K * CIo * k * CIs) * LAI * xl
        )
    return pso


def sunshade_h(tts, tto, psi, Gs, Go, CIs, CIo, LAI, hotspot):
    """
    Calculate sunlit foliage and soil probabilities (with hotspot).

    Parameters
    ----------
    tts : float
        Sun zenith angle [degrees].
    tto : float
        View zenith angle [degrees].
    psi : float
        Relative azimuth angle [degrees].
    Gs : float
        G-function value in the solar direction.
    Go : float
        G-function value in the viewing direction.
    CIs : float
        Clumping index in the solar direction.
    CIo : float
        Clumping index in the viewing direction.
    LAI : float
        Leaf Area Index [m2/m2].
    hotspot : float
        Direct hotspot parameter.

    Returns
    -------
    kc : float
        Probability of viewing sunlit foliage.
    kg : float
        Probability of viewing sunlit soil.
    """
    deg2rad = np.pi / 180.0

    cos_tts = np.cos(tts * deg2rad)
    tan_tto = np.tan(tto * deg2rad)
    cos_tto = np.cos(tto * deg2rad)
    tan_tts = np.tan(tts * deg2rad)

    psi = abs(psi - 360.0 * round(psi / 360.0))

    if tts == tto and psi == 0:
        kc = 1.0 - np.exp(-Gs * CIs / cos_tts * LAI)
        kg = np.exp(-Gs * CIs / cos_tts * LAI)
    else:
        nl = 20
        x = np.arange(-1.0 / nl, -1.0 - 1e-10, -1.0 / nl)
        xl = np.concatenate(([0.0], x))
        dx = 1.0 / nl
        iLAI = LAI / nl

        q = hotspot

        dso = np.sqrt(tan_tts**2 + tan_tto**2 - 2.0 * tan_tts * tan_tto * np.cos(psi * deg2rad))

        k = Gs / cos_tts
        K = Go / cos_tto

        Ps = np.exp(k * xl * CIs * LAI)
        Po = np.exp(K * xl * CIo * LAI)

        Ps[:nl] = Ps[:nl] * (1.0 - np.exp(-k * CIs * LAI * dx)) / (k * CIs * LAI * dx)
        Po[:nl] = Po[:nl] * (1.0 - np.exp(-K * CIo * LAI * dx)) / (K * CIo * LAI * dx)

        Pso = np.zeros_like(Po)

        for j in range(len(xl)):
            result, _ = quad(
                _pso_function_hotspot,
                xl[j] - dx, xl[j],
                args=(K, k, CIs, CIo, LAI, q, dso)
            )
            Pso[j] = result / dx

        mask1 = Pso > Po
        Pso[mask1] = np.minimum(Po[mask1], Ps[mask1])
        mask2 = Pso > Ps
        Pso[mask2] = np.minimum(Po[mask2], Ps[mask2])

        kc = iLAI * CIo * K * np.sum(Pso[:nl])
        kg = Pso[nl]

    return kc, kg
