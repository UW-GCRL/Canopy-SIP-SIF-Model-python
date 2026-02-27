"""
Scattering phase function calculation.
"""

import numpy as np
from .volscat import volscat


def phase(tts, tto, psi, lidf):
    """
    Calculate the scattering phase function.

    Parameters
    ----------
    tts : float
        Sun zenith angle [degrees].
    tto : float
        View zenith angle [degrees].
    psi : float
        Relative azimuth angle [degrees].
    lidf : np.ndarray
        Leaf inclination distribution function (13 elements).

    Returns
    -------
    Gs : float
        G-function in the solar direction.
    Go : float
        G-function in the viewing direction.
    k : float
        Extinction coefficient in direction of sun.
    K : float
        Extinction coefficient in direction of observer.
    sob : float
        Weight of specular-to-directional back scatter coefficient.
    sof : float
        Weight of specular-to-directional forward scatter coefficient.
    """
    deg2rad = np.pi / 180.0

    litab = np.array([5., 15., 25., 35., 45., 55., 65., 75., 81., 83., 85., 87., 89.])

    cos_tts = np.cos(tts * deg2rad)
    cos_tto = np.cos(tto * deg2rad)

    psi = abs(psi - 360.0 * round(psi / 360.0))

    chi_s, chi_o, frho, ftau = volscat(tts, tto, psi, litab)

    ksli = chi_s / cos_tts
    koli = chi_o / cos_tto

    sobli = frho * np.pi / (cos_tts * cos_tto)
    sofli = ftau * np.pi / (cos_tts * cos_tto)

    k = np.dot(ksli, lidf)
    K = np.dot(koli, lidf)
    sob = np.dot(sobli, lidf)
    sof = np.dot(sofli, lidf)

    Go = K * cos_tto
    Gs = k * cos_tts

    return Gs, Go, k, K, sob, sof
