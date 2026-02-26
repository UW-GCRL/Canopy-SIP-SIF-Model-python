"""
Bidirectional between-crown gap probability covariance (hotspot effect).
"""

import numpy as np


def get_hsf_go(par, SZA, SAA, VZA, VAA, Ps_dir_go, Pv_dir_go, z):
    """
    Calculate the hotspot enhancement term for between-crown gap probability.

    The total bidirectional gap probability is:
        KG = Ps * Pv + bg
    where bg = Y * sqrt(Ps * Pv * (1 - Ps) * (1 - Pv)).

    Parameters
    ----------
    par : float
        Crown size or hotspot parameter [m].
    SZA : float
        Sun Zenith Angle [degrees].
    SAA : float
        Sun Azimuth Angle [degrees].
    VZA : float
        View Zenith Angle [degrees].
    VAA : float
        View Azimuth Angle [degrees].
    Ps_dir_go : float
        Directional between-crown gap probability in solar direction.
    Pv_dir_go : float
        Directional between-crown gap probability in view direction.
    z : float
        Crown center height [m].

    Returns
    -------
    bg : float
        The covariance term of the bidirectional gap probability.
    """
    SZA_rad = np.deg2rad(SZA)
    SAA_rad = np.deg2rad(SAA)
    VZA_rad = np.deg2rad(VZA)
    VAA_rad = np.deg2rad(VAA)

    f1 = np.sqrt(Ps_dir_go * Pv_dir_go * (1.0 - Ps_dir_go) * (1.0 - Pv_dir_go))

    cosgamma = (np.cos(SZA_rad) * np.cos(VZA_rad) +
                np.sin(SZA_rad) * np.sin(VZA_rad) * np.cos(VAA_rad - SAA_rad))

    delta = np.sqrt(
        1.0 / np.cos(SZA_rad)**2 +
        1.0 / np.cos(VZA_rad)**2 -
        2.0 * cosgamma / (np.cos(SZA_rad) * np.cos(VZA_rad))
    )

    if delta < 0.00001:
        delta = 0.00001

    Y = np.exp(-delta / par * z)

    bg = f1 * Y

    return bg
