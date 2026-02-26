"""
Hemispherical interceptance calculation.

Calculates hemispherical interceptance from gap fraction data
using trapezoidal integration.
"""

import numpy as np


def get_hemi_interceptance(gap_tot):
    """
    Calculate hemispherical interceptance from gap fraction data.

    Parameters
    ----------
    gap_tot : np.ndarray
        Gap fraction data array where column 3 (index 2) contains
        gap fractions at zenith angles [0, 5, 10, ..., 85, 89] degrees.

    Returns
    -------
    iD : float
        Hemispherical interceptance.
    """
    za = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                   50, 55, 60, 65, 70, 75, 80, 85, 89], dtype=float)

    iv = np.ones_like(za)

    for i in range(len(za)):
        p = gap_tot[i, 2]  # Column index 2 (0-based) = column 3 (1-based)
        iv[i] = (1.0 - p) * np.sin(np.deg2rad(2.0 * za[i]))

    iD = np.trapz(iv, np.deg2rad(za))

    return iD
