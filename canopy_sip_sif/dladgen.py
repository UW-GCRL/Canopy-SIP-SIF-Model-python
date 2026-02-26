"""
Two-parameter leaf angle distribution generation (dladgen).
"""

import numpy as np


def dcum(a, b, t):
    """
    Cumulative leaf angle distribution function.

    Parameters
    ----------
    a, b : float
        Distribution parameters.
    t : float
        Angle in degrees.

    Returns
    -------
    f : float
        Cumulative distribution value.
    """
    rd = np.pi / 180.0
    if a >= 1:
        f = 1.0 - np.cos(rd * t)
    else:
        eps = 1e-8
        delx = 1.0
        x = 2.0 * rd * t
        p = x
        while delx >= eps:
            y = a * np.sin(x) + 0.5 * b * np.sin(2.0 * x)
            dx = 0.5 * (y - x + p)
            x = x + dx
            delx = abs(dx)
        f = (2.0 * y + p) / np.pi
    return f


def dladgen(a, b):
    """
    Generate leaf angle distribution using two-parameter model.

    Parameters
    ----------
    a, b : float
        Distribution parameters.
        Requirement: |a| + |b| < 1.

    Returns
    -------
    freq : np.ndarray
        Leaf angle distribution frequencies (13 elements).
    litab : np.ndarray
        Leaf inclination angle centers (13 elements, degrees).
    """
    litab = np.array([5., 15., 25., 35., 45., 55., 65., 75., 81., 83., 85., 87., 89.])
    freq = np.zeros(13)

    for i1 in range(8):
        t = (i1 + 1) * 10.0
        freq[i1] = dcum(a, b, t)

    for i2 in range(8, 12):
        t = 80.0 + (i2 - 7) * 2.0
        freq[i2] = dcum(a, b, t)

    freq[12] = 1.0
    for i in range(12, 0, -1):
        freq[i] = freq[i] - freq[i - 1]

    return freq, litab
