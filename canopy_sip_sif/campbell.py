"""
Campbell leaf angle distribution function.

Computation of the leaf angle distribution function value (freq).
Ellipsoidal distribution function characterised by the average leaf
inclination angle in degree (ala).
Campbell 1986.
"""

import numpy as np


def campbell(ala):
    """
    Compute leaf angle distribution using the Campbell (1986) ellipsoidal model.

    Parameters
    ----------
    ala : float
        Average leaf inclination angle in degrees.

    Returns
    -------
    freq0 : np.ndarray
        Leaf angle distribution frequencies (13 elements).
    litab : np.ndarray
        Leaf inclination angle centers (13 elements, degrees).
    """
    tx1 = np.array([10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88., 90.])
    tx2 = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88.])

    litab = (tx2 + tx1) / 2.0
    n = len(litab)
    tl1 = tx1 * (np.pi / 180.0)
    tl2 = tx2 * (np.pi / 180.0)
    excent = np.exp(-1.6184e-5 * ala**3 + 2.1145e-3 * ala**2 - 1.2390e-1 * ala + 3.2491)

    freq = np.zeros(n)
    for i in range(n):
        x1 = excent / np.sqrt(1.0 + excent**2 * np.tan(tl1[i])**2)
        x2 = excent / np.sqrt(1.0 + excent**2 * np.tan(tl2[i])**2)
        if excent == 1:
            freq[i] = abs(np.cos(tl1[i]) - np.cos(tl2[i]))
        else:
            alpha = excent / np.sqrt(abs(1.0 - excent**2))
            alpha2 = alpha**2
            x12 = x1**2
            x22 = x2**2
            if excent > 1:
                alpx1 = np.sqrt(alpha2 + x12)
                alpx2 = np.sqrt(alpha2 + x22)
                dum = x1 * alpx1 + alpha2 * np.log(x1 + alpx1)
                freq[i] = abs(dum - (x2 * alpx2 + alpha2 * np.log(x2 + alpx2)))
            else:
                almx1 = np.sqrt(alpha2 - x12)
                almx2 = np.sqrt(alpha2 - x22)
                dum = x1 * almx1 + alpha2 * np.arcsin(x1 / alpha)
                freq[i] = abs(dum - (x2 * almx2 + alpha2 * np.arcsin(x2 / alpha)))

    sum0 = np.sum(freq)
    freq0 = freq / sum0

    return freq0, litab
