"""
Volume scattering functions.

Volscatt version 2.
Created by W. Verhoef
Edited by Joris Timmermans to MATLAB nomenclature.
Translated to Python.
"""

import numpy as np


def volscat(tts, tto, psi, ttli):
    """
    Volume scattering function.

    Parameters
    ----------
    tts : float
        Sun zenith angle in degrees.
    tto : float
        Observation zenith angle in degrees.
    psi : float
        Difference of azimuth angle between solar and viewing position.
    ttli : np.ndarray
        Leaf inclination array (degrees).

    Returns
    -------
    chi_s, chi_o, frho, ftau : np.ndarray
        Scattering coefficients.
    """
    deg2rad = np.pi / 180.0
    nli = len(ttli)

    psi_rad = psi * deg2rad * np.ones(nli)

    cos_psi = np.cos(psi * deg2rad)

    cos_ttli = np.cos(ttli * deg2rad)
    sin_ttli = np.sin(ttli * deg2rad)

    cos_tts = np.cos(tts * deg2rad)
    sin_tts = np.sin(tts * deg2rad)

    cos_tto = np.cos(tto * deg2rad)
    sin_tto = np.sin(tto * deg2rad)

    Cs = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts

    Co = cos_ttli * cos_tto
    So = sin_ttli * sin_tto

    As = np.maximum(Ss, Cs)
    Ao = np.maximum(So, Co)

    bts = np.arccos(-Cs / As)
    bto = np.arccos(-Co / Ao)

    chi_o = (2.0 / np.pi) * ((bto - np.pi / 2.0) * Co + np.sin(bto) * So)
    chi_s = (2.0 / np.pi) * ((bts - np.pi / 2.0) * Cs + np.sin(bts) * Ss)

    delta1 = np.abs(bts - bto)
    delta2 = np.pi - np.abs(bts + bto - np.pi)

    Tot = psi_rad + delta1 + delta2

    bt1 = np.minimum(psi_rad, delta1)
    bt3 = np.maximum(psi_rad, delta2)
    bt2 = Tot - bt1 - bt3

    T1 = 2.0 * Cs * Co + Ss * So * cos_psi
    T2 = np.sin(bt2) * (2.0 * As * Ao + Ss * So * np.cos(bt1) * np.cos(bt3))

    Jmin = bt2 * T1 - T2
    Jplus = (np.pi - bt2) * T1 + T2

    frho = Jplus / (2.0 * np.pi**2)
    ftau = -Jmin / (2.0 * np.pi**2)

    frho = np.maximum(np.zeros(nli), frho)
    ftau = np.maximum(np.zeros(nli), ftau)

    return chi_s, chi_o, frho, ftau
